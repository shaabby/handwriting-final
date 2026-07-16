"""Train and evaluate ViT or CNN models on MNIST with MindSpore."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context

from src.vit_mnist.cnn import cnn_mnist
from src.vit_mnist.dataset import create_datasets
from src.vit_mnist.model import vit_mnist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT or CNN models on MNIST.")
    parser.add_argument("--model", default="vit", choices=["vit", "cnn"], help="Model architecture to train.")
    parser.add_argument("--data-dir", default="./data/MNIST", help="Directory containing or receiving MNIST IDX files.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for checkpoints and reports.")
    parser.add_argument("--device-target", default="Ascend", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def set_runtime(args: argparse.Namespace) -> None:
    ms.set_seed(args.seed)
    mode = context.GRAPH_MODE if args.mode == "GRAPH" else context.PYNATIVE_MODE
    context.set_context(mode=mode, device_target=args.device_target)


def accuracy_from_logits(logits: Tensor, labels: Tensor) -> tuple[int, int]:
    preds = ops.Argmax(axis=1)(logits)
    correct = int(ops.Equal()(preds, labels).asnumpy().sum())
    total = int(labels.shape[0])
    return correct, total


def train_one_epoch(train_ds, net, loss_fn, optimizer) -> tuple[float, float]:
    net.set_train(True)

    def forward_fn(images, labels):
        logits = net(images)
        loss = loss_fn(logits, labels)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    step_count = 0

    for images, labels in train_ds.create_tuple_iterator():
        (loss, logits), grads = grad_fn(images, labels)
        optimizer(grads)
        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += float(loss.asnumpy())
        total_correct += correct
        total_seen += seen
        step_count += 1

    return total_loss / max(step_count, 1), total_correct / max(total_seen, 1)


def evaluate(dataset, net, loss_fn) -> tuple[float, float, np.ndarray]:
    net.set_train(False)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    step_count = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    for images, labels in dataset.create_tuple_iterator():
        logits = net(images)
        loss = loss_fn(logits, labels)
        preds = ops.Argmax(axis=1)(logits).asnumpy()
        label_np = labels.asnumpy()

        total_loss += float(loss.asnumpy())
        total_correct += int((preds == label_np).sum())
        total_seen += int(label_np.shape[0])
        step_count += 1
        for true_label, pred_label in zip(label_np, preds):
            confusion[int(true_label), int(pred_label)] += 1

    return total_loss / max(step_count, 1), total_correct / max(total_seen, 1), confusion


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict]) -> None:
    epochs = [row["epoch"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    val_acc = [row["val_acc"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train acc")
    axes[1].plot(epochs, val_acc, label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(output_dir: Path, confusion: np.ndarray) -> None:
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def create_model(name: str) -> nn.Cell:
    if name == "vit":
        return vit_mnist()
    if name == "cnn":
        return cnn_mnist()
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    args = parse_args()
    set_runtime(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = create_datasets(
        args.data_dir,
        batch_size=args.batch_size,
        num_parallel_workers=args.num_workers,
    )

    net = create_model(args.model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.AdamWeightDecay(
        params=net.trainable_params(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_epoch = 0
    best_ckpt = output_dir / f"best_{args.model}_mnist.ckpt"
    metrics = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(train_ds, net, loss_fn, optimizer)
        val_loss, val_acc, _ = evaluate(val_ds, net, loss_fn)
        epoch_time = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        }
        metrics.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ms.save_checkpoint(net, str(best_ckpt))
            print(f"Saved best checkpoint: epoch={best_epoch}, val_acc={best_val_acc:.4f}")

        save_metrics_csv(output_dir / "metrics.csv", metrics)
        plot_curves(output_dir / "curves.png", metrics)

    ms.load_checkpoint(str(best_ckpt), net)
    test_loss, test_acc, confusion = evaluate(test_ds, net, loss_fn)
    save_confusion_matrix(output_dir, confusion)

    summary = (
        f"Model: {args.model}\n"
        f"Best epoch: {best_epoch}\n"
        f"Best val accuracy: {best_val_acc:.6f}\n"
        f"Test loss: {test_loss:.6f}\n"
        f"Test accuracy: {test_acc:.6f}\n"
        f"Checkpoint: {best_ckpt}\n"
    )
    (output_dir / "result_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
