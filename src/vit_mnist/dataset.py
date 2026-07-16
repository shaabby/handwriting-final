"""MNIST dataset utilities for MindSpore."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore import dtype as mstype


MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


RAW_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def _remove_mnist_archives(data_path: Path) -> None:
    for filename in MNIST_URLS:
        archive_path = data_path / filename
        if archive_path.exists():
            archive_path.unlink()


def prepare_mnist(data_dir: str | Path) -> Path:
    """Download and extract MNIST IDX files when they are missing."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if all((data_path / name).exists() for name in RAW_FILES):
        _remove_mnist_archives(data_path)
        return data_path

    for filename, url in MNIST_URLS.items():
        gz_path = data_path / filename
        raw_path = data_path / filename.removesuffix(".gz")

        if not raw_path.exists():
            if not gz_path.exists():
                print(f"Downloading {filename} ...")
                urllib.request.urlretrieve(url, gz_path)
            print(f"Extracting {filename} ...")
            with gzip.open(gz_path, "rb") as src, raw_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            gz_path.unlink()

    missing = [name for name in RAW_FILES if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing MNIST files: {missing}")
    _remove_mnist_archives(data_path)
    return data_path


def _preprocess(dataset: ds.Dataset, batch_size: int, training: bool) -> ds.Dataset:
    image_ops = [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW(),
    ]
    label_ops = [transforms.TypeCast(mstype.int32)]

    dataset = dataset.map(image_ops, input_columns="image")
    dataset = dataset.map(label_ops, input_columns="label")
    if training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, drop_remainder=training)
    return dataset


def create_datasets(
    data_dir: str | Path,
    batch_size: int,
    train_size: int = 54000,
    val_size: int = 6000,
    num_parallel_workers: int = 4,
) -> tuple[ds.Dataset, ds.Dataset, ds.Dataset]:
    """Create train, validation, and test datasets with fixed MNIST split."""

    data_path = prepare_mnist(data_dir)
    full_train = ds.MnistDataset(
        str(data_path),
        usage="train",
        shuffle=False,
        num_parallel_workers=num_parallel_workers,
    )
    train_ds, val_ds = full_train.split([train_size, val_size], randomize=False)
    test_ds = ds.MnistDataset(
        str(data_path),
        usage="test",
        shuffle=False,
        num_parallel_workers=num_parallel_workers,
    )

    return (
        _preprocess(train_ds, batch_size, training=True),
        _preprocess(val_ds, batch_size, training=False),
        _preprocess(test_ds, batch_size, training=False),
    )
