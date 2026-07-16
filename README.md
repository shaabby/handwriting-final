# MNIST Handwriting Recognition

This project implements compact ViT and CNN models in MindSpore for MNIST digit
classification on a single Ascend device.

## Files

- `src/vit_mnist/dataset.py`: MNIST download, split, preprocessing, and dataloaders.
- `src/vit_mnist/model.py`: Patch embedding, multi-head self-attention,
  Transformer encoder, and ViT classifier.
- `src/vit_mnist/cnn.py`: LeNet-style CNN baseline.
- `train.py`: training, validation, checkpointing, curve plotting, confusion
  matrix, and final test evaluation.

## Run on Huawei Cloud Ascend

Use the platform-provided Python and MindSpore environment directly.

```bash
python train.py --model vit --data-dir ./data/MNIST --output-dir ./outputs/vit --device-target Ascend
python train.py --model cnn --data-dir ./data/MNIST --output-dir ./outputs/cnn --device-target Ascend
```

The script downloads MNIST if the raw files are missing. If the cloud image has
no external network access, place the extracted MNIST files under
`./data/MNIST`:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

## Default Experiment

```text
Train:      54,000 images
Val:         6,000 images
Test:       10,000 images
Batch size:   128
Epochs:        30
Optimizer: AdamW
LR:          1e-3
Weight decay: 1e-2
```

ViT configuration:

```text
patch_size = 4
embed_dim = 64
num_heads = 4
num_layers = 3
mlp_dim = 128
dropout = 0.1
num_classes = 10
```

CNN configuration:

```text
1x28x28
Conv 3x3, 32 channels
ReLU
MaxPool 2x2
Conv 3x3, 64 channels
ReLU
MaxPool 2x2
Flatten
Dense 128
ReLU
Dropout 0.1
Dense 10
```

## Outputs

`train.py` writes the following files to `--output-dir`:

- `best_vit_mnist.ckpt` or `best_cnn_mnist.ckpt`
- `metrics.csv`
- `curves.png`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `result_summary.txt`
