"""Small CNN baseline for MNIST classification."""

from __future__ import annotations

import mindspore.nn as nn
import mindspore.ops as ops


class CNNMNIST(nn.Cell):
    """LeNet-style CNN baseline for 28x28 MNIST images."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(1, 32, kernel_size=3, pad_mode="same", has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, pad_mode="same", has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(128, num_classes),
        )

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


def cnn_mnist(**kwargs) -> CNNMNIST:
    return CNNMNIST(**kwargs)
