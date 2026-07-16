"""Compact Vision Transformer implemented with basic MindSpore layers."""

from __future__ import annotations

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Normal, TruncatedNormal, initializer


class PatchEmbedding(nn.Cell):
    """Convert an image into patch tokens."""

    def __init__(self, image_size: int = 28, patch_size: int = 4, in_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=True,
            weight_init=TruncatedNormal(sigma=0.02),
        )
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.proj(x)
        batch_size = x.shape[0]
        x = self.reshape(x, (batch_size, x.shape[1], -1))
        x = self.transpose(x, (0, 2, 1))
        return x


class MultiHeadSelfAttention(nn.Cell):
    """Multi-head self-attention for token sequences."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Dense(embed_dim, embed_dim * 3)
        self.proj = nn.Dense(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(p=dropout)
        self.proj_drop = nn.Dropout(p=dropout)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = ops.BatchMatMul()

    def construct(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        attn = self.matmul(q, self.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.matmul(attn, v)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (batch_size, seq_len, self.embed_dim))
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Cell):
    """Transformer feed-forward block."""

    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Dense(mlp_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

    def construct(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Cell):
    """Pre-norm Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm((embed_dim,))
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm((embed_dim,))
        self.mlp = Mlp(embed_dim, mlp_dim, dropout)

    def construct(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTMNIST(nn.Cell):
    """Small ViT classifier for 28x28 MNIST images."""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_tokens = self.patch_embed.num_patches + 1

        self.cls_token = Parameter(initializer(Normal(sigma=0.02), (1, 1, embed_dim), ms.float32), name="cls_token")
        self.pos_embed = Parameter(initializer(Normal(sigma=0.02), (1, num_tokens, embed_dim), ms.float32), name="pos_embed")
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.SequentialCell(
            [TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm((embed_dim,))
        self.head = nn.Dense(embed_dim, num_classes)
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.tile(self.cls_token, (batch_size, 1, 1))
        x = self.concat((cls_tokens, x))
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_feature = x[:, 0]
        return self.head(cls_feature)


def vit_mnist(**kwargs) -> ViTMNIST:
    return ViTMNIST(**kwargs)
