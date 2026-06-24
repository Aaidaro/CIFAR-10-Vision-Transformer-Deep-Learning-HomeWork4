"""Vision Transformer implementation written from basic PyTorch modules.

The code avoids using ready-made ViT implementations. Patch projection, class token,
positional embeddings, multi-head self-attention, Transformer encoder blocks, and
classification head are implemented explicitly for the assignment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of flattened patch embeddings.

    Args:
        image_size: Height/width of the square input image.
        patch_size: Height/width of each square patch.
        in_channels: Number of image channels.
        embed_dim: Transformer embedding dimension.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_channels * self.patch_size * self.patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings with shape ``[batch, num_patches, embed_dim]``."""
        batch, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Expected images of size {self.image_size}x{self.image_size}, got {height}x{width}")
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch, self.num_patches, channels * p * p)
        return self.proj(patches)


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention layer.

    This layer returns both attention probabilities and pre-softmax similarity
    scores so that CLS-to-patch attention maps can be visualized later.
    """

    def __init__(self, embed_dim: int, num_heads: int, attention_dropout: float = 0.0, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute self-attention.

        Args:
            x: Token embeddings with shape ``[batch, seq_len, embed_dim]``.

        Returns:
            ``(output, attention_probs, attention_scores)`` where attention tensors
            have shape ``[batch, heads, seq_len, seq_len]``.
        """
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attention = scores.softmax(dim=-1)
        attention = self.attn_drop(attention)

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_drop(self.out(out))
        return out, attention, scores


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block used inside ViT.

    Args:
        embed_dim: Token embedding size.
        num_heads: Number of attention heads.
        mlp_hidden_dim: Width of the hidden linear layer in the feed-forward MLP.
        dropout: Dropout probability for MLP and residual projections.
        attention_dropout: Dropout probability applied to attention probabilities.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one Transformer encoder block and expose attention tensors."""
        attn_out, attention, scores = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attention, scores


class VisionTransformer(nn.Module):
    """Configurable Vision Transformer classifier for CIFAR-10.

    Args mirror the assignment table and case studies. The model uses a learnable
    CLS token; its final representation is fed into a linear classification head.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_hidden_dim, dropout, attention_dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize learnable parameters with ViT-style truncated normal values."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Run the ViT classifier.

        Args:
            x: Image batch with shape ``[batch, channels, image_size, image_size]``.
            return_attention: If true, also returns per-layer attention probabilities
                and QK similarity scores.

        Returns:
            Logits when ``return_attention=False``. Otherwise returns
            ``(logits, attentions, attention_scores)``.
        """
        batch = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attentions: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        for block in self.blocks:
            x, attention, scores = block(x)
            if return_attention:
                attentions.append(attention.detach())
                scores_list.append(scores.detach())

        x = self.norm(x)
        logits = self.head(x[:, 0])
        if return_attention:
            return logits, attentions, scores_list
        return logits


def build_model(model_config: Dict) -> VisionTransformer:
    """Instantiate ``VisionTransformer`` from the ``model`` section of a config."""
    return VisionTransformer(
        image_size=int(model_config["image_size"]),
        patch_size=int(model_config["patch_size"]),
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", 10)),
        embed_dim=int(model_config["embed_dim"]),
        depth=int(model_config["depth"]),
        num_heads=int(model_config["num_heads"]),
        mlp_hidden_dim=int(model_config["mlp_hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_vit_macs(model_config: Dict) -> int:
    """Estimate multiply-accumulate operations for one forward pass.

    This is a simple analytical estimate for comparing cases; it excludes small
    operations such as activations and normalization.
    """
    image_size = int(model_config["image_size"])
    patch_size = int(model_config["patch_size"])
    in_channels = int(model_config.get("in_channels", 3))
    embed_dim = int(model_config["embed_dim"])
    depth = int(model_config["depth"])
    mlp_hidden_dim = int(model_config["mlp_hidden_dim"])
    num_classes = int(model_config.get("num_classes", 10))

    patches = (image_size // patch_size) ** 2
    seq_len = patches + 1
    patch_macs = patches * (patch_size * patch_size * in_channels) * embed_dim
    attn_proj_macs = 4 * seq_len * embed_dim * embed_dim
    attn_matrix_macs = 2 * seq_len * seq_len * embed_dim
    mlp_macs = 2 * seq_len * embed_dim * mlp_hidden_dim
    head_macs = embed_dim * num_classes
    return int(patch_macs + depth * (attn_proj_macs + attn_matrix_macs + mlp_macs) + head_macs)
