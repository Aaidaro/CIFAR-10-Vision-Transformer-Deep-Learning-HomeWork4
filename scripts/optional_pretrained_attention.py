"""Optional Question 2.2(c): attention maps for torchvision's pretrained vit-b-16.

The main homework implementation does not depend on torchvision. This file is kept
separate because the question explicitly allows using pretrained ``vit_b_16`` for
the optional comparison.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import CIFAR10_CLASSES, load_cifar10_arrays
from utils.config import prepare_config
from utils.visualization import plot_attention_maps


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    """Normalize an image tensor with ImageNet statistics."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def _select_raw_test_image(config: Dict, target_class: Optional[int], index: Optional[int]) -> tuple[torch.Tensor, int, int]:
    """Return a raw CIFAR test image resized to 224 for pretrained ViT-B/16."""
    _, _, test_images, test_labels = load_cifar10_arrays(config["data"]["data_dir"])
    if index is not None:
        chosen = int(index)
    elif target_class is not None:
        matches = [i for i, label in enumerate(test_labels.tolist()) if int(label) == int(target_class)]
        chosen = matches[0]
    else:
        chosen = 0
    image = torch.from_numpy(test_images[chosen].copy()).permute(2, 0, 1).float().div(255.0)
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return image, int(test_labels[chosen]), chosen


def _compute_attention_from_mha(mha: torch.nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    """Compute attention probabilities from a torchvision MHA module input.

    Args:
        mha: ``nn.MultiheadAttention`` module inside a torchvision encoder block.
        x: Normalized token tensor with shape ``[batch, seq, embed_dim]``.

    Returns:
        Attention probabilities with shape ``[batch, heads, seq, seq]``.
    """
    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    embed_dim = mha.embed_dim
    q_w, k_w, _ = weight.split(embed_dim, dim=0)
    q_b, k_b, _ = bias.split(embed_dim, dim=0) if bias is not None else (None, None, None)
    q = F.linear(x, q_w, q_b)
    k = F.linear(x, k_w, k_b)
    batch, seq_len, _ = q.shape
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    return scores.softmax(dim=-1)


def pretrained_vit_attention(config_path: str, layers: Iterable[int], target_class=None, index=None) -> Path:
    """Save CLS attention maps for pretrained torchvision ViT-B/16 layers."""
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torchvision to run optional_pretrained_attention.py") from exc

    cfg = prepare_config(config_path=config_path, case="base")
    raw_image, label, chosen = _select_raw_test_image(cfg, target_class, index)
    input_image = _imagenet_normalize(raw_image).unsqueeze(0).to(_device())

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(_device()).eval()
    requested = [int(layer) for layer in layers]

    # Recreate the relevant part of torchvision's forward pass so Q/K similarities
    # can be computed for selected blocks even though the default forward does not
    # return attention weights.
    with torch.no_grad():
        tokens = model._process_input(input_image)
        batch = tokens.shape[0]
        cls_token = model.class_token.expand(batch, -1, -1)
        x = torch.cat([cls_token, tokens], dim=1)
        x = x + model.encoder.pos_embedding
        cls_attn = {}
        for idx, block in enumerate(model.encoder.layers, start=1):
            x_norm = block.ln_1(x)
            if idx in requested:
                attn = _compute_attention_from_mha(block.self_attention, x_norm)
                cls_attn[idx] = attn[0, :, 0, 1:].cpu()
            attn_out, _ = block.self_attention(x_norm, x_norm, x_norm, need_weights=False)
            x = x + block.dropout(attn_out)
            y = block.ln_2(x)
            y = block.mlp(y)
            x = x + y

    output_dir = PROJECT_ROOT / "outputs" / "pretrained_vit_b_16"
    output_path = output_dir / f"attention_idx{chosen}_label{label}.png"
    plot_attention_maps(
        image=_imagenet_normalize(raw_image),
        cls_attentions=cls_attn,
        patch_size=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        output_path=output_path,
        title=f"Pretrained ViT-B/16 CLS attention | true: {CIFAR10_CLASSES[label]}",
    )
    print(f"Saved optional pretrained attention map to: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path
