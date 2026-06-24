"""Plotting utilities for training curves and ViT attention maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_history(history: Dict[str, List[float]], output_dir: str | Path, run_name: str) -> None:
    """Save loss and accuracy curves for a training run.

    Args:
        history: Dictionary containing train/validation loss and accuracy lists.
        output_dir: Directory where figures are written.
        run_name: Prefix used for figure filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"Loss curves - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy curves - {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_accuracy.png", dpi=180)
    plt.close()


def unnormalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Convert a normalized ``[C,H,W]`` tensor into a clipped ``[H,W,C]`` image."""
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = image * std_t + mean_t
    image = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def _draw_patch_grid(ax, image_size: int, patch_size: int) -> None:
    """Overlay patch grid lines on an image axis."""
    for location in range(0, image_size + 1, patch_size):
        ax.axhline(location - 0.5, linewidth=0.35)
        ax.axvline(location - 0.5, linewidth=0.35)


def plot_attention_maps(
    image: torch.Tensor,
    cls_attentions: Dict[int, torch.Tensor],
    patch_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    output_path: str | Path,
    title: str = "CLS attention maps",
) -> None:
    """Save a figure like the assignment example.

    Args:
        image: Normalized image tensor with shape ``[C,H,W]``.
        cls_attentions: Mapping from 1-indexed layer number to attention tensor
            with shape ``[heads, num_patches]``. Values should be attention
            probabilities from the CLS token to patch tokens.
        patch_size: Patch size used by the model.
        mean: Channel means used to unnormalize the image.
        std: Channel standard deviations used to unnormalize the image.
        output_path: Figure path.
        title: Figure title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = unnormalize(image, mean, std)
    image_size = image_np.shape[0]
    grid = image_size // patch_size
    layers = list(cls_attentions.keys())
    rows = len(layers)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3.15 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, layer in enumerate(layers):
        attn = cls_attentions[layer].detach().cpu()
        first_head = attn[0].reshape(grid, grid)
        mean_heads = attn.mean(dim=0).reshape(grid, grid)

        axes[row, 0].imshow(image_np)
        _draw_patch_grid(axes[row, 0], image_size, patch_size)
        axes[row, 0].set_title("Patched Input")
        axes[row, 0].set_ylabel(f"L{layer}", rotation=0, labelpad=25, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(first_head, interpolation="nearest")
        axes[row, 1].set_title("First Attention Head")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        axes[row, 2].imshow(mean_heads, interpolation="nearest")
        axes[row, 2].set_title("Mean Attention of Heads")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
