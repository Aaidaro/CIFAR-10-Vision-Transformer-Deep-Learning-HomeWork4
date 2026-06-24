"""Create CLS-to-patch attention maps for trained ViT checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import CIFAR10_CLASSES, CIFAR10LocalDataset, load_cifar10_arrays
from models.model import build_model
from scripts.evaluate import load_checkpoint
from scripts.train import _device
from utils.config import prepare_config, set_seed
from utils.visualization import plot_attention_maps


def _select_test_image(
    config: Dict,
    target_class: Optional[int] = None,
    index: Optional[int] = None,
) -> tuple[torch.Tensor, int, int]:
    """Select one test image by absolute index or first matching target class."""
    data_cfg = config["data"]
    _, _, test_images, test_labels = load_cifar10_arrays(data_cfg["data_dir"])
    if index is not None:
        chosen = int(index)
    elif target_class is not None:
        matches = [i for i, label in enumerate(test_labels.tolist()) if int(label) == int(target_class)]
        if not matches:
            raise ValueError(f"No test image found for target class {target_class}")
        chosen = matches[0]
    else:
        chosen = 0

    dataset = CIFAR10LocalDataset(
        test_images,
        test_labels,
        image_size=int(config["model"]["image_size"]),
        train=False,
        augment=False,
        normalize=bool(data_cfg.get("normalize", True)),
        mean=data_cfg.get("mean", (0.4914, 0.4822, 0.4465)),
        std=data_cfg.get("std", (0.2470, 0.2435, 0.2616)),
    )
    image, label = dataset[chosen]
    return image, int(label), chosen


def collect_cls_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    layers: Iterable[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Run a model and collect CLS attention vectors from requested layers.

    Layer numbers are 1-indexed to match the assignment notation (L1, L2, L4).
    """
    model.eval()
    requested = list(layers)
    with torch.no_grad():
        logits, attentions, _ = model(image.unsqueeze(0).to(device), return_attention=True)
        prediction = int(logits.argmax(dim=1).item())
    result = {}
    for layer in requested:
        idx = int(layer) - 1
        if idx < 0 or idx >= len(attentions):
            print(f"Skipping layer {layer}: model has {len(attentions)} layers.")
            continue
        # [batch, heads, seq, seq] -> [heads, num_patches], CLS row without CLS column.
        result[int(layer)] = attentions[idx][0, :, 0, 1:].cpu()
    print(f"Predicted class: {prediction} ({CIFAR10_CLASSES[prediction]})")
    return result


def run_attention(config: Dict, checkpoint_path: str | Path, layers, target_class=None, index=None) -> Path:
    """Generate and save the attention figure for a checkpoint and test image."""
    set_seed(int(config.get("seed", 42)))
    device = _device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    image, label, chosen_index = _select_test_image(config, target_class=target_class, index=index)
    cls_attn = collect_cls_attention(model, image, layers, device)
    if not cls_attn:
        raise ValueError("No valid layers were selected for attention plotting.")

    run_name = str(config["experiment"]["name"])
    output_dir = PROJECT_ROOT / str(config["experiment"].get("output_dir", "outputs")) / run_name
    output_path = output_dir / f"attention_idx{chosen_index}_label{label}.png"
    plot_attention_maps(
        image=image,
        cls_attentions=cls_attn,
        patch_size=int(config["model"]["patch_size"]),
        mean=config["data"].get("mean", (0.4914, 0.4822, 0.4465)),
        std=config["data"].get("std", (0.2470, 0.2435, 0.2616)),
        output_path=output_path,
        title=f"CLS attention | true: {CIFAR10_CLASSES[label]} | index: {chosen_index}",
    )
    print(f"Saved attention map to: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def attention_from_args(args) -> Path:
    """CLI adapter used by ``scripts/main.py``."""
    device = _device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = checkpoint.get("config")
    if cfg is None:
        cfg = prepare_config(args.config, args.case, args.quick)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    return run_attention(cfg, args.checkpoint, args.layers, args.target_class, args.index)
