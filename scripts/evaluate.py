"""Evaluation utilities for saved ViT checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import make_dataloaders
from models.model import build_model, count_parameters, estimate_vit_macs
from scripts.train import _device, run_epoch
from utils.config import prepare_config, set_seed


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Dict:
    """Load a checkpoint relative to the project root when needed."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return torch.load(path, map_location=device)


def run_evaluation(config: Dict, checkpoint_path: str | Path) -> Dict:
    """Evaluate a saved checkpoint on the test split."""
    set_seed(int(config.get("seed", 42)))
    device = _device()
    _, _, test_loader = make_dataloaders(config)
    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"].get("label_smoothing", 0.0)))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "params": count_parameters(model),
        "estimated_macs": estimate_vit_macs(config["model"]),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def evaluate_from_args(args) -> Dict:
    """CLI adapter used by ``scripts/main.py``."""
    device = _device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = checkpoint.get("config")
    if cfg is None:
        cfg = prepare_config(args.config, args.case, args.quick)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.subset_test is not None:
        cfg["data"]["subset_test"] = args.subset_test
    return run_evaluation(cfg, args.checkpoint)
