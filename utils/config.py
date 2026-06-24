"""Configuration helpers for YAML files and named experiment cases."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return a dictionary."""
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update ``base`` with ``updates`` and return ``base``."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def prepare_config(
    config_path: str | Path = "config/config.yaml",
    case: str = "base",
    quick: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the base config, apply a named case, quick mode, and CLI overrides."""
    cfg = load_yaml(config_path)
    cases = cfg.get("cases", {})
    if case not in cases:
        available = ", ".join(sorted(cases))
        raise KeyError(f"Unknown case '{case}'. Available cases: {available}")
    cfg = deep_update(copy.deepcopy(cfg), copy.deepcopy(cases[case]))
    cfg["experiment"]["name"] = cfg["experiment"].get("name", case)

    if quick:
        quick_cfg = cfg.get("quick", {})
        cfg["train"]["epochs"] = int(quick_cfg.get("epochs", cfg["train"]["epochs"]))
        cfg["train"]["batch_size"] = int(quick_cfg.get("batch_size", cfg["train"]["batch_size"]))
        cfg["data"]["subset_train"] = quick_cfg.get("subset_train")
        cfg["data"]["subset_val"] = quick_cfg.get("subset_val")
        cfg["data"]["subset_test"] = quick_cfg.get("subset_test")
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_quick"

    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
