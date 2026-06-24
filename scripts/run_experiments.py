"""Run one or more named experiment cases from the YAML config."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import run_training
from utils.config import prepare_config


def run_cases(config_path: str, cases: Iterable[str], quick: bool = False, data_dir: str | None = None) -> List[dict]:
    """Train a sequence of named cases and return their metrics."""
    metrics = []
    for case in cases:
        overrides = {"data": {"data_dir": data_dir}} if data_dir else None
        cfg = prepare_config(config_path=config_path, case=case, quick=quick, overrides=overrides)
        metrics.append(run_training(cfg))
    return metrics
