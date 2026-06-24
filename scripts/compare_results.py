"""Summarize finished experiment metrics for Question 2.2(a)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compare_results(outputs_dir: str | Path = "outputs") -> Path:
    """Collect every ``metrics.json`` under ``outputs_dir`` into one CSV table."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.is_absolute():
        outputs_path = PROJECT_ROOT / outputs_path
    rows: List[dict] = []
    for metrics_path in sorted(outputs_path.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {outputs_path}")

    csv_path = outputs_path / "comparison_summary.csv"
    keys = ["run_name", "best_val_acc", "test_acc", "test_loss", "params", "estimated_macs", "elapsed_seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    print(f"Saved comparison table to: {csv_path.relative_to(PROJECT_ROOT)}")
    return csv_path
