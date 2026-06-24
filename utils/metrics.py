"""Metrics and small bookkeeping helpers for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    """Track the running average of a scalar quantity."""

    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``n`` observations with scalar value ``value``."""
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        """Return the current average, or zero before any updates."""
        return self.total / max(1, self.count)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 classification accuracy for a batch."""
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def num_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Return the number of correct predictions in a batch."""
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())
