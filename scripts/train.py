"""Training loop for the CIFAR-10 Vision Transformer."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import make_dataloaders
from models.model import build_model, count_parameters, estimate_vit_macs
from utils.config import prepare_config, save_yaml, set_seed
from utils.metrics import AverageMeter, num_correct
from utils.visualization import plot_history


def _device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create the configured optimizer."""
    opt_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["learning_rate"])
    wd = float(cfg["train"].get("weight_decay", 0.0))
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a warmup + cosine learning-rate schedule."""
    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 0.0,
    amp: bool = False,
) -> Tuple[float, float]:
    """Run one train or evaluation epoch.

    Args:
        optimizer: When provided, the model is trained; otherwise evaluation mode
            is used with no gradient updates.
    """
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter("loss")
    correct = 0
    total = 0
    scaler_enabled = amp and device.type == "cuda" and training
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)
            correct += num_correct(logits.detach(), targets)
            total += batch_size

    return loss_meter.avg, correct / max(1, total)


def save_history_csv(history: Dict, output_path: Path) -> None:
    """Save training history as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for idx in range(len(history["train_loss"])):
            writer.writerow(
                [
                    idx + 1,
                    history["train_loss"][idx],
                    history["train_acc"][idx],
                    history["val_loss"][idx],
                    history["val_acc"][idx],
                    history["lr"][idx],
                ]
            )


def run_training(config: Dict) -> Dict:
    """Train one configured ViT case and save checkpoints, plots, and metrics."""
    set_seed(int(config.get("seed", 42)))
    device = _device()
    run_name = str(config["experiment"]["name"])
    output_dir = PROJECT_ROOT / str(config["experiment"].get("output_dir", "outputs")) / run_name
    save_dir = PROJECT_ROOT / str(config["experiment"].get("save_dir", "models/saved_models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "resolved_config.yaml")

    train_loader, val_loader, test_loader = make_dataloaders(config)
    model = build_model(config["model"]).to(device)
    params = count_parameters(model)
    macs = estimate_vit_macs(config["model"])
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {params:,}")
    print(f"Estimated MACs / image: {macs:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"].get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    grad_clip = float(config["train"].get("grad_clip_norm", 0.0))
    amp = bool(config["train"].get("amp", False))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_path = save_dir / f"{run_name}_best.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    epochs = int(config["train"]["epochs"])

    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, grad_clip_norm=grad_clip, amp=amp
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {current_lr:.6g}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "history": history,
            "val_acc": val_acc,
            "params": params,
            "estimated_macs": macs,
        }
        torch.save(checkpoint, last_path)
        if val_acc > best_val_acc and bool(config["train"].get("save_best", True)):
            best_val_acc = val_acc
            torch.save(checkpoint, best_path)

    checkpoint_to_test = torch.load(best_path if best_path.exists() else last_path, map_location=device)
    model.load_state_dict(checkpoint_to_test["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    elapsed = time.time() - start_time

    metrics = {
        "run_name": run_name,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "params": params,
        "estimated_macs": macs,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(best_path.relative_to(PROJECT_ROOT)) if best_path.exists() else None,
        "last_checkpoint": str(last_path.relative_to(PROJECT_ROOT)),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_history_csv(history, output_dir / "history.csv")
    plot_history(history, output_dir, run_name)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print(f"Saved outputs to: {output_dir.relative_to(PROJECT_ROOT)}")
    return metrics


def train_from_args(args) -> Dict:
    """CLI adapter used by ``scripts/main.py``."""
    overrides = {"data": {}, "train": {}, "model": {}, "experiment": {}}
    if args.data_dir is not None:
        overrides["data"]["data_dir"] = args.data_dir
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["train"]["learning_rate"] = args.lr
    if args.subset_train is not None:
        overrides["data"]["subset_train"] = args.subset_train
    if args.subset_val is not None:
        overrides["data"]["subset_val"] = args.subset_val
    if args.subset_test is not None:
        overrides["data"]["subset_test"] = args.subset_test
    overrides = {k: v for k, v in overrides.items() if v}
    cfg = prepare_config(args.config, args.case, args.quick, overrides)
    return run_training(cfg)
