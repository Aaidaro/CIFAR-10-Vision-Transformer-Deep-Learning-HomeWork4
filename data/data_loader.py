"""Data loading utilities for the local CIFAR-10 Python archive.

This module intentionally does not use ``torchvision.datasets.CIFAR10`` so that the
assignment can be graded without relying on a prewritten dataset wrapper. It reads
the official CIFAR-10 Python batches, applies lightweight augmentations, and builds
PyTorch ``DataLoader`` objects for train/validation/test splits.
"""

from __future__ import annotations

import pickle
import random
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root without hard-coding absolutes."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _project_root() / path


def _extract_archive_if_needed(data_dir: Path) -> Path:
    """Find or extract the CIFAR-10 Python directory inside ``data_dir``.

    Args:
        data_dir: Directory that contains either ``cifar-10-batches-py`` or the
            archive ``cifar-10-python.tar.gz`` downloaded from the Toronto site.

    Returns:
        Path to the extracted ``cifar-10-batches-py`` directory.

    Raises:
        FileNotFoundError: If neither the extracted directory nor archive exists.
    """
    batches_dir = data_dir / "cifar-10-batches-py"
    if batches_dir.exists():
        return batches_dir

    archive = data_dir / "cifar-10-python.tar.gz"
    if archive.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=data_dir)
        if batches_dir.exists():
            return batches_dir

    raise FileNotFoundError(
        "Could not find CIFAR-10 data. Put 'cifar-10-python.tar.gz' or the "
        "extracted 'cifar-10-batches-py/' folder under: "
        f"{data_dir.as_posix()}"
    )


def _load_pickle_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR-10 pickle batch as image and label arrays."""
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return images.astype(np.uint8), labels


def load_cifar10_arrays(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the local CIFAR-10 Python batches into memory.

    Args:
        data_dir: Relative or absolute directory containing the downloaded archive
            or extracted CIFAR-10 Python batches.

    Returns:
        ``(train_images, train_labels, test_images, test_labels)`` where images are
        ``uint8`` arrays with shape ``[N, H, W, C]`` and labels are ``int64`` arrays.
    """
    data_path = _extract_archive_if_needed(_resolve_path(data_dir))

    train_images, train_labels = [], []
    for batch_idx in range(1, 6):
        images, labels = _load_pickle_batch(data_path / f"data_batch_{batch_idx}")
        train_images.append(images)
        train_labels.append(labels)

    test_images, test_labels = _load_pickle_batch(data_path / "test_batch")
    return (
        np.concatenate(train_images, axis=0),
        np.concatenate(train_labels, axis=0),
        test_images,
        test_labels,
    )


def train_val_indices(num_samples: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic random train/validation indices."""
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    val_size = int(round(num_samples * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


class CIFAR10LocalDataset(Dataset):
    """A minimal CIFAR-10 dataset with built-in augmentation.

    Args:
        images: Array of raw CIFAR images in ``[N, H, W, C]`` format.
        labels: Array of integer class labels.
        image_size: Output size. CIFAR-10 images are resized only when this value
            differs from 32.
        train: Whether to apply training augmentations.
        augment: Enables/disables random crop, flip, and cutout for training.
        random_crop_padding: Number of zero/reflect padding pixels before crop.
        random_horizontal_flip: If true, randomly flips training images.
        cutout: If true, masks a square region after normalization.
        cutout_size: Side length of the cutout square in output pixels.
        normalize: If true, applies channel-wise normalization.
        mean: Normalization mean for RGB channels.
        std: Normalization standard deviation for RGB channels.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_size: int = 32,
        train: bool = False,
        augment: bool = False,
        random_crop_padding: int = 4,
        random_horizontal_flip: bool = True,
        cutout: bool = False,
        cutout_size: int = 8,
        normalize: bool = True,
        mean: Sequence[float] = (0.4914, 0.4822, 0.4465),
        std: Sequence[float] = (0.2470, 0.2435, 0.2616),
    ) -> None:
        self.images = images
        self.labels = labels
        self.image_size = int(image_size)
        self.train = train
        self.augment = augment
        self.random_crop_padding = int(random_crop_padding)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.cutout = bool(cutout)
        self.cutout_size = int(cutout_size)
        self.normalize = bool(normalize)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        """Return the number of examples."""
        return int(self.labels.shape[0])

    def _augment_np_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random crop and horizontal flip to one raw image."""
        if not (self.train and self.augment):
            return image

        if self.random_crop_padding > 0:
            p = self.random_crop_padding
            padded = np.pad(image, ((p, p), (p, p), (0, 0)), mode="reflect")
            top = random.randint(0, 2 * p)
            left = random.randint(0, 2 * p)
            image = padded[top : top + 32, left : left + 32, :]

        if self.random_horizontal_flip and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])

        return image

    def _resize_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Resize a tensor image from 32x32 to ``image_size`` when needed."""
        if self.image_size == image.shape[-1]:
            return image
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return image.squeeze(0)

    def _apply_cutout(self, image: torch.Tensor) -> torch.Tensor:
        """Mask a random square region of a tensor image."""
        if not (self.train and self.augment and self.cutout and self.cutout_size > 0):
            return image
        _, height, width = image.shape
        size = min(self.cutout_size, height, width)
        cy = random.randint(0, height - 1)
        cx = random.randint(0, width - 1)
        y1 = max(0, cy - size // 2)
        y2 = min(height, y1 + size)
        x1 = max(0, cx - size // 2)
        x2 = min(width, x1 + size)
        image[:, y1:y2, x1:x2] = 0.0
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one normalized image tensor and its integer label."""
        image = self._augment_np_image(self.images[index])
        image_t = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0)
        image_t = self._resize_tensor(image_t)
        if self.normalize:
            image_t = (image_t - self.mean) / self.std
        image_t = self._apply_cutout(image_t)
        label_t = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image_t, label_t


def _maybe_subset(dataset: Dataset, subset_size: Optional[int], seed: int) -> Dataset:
    """Return a deterministic subset when ``subset_size`` is set."""
    if subset_size is None:
        return dataset
    subset_size = int(min(subset_size, len(dataset)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def make_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build CIFAR-10 train/validation/test DataLoaders from a config dictionary."""
    seed = int(config.get("seed", 42))
    data_cfg = config["data"]
    train_images, train_labels, test_images, test_labels = load_cifar10_arrays(data_cfg["data_dir"])
    tr_idx, val_idx = train_val_indices(len(train_labels), float(data_cfg["validation_fraction"]), seed)

    common = dict(
        image_size=int(config["model"]["image_size"]),
        random_crop_padding=int(data_cfg.get("random_crop_padding", 4)),
        random_horizontal_flip=bool(data_cfg.get("random_horizontal_flip", True)),
        cutout=bool(data_cfg.get("cutout", False)),
        cutout_size=int(data_cfg.get("cutout_size", 8)),
        normalize=bool(data_cfg.get("normalize", True)),
        mean=data_cfg.get("mean", (0.4914, 0.4822, 0.4465)),
        std=data_cfg.get("std", (0.2470, 0.2435, 0.2616)),
    )

    train_set = CIFAR10LocalDataset(
        train_images[tr_idx], train_labels[tr_idx], train=True, augment=bool(data_cfg.get("augment", True)), **common
    )
    val_set = CIFAR10LocalDataset(train_images[val_idx], train_labels[val_idx], train=False, augment=False, **common)
    test_set = CIFAR10LocalDataset(test_images, test_labels, train=False, augment=False, **common)

    train_set = _maybe_subset(train_set, data_cfg.get("subset_train"), seed)
    val_set = _maybe_subset(val_set, data_cfg.get("subset_val"), seed + 1)
    test_set = _maybe_subset(test_set, data_cfg.get("subset_test"), seed + 2)

    train_cfg = config["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 2))
    pin_memory = torch.cuda.is_available()

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )
