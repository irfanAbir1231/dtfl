"""
utils/augmentation.py
=====================
TITAN Step 4 — train-side image augmentation pipeline for HAM10000
(and any other 32x32 / small-image classification task).

Provides two composable builders:

  build_titan_train_transform(image_size=32, strong=True)
      Train-side transform with strong augmentation:
          Resize -> RandomHorizontalFlip -> RandomVerticalFlip
                   -> RandomCrop (padding=4)
                   -> RandAugment(n=2, m=5)         (if available)
                   -> ToTensor
                   -> Normalize(IMAGENET_MEAN, IMAGENET_STD)
                   -> RandomErasing(p=0.25, scale=(0.02, 0.2))

      If `strong=False`, only basic flips are applied (baseline behaviour).

  build_titan_eval_transform(image_size=32)
      Deterministic eval/test transforms (no augmentation). Identical to the
      baseline test pipeline.

RandAugment import falls back gracefully on older torchvision versions.
This module is import-free of the rest of the project and can be used by any
data loader (HAM10000, CIFAR-10, CIFAR-100, CINIC-10).
"""

from __future__ import annotations

import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _maybe_randaugment():
    """Return RandAugment class if available, else None."""
    try:
        from torchvision.transforms import RandAugment  # type: ignore
        return RandAugment
    except ImportError:
        try:
            from torchvision.transforms.autoaugment import RandAugment  # type: ignore
            return RandAugment
        except ImportError:
            return None


def build_titan_train_transform(
    image_size: int = 32,
    strong: bool = True,
) -> transforms.Compose:
    """Train transforms with optional strong augmentation (TITAN Step 4).

    Args:
        image_size: Square image size used by the model.
        strong:     When True, add RandAugment + RandomErasing on top of
                    the baseline flips + crop. When False, returns the
                    baseline-only pipeline.
    """
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(image_size, padding=4),
    ]
    if strong:
        RandAugment = _maybe_randaugment()
        if RandAugment is not None:
            ops.append(RandAugment(num_ops=2, magnitude=5))
        else:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    if strong:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)))
    return transforms.Compose(ops)


def build_titan_eval_transform(image_size: int = 32) -> transforms.Compose:
    """Deterministic eval/test transforms (untouched from baseline)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )