"""
utils/titan.py
==============
TITAN plan helper utilities for the DTFL training loop.

Implements (in dependency-free, importable form):
  1. Learning-rate warmup + cosine schedule       (Step 2)
  2. FedProx proximal term                        (Step 3)
  3. Server-side EMA of global weights            (Step 5)
  4. MixUp / CutMix image blend                   (Step 7)
  5. Gradient clipping wrapper                    (Step 9)
  6. Test-time augmentation (TTA)                 (Step 11)

RandAugment + RandomErasing (Step 4) live with the data pipeline,
so they are NOT defined here.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Step 2 — Learning rate schedule
# ---------------------------------------------------------------------------

def lr_schedule(
    round_idx: int,
    total_rounds: int,
    base_lr: float,
    warmup_rounds: int = 10,
    min_lr_ratio: float = 0.01,
) -> float:
    """Linear warmup followed by cosine decay.

    Args:
        round_idx:      Zero-based current global round (int).
        total_rounds:   Total number of rounds planned (int).
        base_lr:        Peak learning rate after warmup.
        warmup_rounds:  Number of warmup rounds (default 10).
        min_lr_ratio:   Final LR as a fraction of base_lr (default 0.01).

    Returns:
        lr (float) for this round.
    """
    if total_rounds <= 0:
        return base_lr
    warmup_rounds = max(1, min(warmup_rounds, total_rounds // 5))
    if round_idx < warmup_rounds:
        return base_lr * (round_idx + 1) / warmup_rounds
    progress = (round_idx - warmup_rounds) / max(1, total_rounds - warmup_rounds)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


# ---------------------------------------------------------------------------
# Step 3 — FedProx proximal term
# ---------------------------------------------------------------------------

def fedprox_loss(
    local_params: Dict[str, torch.Tensor],
    global_params: Dict[str, torch.Tensor],
    mu: float,
    base_loss: torch.Tensor,
) -> torch.Tensor:
    """Add the FedProx proximal penalty on top of a base loss.

    Penalty: (mu / 2) * sum_p || w_p - w_global_p ||^2

    Only tensors with requires_grad=True contribute (frozen BN running
    stats etc. are skipped automatically because they appear with the
    same value in both dicts).

    Args:
        local_params:  Current local model state_dict (trainable params).
        global_params: Snapshot of the global state_dict at round start.
        mu:            Proximal coefficient (typical 0.001-0.05).
        base_loss:     The task loss (CrossEntropy etc.) already computed.

    Returns:
        loss (Tensor): base_loss + prox_term (differentiable w.r.t. local_params).
    """
    if mu <= 0.0:
        return base_loss
    prox = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
    for k, v in local_params.items():
        if not v.requires_grad:
            continue
        g = global_params.get(k)
        if g is None:
            continue
        prox = prox + torch.sum((v - g.to(v.device)) ** 2)
    return base_loss + 0.5 * mu * prox


# ---------------------------------------------------------------------------
# Step 5 — Server-side EMA of the global model
# ---------------------------------------------------------------------------

class ServerEMA:
    """Maintain an exponential moving average of the aggregated server weights.

    The EMA is used for evaluation and for broadcasting to clients in the next
    round. The raw aggregated weights are still kept around for the next
    aggregation step.

    Usage:
        ema = ServerEMA(decay=0.9)
        # after aggregating w_avg from clients:
        ema_state = ema.update(w_avg, previous_ema_state)
        # send ema_state to clients; evaluate using ema_state
    """

    def __init__(self, decay: float = 0.9) -> None:
        self.decay = decay
        self._state: Optional[Dict[str, torch.Tensor]] = None

    def initialized(self) -> bool:
        return self._state is not None

    def state(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._state

    def update(
        self,
        w_avg: Dict[str, torch.Tensor],
        prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Combine ``w_avg`` with the previous EMA state.

        On the first call, the EMA simply equals ``w_avg``.
        On subsequent calls:
            ema = decay * prev_ema + (1 - decay) * w_avg
        If ``prev_state`` is provided, it is used as the previous EMA (allows
        callers to roll back / inspect).
        """
        if prev_state is not None:
            self._state = prev_state
        if self._state is None:
            self._state = copy.deepcopy(
                {k: v.detach().clone() for k, v in w_avg.items()}
            )
            return self._state
        new_state: Dict[str, torch.Tensor] = {}
        for k, v in w_avg.items():
            v = v.detach()
            if k in self._state:
                new_state[k] = self.decay * self._state[k].to(v.device) + (
                    1.0 - self.decay
                ) * v
            else:
                new_state[k] = v.clone()
        self._state = new_state
        return self._state


# ---------------------------------------------------------------------------
# Step 7 — MixUp / CutMix image blending
# ---------------------------------------------------------------------------
# MixUp:    x_mix = lam * x_a + (1-lam) * x_b
# CutMix:   paste a random rectangle from x_b onto x_a, label area = 1-lam
# Both are mixed at the BATCH level (one lambda per batch).
# ---------------------------------------------------------------------------

def _rand_bbox(size, lam):
    """Random bbox for CutMix. ``size`` is the tensor shape (B, C, H, W)."""
    H, W = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Apply MixUp to a batch.

    Args:
        images: (B, C, H, W) float tensor (already on the right device).
        labels: (B,) long tensor of class indices.
        alpha:  Beta-distribution alpha. Typical 0.2-0.4. If <=0 returns
                (images, labels, labels, 1.0) so callers can be uniform.

    Returns:
        images_mixed, labels_a, labels_b, lam
        loss = lam * CE(pred, labels_a) + (1-lam) * CE(pred, labels_b)
    """
    if alpha <= 0.0:
        return images, labels, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)  # bias toward less aggressive mixing
    perm = torch.randperm(images.size(0), device=images.device)
    images_mixed = lam * images + (1.0 - lam) * images[perm]
    return images_mixed, labels, labels[perm], lam


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Apply CutMix to a batch.

    Args:
        images: (B, C, H, W) float tensor.
        labels: (B,) long tensor of class indices.
        alpha:  Beta-distribution alpha. Typical 1.0 for CutMix.

    Returns:
        images_mixed, labels_a, labels_b, lam (effective area ratio).
    """
    if alpha <= 0.0:
        return images, labels, labels, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(images.size(0), device=images.device)
    images_mixed = images.clone()
    x1, y1, x2, y2 = _rand_bbox(images.shape, lam)
    images_mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(images.size(2) * images.size(3)))
    return images_mixed, labels, labels[perm], lam


def mix_criterion(
    criterion,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
):
    """Cross-entropy loss for a mixed batch."""
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def maybe_mix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    cutmix_prob: float = 0.5,
    generator: Optional[torch.Generator] = None,
):
    """Pick MixUp or CutMix with ``cutmix_prob``, fall back to plain batch.

    Returns: (images_mixed, labels_a, labels_b, lam, kind)
        kind is one of {"none", "mixup", "cutmix"}.
    """
    if alpha <= 0.0:
        return images, labels, labels, 1.0, "none"
    if generator is not None:
        pick = float(torch.rand((), generator=generator).item())
    else:
        pick = float(torch.rand(()).item())
    if pick < cutmix_prob:
        imgs, a, b, lam = cutmix_batch(images, labels, alpha)
        return imgs, a, b, lam, "cutmix"
    imgs, a, b, lam = mixup_batch(images, labels, alpha)
    return imgs, a, b, lam, "mixup"


# ---------------------------------------------------------------------------
# Step 9 — Gradient clipping
# ---------------------------------------------------------------------------
# A thin wrapper around torch.nn.utils.clip_grad_norm_ so callers can be
# gated by a single CLI flag without sprinkling torch imports in main.py.
# ---------------------------------------------------------------------------

def clip_gradients(parameters, max_norm: float = 1.0, norm_type: float = 2.0):
    """Clip in-place gradients of an iterable of parameters.

    Args:
        parameters: same thing you pass to optimizer.step().
        max_norm:   Max L-norm after clipping. Set <= 0 to disable.
        norm_type:  Inf-norm type (default 2.0 = L2).

    Returns:
        total_norm (float): the pre-clipping total norm of the gradients.
    """
    if max_norm is None or max_norm <= 0.0:
        return 0.0
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)
    return float(total_norm.detach().cpu().item())


# ---------------------------------------------------------------------------
# Step 11 — Test-time augmentation (TTA)
# ---------------------------------------------------------------------------
# At evaluation time we average predictions over N augmented views.
# Default N=1 (no TTA). With --use_titan_tta we use 4 views:
#   original, hflip, vflip, hflip+vflip
# ---------------------------------------------------------------------------

def tta_views(images: torch.Tensor, n_views: int = 4):
    """Yield a list of augmented versions of ``images``.

    n_views = 1 -> just the original.
    n_views = 4 -> original, hflip, vflip, hflip+vflip.
    Other values fall back to a sensible subset.
    All returned tensors are clones (no in-place edits on ``images``).
    """
    if n_views <= 1:
        return [images]
    if n_views >= 4:
        return [
            images,
            torch.flip(images, dims=[3]),
            torch.flip(images, dims=[2]),
            torch.flip(images, dims=[2, 3]),
        ]
    out = [images]
    if n_views >= 2:
        out.append(torch.flip(images, dims=[3]))
    if n_views >= 3:
        out.append(torch.flip(images, dims=[2]))
    return out


def tta_average_logits(model, images: torch.Tensor, n_views: int = 4):
    """Run the model on each TTA view and return averaged softmax probabilities.

    Returns a (B, num_classes) tensor on the same device as ``images``.
    """
    views = tta_views(images, n_views=n_views)
    probs = None
    with torch.no_grad():
        for v in views:
            logits = model(v)
            p = torch.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
    probs = probs / float(len(views))
    return probs


# ---------------------------------------------------------------------------
# Self-test (run with ``python -m utils.titan``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    print("lr schedule:", [round(lr_schedule(i, 100, 0.1, 10), 5) for i in [0, 5, 9, 10, 50, 99]])

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    for kind in ("mixup", "cutmix", "none"):
        if kind == "mixup":
            imgs, a, b, lam = mixup_batch(x, y, 0.2)
        elif kind == "cutmix":
            imgs, a, b, lam = cutmix_batch(x, y, 1.0)
        else:
            imgs, a, b, lam = x, y, y, 1.0
        print(kind, "lam", round(lam, 3), "shape", tuple(imgs.shape))

    views = tta_views(x, n_views=4)
    print("tta views:", len(views), [tuple(v.shape) for v in views])

    p = torch.nn.Linear(4, 2).weight
    p.grad = torch.randn_like(p) * 100
    n = clip_gradients([p], max_norm=1.0)
    print("grad clip norm after:", round(p.grad.norm().item(), 3), "max was:", n)
