"""
utils/privacy.py
================
Privacy helper utilities for DTFL smashed-tensor protection.

This module implements the privacy mechanism described in the thesis:
  1. Adaptive Gaussian noise decay schedule  (§2 of privacy-implementation-plan.md)
  2. L2 per-sample norm clipping             (§3 step 2)
  3. Gaussian noise injection                (§3 step 3)
  4. SNR metric                              (§4)
  5. Correlation-drop metric                 (§5)
  6. Optional (ε, δ) DP accounting           (§6)

All functions are pure (no global state) and operate on CPU or CUDA tensors.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Adaptive noise schedule
# ---------------------------------------------------------------------------

def noise_schedule(round_idx: int, sigma0: float, sigma_min: float, decay: float) -> float:
    """Compute the noise standard-deviation multiplier for the current round.

    Implements the exponential decay schedule recommended in the thesis:
        sigma_t = max(sigma_min, sigma0 * decay ** round_idx)

    Args:
        round_idx:  Zero-based global communication round index.
        sigma0:     Initial (maximum) noise multiplier (--noise_sigma0).
        sigma_min:  Floor value — noise never drops below this (--noise_sigma_min).
        decay:      Multiplicative decay per round in (0, 1) (--noise_decay).

    Returns:
        sigma_t (float): The noise multiplier for this round.
    """
    sigma_t = max(sigma_min, sigma0 * (decay ** round_idx))
    return float(sigma_t)


# ---------------------------------------------------------------------------
# 2. L2 per-sample norm clipping
# ---------------------------------------------------------------------------

def clip_features(fx: Tensor, clip_C: float) -> Tensor:
    """Clip each sample's feature vector to L2 norm ≤ clip_C (in-place safe copy).

    Each row in the batch is independently rescaled so that its L2 norm does not
    exceed *clip_C*.  Samples already within the bound are left unchanged.

    The returned tensor is detached from the computation graph; gradients for
    the clipped tensor are **not** needed because we will re-attach with
    requires_grad=True later for the server backward pass.

    Args:
        fx:      Smashed tensor from the client model, shape (N, *).
        clip_C:  Maximum L2 norm per sample (--clip_C).

    Returns:
        fx_clipped: Clipped tensor, same shape as fx, detached.
    """
    fx_detached = fx.detach().clone()                       # safe copy, no grad
    # Flatten to (N, D) for norm computation, then reshape back
    original_shape = fx_detached.shape
    flat = fx_detached.view(original_shape[0], -1)          # (N, D)
    norms = flat.norm(p=2, dim=1, keepdim=True)             # (N, 1)
    # scale = min(1, C / ||x||); clamp avoids div-by-zero
    scale = torch.clamp(clip_C / norms.clamp(min=1e-10), max=1.0)
    clipped_flat = flat * scale
    fx_clipped = clipped_flat.view(original_shape)
    return fx_clipped


# ---------------------------------------------------------------------------
# 3. Gaussian noise injection
# ---------------------------------------------------------------------------

def add_gaussian_noise(fx_clipped: Tensor, sigma_t: float, clip_C: float) -> Tensor:
    """Add calibrated Gaussian noise to the clipped smashed tensor.

    The noise standard deviation per element is:
        std = sigma_t * clip_C

    This follows the Gaussian mechanism for DP where clip_C is the L2 sensitivity
    and sigma_t is the noise multiplier.

    Args:
        fx_clipped: Clipped smashed tensor (detached), shape (N, *).
        sigma_t:    Noise multiplier for this round (from noise_schedule).
        clip_C:     Clipping bound used for sensitivity calibration.

    Returns:
        fx_noisy: Noisy tensor, same shape as fx_clipped, detached.
    """
    std = sigma_t * clip_C
    noise = torch.randn_like(fx_clipped) * std
    fx_noisy = fx_clipped + noise
    return fx_noisy.detach()


# ---------------------------------------------------------------------------
# 4. SNR metric
# ---------------------------------------------------------------------------

def compute_snr(fx_clipped: Tensor, fx_noisy: Tensor) -> float:
    """Compute the batch-level Signal-to-Noise Ratio.

    SNR = E[||signal||^2_F] / E[||noise||^2_F]
        = ||fx_clipped||_F^2 / ||fx_noisy - fx_clipped||_F^2

    A higher SNR means less distortion relative to the signal energy.
    The metric is computed over the entire batch.

    Args:
        fx_clipped: Clipped (clean) smashed tensor.
        fx_noisy:   Noise-added smashed tensor.

    Returns:
        snr (float): Signal-to-noise ratio (dimensionless, ≥ 0).
                     Returns inf if noise power is zero (no noise was added).
    """
    signal_power = fx_clipped.detach().pow(2).sum().item()
    noise = fx_noisy.detach() - fx_clipped.detach()
    noise_power = noise.pow(2).sum().item()
    if noise_power < 1e-30:
        return float("inf")
    return signal_power / noise_power


# ---------------------------------------------------------------------------
# 5. Correlation-drop metric
# ---------------------------------------------------------------------------

def compute_corr_drop(
    images: Tensor,
    fx_clipped: Tensor,
    fx_noisy: Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute the relative drop in distance correlation after noise injection.

    Uses the existing ``dis_corr`` utility from ``utils.loss``:
        dcor_clean  = dis_corr(images, fx_clipped)
        dcor_noisy  = dis_corr(images, fx_noisy)
        corr_drop   = (dcor_clean - dcor_noisy) / max(dcor_clean, eps)

    A value near 1.0 means the noise has almost fully de-correlated the
    smashed features from the raw inputs — ideal for privacy.
    A value near 0.0 means the noise had little effect.

    Args:
        images:     Raw input batch, shape (N, C, H, W).
        fx_clipped: Clipped smashed tensor, shape (N, *).
        fx_noisy:   Noisy smashed tensor, shape (N, *).
        eps:        Small constant to avoid division by zero.

    Returns:
        corr_drop (float): Relative correlation reduction in [−∞, 1].
                           Negative values (rare) indicate the noise increased
                           the apparent correlation (numerical artifact).
    """
    # Import here to avoid circular imports; dis_corr is in the same package
    from utils.loss import dis_corr  # noqa: PLC0415

    with torch.no_grad():
        dcor_clean = dis_corr(images.detach(), fx_clipped.detach()).item()
        dcor_noisy = dis_corr(images.detach(), fx_noisy.detach()).item()

    denominator = max(abs(dcor_clean), eps)
    corr_drop = (dcor_clean - dcor_noisy) / denominator
    return float(corr_drop)


# ---------------------------------------------------------------------------
# 6. Optional (ε, δ)-DP accounting via Opacus
# ---------------------------------------------------------------------------

def get_dp_epsilon(
    sample_rate: float,
    noise_multiplier: float,
    num_steps: int,
    delta: float,
    mechanism: str = "gaussian",
) -> Optional[float]:
    """Compute the privacy budget ε for the given accounting parameters.

    Uses Opacus' RDP accounting analysis API if the library is installed.
    Returns ``None`` gracefully when opacus is not available, so the rest of
    training is completely unaffected.

    The accounting uses the standard subsampled Gaussian mechanism:
        ε = compute_dp_sgd_privacy(n, batch_size, noise_multiplier,
                                    epochs, delta)

    Here we map:
        - sample_rate   = batch_size / dataset_size  (the subsampling ratio q)
        - noise_multiplier = sigma_t (noise std relative to sensitivity)
        - num_steps     = total optimizer steps consumed this round

    Args:
        sample_rate:       Poisson subsampling rate q = batch_size / N_local.
        noise_multiplier:  Gaussian noise multiplier (sigma_t).
        num_steps:         Number of mechanism applications this round.
        delta:             Target δ for (ε, δ)-DP (--privacy_delta).
        mechanism:         Reserved; only "gaussian" is currently supported.

    Returns:
        epsilon (float | None): The privacy budget ε, or None if opacus is
                                not installed or accounting fails.
    """
    if noise_multiplier <= 0 or sample_rate <= 0 or num_steps <= 0:
        logger.debug("DP accounting skipped: invalid parameters.")
        return None

    try:
        # Opacus ≥ 1.0 exposes RDP analysis through the accountants package.
        from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent  # noqa: PLC0415
    
        # Orders for Rényi DP
        orders = list(range(2, 64)) + [128, 256]
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=noise_multiplier,
            steps=num_steps,
            orders=orders,
        )
        epsilon, _ = get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        return float(epsilon)

    except ImportError:
        # Opacus not installed — skip silently (logged once at first call)
        logger.info(
            "opacus is not installed; (ε, δ)-DP accounting is disabled. "
            "Install it with: pip install opacus"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("DP accounting failed: %s", exc)
        return None
