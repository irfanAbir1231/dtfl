"""
Shared tier profiling constants and naming utilities for DTFL / DEATS.

Code tier numbering (internal):
    1 = deepest client split (most local compute, least server offload)
    N = shallowest client split (least local compute, most server offload)

Paper / thesis tier numbering (display):
    1 = shallowest, N = deepest — inverted relative to code tiers.
"""

from __future__ import annotations

from typing import Dict, List

# Normalized client/server times from DTFL paper (Appendix B), keyed by paper tier.
PAPER_CLIENT_TIME: Dict[int, float] = {
    1: 1.00,
    2: 1.63,
    3: 2.16,
    4: 2.68,
    5: 3.30,
    6: 3.81,
    7: 4.33,
}

PAPER_SERVER_TIME: Dict[int, float] = {
    1: 1.00,
    2: 0.82,
    3: 0.65,
    4: 0.51,
    5: 0.33,
    6: 0.20,
    7: 0.10,
}

MB = 1024 ** 2

CODE_TIER_DATA_SIZE: Dict[int, float] = {
    1: 2.5002 * MB,
    2: 2.5006 * MB,
    3: 1.2512 * MB,
    4: 1.2504 * MB,
    5: 0.629 * MB,
    6: 0.6278 * MB,
    7: 0.6278 * MB,
}

CODE_TIER_CLIENT_PROFILE: Dict[int, float] = {
    1: 0.160,
    2: 0.118,
    3: 0.065,
    4: 0.060,
    5: 0.037,
    6: 0.019,
    7: 0.010,
}

CODE_TIER_SERVER_PROFILE: Dict[int, float] = {
    1: 0.005,
    2: 0.026,
    3: 0.063,
    4: 0.098,
    5: 0.105,
    6: 0.133,
    7: 0.150,
}


def code_tier_to_paper(code_tier: int, num_tiers: int) -> int:
    """Convert internal code tier index to thesis/paper tier index."""
    return num_tiers - code_tier + 1


def paper_tier_to_code(paper_tier: int, num_tiers: int) -> int:
    """Convert thesis/paper tier index to internal code tier index."""
    return num_tiers - paper_tier + 1


def paper_tiers_for_battery(battery_percent: float, num_tiers: int) -> List[int]:
    """
    Return allowed *code* tier indices based on battery level (Table 3.1, thesis).

    High battery   (>70 %): deep tiers  — paper 5–7
    Medium battery (30–70 %): medium tiers — paper 3–4
    Low battery    (≤30 %): shallow tiers — paper 1–2
    """
    if battery_percent > 70.0:
        paper_range = range(max(1, num_tiers - 2), num_tiers + 1)
    elif battery_percent > 30.0:
        mid = num_tiers // 2
        paper_range = range(max(1, mid - 1), min(num_tiers, mid + 1) + 1)
    else:
        paper_range = range(1, min(3, num_tiers + 1))

    return sorted({paper_tier_to_code(p, num_tiers) for p in paper_range})


def normalized_comp_ratio(paper_tier: int) -> float:
    """Client-side compute ratio relative to shallowest tier."""
    return PAPER_CLIENT_TIME[paper_tier] / PAPER_CLIENT_TIME[1]


def normalized_comm_ratio(paper_tier: int) -> float:
    """Server-side / communication ratio relative to shallowest tier."""
    return PAPER_SERVER_TIME[paper_tier] / PAPER_SERVER_TIME[1]
