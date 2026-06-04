"""
Dynamic tier scheduler for split federated learning.

DTFL latency-based tiering (default) plus optional DEATS battery-aware
assignment (thesis Section 3.3).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.tier_profiles import (
    CODE_TIER_CLIENT_PROFILE,
    CODE_TIER_DATA_SIZE,
    CODE_TIER_SERVER_PROFILE,
)
from utils.deats import DEATSScheduler


def _profile_fallback(num_tiers: int) -> int:
    """Shallowest code tier — safe default when history is empty."""
    return num_tiers


def index_of_greatest_smaller(lst, t_max, num_tiers: int):
    """Pick the deepest tier whose estimated time fits within T_max."""
    feasible = [num for num in lst if num <= t_max]
    if not feasible:
        fallback = _profile_fallback(num_tiers)
        return fallback, lst[fallback - 1]
    return list(lst).index(feasible[0]) + 1, feasible[0]


def client_time_tier(computation_time_clients, client_tier, num_users, num_tiers):
    """Build per-(client, tier) computation time history."""
    client_tier_time = {}
    for i in range(num_users):
        for j in range(1, num_tiers + 1):
            client_tier_time[i, j] = []
            for t in range(len(client_tier)):
                if (
                    client_tier[t][i] == j
                    and not np.isnan(computation_time_clients[i][t])
                ):
                    client_tier_time[i, j].append(computation_time_clients[i][t])
    return client_tier_time


def _estimate_tier_times(
    current_tier: int,
    current_comp_estimation_time: float,
    batch_num: float,
    net_speed: float,
    num_tiers: int,
) -> Dict[int, float]:
    """Estimate round completion time for each candidate tier."""
    fallback = _profile_fallback(num_tiers)
    estimates = {}
    for m in range(1, num_tiers + 1):
        client_profile = CODE_TIER_CLIENT_PROFILE.get(
            m, CODE_TIER_CLIENT_PROFILE.get(fallback, 0.019)
        )
        current_profile = CODE_TIER_CLIENT_PROFILE.get(
            current_tier, CODE_TIER_CLIENT_PROFILE.get(fallback, 0.019)
        )
        data_size = CODE_TIER_DATA_SIZE.get(
            m, CODE_TIER_DATA_SIZE.get(fallback, 0)
        )
        server_profile = CODE_TIER_SERVER_PROFILE.get(
            m, CODE_TIER_SERVER_PROFILE.get(fallback, 0.133)
        )

        client_time = (
            client_profile / current_profile * current_comp_estimation_time
            + data_size * batch_num / net_speed
        )
        server_time = (data_size / net_speed + server_profile) * batch_num
        estimates[m] = max(server_time, client_time)
    return estimates


def TierScheduler(
    computation_time_clients,
    t_max,
    deats_scheduler: Optional[DEATSScheduler] = None,
    remaining_rounds: int = 1,
    **kwargs,
):
    """
    Assign tiers for the next FL round.

    With ``deats_scheduler``, uses DEATS (time + energy + fairness + battery).
    Otherwise uses DTFL latency-only policy.
    """
    client_tier = kwargs["client_tier_all"]
    delay_history = kwargs["delay_history"]
    num_tiers = kwargs["num_tiers"]
    num_users = kwargs["num_users"]
    dataset_size = kwargs["dataset_size"]
    batch_size = kwargs["batch_size"]
    net_speed = kwargs["net_speed"]

    batch_num_clients = {
        key: value / batch_size for key, value in dataset_size.items()
    }

    for k in range(num_users):
        current_tier = client_tier[-1][k]
        transfer_data_size = (
            batch_num_clients[k] * CODE_TIER_DATA_SIZE.get(current_tier, 0)
        )
        communication_time = transfer_data_size / net_speed[k]
        last_delay = delay_history[k].iloc[-1]
        if pd.isna(last_delay):
            computation_time_clients[k].append(np.nan)
        else:
            computation_time_clients[k].append(last_delay - communication_time)

    client_tier_next = {}
    time_estimation_client = {}
    client_times_tier = client_time_tier(
        computation_time_clients, client_tier, num_users, num_tiers
    )
    shallow_fallback = _profile_fallback(num_tiers)

    for k in range(num_users):
        current_tier = client_tier[-1][k]
        times_last_tier = client_times_tier[k, current_tier]

        if not times_last_tier:
            client_tier_next[k] = shallow_fallback
            continue

        current_comp_estimation_time = (
            pd.DataFrame({"Times": times_last_tier})["Times"]
            .ewm(span=2, adjust=False)
            .mean()
            .iloc[-1]
        )

        time_estimation = _estimate_tier_times(
            current_tier,
            current_comp_estimation_time,
            batch_num_clients[k],
            net_speed[k],
            num_tiers,
        )
        time_estimation_list = [time_estimation[m] for m in sorted(time_estimation)]
        time_estimation_client[k] = min(time_estimation_list)

        if deats_scheduler is not None:
            if not deats_scheduler.battery.is_alive(k):
                client_tier_next[k] = num_tiers
                continue

            selected, _ = deats_scheduler.select_tier(
                client_id=k,
                time_estimates=time_estimation,
                remaining_rounds=remaining_rounds,
                t_max=t_max,
            )
            client_tier_next[k] = selected
        else:
            client_tier_next[k], _ = index_of_greatest_smaller(
                time_estimation_list, t_max, num_tiers
            )

    t_max = max(time_estimation_client.values()) if time_estimation_client else t_max
    return client_tier_next, t_max, computation_time_clients
