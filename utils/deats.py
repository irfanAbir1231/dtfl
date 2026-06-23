"""
Dynamic Energy-Aware Tier Scheduling (DEATS) — thesis Section 3.3.

  - EMA energy rate tracking          (Eq. 3.1)
  - Energy ratio per tier cost        (Eq. 3.2)
  - Per-round energy estimate         (Eq. 3.3)
  - Survivability prediction          (Eq. 3.4)
  - Combined optimization score       (Eq. 3.5–3.6)
  - Dynamic adaptation & safety floor (Section 3.3.6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.tier_profiles import (
    code_tier_to_paper,
    normalized_comm_ratio,
    normalized_comp_ratio,
    paper_tiers_for_battery,
)


@dataclass
class DEATSConfig:
    """Hyperparameters for DEATS scheduling."""

    ema_alpha: float = 0.3
    energy_comp_weight: float = 0.7
    energy_comm_weight: float = 0.3
    score_time_weight: float = 0.30
    score_energy_weight: float = 0.35
    score_fairness_weight: float = 0.35
    battery_min: float = 5.0
    energy_scale: float = 0.03
    battery_init_min: float = 65.0
    battery_init_max: float = 100.0


@dataclass
class ClientEnergyState:
    """Per-client energy state tracked across FL rounds."""

    battery_percent: float
    ema_energy_rate: float = 0.0
    cumulative_energy: float = 0.0
    tier_history: List[int] = field(default_factory=list)
    dropout: bool = False


class BatterySimulator:
    """Simulates heterogeneous client battery levels during FL training."""

    def __init__(
        self,
        num_clients: int,
        config: Optional[DEATSConfig] = None,
        seed: int = 10,
    ) -> None:
        self.config = config or DEATSConfig()
        self.rng = np.random.default_rng(seed)
        self.clients: Dict[int, ClientEnergyState] = {
            k: ClientEnergyState(
                battery_percent=float(
                    self.rng.uniform(
                        self.config.battery_init_min,
                        self.config.battery_init_max,
                    )
                )
            )
            for k in range(num_clients)
        }

    def get_battery(self, client_id: int) -> float:
        """Return current battery percentage (not below the safety floor display)."""
        return max(
            self.clients[client_id].battery_percent,
            self.config.battery_min,
        )

    def is_alive(self, client_id: int) -> bool:
        """Whether the client can still participate."""
        return not self.clients[client_id].dropout

    def drain(
        self,
        client_id: int,
        duration: float,
        data_transmitted: float,
        net_speed: float,
        delay_coefficient: float,
        code_tier: int,
        num_tiers: int,
    ) -> float:
        """Apply battery drain for one training round; return energy consumed (%)."""
        energy = self._instantaneous_energy(
            duration,
            data_transmitted,
            net_speed,
            delay_coefficient,
            code_tier,
            num_tiers,
        )
        state = self.clients[client_id]
        usable = max(state.battery_percent - self.config.battery_min, 0.0)
        drain = min(energy, usable)

        state.battery_percent -= drain
        state.cumulative_energy += drain
        state.tier_history.append(code_tier)

        if state.battery_percent <= self.config.battery_min:
            state.battery_percent = self.config.battery_min
            state.dropout = True

        return drain

    def _instantaneous_energy(
        self,
        duration: float,
        data_transmitted: float,
        net_speed: float,
        delay_coefficient: float,
        code_tier: int,
        num_tiers: int,
    ) -> float:
        """Instantaneous energy proxy E_t for the current round."""
        paper_tier = code_tier_to_paper(code_tier, num_tiers)
        comp_ratio = normalized_comp_ratio(paper_tier)
        comm_ratio = normalized_comm_ratio(paper_tier)

        cfg = self.config
        raw = (
            duration * delay_coefficient * cfg.energy_scale
            + (data_transmitted / max(net_speed, 1e-9)) * cfg.energy_scale
        )
        ratio = (
            cfg.energy_comp_weight * comp_ratio
            + cfg.energy_comm_weight * comm_ratio
        )
        return raw * ratio


class DEATSScheduler:
    """Battery-aware tier selection (Eq. 3.5–3.6) on top of DTFL time estimates."""

    def __init__(
        self,
        num_clients: int,
        num_tiers: int,
        config: Optional[DEATSConfig] = None,
        battery_simulator: Optional[BatterySimulator] = None,
        seed: int = 10,
    ) -> None:
        self.num_clients = num_clients
        self.num_tiers = num_tiers
        self.config = config or DEATSConfig()
        self.battery = battery_simulator or BatterySimulator(
            num_clients, self.config, seed=seed
        )
        self._relative_energy: Dict[int, float] = {
            k: 0.0 for k in range(num_clients)
        }

    def update_ema(self, client_id: int, observed_energy: float) -> None:
        """Update EMA energy rate after observing a round (Eq. 3.1)."""
        state = self.battery.clients[client_id]
        alpha = self.config.ema_alpha
        if state.ema_energy_rate == 0.0:
            state.ema_energy_rate = observed_energy
        else:
            state.ema_energy_rate = (
                alpha * observed_energy
                + (1.0 - alpha) * state.ema_energy_rate
            )

    def energy_ratio(self, code_tier: int) -> float:
        """Per-tier energy scaling factor (Eq. 3.2)."""
        paper_tier = code_tier_to_paper(code_tier, self.num_tiers)
        cfg = self.config
        comp = normalized_comp_ratio(paper_tier)
        comm = normalized_comm_ratio(paper_tier)
        return cfg.energy_comp_weight * comp + cfg.energy_comm_weight * comm

    def round_energy(self, client_id: int, code_tier: int) -> float:
        """Estimated energy cost per round at tier m (Eq. 3.3)."""
        ema = self.battery.clients[client_id].ema_energy_rate
        if ema <= 0.0:
            # Conservative bootstrap before first measurements arrive.
            # A larger default avoids optimistic survivability estimates
            # that can over-assign heavy tiers in early rounds.
            ema = 2.0
        return ema * self.energy_ratio(code_tier)

    def survivable_rounds(self, client_id: int, code_tier: int) -> float:
        """Predict how many rounds the device survives at tier m (Eq. 3.4)."""
        battery = self.battery.get_battery(client_id)
        usable = max(battery - self.config.battery_min, 0.0)
        cost = self.round_energy(client_id, code_tier)
        if cost <= 0.0:
            return float("inf")
        return usable / cost

    def record_relative_energy(self, client_id: int, energy_drain: float) -> None:
        """Track relative energy consumption for fairness scoring."""
        battery = self.battery.get_battery(client_id)
        if battery > 0.0:
            self._relative_energy[client_id] = energy_drain / battery

    def fairness_deviation(self, client_id: int, predicted_relative: float) -> float:
        """Fairness penalty: deviation from mean relative energy."""
        peers = [
            v
            for cid, v in self._relative_energy.items()
            if cid != client_id and v > 0.0
        ]
        if not peers:
            return 0.0
        mean_rel = float(np.mean(peers))
        return (predicted_relative - mean_rel) ** 2

    def select_tier(
        self,
        client_id: int,
        time_estimates: Dict[int, float],
        remaining_rounds: int,
        t_max: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose optimal code tier for a client (Eq. 3.5–3.6)."""
        battery = self.battery.get_battery(client_id)
        allowed = paper_tiers_for_battery(battery, self.num_tiers)
        candidates = [m for m in sorted(time_estimates) if m in allowed]

        if not candidates:
            return self.num_tiers, {
                "reason": "no_battery_feasible_tier",
                "battery": battery,
            }

        cfg = self.config
        scores: Dict[int, float] = {}
        metrics: Dict[str, Any] = {"battery": battery}

        for code_tier in candidates:
            if t_max is not None and time_estimates[code_tier] > t_max:
                continue

            if self.survivable_rounds(client_id, code_tier) < remaining_rounds:
                continue

            predicted_drain = self.round_energy(client_id, code_tier)
            predicted_relative = (
                predicted_drain / battery if battery > 0.0 else 0.0
            )

            scores[code_tier] = (
                cfg.score_time_weight * time_estimates[code_tier]
                + cfg.score_energy_weight * predicted_drain
                + cfg.score_fairness_weight
                * self.fairness_deviation(client_id, predicted_relative)
            )
            metrics[f"survive_t{code_tier}"] = self.survivable_rounds(
                client_id, code_tier
            )

        if not scores:
            feasible = [m for m in allowed if m in time_estimates]
            selected = max(feasible) if feasible else self.num_tiers
            return selected, {**metrics, "reason": "survivability_fallback"}

        selected = min(scores, key=scores.get)
        metrics.update(
            {
                "selected_tier": selected,
                "score": scores[selected],
                "allowed_tiers": allowed,
                "feasible_tiers": list(scores.keys()),
            }
        )
        return selected, metrics
