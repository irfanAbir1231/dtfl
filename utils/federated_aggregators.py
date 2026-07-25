"""Server-side aggregation optimizers for DTFL.

The existing heterogeneous client/server state dictionaries are first reduced
to a sample-weighted model by ``utils.fedavg.aggregated_fedavg``.  The classes
in this module then apply the selected server update while keeping the rest of
the DTFL training pipeline unchanged.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Dict

import torch


SUPPORTED_ALGORITHMS = ("fedavg", "fedavgm", "fedyogi", "fedadagrad")

_DEFAULT_SERVER_LR = {
    "fedavg": 1.0,
    "fedavgm": 1.0,
    "fedyogi": 0.01,
    "fedadagrad": 0.1,
}


StateDict = Mapping[str, torch.Tensor]

# BatchNorm running statistics are buffers, not gradient-trained parameters.
# Stateful server optimizers must not build momentum/adaptive state for them.
_BATCH_NORM_BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than 0, got {value}.")


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")


def _is_batch_norm_buffer(key: str) -> bool:
    return key.endswith(_BATCH_NORM_BUFFER_SUFFIXES)


class ServerAggregator(ABC):
    """Base interface for applying a server-side federated update."""

    name: str

    @abstractmethod
    def aggregate(
        self,
        current_state: StateDict,
        averaged_state: StateDict,
    ) -> Dict[str, torch.Tensor]:
        """Return the next global state without modifying either input."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FedAvgAggregator(ServerAggregator):
    """Apply the existing sample-weighted FedAvg result directly."""

    name = "fedavg"

    def aggregate(
        self,
        current_state: StateDict,
        averaged_state: StateDict,
    ) -> Dict[str, torch.Tensor]:
        del current_state
        return copy.deepcopy(averaged_state)


class _StatefulServerAggregator(ServerAggregator):
    """Common tensor handling for stateful server optimizers."""

    def _prepare_update(
        self,
        key: str,
        current: torch.Tensor,
        averaged: torch.Tensor,
    ) -> torch.Tensor:
        if current.shape != averaged.shape:
            raise ValueError(
                f"Cannot aggregate parameter '{key}': current shape "
                f"{tuple(current.shape)} does not match averaged shape "
                f"{tuple(averaged.shape)}."
            )
        averaged = averaged.to(device=current.device, dtype=current.dtype)
        return averaged - current

    @staticmethod
    def _state_tensor(
        state: Dict[str, torch.Tensor],
        key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = state.get(key)
        if (
            value is None
            or value.shape != reference.shape
            or value.device != reference.device
            or value.dtype != reference.dtype
        ):
            value = torch.zeros_like(reference)
        return value

    def aggregate(
        self,
        current_state: StateDict,
        averaged_state: StateDict,
    ) -> Dict[str, torch.Tensor]:
        result = copy.deepcopy(current_state)

        with torch.no_grad():
            for key, averaged in averaged_state.items():
                if key not in current_state:
                    result[key] = averaged.detach().clone()
                    continue

                current = current_state[key]
                if (
                    _is_batch_norm_buffer(key)
                    or not torch.is_floating_point(current)
                ):
                    if current.shape != averaged.shape:
                        raise ValueError(
                            f"Cannot aggregate parameter '{key}': current "
                            f"shape {tuple(current.shape)} does not match "
                            f"averaged shape {tuple(averaged.shape)}."
                        )
                    result[key] = averaged.to(
                        device=current.device,
                        dtype=current.dtype,
                    ).detach().clone()
                    continue

                update = self._prepare_update(key, current, averaged)
                result[key] = self._apply_update(key, current, update)

        return result

    @abstractmethod
    def _apply_update(
        self,
        key: str,
        current: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one parameter update and return a detached tensor."""


class FedAvgMAggregator(_StatefulServerAggregator):
    """Federated Averaging with server momentum."""

    name = "fedavgm"

    def __init__(self, server_lr: float = 1.0, momentum: float = 0.9) -> None:
        _validate_positive("server_lr", server_lr)
        _validate_unit_interval("momentum", momentum)
        self.server_lr = server_lr
        self.momentum = momentum
        self._velocity: Dict[str, torch.Tensor] = {}

    def _apply_update(
        self,
        key: str,
        current: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        velocity = self._state_tensor(self._velocity, key, update)
        velocity = self.momentum * velocity + update
        self._velocity[key] = velocity.detach().clone()
        return (current + self.server_lr * velocity).detach().clone()

    def __repr__(self) -> str:
        return (
            f"FedAvgMAggregator(server_lr={self.server_lr}, "
            f"momentum={self.momentum})"
        )


class FedAdagradAggregator(_StatefulServerAggregator):
    """FedAdagrad adaptive server optimizer."""

    name = "fedadagrad"

    def __init__(self, server_lr: float = 0.1, tau: float = 1e-3) -> None:
        _validate_positive("server_lr", server_lr)
        _validate_positive("tau", tau)
        self.server_lr = server_lr
        self.tau = tau
        self._second_moment: Dict[str, torch.Tensor] = {}

    def _apply_update(
        self,
        key: str,
        current: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        second_moment = self._state_tensor(
            self._second_moment,
            key,
            update,
        )
        second_moment = second_moment + update.square()
        self._second_moment[key] = second_moment.detach().clone()
        next_value = current + (
            self.server_lr * update / (second_moment.sqrt() + self.tau)
        )
        return next_value.detach().clone()

    def __repr__(self) -> str:
        return (
            f"FedAdagradAggregator(server_lr={self.server_lr}, "
            f"tau={self.tau})"
        )


class FedYogiAggregator(_StatefulServerAggregator):
    """FedYogi adaptive server optimizer."""

    name = "fedyogi"

    def __init__(
        self,
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        _validate_positive("server_lr", server_lr)
        _validate_unit_interval("beta1", beta1)
        _validate_unit_interval("beta2", beta2)
        _validate_positive("tau", tau)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self._first_moment: Dict[str, torch.Tensor] = {}
        self._second_moment: Dict[str, torch.Tensor] = {}

    def _apply_update(
        self,
        key: str,
        current: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        first_moment = self._state_tensor(
            self._first_moment,
            key,
            update,
        )
        second_moment = self._state_tensor(
            self._second_moment,
            key,
            update,
        )

        first_moment = (
            self.beta1 * first_moment
            + (1.0 - self.beta1) * update
        )
        update_squared = update.square()
        second_moment = second_moment - (
            (1.0 - self.beta2)
            * update_squared
            * torch.sign(second_moment - update_squared)
        )

        self._first_moment[key] = first_moment.detach().clone()
        self._second_moment[key] = second_moment.detach().clone()
        next_value = current + (
            self.server_lr
            * first_moment
            / (second_moment.sqrt() + self.tau)
        )
        return next_value.detach().clone()

    def __repr__(self) -> str:
        return (
            f"FedYogiAggregator(server_lr={self.server_lr}, "
            f"beta1={self.beta1}, beta2={self.beta2}, tau={self.tau})"
        )


def create_server_aggregator(
    algorithm: str = "fedavg",
    *,
    server_lr: float | None = None,
    server_momentum: float = 0.9,
    server_beta1: float = 0.9,
    server_beta2: float = 0.99,
    server_tau: float = 1e-3,
) -> ServerAggregator:
    """Build the requested server aggregator.

    ``server_lr=None`` selects the conventional default for each algorithm:
    1.0 for FedAvg/FedAvgM, 0.01 for FedYogi, and 0.1 for FedAdagrad.
    """

    normalized = algorithm.strip().lower()
    if normalized not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(SUPPORTED_ALGORITHMS)
        raise ValueError(
            f"Unsupported federated algorithm '{algorithm}'. "
            f"Choose one of: {supported}."
        )

    resolved_lr = (
        _DEFAULT_SERVER_LR[normalized]
        if server_lr is None
        else server_lr
    )

    if normalized == "fedavg":
        return FedAvgAggregator()
    if normalized == "fedavgm":
        return FedAvgMAggregator(
            server_lr=resolved_lr,
            momentum=server_momentum,
        )
    if normalized == "fedadagrad":
        return FedAdagradAggregator(
            server_lr=resolved_lr,
            tau=server_tau,
        )
    return FedYogiAggregator(
        server_lr=resolved_lr,
        beta1=server_beta1,
        beta2=server_beta2,
        tau=server_tau,
    )
