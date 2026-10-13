"""Type contract reserved for a future path-dependent material model.

No implementation is provided in Phase 2. A solver must not substitute
``evaluate_curve`` for this update operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class UpdateResult[StateT]:
    stress_pa: float
    tangent_pa: float
    state: StateT


class HistoryModel(Protocol[StateT]):
    def update(
        self, *, total_strain: float, temperature_k: float, previous_state: StateT
    ) -> UpdateResult[StateT]:
        """Advance one loading step; never mutate the caller's previous state."""
        ...
