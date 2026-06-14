"""Dynamic obstacle primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicObstacle:
    """Circular moving obstacle with a time-indexed reference trajectory."""

    id: str
    radius: float
    trajectory: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        """Validate trajectory waypoints for piecewise-linear interpolation."""
        if self.radius <= 0.0:
            raise ValueError("Dynamic obstacle radius must be positive")
        if not self.trajectory:
            raise ValueError(f"Dynamic obstacle {self.id} has no trajectory")

        times = [point[0] for point in self.trajectory]
        if times != sorted(times):
            raise ValueError(f"Dynamic obstacle {self.id} trajectory times must be sorted")

    def position_at(self, time: float) -> tuple[float, float]:
        """Return the interpolated obstacle position at a given time."""
        times = np.array([point[0] for point in self.trajectory], dtype=float)
        xs = np.array([point[1] for point in self.trajectory], dtype=float)
        ys = np.array([point[2] for point in self.trajectory], dtype=float)
        x = float(np.interp(time, times, xs))
        y = float(np.interp(time, times, ys))
        return (x, y)

    def predicted_position(self, time: float) -> tuple[float, float]:
        """Return the predicted obstacle position ``p_h(t)``."""
        return self.position_at(time)
