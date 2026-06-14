"""Occupancy-grid warehouse map representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class RectangleObstacle:
    """Axis-aligned rectangular shelf or wall obstacle in world coordinates."""

    id: str
    x: float
    y: float
    width: float
    height: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return bounds as ``(xmin, ymin, xmax, ymax)``."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return whether a point lies inside the obstacle plus a margin."""
        xmin, ymin, xmax, ymax = self.bounds
        return (
            xmin - margin <= x <= xmax + margin
            and ymin - margin <= y <= ymax + margin
        )


@dataclass(frozen=True)
class WarehouseMap:
    """Static 2D warehouse represented as a binary occupancy grid.

    Occupancy values use ``True`` for blocked cells and ``False`` for free
    cells. The grid row index corresponds to y and the column index to x.
    """

    width: float
    height: float
    resolution: float
    static_obstacles: tuple[RectangleObstacle, ...] = ()
    occupancy_grid: NDArray[np.bool_] = field(init=False, repr=False)
    obstacle_distance_grid: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Rasterize rectangular obstacles into the occupancy grid."""
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Warehouse dimensions must be positive")
        if self.resolution <= 0.0:
            raise ValueError("Warehouse resolution must be positive")

        occupancy_grid = self._build_occupancy_grid()
        object.__setattr__(self, "occupancy_grid", occupancy_grid)
        object.__setattr__(
            self,
            "obstacle_distance_grid",
            self._build_obstacle_distance_grid(occupancy_grid),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the occupancy grid shape as ``(rows, cols)``."""
        rows = int(ceil(self.height / self.resolution))
        cols = int(ceil(self.width / self.resolution))
        return rows, cols

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return the Matplotlib image extent for the grid."""
        return (0.0, self.width, 0.0, self.height)

    def _build_occupancy_grid(self) -> NDArray[np.bool_]:
        """Create the binary occupancy grid from rectangular obstacles."""
        grid = np.zeros(self.shape, dtype=np.bool_)
        for obstacle in self.static_obstacles:
            row_min, col_min = self.world_to_grid(obstacle.x, obstacle.y)
            row_max, col_max = self.world_to_grid(
                obstacle.x + obstacle.width,
                obstacle.y + obstacle.height,
            )
            row_start = max(0, min(row_min, row_max))
            row_stop = min(grid.shape[0], max(row_min, row_max) + 1)
            col_start = max(0, min(col_min, col_max))
            col_stop = min(grid.shape[1], max(col_min, col_max) + 1)
            for row in range(row_start, row_stop):
                for col in range(col_start, col_stop):
                    x, y = self.grid_to_world(row, col)
                    if obstacle.contains_point(x, y):
                        grid[row, col] = True
        return grid

    def _build_obstacle_distance_grid(
        self,
        occupancy_grid: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Create a grid of distances from each free cell to occupied space."""
        if not occupancy_grid.any():
            return np.full(occupancy_grid.shape, np.inf, dtype=np.float64)
        return distance_transform_edt(~occupancy_grid) * self.resolution

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to ``(row, col)`` grid indices."""
        col = int(floor(x / self.resolution))
        row = int(floor(y / self.resolution))
        rows, cols = self.shape
        return max(0, min(row, rows - 1)), max(0, min(col, cols - 1))

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid indices to the world coordinate of the cell center."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return (x, y)

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return whether a circular footprint center is inside map bounds."""
        return margin <= x <= self.width - margin and margin <= y <= self.height - margin

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return whether a point or circular footprint intersects occupied space."""
        if not self.in_bounds(x, y):
            return True

        if margin <= 0.0:
            row, col = self.world_to_grid(x, y)
            return bool(self.occupancy_grid[row, col])

        return self.distance_to_nearest_obstacle(x, y) <= margin

    def distance_to_nearest_obstacle(self, x: float, y: float) -> float:
        """Return grid-based distance from a world point to occupied space."""
        if not self.in_bounds(x, y):
            return 0.0

        row, col = self.world_to_grid(x, y)
        return float(self.obstacle_distance_grid[row, col])
