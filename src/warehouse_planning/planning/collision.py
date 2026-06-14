"""Collision checking utilities for static and dynamic scene elements."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from warehouse_planning.maps.warehouse_map import WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotState


@dataclass(frozen=True)
class CollisionChecker:
    """Check robot footprints against map bounds, static geometry, and movers."""

    warehouse: WarehouseMap
    dynamic_obstacles: tuple[DynamicObstacle, ...] = ()

    def is_state_valid(
        self,
        robot: Robot,
        state: RobotState,
        time: float = 0.0,
        other_robot_states: Iterable[RobotState] = (),
    ) -> bool:
        """Return whether a robot state is collision-free at a given time."""
        if self.collides_with_static_obstacle(state):
            return False
        if self.collides_with_any_robot(state, other_robot_states):
            return False
        return not self.collides_with_dynamic_obstacle(state, time)

    def collides_with_static_obstacle(self, state: RobotState) -> bool:
        """Return whether a robot footprint overlaps map bounds or obstacles."""
        if not self.warehouse.in_bounds(state.x, state.y, margin=state.radius):
            return True
        return self.warehouse.is_occupied(state.x, state.y, margin=state.radius)

    @staticmethod
    def collides_with_robot(state_a: RobotState, state_b: RobotState) -> bool:
        """Return whether two circular robot footprints overlap."""
        distance = hypot(state_a.x - state_b.x, state_a.y - state_b.y)
        return distance <= state_a.radius + state_b.radius

    def collides_with_any_robot(
        self,
        state: RobotState,
        other_robot_states: Iterable[RobotState],
    ) -> bool:
        """Return whether a robot footprint overlaps any other robot footprint."""
        return any(
            self.collides_with_robot(state, other_state)
            for other_state in other_robot_states
        )

    def collides_with_dynamic_obstacle(
        self,
        state: RobotState,
        time: float,
    ) -> bool:
        """Return whether a robot footprint overlaps any dynamic obstacle."""
        for obstacle in self.dynamic_obstacles:
            ox, oy = obstacle.predicted_position(time)
            if hypot(state.x - ox, state.y - oy) <= state.radius + obstacle.radius:
                return True
        return False
