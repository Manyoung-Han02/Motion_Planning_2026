"""Conflict-Based Search style multi-robot planner interface."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_planning.models.robot import Robot
from warehouse_planning.planning.prioritized import MultiRobotPlanResult, PrioritizedPlanner


@dataclass(frozen=True)
class Conflict:
    """High-level CBS conflict descriptor."""

    robot_a: str
    robot_b: str
    time: float
    location: tuple[float, float]


@dataclass
class CBSPlanner:
    """CBS-style planner skeleton.

    The current version delegates to prioritized planning so the surrounding
    simulator can run before the full CBS search tree is implemented.
    """

    fallback_planner: PrioritizedPlanner

    def plan(self, robots: tuple[Robot, ...]) -> MultiRobotPlanResult:
        """Return a first feasible set of placeholder paths."""
        return self.fallback_planner.plan(robots)
