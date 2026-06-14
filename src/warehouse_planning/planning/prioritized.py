"""Multi-robot baseline planners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from warehouse_planning.models.robot import Robot
from warehouse_planning.planning.kinodynamic_astar import (
    ContinuousPose,
    KinodynamicAStarPlanner,
    ReservationTable,
)


@dataclass(frozen=True)
class MultiRobotPlanResult:
    """Paths and bookkeeping returned by a multi-robot baseline."""

    paths: dict[str, list[ContinuousPose]]
    planning_time: float
    success: bool
    failed_robot_id: str | None = None


@dataclass
class IndependentPlanner:
    """Plan each robot independently with kinodynamic A*.

    Robot-robot conflicts are intentionally not constrained during planning;
    they are measured afterward by the evaluation metrics.
    """

    single_robot_planner: KinodynamicAStarPlanner

    def plan(self, robots: tuple[Robot, ...]) -> MultiRobotPlanResult:
        """Plan one path per robot without robot-robot constraints."""
        start_time = perf_counter()
        paths: dict[str, list[ContinuousPose]] = {}
        for robot in robots:
            planner = replace(self.single_robot_planner, reservation_table=None)
            try:
                paths[robot.id] = planner.plan(robot)
            except ValueError:
                return MultiRobotPlanResult(
                    paths=paths,
                    planning_time=perf_counter() - start_time,
                    success=False,
                    failed_robot_id=robot.id,
                )

        return MultiRobotPlanResult(
            paths=paths,
            planning_time=perf_counter() - start_time,
            success=True,
        )


@dataclass
class PrioritizedPlanner:
    """Plan robots sequentially using reservations from earlier robots."""

    single_robot_planner: KinodynamicAStarPlanner

    def plan(self, robots: tuple[Robot, ...]) -> MultiRobotPlanResult:
        """Plan robots in tuple order with vertex and edge-swap constraints."""
        start_time = perf_counter()
        paths: dict[str, list[ContinuousPose]] = {}
        reservation_table: ReservationTable | None = None

        for robot in robots:
            planner = replace(
                self.single_robot_planner,
                reservation_table=reservation_table,
            )
            try:
                paths[robot.id] = planner.plan(robot)
            except ValueError:
                return MultiRobotPlanResult(
                    paths=paths,
                    planning_time=perf_counter() - start_time,
                    success=False,
                    failed_robot_id=robot.id,
                )

            reservation_table = ReservationTable.from_paths(
                paths,
                planner.collision_checker,
                planner.dt,
                planner.max_time_steps,
            )

        return MultiRobotPlanResult(
            paths=paths,
            planning_time=perf_counter() - start_time,
            success=True,
        )
