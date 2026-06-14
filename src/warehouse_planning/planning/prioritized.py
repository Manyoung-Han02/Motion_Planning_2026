"""Multi-robot baseline planners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from warehouse_planning.models.robot import Robot
from warehouse_planning.planning.coordination import (
    coordinate_local_waits,
    count_sampled_path_conflicts,
)
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
                cell_padding=planner.reservation_padding,
            )

        return MultiRobotPlanResult(
            paths=paths,
            planning_time=perf_counter() - start_time,
            success=True,
        )


@dataclass
class ConcurrentLocalWaitPlanner:
    """Plan all robots, then insert local waits only near conflicts.

    This keeps most robots moving concurrently while using explicit sampled
    space-time conflict checks. It is intentionally lighter than full CBS and
    works well for the small presentation and benchmark scenarios.
    """

    single_robot_planner: KinodynamicAStarPlanner
    time_step: float = 0.2
    wait_step: float = 0.4
    max_total_wait: float = 6.0
    clearance_margin: float = 0.14

    def plan(self, robots: tuple[Robot, ...]) -> MultiRobotPlanResult:
        """Return paths with coordinated local waits."""
        start_time = perf_counter()
        base_paths: dict[str, list[ContinuousPose]] = {}
        reservation_table: ReservationTable | None = None
        for robot in robots:
            planner = replace(
                self.single_robot_planner,
                reservation_table=reservation_table,
            )
            try:
                base_paths[robot.id] = planner.plan(robot)
            except ValueError:
                return MultiRobotPlanResult(
                    paths=base_paths,
                    planning_time=perf_counter() - start_time,
                    success=False,
                    failed_robot_id=robot.id,
                )
            reservation_table = ReservationTable.from_paths(
                base_paths,
                planner.collision_checker,
                planner.dt,
                planner.max_time_steps,
                cell_padding=planner.reservation_padding,
            )

        scheduled = coordinate_local_waits(
            base_paths,
            robots,
            time_step=self.time_step,
            wait_step=self.wait_step,
            max_total_wait=self.max_total_wait,
            clearance_margin=self.clearance_margin,
        )
        success = (
            count_sampled_path_conflicts(
                scheduled,
                robots,
                time_step=self.time_step,
                clearance_margin=self.clearance_margin,
            )
            == 0
        )

        return MultiRobotPlanResult(
            paths={robot.id: scheduled[robot.id] for robot in robots},
            planning_time=perf_counter() - start_time,
            success=success,
        )
