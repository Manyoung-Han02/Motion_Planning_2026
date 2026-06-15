"""Windowed multi-robot receding-horizon planner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from time import perf_counter

from warehouse_planning.models.robot import Robot, RobotState
from warehouse_planning.planning.kinodynamic_astar import (
    ContinuousPose,
    KinodynamicAStarPlanner,
    ReservationTable,
)
from warehouse_planning.planning.prioritized import MultiRobotPlanResult, PrioritizedPlanner


@dataclass
class WindowedRecedingHorizonPlanner:
    """Approximate simultaneous multi-robot planning in rolling time windows.

    Each cycle replans all unfinished robots from their current states over a
    short horizon. Earlier paths in the current window become space-time
    reservations for later robots, and only the first few steps of each window
    are executed before the next replan. This is a practical approximation to
    windowed MAPF for nonholonomic lattice paths.
    """

    single_robot_planner: KinodynamicAStarPlanner
    window_steps: int = 24
    execute_steps: int = 4
    max_cycles: int = 18
    goal_tolerance: float = 0.7

    def plan(self, robots: tuple[Robot, ...]) -> MultiRobotPlanResult:
        """Plan timed paths for all robots with receding-horizon reservations."""
        if self.window_steps <= 0:
            raise ValueError("window_steps must be positive")
        if self.execute_steps <= 0:
            raise ValueError("execute_steps must be positive")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")

        start_time = perf_counter()
        dt = self.single_robot_planner.dt
        current_time = 0.0
        current_states = {robot.id: robot.start for robot in robots}
        paths: dict[str, list[ContinuousPose]] = {
            robot.id: [(robot.start.x, robot.start.y, robot.start.theta, 0.0)]
            for robot in robots
        }
        reached = {
            robot.id: self._is_goal_reached(robot.start, robot.goal)
            for robot in robots
        }

        for _ in range(self.max_cycles):
            if all(reached.values()):
                break

            window_paths = self._plan_window(
                robots,
                current_states,
                reached,
                current_time,
            )
            next_time = current_time + self.execute_steps * dt
            for robot in robots:
                executed = self._segment_until(
                    window_paths.get(robot.id, []),
                    current_time,
                    next_time,
                )
                if len(executed) <= 1:
                    executed = [
                        (
                            current_states[robot.id].x,
                            current_states[robot.id].y,
                            current_states[robot.id].theta,
                            next_time,
                        )
                    ]
                self._append_executed_segment(paths[robot.id], executed)
                last_pose = paths[robot.id][-1]
                current_states[robot.id] = RobotState(
                    last_pose[0],
                    last_pose[1],
                    last_pose[2],
                    robot.radius,
                )
                reached[robot.id] = self._is_goal_reached(
                    current_states[robot.id],
                    robot.goal,
                )
            current_time = next_time

        success = all(reached.values())
        return MultiRobotPlanResult(
            paths=paths,
            planning_time=perf_counter() - start_time,
            success=success,
            failed_robot_id=None if success else self._first_unreached(robots, reached),
        )

    def _plan_window(
        self,
        robots: tuple[Robot, ...],
        current_states: dict[str, RobotState],
        reached: dict[str, bool],
        current_time: float,
    ) -> dict[str, list[ContinuousPose]]:
        """Plan one rolling window with space-time reservations."""
        window_paths: dict[str, list[ContinuousPose]] = {}
        reservation_table: ReservationTable | None = None
        ordered_robots = sorted(
            robots,
            key=lambda robot: (
                reached[robot.id],
                -self._remaining_distance(current_states[robot.id], robot.goal),
                robot.id,
            ),
        )

        for robot in ordered_robots:
            state = current_states[robot.id]
            if reached[robot.id]:
                path = self._hold_path(state, current_time)
            else:
                planning_robot = replace(robot, start=state)
                planner = replace(
                    self.single_robot_planner,
                    reservation_table=reservation_table,
                    time_origin=current_time,
                    max_time_steps=self.window_steps,
                    allow_partial=True,
                )
                try:
                    path = planner.plan(planning_robot)
                except ValueError:
                    path = self._hold_path(state, current_time)

            window_paths[robot.id] = path
            reservation_table = ReservationTable.from_paths(
                window_paths,
                self.single_robot_planner.collision_checker,
                self.single_robot_planner.dt,
                self.window_steps,
                cell_padding=self.single_robot_planner.reservation_padding,
                time_origin=current_time,
            )

        return {robot.id: window_paths[robot.id] for robot in robots}

    def _hold_path(
        self,
        state: RobotState,
        current_time: float,
    ) -> list[ContinuousPose]:
        """Return a stationary path over the current window."""
        dt = self.single_robot_planner.dt
        return [
            (state.x, state.y, state.theta, current_time + step * dt)
            for step in range(self.window_steps + 1)
        ]

    def _segment_until(
        self,
        path: list[ContinuousPose],
        current_time: float,
        next_time: float,
    ) -> list[ContinuousPose]:
        """Return the executed prefix of a window path."""
        if not path:
            return []
        segment = [
            pose
            for pose in path
            if current_time <= pose[3] <= next_time
        ]
        endpoint = self._pose_at_time(path, next_time)
        if endpoint is not None and (
            not segment or abs(segment[-1][3] - next_time) > 1e-9
        ):
            segment.append(endpoint)
        return segment

    @staticmethod
    def _append_executed_segment(
        path: list[ContinuousPose],
        segment: list[ContinuousPose],
    ) -> None:
        """Append a segment while avoiding duplicate timestamps."""
        for pose in segment:
            if path and pose[3] <= path[-1][3] + 1e-9:
                continue
            path.append(pose)

    @staticmethod
    def _pose_at_time(
        path: list[ContinuousPose],
        time: float,
    ) -> ContinuousPose | None:
        """Return linearly interpolated pose at a global time."""
        if not path:
            return None
        if time <= path[0][3]:
            return path[0]
        if time >= path[-1][3]:
            return path[-1]
        for start, end in zip(path, path[1:]):
            if start[3] <= time <= end[3]:
                duration = max(end[3] - start[3], 1e-9)
                alpha = (time - start[3]) / duration
                return (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                    start[2] + alpha * (end[2] - start[2]),
                    time,
                )
        return path[-1]

    def _is_goal_reached(
        self,
        state: RobotState,
        goal: RobotState,
    ) -> bool:
        """Return whether a robot is within the position goal tolerance."""
        return self._remaining_distance(state, goal) <= self.goal_tolerance

    @staticmethod
    def _remaining_distance(
        state: RobotState,
        goal: RobotState,
    ) -> float:
        """Return Euclidean remaining distance."""
        return hypot(goal.x - state.x, goal.y - state.y)

    @staticmethod
    def _first_unreached(
        robots: tuple[Robot, ...],
        reached: dict[str, bool],
    ) -> str | None:
        """Return the first robot id that did not reach its goal."""
        for robot in robots:
            if not reached.get(robot.id, False):
                return robot.id
        return None


@dataclass
class WindowedConflictReplanner:
    """Prioritized planner with bounded conflict-window replanning.

    This planner first builds complete nonholonomic paths with prioritized
    space-time reservations, then scans a finite future window for sampled
    robot conflicts. When a conflict is found, the lower-priority robot is
    replanned from shortly before the conflict while treating all other paths
    as local time-expanded reservations.
    """

    single_robot_planner: KinodynamicAStarPlanner
    window_steps: int = 24
    repair_iterations: int = 3
    lookback_steps: int = 3
    clearance_margin: float = 0.14

    def plan(self, robots: tuple[Robot, ...]) -> MultiRobotPlanResult:
        """Return paths after bounded windowed conflict repair."""
        start_time = perf_counter()
        initial = PrioritizedPlanner(self.single_robot_planner).plan(robots)
        if not initial.success:
            return initial

        paths = {robot_id: list(path) for robot_id, path in initial.paths.items()}
        robot_by_id = {robot.id: robot for robot in robots}
        for _ in range(self.repair_iterations):
            conflict = self._first_window_conflict(paths, robots)
            if conflict is None:
                break
            conflict_time, _, lower_priority_id = conflict
            robot = robot_by_id[lower_priority_id]
            replan_time = max(
                0.0,
                conflict_time - self.lookback_steps * self.single_robot_planner.dt,
            )
            repaired = self._replan_suffix(
                robot,
                paths,
                replan_time=replan_time,
            )
            if repaired is not None:
                paths[robot.id] = repaired

        success = self._first_window_conflict(paths, robots) is None
        return MultiRobotPlanResult(
            paths=paths,
            planning_time=perf_counter() - start_time,
            success=success,
            failed_robot_id=None if success else "window_conflict",
        )

    def _replan_suffix(
        self,
        robot: Robot,
        paths: dict[str, list[ContinuousPose]],
        replan_time: float,
    ) -> list[ContinuousPose] | None:
        """Replan one robot suffix in the local conflict window."""
        current_pose = self._pose_at_time(paths[robot.id], replan_time)
        if current_pose is None:
            return None

        prefix = [pose for pose in paths[robot.id] if pose[3] < replan_time]
        prefix.append(current_pose)
        planning_robot = replace(
            robot,
            start=RobotState(
                current_pose[0],
                current_pose[1],
                current_pose[2],
                robot.radius,
            ),
        )
        reservation_table = ReservationTable.from_paths(
            {
                robot_id: path
                for robot_id, path in paths.items()
                if robot_id != robot.id
            },
            self.single_robot_planner.collision_checker,
            self.single_robot_planner.dt,
            self.single_robot_planner.max_time_steps,
            cell_padding=self.single_robot_planner.reservation_padding,
            time_origin=replan_time,
        )
        planner = replace(
            self.single_robot_planner,
            reservation_table=reservation_table,
            time_origin=replan_time,
            allow_partial=True,
        )
        try:
            suffix = planner.plan(planning_robot)
        except ValueError:
            return None
        return prefix + [pose for pose in suffix if pose[3] > replan_time]

    def _first_window_conflict(
        self,
        paths: dict[str, list[ContinuousPose]],
        robots: tuple[Robot, ...],
    ) -> tuple[float, str, str] | None:
        """Return the first sampled conflict within the finite lookahead."""
        if not paths:
            return None
        dt = self.single_robot_planner.dt
        latest_time = min(
            max((path[-1][3] for path in paths.values() if path), default=0.0),
            self.window_steps * dt,
        )
        for step in range(int(round(latest_time / dt)) + 1):
            time = step * dt
            for index, first_robot in enumerate(robots):
                first_pose = self._pose_at_time(paths.get(first_robot.id, []), time)
                if first_pose is None:
                    continue
                for second_robot in robots[index + 1 :]:
                    second_pose = self._pose_at_time(paths.get(second_robot.id, []), time)
                    if second_pose is None:
                        continue
                    clearance = (
                        first_robot.radius
                        + second_robot.radius
                        + self.clearance_margin
                    )
                    if hypot(
                        first_pose[0] - second_pose[0],
                        first_pose[1] - second_pose[1],
                    ) < clearance:
                        return (time, first_robot.id, second_robot.id)
        return None

    @staticmethod
    def _pose_at_time(
        path: list[ContinuousPose],
        time: float,
    ) -> ContinuousPose | None:
        """Return linearly interpolated pose at a global time."""
        if not path:
            return None
        if time <= path[0][3]:
            return path[0]
        if time >= path[-1][3]:
            return path[-1]
        for start, end in zip(path, path[1:]):
            if start[3] <= time <= end[3]:
                duration = max(end[3] - start[3], 1e-9)
                alpha = (time - start[3]) / duration
                return (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                    start[2] + alpha * (end[2] - start[2]),
                    time,
                )
        return path[-1]
