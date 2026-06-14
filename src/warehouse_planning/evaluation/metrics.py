"""Basic trajectory evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import pandas as pd

from warehouse_planning.maps.warehouse_map import WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose
from warehouse_planning.planning.prioritized import MultiRobotPlanResult


@dataclass(frozen=True)
class PathMetrics:
    """Summary metrics for a planned trajectory."""

    robot_id: str
    path_length: float
    duration: float
    num_states: int


@dataclass(frozen=True)
class MultiRobotMetrics:
    """Summary metrics for one multi-robot planning run."""

    success_rate: float
    robot_robot_collision_count: int
    dynamic_obstacle_near_miss_count: int
    same_cell_conflict_count: int
    edge_swap_conflict_count: int
    total_path_length: float
    makespan: float
    planning_time: float


def compute_path_metrics(robot_id: str, path: list[ContinuousPose]) -> PathMetrics:
    """Compute simple geometric metrics for a single planned path."""
    length = 0.0
    for previous, current in zip(path, path[1:]):
        length += hypot(current[0] - previous[0], current[1] - previous[1])

    duration = 0.0
    if path:
        duration = path[-1][3] - path[0][3]

    return PathMetrics(
        robot_id=robot_id,
        path_length=length,
        duration=duration,
        num_states=len(path),
    )


def metrics_to_frame(metrics: list[PathMetrics]) -> pd.DataFrame:
    """Convert path metrics to a pandas DataFrame for experiment logging."""
    return pd.DataFrame([metric.__dict__ for metric in metrics])


def evaluate_multi_robot_plan(
    robots: tuple[Robot, ...],
    result: MultiRobotPlanResult,
    dt: float,
    warehouse: WarehouseMap,
    dynamic_obstacles: tuple[DynamicObstacle, ...] = (),
    near_miss_distance: float = 0.75,
) -> MultiRobotMetrics:
    """Compute baseline metrics for synchronized multi-robot paths."""
    robot_by_id = {robot.id: robot for robot in robots}
    path_lengths = [
        compute_path_metrics(robot_id, path).path_length
        for robot_id, path in result.paths.items()
    ]
    makespan = max((path[-1][3] for path in result.paths.values() if path), default=0.0)
    max_time_index = int(round(makespan / dt)) if dt > 0.0 else 0
    planned_robot_count = len(result.paths)
    success_rate = planned_robot_count / len(robots) if robots else 0.0

    circular_collisions = 0
    dynamic_near_misses = 0
    same_cell_conflicts = 0
    edge_swap_conflicts = 0
    robot_ids = sorted(result.paths)

    for time_index in range(max_time_index + 1):
        time = time_index * dt
        for robot_id in robot_ids:
            pose = _pose_at_time_index(result.paths[robot_id], time_index, dt)
            if pose is None:
                continue
            robot = robot_by_id[robot_id]
            for obstacle in dynamic_obstacles:
                ox, oy = obstacle.predicted_position(time)
                clearance = hypot(pose[0] - ox, pose[1] - oy)
                collision_distance = robot.radius + obstacle.radius
                if collision_distance < clearance <= collision_distance + near_miss_distance:
                    dynamic_near_misses += 1

        for index, robot_a_id in enumerate(robot_ids):
            for robot_b_id in robot_ids[index + 1 :]:
                pose_a = _pose_at_time_index(result.paths[robot_a_id], time_index, dt)
                pose_b = _pose_at_time_index(result.paths[robot_b_id], time_index, dt)
                if pose_a is None or pose_b is None:
                    continue

                robot_a = robot_by_id[robot_a_id]
                robot_b = robot_by_id[robot_b_id]
                distance = hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1])
                if distance <= robot_a.radius + robot_b.radius:
                    circular_collisions += 1

                row_a, col_a = warehouse.world_to_grid(pose_a[0], pose_a[1])
                row_b, col_b = warehouse.world_to_grid(pose_b[0], pose_b[1])
                if row_a == row_b and col_a == col_b:
                    same_cell_conflicts += 1

                if time_index == max_time_index:
                    continue
                next_a = _pose_at_time_index(
                    result.paths[robot_a_id],
                    time_index + 1,
                    dt,
                )
                next_b = _pose_at_time_index(
                    result.paths[robot_b_id],
                    time_index + 1,
                    dt,
                )
                if next_a is None or next_b is None:
                    continue
                next_row_a, next_col_a = warehouse.world_to_grid(next_a[0], next_a[1])
                next_row_b, next_col_b = warehouse.world_to_grid(next_b[0], next_b[1])
                if (
                    row_a == next_row_b
                    and col_a == next_col_b
                    and row_b == next_row_a
                    and col_b == next_col_a
                ):
                    edge_swap_conflicts += 1

    return MultiRobotMetrics(
        success_rate=success_rate,
        robot_robot_collision_count=circular_collisions,
        dynamic_obstacle_near_miss_count=dynamic_near_misses,
        same_cell_conflict_count=same_cell_conflicts,
        edge_swap_conflict_count=edge_swap_conflicts,
        total_path_length=sum(path_lengths),
        makespan=makespan,
        planning_time=result.planning_time,
    )


def multi_robot_metrics_to_frame(metrics: list[MultiRobotMetrics]) -> pd.DataFrame:
    """Convert multi-robot metrics to a pandas DataFrame."""
    return pd.DataFrame([metric.__dict__ for metric in metrics])


def _pose_at_time_index(
    path: list[ContinuousPose],
    time_index: int,
    dt: float,
) -> ContinuousPose | None:
    """Return the path pose at a time index, holding the final pose afterward."""
    if not path:
        return None

    for pose in path:
        if int(round(pose[3] / dt)) == time_index:
            return pose

    if int(round(path[-1][3] / dt)) < time_index:
        return path[-1]
    return None
