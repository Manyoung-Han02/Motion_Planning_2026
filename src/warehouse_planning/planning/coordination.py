"""Lightweight time coordination for multi-robot paths."""

from __future__ import annotations

from math import hypot

import numpy as np

from warehouse_planning.models.robot import Robot
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose


def coordinate_local_waits(
    paths: dict[str, list[ContinuousPose]],
    robots: tuple[Robot, ...],
    time_step: float = 0.2,
    wait_step: float = 0.4,
    max_total_wait: float = 6.0,
    clearance_margin: float = 0.14,
) -> dict[str, list[ContinuousPose]]:
    """Insert local waits near conflicts while preserving simultaneous starts."""
    scheduled: dict[str, list[ContinuousPose]] = {}
    robot_by_id = {robot.id: robot for robot in robots}

    for robot in robots:
        candidate = list(paths.get(robot.id, []))
        inserted_wait = 0.0
        while inserted_wait + wait_step <= max_total_wait:
            conflict_time = first_conflict_against_scheduled(
                candidate,
                robot,
                scheduled,
                robot_by_id,
                time_step=time_step,
                clearance_margin=clearance_margin,
            )
            if conflict_time is None:
                break

            wait_candidates: list[tuple[float, list[ContinuousPose]]] = []
            for lookahead in (1.2, 2.0, 2.8, 3.8, 4.8, 5.8):
                wait_time = min(
                    max(candidate[0][3] + time_step, conflict_time - lookahead),
                    max(candidate[-1][3] - time_step, candidate[0][3] + time_step),
                )
                if wait_segment_conflicts(
                    candidate,
                    robot,
                    scheduled,
                    robot_by_id,
                    wait_time=wait_time,
                    duration=max_total_wait - inserted_wait,
                    time_step=time_step,
                    clearance_margin=clearance_margin,
                ):
                    continue
                waited_path = insert_wait(candidate, wait_time, wait_step)
                conflicts = count_conflicts_against_scheduled(
                    waited_path,
                    robot,
                    scheduled,
                    robot_by_id,
                    time_step=time_step,
                    clearance_margin=clearance_margin,
                )
                wait_distance_from_start = max(wait_time - candidate[0][3], 0.0)
                score = conflicts * 1000.0 + waited_path[-1][3] - 0.03 * wait_distance_from_start
                wait_candidates.append((score, waited_path))

            if not wait_candidates:
                break
            candidate = min(wait_candidates, key=lambda item: item[0])[1]
            inserted_wait += wait_step

        scheduled[robot.id] = candidate

    return {robot.id: scheduled.get(robot.id, []) for robot in robots}


def wait_segment_conflicts(
    path: list[ContinuousPose],
    robot: Robot,
    scheduled_paths: dict[str, list[ContinuousPose]],
    robot_by_id: dict[str, Robot],
    wait_time: float,
    duration: float,
    time_step: float,
    clearance_margin: float,
) -> bool:
    """Return whether holding a pose would overlap scheduled robots."""
    wait_pose = pose_at_time(path, wait_time)
    if wait_pose is None:
        return False
    for sample_time in np.arange(wait_time, wait_time + duration + time_step, time_step):
        time = float(sample_time)
        for other_id, other_path in scheduled_paths.items():
            other_pose = pose_at_time(other_path, time)
            if other_pose is None:
                continue
            other_robot = robot_by_id[other_id]
            clearance = robot.radius + other_robot.radius + clearance_margin
            if hypot(wait_pose[0] - other_pose[0], wait_pose[1] - other_pose[1]) < clearance:
                return True
    return False


def first_conflict_against_scheduled(
    candidate_path: list[ContinuousPose],
    robot: Robot,
    scheduled_paths: dict[str, list[ContinuousPose]],
    robot_by_id: dict[str, Robot],
    time_step: float,
    clearance_margin: float,
) -> float | None:
    """Return the first sampled conflict against already scheduled robots."""
    if not candidate_path or not scheduled_paths:
        return None
    latest_time = max(
        [candidate_path[-1][3]]
        + [path[-1][3] for path in scheduled_paths.values() if path]
    )
    for sample_time in np.arange(0.0, latest_time + time_step, time_step):
        time = float(sample_time)
        pose = pose_at_time(candidate_path, time)
        if pose is None:
            continue
        for other_id, other_path in scheduled_paths.items():
            other_pose = pose_at_time(other_path, time)
            if other_pose is None:
                continue
            other_robot = robot_by_id[other_id]
            clearance = robot.radius + other_robot.radius + clearance_margin
            if hypot(pose[0] - other_pose[0], pose[1] - other_pose[1]) < clearance:
                return time
    return None


def count_conflicts_against_scheduled(
    candidate_path: list[ContinuousPose],
    robot: Robot,
    scheduled_paths: dict[str, list[ContinuousPose]],
    robot_by_id: dict[str, Robot],
    time_step: float,
    clearance_margin: float,
) -> int:
    """Count sampled conflicts against already scheduled robots."""
    if not candidate_path or not scheduled_paths:
        return 0
    conflict_count = 0
    latest_time = max(
        [candidate_path[-1][3]]
        + [path[-1][3] for path in scheduled_paths.values() if path]
    )
    for sample_time in np.arange(0.0, latest_time + time_step, time_step):
        time = float(sample_time)
        pose = pose_at_time(candidate_path, time)
        if pose is None:
            continue
        for other_id, other_path in scheduled_paths.items():
            other_pose = pose_at_time(other_path, time)
            if other_pose is None:
                continue
            other_robot = robot_by_id[other_id]
            clearance = robot.radius + other_robot.radius + clearance_margin
            if hypot(pose[0] - other_pose[0], pose[1] - other_pose[1]) < clearance:
                conflict_count += 1
    return conflict_count


def pose_at_time(
    path: list[ContinuousPose],
    time: float,
) -> ContinuousPose | None:
    """Return a smooth interpolated pose from a timed path."""
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
            beta = alpha * alpha * (3.0 - 2.0 * alpha)
            return (
                start[0] + beta * (end[0] - start[0]),
                start[1] + beta * (end[1] - start[1]),
                start[2] + beta * (end[2] - start[2]),
                time,
            )
    return path[-1]


def insert_wait(
    path: list[ContinuousPose],
    wait_time: float,
    duration: float,
) -> list[ContinuousPose]:
    """Insert a stationary segment into a timed path."""
    if not path or duration <= 0.0:
        return list(path)
    wait_pose = pose_at_time(path, wait_time)
    if wait_pose is None:
        return list(path)
    before = [pose for pose in path if pose[3] < wait_time]
    after = [
        (x, y, theta, time + duration)
        for x, y, theta, time in path
        if time > wait_time
    ]
    return before + [
        wait_pose,
        (wait_pose[0], wait_pose[1], wait_pose[2], wait_time + duration),
    ] + after


def count_sampled_path_conflicts(
    paths: dict[str, list[ContinuousPose]],
    robots: tuple[Robot, ...],
    time_step: float = 0.2,
    clearance_margin: float = 0.14,
) -> int:
    """Count sampled robot-robot conflicts in already timed paths."""
    active_robots = tuple(robot for robot in robots if paths.get(robot.id))
    if not active_robots:
        return 0
    latest_time = max(paths[robot.id][-1][3] for robot in active_robots)
    conflict_count = 0
    for sample_time in np.arange(0.0, latest_time + time_step, time_step):
        time = float(sample_time)
        for index, robot in enumerate(active_robots):
            first_pose = pose_at_time(paths[robot.id], time)
            if first_pose is None:
                continue
            for other in active_robots[index + 1 :]:
                other_pose = pose_at_time(paths[other.id], time)
                if other_pose is None:
                    continue
                clearance = robot.radius + other.radius + clearance_margin
                distance = hypot(
                    first_pose[0] - other_pose[0],
                    first_pose[1] - other_pose[1],
                )
                if distance < clearance:
                    conflict_count += 1
    return conflict_count
