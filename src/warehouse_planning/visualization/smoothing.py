"""Smooth trajectory utilities for presentation-quality visualization."""

from __future__ import annotations

from math import atan2, cos, hypot, pi, sin

from warehouse_planning.planning.kinodynamic_astar import ContinuousPose


def interpolate_path(
    path: list[ContinuousPose],
    samples_per_segment: int = 8,
    corner_cut: float = 0.38,
    smoothing_iterations: int = 4,
) -> list[ContinuousPose]:
    """Roll out a coarse planner path into smooth continuous poses.

    The planner already uses nonholonomic forward/turn primitives, but its
    output lives on a lattice. This helper first rounds the lattice corners
    and then adds a continuous visual rollout with heading-aware cubic Hermite
    interpolation and tangent-based headings.
    """
    if len(path) <= 1:
        return list(path)
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")
    if not 0.0 <= corner_cut < 0.5:
        raise ValueError("corner_cut must be in the range [0.0, 0.5)")
    if smoothing_iterations < 0:
        raise ValueError("smoothing_iterations must be non-negative")

    control_path = _round_path_corners(
        path,
        corner_cut=corner_cut,
        iterations=smoothing_iterations,
    )
    tangents = _estimate_tangents(control_path)

    positions: list[tuple[float, float, float]] = []
    for index, (start, end) in enumerate(zip(control_path, control_path[1:])):
        x0, y0, _, t0 = start
        x1, y1, _, t1 = end
        m0x, m0y = tangents[index]
        m1x, m1y = tangents[index + 1]
        duration = max(t1 - t0, 1e-9)
        for sample in range(samples_per_segment):
            alpha = sample / samples_per_segment
            x, y = _hermite_position(
                alpha,
                (x0, y0),
                (x1, y1),
                (m0x * duration, m0y * duration),
                (m1x * duration, m1y * duration),
            )
            t = t0 + alpha * (t1 - t0)
            positions.append((x, y, t))

    positions.append((control_path[-1][0], control_path[-1][1], control_path[-1][3]))
    headings = _headings_from_positions(positions, fallback_theta=control_path[0][2])
    return [
        (x, y, theta, t)
        for (x, y, t), theta in zip(positions, headings)
    ]


def display_path(
    path: list[ContinuousPose],
    samples_per_segment: int = 14,
    dense_time_step: float = 0.08,
    corner_cut: float = 0.42,
    smoothing_iterations: int = 5,
) -> list[ContinuousPose]:
    """Return a presentation path with enough samples for smooth rendering."""
    if len(path) <= 1:
        return list(path)
    if _is_dense_enough(path, dense_time_step) and smoothing_iterations == 0:
        return list(path)
    return interpolate_path(
        path,
        samples_per_segment=samples_per_segment,
        corner_cut=corner_cut,
        smoothing_iterations=smoothing_iterations,
    )


def pose_at_time(path: list[ContinuousPose], time: float) -> ContinuousPose | None:
    """Return an interpolated pose at a given continuous time."""
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
            beta = _smoothstep(alpha)
            theta = start[2] + beta * _wrap_angle(end[2] - start[2])
            return (
                start[0] + beta * (end[0] - start[0]),
                start[1] + beta * (end[1] - start[1]),
                theta,
                time,
            )
    return path[-1]


def path_until(path: list[ContinuousPose], time: float) -> list[ContinuousPose]:
    """Return path samples up to a time, including an interpolated endpoint."""
    pose = pose_at_time(path, time)
    if pose is None:
        return []
    return [sample for sample in path if sample[3] <= time] + [pose]


def path_between(
    path: list[ContinuousPose],
    start_time: float,
    end_time: float,
) -> list[ContinuousPose]:
    """Return path samples between two times with interpolated endpoints."""
    start_pose = pose_at_time(path, start_time)
    end_pose = pose_at_time(path, end_time)
    if start_pose is None or end_pose is None:
        return []
    middle = [sample for sample in path if start_time < sample[3] < end_time]
    return [start_pose] + middle + [end_pose]


def _smoothstep(alpha: float) -> float:
    """Cubic easing for smooth visual interpolation."""
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _blend_angle(first: float, second: float, second_weight: float) -> float:
    """Blend two angles along the shorter angular direction."""
    return first + second_weight * _wrap_angle(second - first)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + pi) % (2.0 * pi) - pi


def _round_path_corners(
    path: list[ContinuousPose],
    corner_cut: float,
    iterations: int,
) -> list[ContinuousPose]:
    """Round lattice corners with repeated open Chaikin subdivision."""
    rounded = _dedupe_adjacent_poses(path)
    if len(rounded) <= 2 or corner_cut == 0.0 or iterations == 0:
        return rounded

    for _ in range(iterations):
        next_path: list[ContinuousPose] = [rounded[0]]
        for start, end in zip(rounded, rounded[1:]):
            next_path.append(_interpolate_pose(start, end, corner_cut))
            next_path.append(_interpolate_pose(start, end, 1.0 - corner_cut))
        next_path.append(rounded[-1])
        rounded = _dedupe_adjacent_poses(next_path)
        if len(rounded) <= 2:
            break
    return rounded


def _dedupe_adjacent_poses(path: list[ContinuousPose]) -> list[ContinuousPose]:
    """Drop exact adjacent duplicates that can create zero-time segments."""
    if not path:
        return []
    deduped = [path[0]]
    for pose in path[1:]:
        previous = deduped[-1]
        same_position = hypot(pose[0] - previous[0], pose[1] - previous[1]) <= 1e-9
        same_time = abs(pose[3] - previous[3]) <= 1e-9
        if same_position and same_time:
            continue
        deduped.append(pose)
    return deduped


def _interpolate_pose(
    start: ContinuousPose,
    end: ContinuousPose,
    alpha: float,
) -> ContinuousPose:
    """Interpolate a pose while blending heading along the shortest angle."""
    return (
        start[0] + alpha * (end[0] - start[0]),
        start[1] + alpha * (end[1] - start[1]),
        _blend_angle(start[2], end[2], alpha),
        start[3] + alpha * (end[3] - start[3]),
    )


def _estimate_tangents(path: list[ContinuousPose]) -> list[tuple[float, float]]:
    """Estimate bounded, heading-aware path tangents for cubic interpolation."""
    points = [(pose[0], pose[1]) for pose in path]
    times = [pose[3] for pose in path]
    tangents: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if index == 0:
            tangent = _endpoint_tangent(path[index], points[1], times[1])
        elif index == len(points) - 1:
            tangent = _endpoint_tangent(
                path[index],
                points[index - 1],
                times[index - 1],
                reverse=True,
            )
        else:
            previous = points[index - 1]
            neighbor = points[index + 1]
            dt = max(times[index + 1] - times[index - 1], 1e-9)
            chord_tangent = (
                (neighbor[0] - previous[0]) / dt,
                (neighbor[1] - previous[1]) / dt,
            )
            heading_tangent = _heading_tangent(path[index], _norm(chord_tangent))
            tangent = _blend_vectors(chord_tangent, heading_tangent, second_weight=0.35)
        tangents.append(_limit_tangent(tangent, max_norm=2.0))
    return tangents


def _endpoint_tangent(
    pose: ContinuousPose,
    neighbor: tuple[float, float],
    neighbor_time: float,
    reverse: bool = False,
) -> tuple[float, float]:
    """Use the endpoint heading without allowing it to dominate the curve."""
    x, y, theta, time = pose
    dt = max(abs(neighbor_time - time), 1e-9)
    if reverse:
        chord = ((x - neighbor[0]) / dt, (y - neighbor[1]) / dt)
    else:
        chord = ((neighbor[0] - x) / dt, (neighbor[1] - y) / dt)
    heading = _heading_tangent(pose, _norm(chord))
    return _blend_vectors(chord, heading, second_weight=0.55)


def _heading_tangent(pose: ContinuousPose, norm: float) -> tuple[float, float]:
    """Return a tangent vector aligned with a pose heading."""
    _, _, theta, _ = pose
    return (norm * cos(theta), norm * sin(theta))


def _blend_vectors(
    first: tuple[float, float],
    second: tuple[float, float],
    second_weight: float,
) -> tuple[float, float]:
    """Blend two 2D vectors."""
    first_weight = 1.0 - second_weight
    return (
        first_weight * first[0] + second_weight * second[0],
        first_weight * first[1] + second_weight * second[1],
    )


def _norm(vector: tuple[float, float]) -> float:
    """Return the Euclidean norm of a 2D vector."""
    return hypot(vector[0], vector[1])


def _is_dense_enough(path: list[ContinuousPose], dense_time_step: float) -> bool:
    """Return whether the path already has enough samples for rendering."""
    if dense_time_step <= 0.0:
        return False
    gaps = [current[3] - previous[3] for previous, current in zip(path, path[1:])]
    positive_gaps = [gap for gap in gaps if gap > 1e-9]
    if not positive_gaps:
        return False
    return max(positive_gaps) <= dense_time_step


def _limit_tangent(
    tangent: tuple[float, float],
    max_norm: float,
) -> tuple[float, float]:
    """Limit tangent magnitude to avoid visual overshoot around corners."""
    norm = hypot(tangent[0], tangent[1])
    if norm <= max_norm or norm <= 1e-9:
        return tangent
    scale = max_norm / norm
    return (tangent[0] * scale, tangent[1] * scale)


def _hermite_position(
    alpha: float,
    p0: tuple[float, float],
    p1: tuple[float, float],
    m0: tuple[float, float],
    m1: tuple[float, float],
) -> tuple[float, float]:
    """Evaluate a cubic Hermite position."""
    a2 = alpha * alpha
    a3 = a2 * alpha
    h00 = 2.0 * a3 - 3.0 * a2 + 1.0
    h10 = a3 - 2.0 * a2 + alpha
    h01 = -2.0 * a3 + 3.0 * a2
    h11 = a3 - a2
    return (
        h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
        h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1],
    )


def _headings_from_positions(
    positions: list[tuple[float, float, float]],
    fallback_theta: float,
) -> list[float]:
    """Derive continuous headings from position tangents."""
    headings: list[float] = []
    previous_heading = fallback_theta
    for index, position in enumerate(positions):
        if index < len(positions) - 1:
            next_position = positions[index + 1]
            dx = next_position[0] - position[0]
            dy = next_position[1] - position[1]
        else:
            previous_position = positions[index - 1]
            dx = position[0] - previous_position[0]
            dy = position[1] - previous_position[1]

        if hypot(dx, dy) > 1e-9:
            raw_heading = atan2(dy, dx)
            previous_heading = previous_heading + 0.45 * _wrap_angle(
                raw_heading - previous_heading
            )
        headings.append(previous_heading)
    return headings
