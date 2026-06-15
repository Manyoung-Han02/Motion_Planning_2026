"""Single-robot kinodynamic A* planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import cos, exp, hypot, pi, sin

from warehouse_planning.models.robot import Robot, RobotState
from warehouse_planning.planning.collision import CollisionChecker

ContinuousPose = tuple[float, float, float, float]


@dataclass(frozen=True, order=True)
class DiscreteState:
    """Grid-time state used by kinodynamic A*."""

    x_index: int
    y_index: int
    theta_index: int
    time_index: int


@dataclass(frozen=True)
class MotionPrimitive:
    """Discrete control primitive for the lattice search."""

    name: str
    theta_delta: int
    distance: float
    turn_cost: float = 0.0
    wait_cost: float = 0.0


@dataclass(frozen=True)
class ReservationTable:
    """Time-indexed cell and edge reservations from already planned robots."""

    vertices: frozenset[tuple[int, int, int]] = frozenset()
    edges: frozenset[tuple[int, int, int, int, int]] = frozenset()

    @classmethod
    def from_paths(
        cls,
        paths: dict[str, list[ContinuousPose]],
        collision_checker: CollisionChecker,
        dt: float,
        max_time_steps: int,
        cell_padding: int = 0,
        time_origin: float = 0.0,
    ) -> "ReservationTable":
        """Build reservations from planned robot paths.

        Vertex key format is ``(time_index, row, col)``. Edge key format is
        ``(time_index, from_row, from_col, to_row, to_col)``. ``time_origin``
        lets rolling-horizon planners build reservations in the same local
        time frame used by the lattice search.
        """
        if cell_padding < 0:
            raise ValueError("cell_padding must be non-negative")
        if time_origin < 0.0:
            raise ValueError("time_origin must be non-negative")
        vertices: set[tuple[int, int, int]] = set()
        edges: set[tuple[int, int, int, int, int]] = set()
        for path in paths.values():
            indexed_cells = cls._path_to_indexed_cells(
                path,
                collision_checker,
                dt,
                max_time_steps,
                time_origin,
            )
            for time_index, row, col in indexed_cells:
                vertices.update(
                    cls._padded_vertices(
                        time_index,
                        row,
                        col,
                        cell_padding,
                        collision_checker,
                    )
                )
            for previous, current in zip(indexed_cells, indexed_cells[1:]):
                previous_time, previous_row, previous_col = previous
                _, current_row, current_col = current
                edges.add(
                    (
                        previous_time,
                        previous_row,
                        previous_col,
                        current_row,
                        current_col,
                    )
                )
        return cls(vertices=frozenset(vertices), edges=frozenset(edges))

    @staticmethod
    def _padded_vertices(
        time_index: int,
        row: int,
        col: int,
        cell_padding: int,
        collision_checker: CollisionChecker,
    ) -> set[tuple[int, int, int]]:
        """Return reserved cells around an occupied grid cell."""
        vertices: set[tuple[int, int, int]] = set()
        rows, cols = collision_checker.warehouse.shape
        max_row = rows - 1
        max_col = cols - 1
        for row_offset in range(-cell_padding, cell_padding + 1):
            for col_offset in range(-cell_padding, cell_padding + 1):
                padded_row = row + row_offset
                padded_col = col + col_offset
                if 0 <= padded_row <= max_row and 0 <= padded_col <= max_col:
                    vertices.add((time_index, padded_row, padded_col))
        return vertices

    @staticmethod
    def _path_to_indexed_cells(
        path: list[ContinuousPose],
        collision_checker: CollisionChecker,
        dt: float,
        max_time_steps: int,
        time_origin: float = 0.0,
    ) -> list[tuple[int, int, int]]:
        """Convert a continuous path to one reserved grid cell per time step."""
        if not path:
            return []

        pose_by_time = {
            int(round((pose[3] - time_origin) / dt)): pose
            for pose in path
            if 0 <= int(round((pose[3] - time_origin) / dt)) <= max_time_steps
        }
        indexed_cells: list[tuple[int, int, int]] = []
        last_pose = path[0]
        for time_index in range(max_time_steps + 1):
            pose = pose_by_time.get(time_index, last_pose)
            if time_index in pose_by_time:
                last_pose = pose
            row, col = collision_checker.warehouse.world_to_grid(pose[0], pose[1])
            indexed_cells.append((time_index, row, col))
        return indexed_cells

    def reserves_vertex(self, state: DiscreteState) -> bool:
        """Return whether a discrete state occupies a reserved cell."""
        return (state.time_index, state.y_index, state.x_index) in self.vertices

    def reserves_reverse_edge(
        self,
        current: DiscreteState,
        neighbor: DiscreteState,
    ) -> bool:
        """Return whether a transition swaps through a reserved edge."""
        reverse_edge = (
            current.time_index,
            neighbor.y_index,
            neighbor.x_index,
            current.y_index,
            current.x_index,
        )
        return reverse_edge in self.edges


@dataclass
class KinodynamicAStarPlanner:
    """Deterministic kinodynamic A* planner over simple motion primitives."""

    collision_checker: CollisionChecker
    dt: float
    theta_bins: int = 16
    step_distance: float | None = None
    goal_tolerance: float | None = None
    max_time_steps: int = 250
    distance_cost_weight: float = 1.0
    turn_cost: float = 0.15
    wait_cost: float = 0.5
    risk_weight: float = 0.0
    safety_distance: float = 1.0
    risk_sigma: float | None = None
    risk_time_offsets: tuple[float, ...] = (0.0,)
    risk_time_decay: float = 0.55
    heuristic_weight: float = 1.0
    reservation_padding: int = 0
    time_origin: float = 0.0
    allow_partial: bool = False
    reservation_table: ReservationTable | None = None
    _motion_primitives: tuple[MotionPrimitive, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize deterministic primitive order."""
        if self.theta_bins <= 0:
            raise ValueError("theta_bins must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.risk_weight < 0.0:
            raise ValueError("risk_weight must be non-negative")
        if self.heuristic_weight <= 0.0:
            raise ValueError("heuristic_weight must be positive")
        if self.reservation_padding < 0:
            raise ValueError("reservation_padding must be non-negative")
        if self.safety_distance <= 0.0:
            raise ValueError("safety_distance must be positive")
        if self.risk_time_decay < 0.0:
            raise ValueError("risk_time_decay must be non-negative")
        if self.time_origin < 0.0:
            raise ValueError("time_origin must be non-negative")

        distance = self.step_distance
        if distance is None:
            distance = max(self.collision_checker.warehouse.resolution, 0.5)
        if distance <= 0.0:
            raise ValueError("step_distance must be positive")
        self.step_distance = distance

        if self.goal_tolerance is None:
            self.goal_tolerance = max(distance, self.collision_checker.warehouse.resolution)
        if self.risk_sigma is None:
            self.risk_sigma = max(self.safety_distance / 2.0, 1e-6)
        if self.risk_sigma <= 0.0:
            raise ValueError("risk_sigma must be positive")

        self._motion_primitives = (
            MotionPrimitive("forward", theta_delta=0, distance=distance),
            MotionPrimitive(
                "forward_left",
                theta_delta=1,
                distance=distance,
                turn_cost=self.turn_cost,
            ),
            MotionPrimitive(
                "forward_right",
                theta_delta=-1,
                distance=distance,
                turn_cost=self.turn_cost,
            ),
            MotionPrimitive("wait", theta_delta=0, distance=0.0, wait_cost=self.wait_cost),
        )

    def plan(self, robot: Robot) -> list[ContinuousPose]:
        """Plan a collision-free path from start to goal.

        Returns:
            A list of continuous poses ``[(x, y, theta, t), ...]``.
        """
        if not self.collision_checker.is_state_valid(
            robot,
            robot.start,
            time=self.time_origin,
        ):
            raise ValueError(f"Invalid start state for robot {robot.id}")

        start = self._state_to_discrete(robot.start, time_index=0)
        if self._violates_reservation(start):
            raise ValueError(f"Reserved start state for robot {robot.id}")

        open_heap: list[tuple[float, int, DiscreteState]] = []
        tie_breaker = count()
        heappush(
            open_heap,
            (self.heuristic_weight * self._heuristic(robot.start.x, robot.start.y, robot), 0, start),
        )

        came_from: dict[DiscreteState, DiscreteState | None] = {start: None}
        cost_so_far: dict[DiscreteState, float] = {start: 0.0}
        best_state = start
        best_heuristic = self._heuristic(robot.start.x, robot.start.y, robot)

        while open_heap:
            _, _, current = heappop(open_heap)
            current_pose = self._discrete_to_pose(current, robot)
            current_heuristic = self._heuristic(current_pose[0], current_pose[1], robot)
            if current.time_index > 0 and current_heuristic < best_heuristic:
                best_state = current
                best_heuristic = current_heuristic

            if self._is_goal(current_pose, robot):
                return self._reconstruct_path(current, came_from, robot)

            if current.time_index >= self.max_time_steps:
                continue

            for primitive in self._motion_primitives:
                neighbor = self._apply_primitive(current, primitive)
                if neighbor.time_index > self.max_time_steps:
                    continue
                if self._violates_reservation(neighbor):
                    continue
                if self._violates_edge_swap(current, neighbor):
                    continue
                if not self._is_transition_valid(current, neighbor, robot):
                    continue

                risk_cost = self._transition_risk_cost(current, neighbor, robot)
                new_cost = cost_so_far[current] + self._primitive_cost(primitive) + risk_cost
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue

                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                nx, ny, _, _ = self._discrete_to_pose(neighbor, robot)
                priority = new_cost + self.heuristic_weight * self._heuristic(nx, ny, robot)
                heappush(open_heap, (priority, next(tie_breaker), neighbor))

        if self.allow_partial and best_state != start:
            return self._reconstruct_path(best_state, came_from, robot)

        raise ValueError(f"No kinodynamic A* path found for robot {robot.id}")

    def _state_to_discrete(self, state: RobotState, time_index: int) -> DiscreteState:
        """Convert a continuous robot state to a grid-time state."""
        row, col = self.collision_checker.warehouse.world_to_grid(state.x, state.y)
        return DiscreteState(
            x_index=col,
            y_index=row,
            theta_index=self._theta_to_index(state.theta),
            time_index=time_index,
        )

    def _discrete_to_pose(
        self,
        state: DiscreteState,
        robot: Robot,
    ) -> ContinuousPose:
        """Convert a discrete state to a continuous pose tuple."""
        x, y = self.collision_checker.warehouse.grid_to_world(
            state.y_index,
            state.x_index,
        )
        theta = self._index_to_theta(state.theta_index)
        return (x, y, theta, self.time_origin + state.time_index * self.dt)

    def _theta_to_index(self, theta: float) -> int:
        """Convert a continuous heading to the nearest theta bin."""
        wrapped = theta % (2.0 * pi)
        return int(round(wrapped / self._theta_resolution())) % self.theta_bins

    def _index_to_theta(self, theta_index: int) -> float:
        """Convert a theta bin to a continuous heading in radians."""
        return (theta_index % self.theta_bins) * self._theta_resolution()

    def _theta_resolution(self) -> float:
        """Return the angular width of one heading bin."""
        return 2.0 * pi / self.theta_bins

    def _apply_primitive(
        self,
        state: DiscreteState,
        primitive: MotionPrimitive,
    ) -> DiscreteState:
        """Apply one primitive to a discrete state."""
        next_theta_index = (state.theta_index + primitive.theta_delta) % self.theta_bins
        if primitive.distance == 0.0:
            return DiscreteState(
                x_index=state.x_index,
                y_index=state.y_index,
                theta_index=next_theta_index,
                time_index=state.time_index + 1,
            )

        x, y, _, _ = self._discrete_to_pose_indexed(state)
        theta = self._index_to_theta(next_theta_index)
        next_x = x + primitive.distance * cos(theta)
        next_y = y + primitive.distance * sin(theta)
        row, col = self.collision_checker.warehouse.world_to_grid(next_x, next_y)
        return DiscreteState(
            x_index=col,
            y_index=row,
            theta_index=next_theta_index,
            time_index=state.time_index + 1,
        )

    def _discrete_to_pose_indexed(self, state: DiscreteState) -> ContinuousPose:
        """Convert a discrete state without needing a robot instance."""
        x, y = self.collision_checker.warehouse.grid_to_world(
            state.y_index,
            state.x_index,
        )
        return (
            x,
            y,
            self._index_to_theta(state.theta_index),
            self.time_origin + state.time_index * self.dt,
        )

    def _is_transition_valid(
        self,
        current: DiscreteState,
        neighbor: DiscreteState,
        robot: Robot,
    ) -> bool:
        """Check sampled static and dynamic hard collisions along a transition."""
        start_x, start_y, start_theta, start_t = self._discrete_to_pose(current, robot)
        end_x, end_y, end_theta, end_t = self._discrete_to_pose(neighbor, robot)
        distance = hypot(end_x - start_x, end_y - start_y)
        sample_spacing = max(self.collision_checker.warehouse.resolution * 0.25, 1e-6)
        sample_count = max(1, int(distance / sample_spacing))

        for sample_index in range(sample_count + 1):
            alpha = sample_index / sample_count
            x = start_x + alpha * (end_x - start_x)
            y = start_y + alpha * (end_y - start_y)
            theta = self._interpolate_angle(start_theta, end_theta, alpha)
            t = start_t + alpha * (end_t - start_t)
            state = RobotState(x=x, y=y, theta=theta, radius=robot.radius)
            if self.collision_checker.collides_with_static_obstacle(state):
                return False
            if self.collision_checker.collides_with_dynamic_obstacle(state, t):
                return False
        return True

    def _violates_reservation(self, state: DiscreteState) -> bool:
        """Return whether a state violates a vertex reservation."""
        if self.reservation_table is None:
            return False
        return self.reservation_table.reserves_vertex(state)

    def _violates_edge_swap(
        self,
        current: DiscreteState,
        neighbor: DiscreteState,
    ) -> bool:
        """Return whether a transition violates an edge-swap reservation."""
        if self.reservation_table is None:
            return False
        return self.reservation_table.reserves_reverse_edge(current, neighbor)

    def _transition_risk_cost(
        self,
        current: DiscreteState,
        neighbor: DiscreteState,
        robot: Robot,
    ) -> float:
        """Return Gaussian dynamic-obstacle risk accumulated along a transition."""
        if self.risk_weight == 0.0 or not self.collision_checker.dynamic_obstacles:
            return 0.0

        start_x, start_y, start_theta, start_t = self._discrete_to_pose(current, robot)
        end_x, end_y, end_theta, end_t = self._discrete_to_pose(neighbor, robot)
        distance = hypot(end_x - start_x, end_y - start_y)
        sample_spacing = max(self.collision_checker.warehouse.resolution * 0.25, 1e-6)
        sample_count = max(1, int(distance / sample_spacing))
        total_risk = 0.0

        for sample_index in range(sample_count + 1):
            alpha = sample_index / sample_count
            x = start_x + alpha * (end_x - start_x)
            y = start_y + alpha * (end_y - start_y)
            theta = self._interpolate_angle(start_theta, end_theta, alpha)
            t = start_t + alpha * (end_t - start_t)
            state = RobotState(x=x, y=y, theta=theta, radius=robot.radius)
            total_risk += self.dynamic_risk_cost(state, t)

        return total_risk / (sample_count + 1)

    def dynamic_risk_cost(self, state: RobotState, time: float) -> float:
        """Return soft Gaussian risk cost near predicted dynamic obstacles."""
        if self.risk_weight == 0.0:
            return 0.0

        cost = 0.0
        for obstacle in self.collision_checker.dynamic_obstacles:
            for offset in self.risk_time_offsets:
                ox, oy = obstacle.predicted_position(max(0.0, time + offset))
                center_distance = hypot(state.x - ox, state.y - oy)
                collision_distance = state.radius + obstacle.radius
                clearance = center_distance - collision_distance
                if clearance <= 0.0:
                    return float("inf")
                if clearance <= self.safety_distance:
                    gaussian = self._gaussian_risk(clearance)
                    temporal_weight = 1.0 if abs(offset) <= 1e-9 else self.risk_time_decay
                    cost += self.risk_weight * temporal_weight * gaussian
        return cost

    def _gaussian_risk(self, clearance: float) -> float:
        """Evaluate a Gaussian risk field from obstacle-boundary clearance."""
        sigma = self.risk_sigma if self.risk_sigma is not None else 1.0
        return exp(-0.5 * (clearance / sigma) ** 2)

    def _primitive_cost(self, primitive: MotionPrimitive) -> float:
        """Return deterministic weighted cost for a primitive."""
        distance_cost = self.distance_cost_weight * primitive.distance
        return distance_cost + primitive.turn_cost + primitive.wait_cost

    def _heuristic(self, x: float, y: float, robot: Robot) -> float:
        """Euclidean distance-to-goal heuristic."""
        return self.distance_cost_weight * hypot(robot.goal.x - x, robot.goal.y - y)

    def _is_goal(self, pose: ContinuousPose, robot: Robot) -> bool:
        """Return whether a pose is within the configured goal tolerance."""
        x, y, theta, _ = pose
        position_close = hypot(robot.goal.x - x, robot.goal.y - y) <= self.goal_tolerance
        if not position_close:
            return False

        heading_error = abs(self._wrap_angle(theta - robot.goal.theta))
        return heading_error <= self._theta_resolution()

    def _reconstruct_path(
        self,
        goal: DiscreteState,
        came_from: dict[DiscreteState, DiscreteState | None],
        robot: Robot,
    ) -> list[ContinuousPose]:
        """Reconstruct a continuous pose path from the predecessor map."""
        states: list[DiscreteState] = []
        current: DiscreteState | None = goal
        while current is not None:
            states.append(current)
            current = came_from[current]
        states.reverse()

        path = [self._discrete_to_pose(state, robot) for state in states]
        if path:
            path[0] = (
                robot.start.x,
                robot.start.y,
                robot.start.theta,
                self.time_origin,
            )
        return path

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to ``[-pi, pi)``."""
        return (angle + pi) % (2.0 * pi) - pi

    def _interpolate_angle(self, start: float, end: float, alpha: float) -> float:
        """Interpolate heading along the shortest angular displacement."""
        delta = self._wrap_angle(end - start)
        return start + alpha * delta
