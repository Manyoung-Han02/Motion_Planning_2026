"""Receding-horizon simulation for dynamic-obstacle-aware planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from warehouse_planning.config import ScenarioConfig
from warehouse_planning.models.robot import Robot, RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import (
    ContinuousPose,
    KinodynamicAStarPlanner,
)


@dataclass(frozen=True)
class SimulationSnapshot:
    """Observed state of robots and dynamic obstacles at one simulation time."""

    time: float
    robot_positions: dict[str, tuple[float, float]]
    obstacle_positions: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class RecedingHorizonConfig:
    """Configuration for closed-loop receding-horizon execution."""

    horizon_steps: int = 24
    execute_steps: int = 3
    max_cycles: int = 25
    risk_weight: float = 8.0
    safety_distance: float = 1.2
    risk_sigma: float | None = None
    step_distance: float | None = None
    goal_tolerance: float | None = None


@dataclass(frozen=True)
class RecedingHorizonStep:
    """Data recorded for one observe-predict-plan-execute cycle."""

    cycle: int
    time: float
    robot_state: RobotState
    obstacle_predictions: dict[str, list[tuple[float, float, float]]]
    planned_path: list[ContinuousPose]
    executed_path: list[ContinuousPose]


@dataclass(frozen=True)
class RecedingHorizonResult:
    """Closed-loop simulation result."""

    robot_id: str
    executed_path: list[ContinuousPose]
    steps: list[RecedingHorizonStep]
    reached_goal: bool


@dataclass
class Simulator:
    """Simulator for receding-horizon single-robot planning experiments."""

    scenario: ScenarioConfig

    def snapshot(
        self,
        time: float = 0.0,
        robot_states: dict[str, RobotState] | None = None,
    ) -> SimulationSnapshot:
        """Return observed robot and dynamic obstacle positions at a time."""
        if robot_states is None:
            robot_states = {robot.id: robot.start for robot in self.scenario.robots}

        return SimulationSnapshot(
            time=time,
            robot_positions={
                robot_id: (state.x, state.y) for robot_id, state in robot_states.items()
            },
            obstacle_positions={
                obstacle.id: obstacle.predicted_position(time)
                for obstacle in self.scenario.dynamic_obstacles
            },
        )

    def run_receding_horizon_single_robot(
        self,
        config: RecedingHorizonConfig,
        robot_index: int = 0,
    ) -> RecedingHorizonResult:
        """Run observe-predict-plan-execute cycles for one robot."""
        if not self.scenario.robots:
            raise ValueError("Scenario has no robots")
        if not self.scenario.dynamic_obstacles:
            raise ValueError("Scenario has no dynamic obstacles")
        if config.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if config.execute_steps <= 0:
            raise ValueError("execute_steps must be positive")
        if config.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")

        base_robot = self.scenario.robots[robot_index]
        dt = self.scenario.simulation.dt
        current_time = 0.0
        current_state = base_robot.start
        executed_path: list[ContinuousPose] = [
            (current_state.x, current_state.y, current_state.theta, current_time)
        ]
        steps: list[RecedingHorizonStep] = []
        goal_tolerance = config.goal_tolerance or max(
            self.scenario.warehouse.resolution,
            config.step_distance or 0.5,
        )

        for cycle in range(config.max_cycles):
            robot = replace(base_robot, start=current_state)
            predictions = self.predict_dynamic_obstacles(
                start_time=current_time,
                horizon_steps=config.horizon_steps,
                dt=dt,
            )
            collision_checker = CollisionChecker(
                warehouse=self.scenario.warehouse,
                dynamic_obstacles=self.scenario.dynamic_obstacles,
            )
            planner = KinodynamicAStarPlanner(
                collision_checker=collision_checker,
                dt=dt,
                step_distance=config.step_distance,
                goal_tolerance=goal_tolerance,
                max_time_steps=config.horizon_steps,
                risk_weight=config.risk_weight,
                safety_distance=config.safety_distance,
                risk_sigma=config.risk_sigma,
                time_origin=current_time,
                allow_partial=True,
            )
            planned_path = planner.plan(robot)
            executed_segment = planned_path[1 : config.execute_steps + 1]
            if not executed_segment:
                break

            executed_path.extend(executed_segment)
            last_pose = executed_segment[-1]
            current_time = last_pose[3]
            current_state = RobotState(
                x=last_pose[0],
                y=last_pose[1],
                theta=last_pose[2],
                radius=current_state.radius,
            )
            steps.append(
                RecedingHorizonStep(
                    cycle=cycle,
                    time=planned_path[0][3],
                    robot_state=robot.start,
                    obstacle_predictions=predictions,
                    planned_path=planned_path,
                    executed_path=list(executed_segment),
                )
            )

            if self._is_goal_reached(current_state, base_robot.goal, goal_tolerance):
                return RecedingHorizonResult(
                    robot_id=base_robot.id,
                    executed_path=executed_path,
                    steps=steps,
                    reached_goal=True,
                )

        return RecedingHorizonResult(
            robot_id=base_robot.id,
            executed_path=executed_path,
            steps=steps,
            reached_goal=self._is_goal_reached(current_state, base_robot.goal, goal_tolerance),
        )

    def predict_dynamic_obstacles(
        self,
        start_time: float,
        horizon_steps: int,
        dt: float,
    ) -> dict[str, list[tuple[float, float, float]]]:
        """Predict dynamic obstacle positions over a finite horizon."""
        predictions: dict[str, list[tuple[float, float, float]]] = {}
        for obstacle in self.scenario.dynamic_obstacles:
            points = []
            for step in range(horizon_steps + 1):
                time = start_time + step * dt
                x, y = obstacle.predicted_position(time)
                points.append((time, x, y))
            predictions[obstacle.id] = points
        return predictions

    def export_animation(
        self,
        result: RecedingHorizonResult,
        output_path: str | Path,
        fps: int = 6,
    ) -> None:
        """Export a GIF or MP4 animation of the receding-horizon run."""
        frames = [
            self.render_receding_horizon_frame(result, frame_index)
            for frame_index in range(len(result.executed_path))
        ]
        if not frames:
            raise ValueError("Cannot export animation with no frames")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".gif":
            imageio.mimsave(output, frames, duration=1000 / fps)
        else:
            imageio.mimsave(output, frames, fps=fps)

    def render_receding_horizon_frame(
        self,
        result: RecedingHorizonResult,
        frame_index: int,
    ) -> np.ndarray:
        """Render one animation frame as an RGB image array."""
        os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from warehouse_planning.visualization.plotting import WarehousePlotter

        pose = result.executed_path[min(frame_index, len(result.executed_path) - 1)]
        robot = self._robot_at_pose(self.scenario.robots[0], pose)
        scenario = replace(self.scenario, robots=(robot,))
        plotter = WarehousePlotter()
        fig, ax = plotter.draw_scenario(scenario, time=pose[3])
        plotter.draw_path(
            ax,
            result.executed_path[: frame_index + 1],
            label="executed",
            color="#ff7f0e",
        )
        current_step = self._step_for_time(result, pose[3])
        if current_step is not None:
            plotter.draw_path(
                ax,
                current_step.planned_path,
                label="current horizon",
                color="#2ca02c",
            )
            for obstacle_id, prediction in current_step.obstacle_predictions.items():
                xs = [point[1] for point in prediction]
                ys = [point[2] for point in prediction]
                ax.plot(xs, ys, color="#d62728", linewidth=1.4, alpha=0.35)
                if prediction:
                    ax.text(xs[-1], ys[-1], f"{obstacle_id} H", fontsize=7)

        ax.set_title(f"Receding-horizon simulation at t={pose[3]:.1f}s")
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = rgba.reshape((height, width, 4))[:, :, :3].copy()
        plt.close(fig)
        return frame

    @staticmethod
    def _robot_at_pose(robot: Robot, pose: ContinuousPose) -> Robot:
        """Return a robot whose start state is set to a pose tuple."""
        state = RobotState(x=pose[0], y=pose[1], theta=pose[2], radius=robot.radius)
        return replace(robot, start=state)

    @staticmethod
    def _is_goal_reached(
        state: RobotState,
        goal: RobotState,
        tolerance: float,
    ) -> bool:
        """Return whether a state is close enough to the goal position."""
        return hypot(state.x - goal.x, state.y - goal.y) <= tolerance

    @staticmethod
    def _step_for_time(
        result: RecedingHorizonResult,
        time: float,
    ) -> RecedingHorizonStep | None:
        """Return the latest receding-horizon step active at a given time."""
        active_step = None
        for step in result.steps:
            if step.time <= time:
                active_step = step
            else:
                break
        return active_step
