"""Benchmark scenarios and evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from warehouse_planning.config import ScenarioConfig, SimulationConfig
from warehouse_planning.evaluation.metrics import evaluate_multi_robot_plan
from warehouse_planning.maps.warehouse_map import RectangleObstacle, WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import KinodynamicAStarPlanner
from warehouse_planning.planning.prioritized import (
    IndependentPlanner,
    MultiRobotPlanResult,
    PrioritizedPlanner,
)
from warehouse_planning.planning.windowed import WindowedConflictReplanner


@dataclass(frozen=True)
class BenchmarkScenario:
    """Named scenario used in benchmark tables and plots."""

    name: str
    scenario: ScenarioConfig


@dataclass(frozen=True)
class BenchmarkMethod:
    """Named planner factory used by the benchmark runner."""

    name: str
    planner_factory: Callable[[ScenarioConfig], object]


def build_benchmark_scenarios() -> list[BenchmarkScenario]:
    """Return deterministic scenarios for multi-robot benchmark evaluation."""
    return [
        BenchmarkScenario("narrow_aisle_crossing", _narrow_aisle_crossing()),
        BenchmarkScenario("human_crossing_path", _human_crossing_path()),
        BenchmarkScenario("bottleneck_warehouse", _bottleneck_warehouse()),
        BenchmarkScenario("random_start_goal_tasks", _random_start_goal_tasks()),
    ]


def build_benchmark_methods(risk_weight: float = 8.0) -> list[BenchmarkMethod]:
    """Return planner methods compared by the benchmark."""
    return [
        BenchmarkMethod(
            "Independent A*",
            lambda scenario: IndependentPlanner(_base_planner(scenario, risk_weight=0.0)),
        ),
        BenchmarkMethod(
            "Prioritized Planning",
            lambda scenario: PrioritizedPlanner(_base_planner(scenario, risk_weight=0.0)),
        ),
        BenchmarkMethod(
            "Risk-aware Prioritized Planning",
            lambda scenario: PrioritizedPlanner(
                _base_planner(scenario, risk_weight=risk_weight)
            ),
        ),
        BenchmarkMethod(
            "Proposed Windowed Risk-Aware",
            lambda scenario: WindowedConflictReplanner(
                _base_planner(scenario, risk_weight=risk_weight),
                window_steps=24,
                repair_iterations=2,
                lookback_steps=3,
                clearance_margin=0.14,
            ),
        ),
    ]


def run_benchmark(
    output_dir: str | Path,
    risk_weight: float = 8.0,
) -> pd.DataFrame:
    """Run all benchmark methods and save CSV and bar plots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for benchmark_scenario in build_benchmark_scenarios():
        scenario = benchmark_scenario.scenario
        for method in build_benchmark_methods(risk_weight=risk_weight):
            planner = method.planner_factory(scenario)
            result = planner.plan(scenario.robots)
            if not isinstance(result, MultiRobotPlanResult):
                raise TypeError(f"{method.name} returned unsupported result type")

            metrics = evaluate_multi_robot_plan(
                scenario.robots,
                result,
                dt=scenario.simulation.dt,
                warehouse=scenario.warehouse,
                dynamic_obstacles=scenario.dynamic_obstacles,
            )
            rows.append(
                {
                    "scenario": benchmark_scenario.name,
                    "method": method.name,
                    "success_rate": metrics.success_rate,
                    "robot_robot_collision_count": metrics.robot_robot_collision_count,
                    "dynamic_obstacle_near_miss_count": (
                        metrics.dynamic_obstacle_near_miss_count
                    ),
                    "path_length": metrics.total_path_length,
                    "makespan": metrics.makespan,
                    "computation_time": metrics.planning_time,
                    "same_cell_conflict_count": metrics.same_cell_conflict_count,
                    "edge_swap_conflict_count": metrics.edge_swap_conflict_count,
                    "success": result.success,
                    "failed_robot_id": result.failed_robot_id or "",
                }
            )

    frame = pd.DataFrame(rows)
    csv_path = output / "benchmark_results.csv"
    frame.to_csv(csv_path, index=False)
    save_benchmark_bar_plots(frame, output)
    return frame


def save_benchmark_bar_plots(frame: pd.DataFrame, output_dir: str | Path) -> None:
    """Save grouped bar plots for benchmark metrics."""
    output = Path(output_dir)
    metrics = [
        "success_rate",
        "robot_robot_collision_count",
        "dynamic_obstacle_near_miss_count",
        "path_length",
        "makespan",
        "computation_time",
    ]
    for metric in metrics:
        pivot = frame.pivot(index="scenario", columns="method", values=metric)
        ax = pivot.plot(kind="bar", figsize=(11, 5), rot=25)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)


def _base_planner(
    scenario: ScenarioConfig,
    risk_weight: float,
) -> KinodynamicAStarPlanner:
    """Create a kinodynamic A* planner with benchmark defaults."""
    collision_checker = CollisionChecker(
        warehouse=scenario.warehouse,
        dynamic_obstacles=scenario.dynamic_obstacles,
    )
    return KinodynamicAStarPlanner(
        collision_checker=collision_checker,
        dt=scenario.simulation.dt,
        theta_bins=8,
        step_distance=0.5,
        goal_tolerance=0.55,
        max_time_steps=int(scenario.simulation.horizon / scenario.simulation.dt),
        risk_weight=risk_weight,
        safety_distance=1.0,
        risk_sigma=0.5,
        risk_time_offsets=(-0.4, 0.0, 0.4),
        risk_time_decay=0.5,
        wait_cost=2.0 if risk_weight > 0.0 else 0.5,
        heuristic_weight=2.0 if risk_weight > 0.0 else 1.0,
        reservation_padding=1 if risk_weight > 0.0 else 0,
        allow_partial=risk_weight > 0.0,
    )


def _narrow_aisle_crossing() -> ScenarioConfig:
    spec = _robot_spec()
    robots = (
        Robot("r1", spec, RobotState(0.75, 2.0, 0.0, 0.2), RobotState(5.25, 2.0, 0.0, 0.2)),
        Robot("r2", spec, RobotState(5.25, 2.0, 3.14159, 0.2), RobotState(0.75, 2.0, 3.14159, 0.2)),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=8.0),
        warehouse=WarehouseMap(
            width=6.0,
            height=4.0,
            resolution=0.25,
            static_obstacles=(
                RectangleObstacle("upper_block", 0.0, 2.75, 6.0, 1.25),
                RectangleObstacle("lower_block", 0.0, 0.0, 6.0, 1.25),
            ),
        ),
        robots=robots,
        dynamic_obstacles=(),
    )


def _human_crossing_path() -> ScenarioConfig:
    spec = _robot_spec()
    robots = (
        Robot("r1", spec, RobotState(0.75, 1.0, 0.0, 0.2), RobotState(5.25, 1.0, 0.0, 0.2)),
        Robot("r2", spec, RobotState(0.75, 3.0, 0.0, 0.2), RobotState(5.25, 3.0, 0.0, 0.2)),
    )
    human = DynamicObstacle(
        id="human",
        radius=0.25,
        trajectory=((0.0, 3.0, 0.5), (4.0, 3.0, 3.5), (8.0, 3.0, 0.5)),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=8.0),
        warehouse=WarehouseMap(width=6.0, height=4.0, resolution=0.25),
        robots=robots,
        dynamic_obstacles=(human,),
    )


def _bottleneck_warehouse() -> ScenarioConfig:
    spec = _robot_spec()
    robots = (
        Robot("r1", spec, RobotState(0.75, 0.75, 0.0, 0.2), RobotState(6.25, 4.25, 0.0, 0.2)),
        Robot("r2", spec, RobotState(6.25, 0.75, 3.14159, 0.2), RobotState(0.75, 4.25, 3.14159, 0.2)),
        Robot("r3", spec, RobotState(0.75, 4.25, 0.0, 0.2), RobotState(6.25, 0.75, 0.0, 0.2)),
    )
    obstacles = (
        RectangleObstacle("left_shelf", 2.5, 0.0, 0.5, 2.2),
        RectangleObstacle("right_shelf", 4.0, 2.8, 0.5, 2.2),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=10.0),
        warehouse=WarehouseMap(width=7.0, height=5.0, resolution=0.25, static_obstacles=obstacles),
        robots=robots,
        dynamic_obstacles=(),
    )


def _random_start_goal_tasks() -> ScenarioConfig:
    spec = _robot_spec()
    starts = ((0.75, 0.75, 0.0), (0.75, 4.25, 0.0), (3.5, 0.75, 1.5708))
    goals = ((6.25, 4.25, 0.0), (6.25, 0.75, 0.0), (3.5, 4.25, 1.5708))
    robots = tuple(
        Robot(
            f"r{index + 1}",
            spec,
            RobotState(start[0], start[1], start[2], 0.2),
            RobotState(goal[0], goal[1], goal[2], 0.2),
        )
        for index, (start, goal) in enumerate(zip(starts, goals))
    )
    obstacle = DynamicObstacle(
        id="worker",
        radius=0.25,
        trajectory=((0.0, 1.5, 2.5), (5.0, 5.5, 2.5), (10.0, 1.5, 2.5)),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=10.0),
        warehouse=WarehouseMap(
            width=7.0,
            height=5.0,
            resolution=0.25,
            static_obstacles=(
                RectangleObstacle("shelf_a", 2.0, 1.4, 0.5, 2.2),
                RectangleObstacle("shelf_b", 4.5, 1.4, 0.5, 2.2),
            ),
        ),
        robots=robots,
        dynamic_obstacles=(obstacle,),
    )


def _robot_spec() -> RobotSpec:
    """Return the common benchmark robot specification."""
    return RobotSpec(wheelbase=0.7, max_speed=1.0, max_steering_angle=0.5)
