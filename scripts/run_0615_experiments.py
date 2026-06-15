"""Run the 0615 paper experiments and save all outputs in one folder."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import replace
from math import hypot
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warehouse_planning.config import ScenarioConfig, SimulationConfig, load_scenario_config
from warehouse_planning.evaluation.metrics import compute_path_metrics
from warehouse_planning.maps.warehouse_map import RectangleObstacle, WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose, KinodynamicAStarPlanner
from warehouse_planning.planning.prioritized import (
    IndependentPlanner,
    MultiRobotPlanResult,
    PrioritizedPlanner,
)
from warehouse_planning.planning.windowed import WindowedConflictReplanner
from warehouse_planning.visualization.clean_demo import generate_pedestrian_paths
from warehouse_planning.visualization.plotting import ROBOT_COLORS, WarehousePlotter
from warehouse_planning.visualization.smoothing import interpolate_path, pose_at_time


OUTPUT_DIR = PROJECT_ROOT / "results" / "0615_results"
SCENARIO_PATH = PROJECT_ROOT / "configs" / "warehouse_clean_demo.yaml"
NEAR_SMALL_M = 0.35
NEAR_LARGE_M = 1.00


def main() -> None:
    """Run all 0615 experiments and write tables/figures."""
    set_style()
    prepare_output_dir(OUTPUT_DIR)
    base_scenario = obstacle_aware_pedestrian_scenario(load_scenario_config(SCENARIO_PATH))
    ablation_scenario = make_human_crossing_ablation_scenario()

    comparison = run_algorithm_comparison(base_scenario)
    comparison.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)
    save_algorithm_comparison_plot(comparison)
    save_table_image(comparison, "algorithm_comparison_table")

    risk_ablation = run_risk_ablation(ablation_scenario)
    risk_ablation.to_csv(OUTPUT_DIR / "risk_ablation_metrics.csv", index=False)
    save_risk_ablation_plot(risk_ablation)
    save_risk_heatmap_overlay(ablation_scenario)

    window_ablation = run_window_ablation(base_scenario)
    window_ablation.to_csv(OUTPUT_DIR / "planning_window_ablation.csv", index=False)
    save_window_ablation_plot(window_ablation)

    diversity = run_scenario_diversity(base_scenario)
    diversity.to_csv(OUTPUT_DIR / "scenario_diversity_metrics.csv", index=False)
    save_scenario_diversity_plot(diversity)
    save_scenario_overview(base_scenario)

    write_experiment_notes()
    print(f"Saved 0615 experiments to {OUTPUT_DIR}")


def set_style() -> None:
    """Use a compact academic plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2933",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def prepare_output_dir(output_dir: Path) -> None:
    """Recreate the dedicated 0615 results directory."""
    resolved_output = output_dir.resolve()
    resolved_results = (PROJECT_ROOT / "results").resolve()
    if not str(resolved_output).startswith(str(resolved_results)):
        raise ValueError(f"Refusing to clean outside results/: {resolved_output}")
    if resolved_output.exists():
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def obstacle_aware_pedestrian_scenario(scenario: ScenarioConfig) -> ScenarioConfig:
    """Use valid obstacle-aware pedestrian rollouts as dynamic obstacles."""
    if not scenario.dynamic_obstacles:
        return scenario
    pedestrian_paths = generate_pedestrian_paths(
        scenario,
        duration=scenario.simulation.horizon,
        dt=scenario.simulation.dt,
    )
    dynamic_obstacles = tuple(
        DynamicObstacle(
            id=pedestrian.id,
            radius=pedestrian.radius,
            trajectory=tuple((pose[3], pose[0], pose[1]) for pose in pedestrian_paths[pedestrian.id]),
        )
        for pedestrian in scenario.dynamic_obstacles
    )
    return replace(scenario, dynamic_obstacles=dynamic_obstacles)


def run_algorithm_comparison(scenario: ScenarioConfig) -> pd.DataFrame:
    """Compare general, dynamic-obstacle, and proposed planners."""
    rows = []
    methods = make_algorithm_methods(scenario, window_steps=32)
    for method_name, planner in methods:
        result = planner.plan(scenario.robots)
        rows.append({"scenario": "warehouse_4r3h", "method": method_name, **metric_row(scenario, result)})
    return pd.DataFrame(rows)


def make_algorithm_methods(
    scenario: ScenarioConfig,
    window_steps: int,
) -> list[tuple[str, object]]:
    """Return planners used in the main comparison."""
    return [
        (
            "Independent A*",
            IndependentPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=0)),
        ),
        (
            "Prioritized Planning",
            PrioritizedPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=0)),
        ),
        (
            "Risk-aware Independent A*",
            IndependentPlanner(make_planner(scenario, risk_weight=8.0, reservation_padding=0)),
        ),
        (
            "Risk-aware Prioritized",
            PrioritizedPlanner(make_planner(scenario, risk_weight=8.0, reservation_padding=2)),
        ),
        (
            "CBS-style Reservations",
            PrioritizedPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=3)),
        ),
        (
            "Proposed Windowed Risk-Aware",
            WindowedConflictReplanner(
                make_planner(scenario, risk_weight=8.0, reservation_padding=3),
                window_steps=window_steps,
                repair_iterations=2,
                lookback_steps=4,
                clearance_margin=0.14,
            ),
        ),
    ]


def run_risk_ablation(scenario: ScenarioConfig) -> pd.DataFrame:
    """Run risk-weight and risk-spread ablations for one robot."""
    rows = []
    robot = scenario.robots[0]
    for weight in (0.0, 2.0, 4.0, 8.0, 12.0):
        planner = make_planner(
            scenario,
            risk_weight=weight,
            reservation_padding=0,
            safety_distance=1.6,
            risk_sigma=0.75,
            planner_horizon=55,
            theta_bins=12,
        )
        start_time = perf_counter()
        path = planner.plan(robot)
        result = single_robot_result(robot.id, path, planner_time=perf_counter() - start_time)
        rows.append({"ablation": "risk_weight", "value": weight, **metric_row(scenario, result)})

    for sigma in (0.45, 0.75, 1.05, 1.35):
        planner = make_planner(
            scenario,
            risk_weight=8.0,
            reservation_padding=0,
            safety_distance=1.6,
            risk_sigma=sigma,
            planner_horizon=55,
            theta_bins=12,
        )
        start_time = perf_counter()
        path = planner.plan(robot)
        result = single_robot_result(robot.id, path, planner_time=perf_counter() - start_time)
        rows.append({"ablation": "risk_sigma", "value": sigma, **metric_row(scenario, result)})
    return pd.DataFrame(rows)


def run_window_ablation(scenario: ScenarioConfig) -> pd.DataFrame:
    """Run the receding/conflict-window step count ablation."""
    rows = []
    for window_steps in (16, 24, 32, 40):
        planner = WindowedConflictReplanner(
            make_planner(scenario, risk_weight=8.0, reservation_padding=3),
            window_steps=window_steps,
            repair_iterations=2,
            lookback_steps=4,
            clearance_margin=0.14,
        )
        result = planner.plan(scenario.robots)
        rows.append(
            {
                "planning_window_steps": window_steps,
                "planning_window_seconds": window_steps * scenario.simulation.dt,
                **metric_row(scenario, result),
            }
        )
    return pd.DataFrame(rows)


def run_scenario_diversity(base_scenario: ScenarioConfig) -> pd.DataFrame:
    """Evaluate the proposed method on several map scales."""
    rows = []
    scenarios = [
        ("warehouse_4r3h", base_scenario),
        ("medium_clutter_5r4h", obstacle_aware_pedestrian_scenario(make_medium_clutter_scenario())),
        ("large_dense_6r5h", obstacle_aware_pedestrian_scenario(make_large_dense_scenario())),
    ]
    for scenario_name, scenario in scenarios:
        is_base = scenario_name == "warehouse_4r3h"
        planner = WindowedConflictReplanner(
            make_planner(
                scenario,
                risk_weight=8.0 if is_base else 7.0,
                reservation_padding=3 if is_base else 2,
                safety_distance=2.4 if is_base else 1.8,
                risk_sigma=1.0 if is_base else 0.85,
                planner_horizon=(
                    min(70, int(scenario.simulation.horizon / scenario.simulation.dt))
                    if is_base
                    else min(65, int(scenario.simulation.horizon / scenario.simulation.dt))
                ),
                theta_bins=12 if is_base else 8,
                step_distance=0.5 if is_base else 0.6,
                goal_tolerance=0.65 if is_base else 0.85,
            ),
            window_steps=32 if is_base else 28,
            repair_iterations=2,
            lookback_steps=4 if is_base else 3,
            clearance_margin=0.14,
        )
        result = planner.plan(scenario.robots)
        rows.append(
            {
                "scenario": scenario_name,
                "robots": len(scenario.robots),
                "pedestrians": len(scenario.dynamic_obstacles),
                "obstacles": len(scenario.warehouse.static_obstacles),
                "map_width": scenario.warehouse.width,
                "map_height": scenario.warehouse.height,
                **metric_row(scenario, result),
            }
        )
    return pd.DataFrame(rows)


def make_planner(
    scenario: ScenarioConfig,
    risk_weight: float,
    reservation_padding: int,
    safety_distance: float = 2.4,
    risk_sigma: float = 1.0,
    planner_horizon: int | None = None,
    theta_bins: int = 12,
    step_distance: float = 0.5,
    goal_tolerance: float = 0.65,
) -> KinodynamicAStarPlanner:
    """Create a kinodynamic lattice planner with paper-experiment defaults."""
    dynamic_obstacles = scenario.dynamic_obstacles if risk_weight > 0.0 else ()
    return KinodynamicAStarPlanner(
        collision_checker=CollisionChecker(
            warehouse=scenario.warehouse,
            dynamic_obstacles=dynamic_obstacles,
        ),
        dt=scenario.simulation.dt,
        theta_bins=theta_bins,
        step_distance=step_distance,
        goal_tolerance=goal_tolerance,
        max_time_steps=planner_horizon
        or min(70, int(scenario.simulation.horizon / scenario.simulation.dt)),
        risk_weight=risk_weight,
        safety_distance=safety_distance,
        risk_sigma=risk_sigma,
        risk_time_offsets=(-0.6, 0.0, 0.6, 1.2) if risk_weight > 0.0 else (0.0,),
        risk_time_decay=0.55,
        wait_cost=3.0 if risk_weight > 0.0 else 0.5,
        heuristic_weight=3.0 if risk_weight > 0.0 else 1.0,
        reservation_padding=reservation_padding,
        allow_partial=risk_weight > 0.0,
    )


def single_robot_result(
    robot_id: str,
    path: list[ContinuousPose],
    planner_time: float,
) -> MultiRobotPlanResult:
    """Wrap a single path in the common result container."""
    return MultiRobotPlanResult(
        paths={robot_id: path},
        planning_time=planner_time,
        success=bool(path),
    )


def metric_row(
    scenario: ScenarioConfig,
    result: MultiRobotPlanResult,
) -> dict[str, float | bool | str]:
    """Compute metrics with two human-near thresholds."""
    sampled = sample_paths(result.paths, scenario.simulation.dt)
    path_length = sum(compute_path_metrics(robot_id, path).path_length for robot_id, path in result.paths.items())
    makespan = max((path[-1][3] for path in result.paths.values() if path), default=0.0)
    robot_collisions = count_robot_collisions(sampled, scenario.robots)
    wall_collisions = count_wall_collisions(sampled, scenario.robots, scenario.warehouse)
    human_small = count_human_threshold_entries(sampled, scenario.robots, scenario.dynamic_obstacles, NEAR_SMALL_M)
    human_large = count_human_threshold_entries(sampled, scenario.robots, scenario.dynamic_obstacles, NEAR_LARGE_M)
    min_human_clearance = min_robot_human_clearance(sampled, scenario.robots, scenario.dynamic_obstacles)
    return {
        "success": result.success,
        "total_distance": path_length,
        "makespan": makespan,
        "computation_time": result.planning_time,
        "wall_collision_count": wall_collisions,
        "robot_collision_count": robot_collisions,
        "wall_or_robot_collision_count": wall_collisions + robot_collisions,
        f"human_near_count_{NEAR_SMALL_M:.2f}m": human_small,
        f"human_near_count_{NEAR_LARGE_M:.2f}m": human_large,
        "min_human_clearance": min_human_clearance,
        "failed_robot_id": result.failed_robot_id or "",
    }


def sample_paths(
    paths: dict[str, list[ContinuousPose]],
    dt: float,
) -> dict[str, list[ContinuousPose]]:
    """Sample all paths at synchronized dt timestamps."""
    makespan = max((path[-1][3] for path in paths.values() if path), default=0.0)
    times = np.arange(0.0, makespan + 0.5 * dt, dt)
    sampled: dict[str, list[ContinuousPose]] = {}
    for robot_id, path in paths.items():
        sampled[robot_id] = [
            pose
            for pose in (pose_at_time(path, float(time)) for time in times)
            if pose is not None
        ]
    return sampled


def count_robot_collisions(
    sampled: dict[str, list[ContinuousPose]],
    robots: tuple[Robot, ...],
) -> int:
    """Count synchronized robot-robot footprint collisions."""
    robot_by_id = {robot.id: robot for robot in robots}
    robot_ids = sorted(sampled)
    collision_count = 0
    for index, first_id in enumerate(robot_ids):
        for second_id in robot_ids[index + 1 :]:
            first_path = sampled[first_id]
            second_path = sampled[second_id]
            for first_pose, second_pose in zip(first_path, second_path):
                clearance = robot_by_id[first_id].radius + robot_by_id[second_id].radius
                if hypot(first_pose[0] - second_pose[0], first_pose[1] - second_pose[1]) <= clearance:
                    collision_count += 1
    return collision_count


def count_wall_collisions(
    sampled: dict[str, list[ContinuousPose]],
    robots: tuple[Robot, ...],
    warehouse: WarehouseMap,
) -> int:
    """Count sampled robot-wall or robot-shelf collisions.

    This uses the same occupancy-grid footprint model as the planner so the
    reported wall metric matches the algorithm's collision constraint.
    """
    robot_by_id = {robot.id: robot for robot in robots}
    count = 0
    for robot_id, path in sampled.items():
        robot = robot_by_id[robot_id]
        for x, y, _, _ in path:
            if warehouse.is_occupied(x, y, margin=robot.radius):
                count += 1
    return count


def count_human_threshold_entries(
    sampled: dict[str, list[ContinuousPose]],
    robots: tuple[Robot, ...],
    dynamic_obstacles: tuple[DynamicObstacle, ...],
    threshold: float,
) -> int:
    """Count samples where robot-human clearance is within a threshold."""
    robot_by_id = {robot.id: robot for robot in robots}
    count = 0
    for robot_id, path in sampled.items():
        robot = robot_by_id[robot_id]
        for x, y, _, time in path:
            for obstacle in dynamic_obstacles:
                ox, oy = obstacle.position_at(time)
                clearance = hypot(x - ox, y - oy) - (robot.radius + obstacle.radius)
                if clearance <= threshold:
                    count += 1
    return count


def min_robot_human_clearance(
    sampled: dict[str, list[ContinuousPose]],
    robots: tuple[Robot, ...],
    dynamic_obstacles: tuple[DynamicObstacle, ...],
) -> float:
    """Return minimum robot-human clearance over all samples."""
    if not dynamic_obstacles or not sampled:
        return float("inf")
    robot_by_id = {robot.id: robot for robot in robots}
    minimum = float("inf")
    for robot_id, path in sampled.items():
        robot = robot_by_id[robot_id]
        for x, y, _, time in path:
            for obstacle in dynamic_obstacles:
                ox, oy = obstacle.position_at(time)
                minimum = min(minimum, hypot(x - ox, y - oy) - (robot.radius + obstacle.radius))
    return minimum


def circle_overlaps_rectangle(
    x: float,
    y: float,
    radius: float,
    obstacle: RectangleObstacle,
) -> bool:
    """Return whether a circle overlaps an axis-aligned rectangle."""
    xmin, ymin, xmax, ymax = obstacle.bounds
    nearest_x = min(max(x, xmin), xmax)
    nearest_y = min(max(y, ymin), ymax)
    return hypot(x - nearest_x, y - nearest_y) <= radius


def save_algorithm_comparison_plot(frame: pd.DataFrame) -> None:
    """Save a multi-panel algorithm comparison plot."""
    metrics = [
        ("total_distance", "Total distance [m]"),
        ("computation_time", "Computation time [s]"),
        ("wall_or_robot_collision_count", "Wall/robot collisions"),
        (f"human_near_count_{NEAR_SMALL_M:.2f}m", f"Human near count <= {NEAR_SMALL_M:.2f}m"),
        (f"human_near_count_{NEAR_LARGE_M:.2f}m", f"Human near count <= {NEAR_LARGE_M:.2f}m"),
        ("min_human_clearance", "Min human clearance [m]"),
    ]
    colors = ["#6b7280", "#4b5563", "#94a3b8", "#0891b2", "#64748b", "#2563eb"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.bar(frame["method"], frame[metric], color=colors, width=0.72)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", linewidth=0.35, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Algorithm Comparison on 4-Robot / 3-Pedestrian Warehouse", y=1.02, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, "algorithm_comparison")


def save_risk_ablation_plot(frame: pd.DataFrame) -> None:
    """Save risk-parameter ablation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4))
    for ax, ablation, title in zip(axes, ("risk_weight", "risk_sigma"), ("Risk weight", "Risk sigma")):
        subset = frame[frame["ablation"] == ablation].sort_values("value")
        ax.plot(subset["value"], subset["total_distance"], marker="o", label="distance")
        ax.plot(subset["value"], subset["min_human_clearance"], marker="s", label="min clearance")
        ax.plot(subset["value"], subset[f"human_near_count_{NEAR_LARGE_M:.2f}m"], marker="^", label="near <= 1.0m")
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.grid(axis="y", linewidth=0.35, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Risk Parameter Ablation", y=1.05, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, "risk_ablation_plot")


def save_risk_heatmap_overlay(scenario: ScenarioConfig) -> None:
    """Show how paths deform as the human-risk weight changes."""
    robot = scenario.robots[0]
    weights = (0.0, 2.0, 4.0, 8.0, 12.0)
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(weights)))
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    plotter = WarehousePlotter(style="clean")
    plotter.configure_clean_axis(ax, scenario)
    plotter.draw_risk_field(ax, scenario, time=3.6, sigma=0.8, resolution=0.08)
    plotter.draw_clean_obstacles(ax, scenario)
    plotter.draw_pedestrians(ax, scenario, time=3.6)
    for index, weight in enumerate(weights):
        planner = make_planner(
            scenario,
            risk_weight=weight,
            reservation_padding=0,
            safety_distance=1.6,
            risk_sigma=0.75,
            planner_horizon=55,
            theta_bins=12,
        )
        path = interpolate_path(planner.plan(robot), samples_per_segment=12)
        ax.plot(
            [pose[0] for pose in path],
            [pose[1] for pose in path],
            color=colors[index],
            linewidth=1.7 + 0.18 * index,
            alpha=0.92,
            label=f"risk={weight:g}",
            solid_capstyle="round",
            zorder=6,
        )
    ax.scatter(robot.start.x, robot.start.y, s=58, facecolors="white", edgecolors="#111827", linewidths=1.2, zorder=10)
    ax.scatter(robot.goal.x, robot.goal.y, s=70, marker="x", color="#111827", linewidths=2.0, zorder=10)
    for pedestrian in scenario.dynamic_obstacles:
        xs = [point[1] for point in pedestrian.trajectory]
        ys = [point[2] for point in pedestrian.trajectory]
        ax.plot(xs, ys, color="#111827", linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.55, zorder=5)
    ax.set_title("Human-Risk Heatmap and Risk-Weight Path Overlay")
    ax.legend(loc="upper center", ncol=len(weights), bbox_to_anchor=(0.5, -0.03), fontsize=8)
    save_figure(fig, "risk_heatmap_path_overlay")


def save_window_ablation_plot(frame: pd.DataFrame) -> None:
    """Save planning-window ablation curves."""
    fig, ax1 = plt.subplots(figsize=(8.8, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(frame["planning_window_steps"], frame["total_distance"], marker="o", color="#2563eb", label="distance")
    ax1.plot(frame["planning_window_steps"], frame["makespan"], marker="s", color="#0891b2", label="makespan")
    ax2.plot(
        frame["planning_window_steps"],
        frame["wall_or_robot_collision_count"],
        marker="^",
        color="#dc2626",
        label="wall/robot collisions",
    )
    ax1.set_xlabel("planning window [steps]")
    ax1.set_ylabel("distance / makespan")
    ax2.set_ylabel("collision count")
    ax1.grid(axis="y", linewidth=0.35, alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax1.set_title("Receding-Horizon / Conflict-Window Ablation")
    fig.tight_layout()
    save_figure(fig, "planning_window_ablation")


def save_scenario_diversity_plot(frame: pd.DataFrame) -> None:
    """Save compact scalability/diversity metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    metrics = [
        ("total_distance", "Total distance [m]"),
        ("computation_time", "Computation time [s]"),
        (f"human_near_count_{NEAR_LARGE_M:.2f}m", "Human near count <= 1.0m"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(frame["scenario"], frame[metric], color="#2563eb", width=0.65)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", linewidth=0.35, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Scenario Diversity with Proposed Planner", y=1.04, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, "scenario_diversity_plot")


def save_scenario_overview(base_scenario: ScenarioConfig) -> None:
    """Save map snapshots for the diverse scenario set."""
    scenarios = [
        ("4R / 3H warehouse", base_scenario),
        ("5R / 4H clutter", obstacle_aware_pedestrian_scenario(make_medium_clutter_scenario())),
        ("6R / 5H dense", obstacle_aware_pedestrian_scenario(make_large_dense_scenario())),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    plotter = WarehousePlotter(style="clean")
    for ax, (title, scenario) in zip(axes, scenarios):
        plotter.configure_clean_axis(ax, scenario)
        plotter.draw_clean_obstacles(ax, scenario)
        plotter.draw_pedestrians(ax, scenario, time=3.0)
        for robot_index, robot in enumerate(scenario.robots):
            color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
            ax.scatter(robot.start.x, robot.start.y, s=28, facecolors="white", edgecolors=color, linewidths=1.1, zorder=8)
            ax.scatter(robot.goal.x, robot.goal.y, s=36, marker="x", color=color, linewidths=1.5, zorder=8)
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, "scenario_overview")


def save_table_image(frame: pd.DataFrame, name: str) -> None:
    """Save a readable table image for slides."""
    display_columns = [
        "method",
        "total_distance",
        "computation_time",
        "wall_or_robot_collision_count",
        f"human_near_count_{NEAR_SMALL_M:.2f}m",
        f"human_near_count_{NEAR_LARGE_M:.2f}m",
        "min_human_clearance",
    ]
    table = frame[display_columns].copy()
    for column in ("total_distance", "computation_time", "min_human_clearance"):
        table[column] = table[column].map(lambda value: f"{value:.2f}")
    fig, ax = plt.subplots(figsize=(12.2, 2.8))
    ax.axis("off")
    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(7.6)
    mpl_table.scale(1.0, 1.34)
    for (row, _), cell in mpl_table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
    save_figure(fig, name)


def save_figure(fig, name: str) -> None:
    """Save a figure as PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_human_crossing_ablation_scenario() -> ScenarioConfig:
    """Return a small single-robot scenario that exposes risk sensitivity."""
    spec = RobotSpec(wheelbase=0.7, max_speed=1.05, max_steering_angle=0.55)
    robot = Robot(
        "robot_1",
        spec,
        RobotState(0.75, 2.45, 0.0, 0.24),
        RobotState(8.15, 2.45, 0.0, 0.24),
    )
    pedestrian = DynamicObstacle(
        "pedestrian_1",
        0.24,
        ((0.0, 4.2, 0.55), (3.6, 4.2, 2.45), (7.5, 4.2, 4.35), (10.0, 4.2, 0.55)),
    )
    scenario = ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=10.0),
        warehouse=WarehouseMap(
            width=9.0,
            height=5.0,
            resolution=0.25,
            static_obstacles=(
                RectangleObstacle("lower_shelf", 2.2, 0.0, 0.55, 1.25),
                RectangleObstacle("upper_shelf", 6.0, 3.75, 0.55, 1.25),
            ),
        ),
        robots=(robot,),
        dynamic_obstacles=(pedestrian,),
    )
    return obstacle_aware_pedestrian_scenario(scenario)


def make_medium_clutter_scenario() -> ScenarioConfig:
    """Return a medium scenario with more robots, pedestrians, and shelves."""
    spec = RobotSpec(wheelbase=0.7, max_speed=1.05, max_steering_angle=0.55)
    starts = ((0.8, 0.8, 0.0), (13.2, 2.6, 3.14), (0.8, 4.3, 0.0), (13.2, 6.0, 3.14), (0.8, 7.8, 0.0))
    goals = ((13.2, 0.8, 0.0), (0.8, 2.6, 3.14), (13.2, 4.3, 0.0), (0.8, 6.0, 3.14), (13.2, 7.8, 0.0))
    robots = tuple(
        Robot(
            f"robot_{index + 1}",
            spec,
            RobotState(start[0], start[1], start[2], 0.24),
            RobotState(goal[0], goal[1], goal[2], 0.24),
        )
        for index, (start, goal) in enumerate(zip(starts, goals))
    )
    obstacles = tuple(
        RectangleObstacle(f"shelf_{index + 1}", x, y, w, h)
        for index, (x, y, w, h) in enumerate(
            (
                (2.6, 1.25, 1.2, 0.55),
                (5.1, 1.25, 1.2, 0.55),
                (7.6, 1.25, 1.2, 0.55),
                (10.1, 1.25, 1.2, 0.55),
                (2.6, 5.0, 1.2, 0.55),
                (5.1, 5.0, 1.2, 0.55),
                (7.6, 5.0, 1.2, 0.55),
                (10.1, 5.0, 1.2, 0.55),
            )
        )
    )
    pedestrians = (
        DynamicObstacle("ped_1", 0.23, ((0.0, 3.0, 0.7), (6.0, 3.0, 8.0), (12.0, 3.0, 0.7))),
        DynamicObstacle("ped_2", 0.23, ((0.0, 11.0, 8.0), (6.0, 9.2, 1.0), (12.0, 11.0, 8.0))),
        DynamicObstacle("ped_3", 0.23, ((0.0, 6.8, 0.7), (6.0, 6.8, 8.0), (12.0, 6.8, 0.7))),
        DynamicObstacle("ped_4", 0.23, ((0.0, 1.4, 4.2), (6.0, 12.2, 4.2), (12.0, 1.4, 4.2))),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=14.0),
        warehouse=WarehouseMap(14.0, 8.6, 0.25, obstacles),
        robots=robots,
        dynamic_obstacles=pedestrians,
    )


def make_large_dense_scenario() -> ScenarioConfig:
    """Return a larger dense scenario for scalability visualization."""
    spec = RobotSpec(wheelbase=0.7, max_speed=1.1, max_steering_angle=0.55)
    starts = (
        (0.9, 0.9, 0.0),
        (17.1, 2.8, 3.14),
        (0.9, 4.6, 0.0),
        (17.1, 7.4, 3.14),
        (0.9, 9.2, 0.0),
        (17.1, 11.1, 3.14),
    )
    goals = (
        (17.1, 0.9, 0.0),
        (0.9, 2.8, 3.14),
        (17.1, 4.6, 0.0),
        (0.9, 7.4, 3.14),
        (17.1, 9.2, 0.0),
        (0.9, 11.1, 3.14),
    )
    robots = tuple(
        Robot(
            f"robot_{index + 1}",
            spec,
            RobotState(start[0], start[1], start[2], 0.24),
            RobotState(goal[0], goal[1], goal[2], 0.24),
        )
        for index, (start, goal) in enumerate(zip(starts, goals))
    )
    obstacles = tuple(
        RectangleObstacle(f"shelf_{index + 1}", x, y, w, h)
        for index, (x, y, w, h) in enumerate(
            (
                (2.8, 1.65, 1.5, 0.55),
                (6.0, 1.65, 1.5, 0.55),
                (9.2, 1.65, 1.5, 0.55),
                (12.4, 1.65, 1.5, 0.55),
                (2.8, 5.6, 1.5, 0.55),
                (6.0, 5.6, 1.5, 0.55),
                (9.2, 5.6, 1.5, 0.55),
                (12.4, 5.6, 1.5, 0.55),
                (2.8, 9.95, 1.5, 0.55),
                (6.0, 9.95, 1.5, 0.55),
                (9.2, 9.95, 1.5, 0.55),
                (12.4, 9.95, 1.5, 0.55),
            )
        )
    )
    pedestrians = (
        DynamicObstacle("ped_1", 0.23, ((0.0, 4.8, 0.7), (7.0, 4.8, 11.0), (14.0, 4.8, 0.7))),
        DynamicObstacle("ped_2", 0.23, ((0.0, 8.2, 11.0), (7.0, 8.2, 0.8), (14.0, 8.2, 11.0))),
        DynamicObstacle("ped_3", 0.23, ((0.0, 11.6, 0.7), (7.0, 11.6, 11.0), (14.0, 11.6, 0.7))),
        DynamicObstacle("ped_4", 0.23, ((0.0, 1.4, 6.0), (7.0, 16.5, 6.0), (14.0, 1.4, 6.0))),
        DynamicObstacle("ped_5", 0.23, ((0.0, 16.5, 3.0), (7.0, 2.0, 8.6), (14.0, 16.5, 3.0))),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=16.0),
        warehouse=WarehouseMap(18.0, 12.0, 0.25, obstacles),
        robots=robots,
        dynamic_obstacles=pedestrians,
    )


def write_experiment_notes() -> None:
    """Write a concise experiment manifest."""
    notes = [
        "# 0615 Results",
        "",
        "All newly generated experiment outputs are stored in this folder.",
        "",
        "Human-near thresholds:",
        f"- Small threshold: clearance <= {NEAR_SMALL_M:.2f} m beyond combined robot/person radii.",
        f"- Large threshold: clearance <= {NEAR_LARGE_M:.2f} m beyond combined robot/person radii.",
        "",
        "Algorithms compared:",
        "- Independent A*: no robot coordination and no dynamic-obstacle risk.",
        "- Prioritized Planning: time-expanded robot reservations, no human-risk cost.",
        "- Risk-aware Independent A*: dynamic-obstacle risk but no robot coordination.",
        "- Risk-aware Prioritized: dynamic-obstacle risk plus prioritized reservations.",
        "- CBS-style Reservations: a lightweight reservation proxy, not full optimal CBS.",
        "- Proposed Windowed Risk-Aware: risk-aware lattice A* with bounded conflict-window repair.",
        "",
        "Additional experiment ideas:",
        "- Add random seeds for each scenario family.",
        "- Add ORCA/RVO as a holonomic dynamic-obstacle baseline if a fair nonholonomic projection layer is added.",
        "- Increase scenario-count sweeps after the paper figures are finalized.",
    ]
    (OUTPUT_DIR / "experiment_notes.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
