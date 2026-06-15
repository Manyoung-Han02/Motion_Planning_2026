"""Run compact CVPR-style experiments and ablations."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warehouse_planning.config import ScenarioConfig, SimulationConfig, load_scenario_config
from warehouse_planning.evaluation.benchmark import build_benchmark_scenarios
from warehouse_planning.evaluation.metrics import evaluate_multi_robot_plan
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
from warehouse_planning.visualization.smoothing import (
    interpolate_path,
    path_between,
    path_until,
    pose_at_time,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "cvpr_experiments"
SCENARIO_PATH = PROJECT_ROOT / "configs" / "warehouse_clean_demo.yaml"


def main() -> None:
    """Generate compact experiment tables and figures for the final paper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Regenerate only the obstacle-aware demo video in results/cvpr_experiments.",
    )
    args = parser.parse_args()

    set_style()
    if args.video_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_windowed_demo_video(load_scenario_config(SCENARIO_PATH))
        print(f"Saved CVPR demo video to {OUTPUT_DIR / 'demo_video.mp4'}")
        return

    prepare_output_dir(OUTPUT_DIR)
    scenario = load_scenario_config(SCENARIO_PATH)
    comparison = run_algorithm_comparison(scenario)
    comparison.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)
    save_algorithm_comparison_plot(comparison)

    ablation = run_ablation_study(scenario)
    ablation.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False)
    save_ablation_plot(ablation)
    save_risk_weight_overlay(scenario)
    save_windowed_demo_video(scenario)
    save_scenario_catalog()
    write_experiment_notes()
    print(f"Saved CVPR-style experiments to {OUTPUT_DIR}")


def set_style() -> None:
    """Configure paper-style Matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2933",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_output_dir(output_dir: Path) -> None:
    """Recreate the CVPR experiment output directory."""
    resolved_output = output_dir.resolve()
    resolved_results = (PROJECT_ROOT / "results").resolve()
    if not str(resolved_output).startswith(str(resolved_results)):
        raise ValueError(f"Refusing to clean outside results/: {resolved_output}")
    if resolved_output.exists():
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def run_algorithm_comparison(scenario: ScenarioConfig) -> pd.DataFrame:
    """Run the main four-method comparison on the final scenario."""
    rows = []
    methods = (
        (
            "Independent A*",
            IndependentPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=0)),
        ),
        (
            "Prioritized Planning",
            PrioritizedPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=0)),
        ),
        (
            "CBS-style proxy",
            PrioritizedPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=2)),
        ),
        (
            "Proposed Windowed Risk-Aware",
            WindowedConflictReplanner(
                make_planner(scenario, risk_weight=8.0, reservation_padding=3),
                window_steps=32,
                repair_iterations=2,
                lookback_steps=4,
                clearance_margin=0.14,
            ),
        ),
    )
    for method, planner in methods:
        result = planner.plan(scenario.robots)
        rows.append({"method": method, **metrics_row(scenario, result)})
    return pd.DataFrame(rows)


def run_ablation_study(scenario: ScenarioConfig) -> pd.DataFrame:
    """Run a compact one-factor-at-a-time ablation set."""
    rows = []
    for risk_weight in (0.0, 4.0, 8.0, 12.0):
        result = run_proposed(
            scenario,
            risk_weight=risk_weight,
            safety_distance=2.4,
            risk_sigma=1.0,
            planner_horizon=70,
            window_steps=32,
        )
        rows.append(
            {
                "ablation": "risk_weight",
                "value": risk_weight,
                **metrics_row(scenario, result),
            }
        )
    for safety_distance, risk_sigma in ((1.2, 0.6), (1.8, 0.8), (2.4, 1.0)):
        result = run_proposed(
            scenario,
            risk_weight=8.0,
            safety_distance=safety_distance,
            risk_sigma=risk_sigma,
            planner_horizon=70,
            window_steps=32,
        )
        rows.append(
            {
                "ablation": "safety_radius",
                "value": safety_distance,
                **metrics_row(scenario, result),
            }
        )
    for window_steps in (20, 28, 36):
        result = run_proposed(
            scenario,
            risk_weight=8.0,
            safety_distance=2.4,
            risk_sigma=1.0,
            planner_horizon=70,
            window_steps=window_steps,
        )
        rows.append(
            {
                "ablation": "planning_window",
                "value": window_steps,
                **metrics_row(scenario, result),
            }
        )
    return pd.DataFrame(rows)


def run_proposed(
    scenario: ScenarioConfig,
    risk_weight: float,
    safety_distance: float,
    risk_sigma: float,
    planner_horizon: int,
    window_steps: int,
) -> MultiRobotPlanResult:
    """Run the proposed windowed risk-aware planner."""
    planner = make_planner(
        scenario,
        risk_weight=risk_weight,
        reservation_padding=3,
        safety_distance=safety_distance,
        risk_sigma=risk_sigma,
        planner_horizon=planner_horizon,
    )
    return WindowedConflictReplanner(
        planner,
        window_steps=window_steps,
        repair_iterations=2,
        lookback_steps=4,
        clearance_margin=0.14,
    ).plan(scenario.robots)


def make_planner(
    scenario: ScenarioConfig,
    risk_weight: float,
    reservation_padding: int,
    safety_distance: float = 2.4,
    risk_sigma: float = 1.0,
    planner_horizon: int | None = None,
) -> KinodynamicAStarPlanner:
    """Create a kinodynamic lattice planner with experiment defaults."""
    dynamic_obstacles = scenario.dynamic_obstacles if risk_weight > 0.0 else ()
    return KinodynamicAStarPlanner(
        collision_checker=CollisionChecker(
            warehouse=scenario.warehouse,
            dynamic_obstacles=dynamic_obstacles,
        ),
        dt=scenario.simulation.dt,
        theta_bins=16,
        step_distance=0.5,
        goal_tolerance=0.65,
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


def scenario_with_obstacle_aware_pedestrians(scenario: ScenarioConfig) -> ScenarioConfig:
    """Replace straight pedestrian references with obstacle-aware rollouts.

    The YAML waypoints are treated as pedestrian intent. This helper rolls them
    out through valid free space and then reuses the valid rollout as the
    dynamic obstacle trajectory seen by both planning and rendering.
    """
    if not scenario.dynamic_obstacles:
        return scenario
    rollout_margin = 0.12
    rollout_scenario = replace(
        scenario,
        dynamic_obstacles=tuple(
            DynamicObstacle(
                id=pedestrian.id,
                radius=pedestrian.radius + rollout_margin,
                trajectory=tuple(
                    (
                        waypoint[0],
                        *nearest_valid_pedestrian_point(
                            scenario,
                            waypoint[1],
                            waypoint[2],
                            pedestrian.radius + rollout_margin,
                        ),
                    )
                    for waypoint in pedestrian.trajectory
                ),
            )
            for pedestrian in scenario.dynamic_obstacles
        ),
    )
    pedestrian_paths = generate_pedestrian_paths(
        rollout_scenario,
        duration=rollout_scenario.simulation.horizon,
        dt=min(0.1, rollout_scenario.simulation.dt),
    )
    safe_obstacles = []
    for pedestrian in scenario.dynamic_obstacles:
        path = pedestrian_paths[pedestrian.id]
        trajectory = tuple((pose[3], pose[0], pose[1]) for pose in path)
        safe_obstacles.append(
            DynamicObstacle(
                id=pedestrian.id,
                radius=pedestrian.radius,
                trajectory=trajectory,
            )
        )
    return replace(scenario, dynamic_obstacles=tuple(safe_obstacles))


def nearest_valid_pedestrian_point(
    scenario: ScenarioConfig,
    x: float,
    y: float,
    radius: float,
) -> tuple[float, float]:
    """Project an invalid pedestrian waypoint into nearby open space."""
    if is_valid_pedestrian_point(scenario, x, y, radius):
        return (x, y)

    resolution = scenario.warehouse.resolution
    max_radius = max(scenario.warehouse.width, scenario.warehouse.height)
    angle_count = 96
    search_radii = np.arange(resolution, max_radius + resolution, resolution)
    for search_radius in search_radii:
        for angle in np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False):
            candidate_x = x + search_radius * np.cos(angle)
            candidate_y = y + search_radius * np.sin(angle)
            if is_valid_pedestrian_point(scenario, candidate_x, candidate_y, radius):
                return (float(candidate_x), float(candidate_y))
    raise ValueError(f"Could not project pedestrian waypoint ({x:.2f}, {y:.2f}) to free space")


def is_valid_pedestrian_point(
    scenario: ScenarioConfig,
    x: float,
    y: float,
    radius: float,
) -> bool:
    """Return whether a pedestrian center has clear static-map clearance."""
    if not scenario.warehouse.in_bounds(x, y, margin=radius):
        return False
    if scenario.warehouse.is_occupied(x, y, margin=radius):
        return False
    return not any(
        obstacle.contains_point(x, y, margin=radius)
        for obstacle in scenario.warehouse.static_obstacles
    )


def metrics_row(
    scenario: ScenarioConfig,
    result: MultiRobotPlanResult,
) -> dict[str, float | bool | str]:
    """Return a compact metric row for one run."""
    metrics = evaluate_multi_robot_plan(
        scenario.robots,
        result,
        dt=scenario.simulation.dt,
        warehouse=scenario.warehouse,
        dynamic_obstacles=scenario.dynamic_obstacles,
    )
    return {
        "success": result.success,
        "robot_collisions": metrics.robot_robot_collision_count,
        "human_near_misses": metrics.dynamic_obstacle_near_miss_count,
        "makespan": metrics.makespan,
        "path_length": metrics.total_path_length,
        "planning_time": metrics.planning_time,
        "failed_robot_id": result.failed_robot_id or "",
    }


def save_algorithm_comparison_plot(frame: pd.DataFrame) -> None:
    """Save a four-panel method comparison figure."""
    metrics = [
        ("robot_collisions", "Robot Collisions"),
        ("human_near_misses", "Human Near-Misses"),
        ("makespan", "Makespan [s]"),
        ("path_length", "Total Path Length [m]"),
    ]
    colors = ["#6b7280", "#4b5563", "#64748b", "#2563eb"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.3))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.bar(frame["method"], frame[metric], color=colors, width=0.68)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.35, alpha=0.3)
        ax.tick_params(axis="x", rotation=18)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Compact Algorithm Comparison", y=1.02, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, "algorithm_comparison")


def save_ablation_plot(frame: pd.DataFrame) -> None:
    """Save ablation curves for risk, safety radius, and window size."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, ablation, title in zip(
        axes,
        ("risk_weight", "safety_radius", "planning_window"),
        ("Risk Weight", "Safety Radius", "Planning Window"),
    ):
        subset = frame[frame["ablation"] == ablation].sort_values("value")
        ax.plot(subset["value"], subset["human_near_misses"], marker="o", label="near-miss")
        ax.plot(subset["value"], subset["path_length"], marker="s", label="path length")
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.35, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("value")
    axes[0].set_ylabel("metric value")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Ablation Study", y=1.05, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, "ablation_summary")


def save_risk_weight_overlay(scenario: ScenarioConfig) -> None:
    """Save an overlay showing path deformation as risk weight changes."""
    robot = scenario.robots[0]
    weights = (0.0, 2.0, 4.0, 8.0, 12.0)
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    plotter = WarehousePlotter(style="clean")
    plotter.configure_clean_axis(ax, scenario)
    plotter.draw_clean_obstacles(ax, scenario)
    for index, weight in enumerate(weights):
        planner = make_planner(
            scenario,
            risk_weight=weight,
            reservation_padding=0,
            safety_distance=2.4,
            risk_sigma=1.0,
        )
        start = perf_counter()
        path = planner.plan(robot)
        _ = perf_counter() - start
        smooth_path = interpolate_path(path, samples_per_segment=12)
        color = cmap(0.25 + 0.65 * index / max(len(weights) - 1, 1))
        ax.plot(
            [pose[0] for pose in smooth_path],
            [pose[1] for pose in smooth_path],
            color=color,
            linewidth=1.8 + 0.25 * index,
            alpha=0.9,
            label=f"risk={weight:g}",
            solid_capstyle="round",
        )
    ax.scatter(robot.start.x, robot.start.y, s=50, facecolors="white", edgecolors="#111827", zorder=8)
    ax.scatter(robot.goal.x, robot.goal.y, s=58, color="#111827", marker="x", zorder=8)
    ax.set_title("Path Overlay by Human Risk Weight")
    ax.legend(loc="upper center", ncol=len(weights), bbox_to_anchor=(0.5, -0.03), fontsize=8)
    save_figure(fig, "risk_weight_path_overlay")


def save_windowed_demo_video(scenario: ScenarioConfig, fps: int = 10) -> None:
    """Save a clean MP4 demo using the proposed windowed risk-aware planner."""
    scenario = scenario_with_obstacle_aware_pedestrians(scenario)
    result = run_proposed(
        scenario,
        risk_weight=8.0,
        safety_distance=2.4,
        risk_sigma=1.0,
        planner_horizon=70,
        window_steps=32,
    )
    robot_paths = {
        robot_id: interpolate_path(path, samples_per_segment=10)
        for robot_id, path in result.paths.items()
    }
    duration = min(12.0, scenario.simulation.horizon)
    frame_times = np.linspace(0.0, duration, max(2, int(duration * fps)))
    frames = [
        render_windowed_demo_frame(
            scenario=scenario,
            robot_paths=robot_paths,
            time=float(time),
        )
        for time in frame_times
    ]
    imageio.mimsave(OUTPUT_DIR / "demo_video.mp4", frames, fps=fps)


def render_windowed_demo_frame(
    scenario: ScenarioConfig,
    robot_paths: dict[str, list[ContinuousPose]],
    time: float,
) -> np.ndarray:
    """Render one CVPR-style video frame for the windowed planner."""
    fig, ax = plt.subplots(figsize=(12.0, 7.68))
    plotter = WarehousePlotter(style="clean")
    plotter.configure_clean_axis(ax, scenario)
    plotter.draw_clean_obstacles(ax, scenario)
    plotter.draw_pedestrians(ax, scenario, time=time)

    for robot_index, robot in enumerate(scenario.robots):
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        path = robot_paths.get(robot.id, [])
        history = path_until(path, time)
        future = path_between(path, time, time + 2.0)
        if len(history) > 1:
            ax.plot(
                [pose[0] for pose in history],
                [pose[1] for pose in history],
                color=color,
                linewidth=2.2,
                alpha=0.88,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )
        if len(future) > 1:
            ax.plot(
                [pose[0] for pose in future],
                [pose[1] for pose in future],
                color=color,
                linewidth=1.6,
                alpha=0.42,
                linestyle=(0, (4, 4)),
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4,
            )
        pose = pose_at_time(path, time)
        if pose is not None:
            plotter.draw_robot_rectangle(ax, pose[0], pose[1], pose[2], robot.radius, color)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = rgba.reshape((height, width, 4))[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_scenario_catalog() -> None:
    """Save scenario diversity metadata and a proposed experiment design."""
    scenarios = build_benchmark_scenarios() + [
        ("cluttered_warehouse", make_cluttered_warehouse()),
        ("random_dense_warehouse", make_random_warehouse(seed=7)),
    ]
    rows = []
    for item in scenarios:
        if isinstance(item, tuple):
            name, scenario = item
        else:
            name, scenario = item.name, item.scenario
        rows.append(
            {
                "scenario": name,
                "robots": len(scenario.robots),
                "pedestrians": len(scenario.dynamic_obstacles),
                "obstacles": len(scenario.warehouse.static_obstacles),
                "width": scenario.warehouse.width,
                "height": scenario.warehouse.height,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "scenario_catalog.csv", index=False)


def make_cluttered_warehouse() -> ScenarioConfig:
    """Return a small cluttered scenario for future experiments."""
    spec = RobotSpec(wheelbase=0.7, max_speed=1.0, max_steering_angle=0.5)
    obstacles = tuple(
        RectangleObstacle(f"box_{idx}", x, y, 0.45, 1.0)
        for idx, (x, y) in enumerate(
            ((2.0, 0.8), (3.2, 2.6), (4.4, 0.9), (5.6, 2.7), (6.8, 1.1))
        )
    )
    robots = (
        Robot("r1", spec, RobotState(0.75, 0.75, 0.0, 0.2), RobotState(7.5, 3.4, 0.0, 0.2)),
        Robot("r2", spec, RobotState(7.5, 0.75, 3.14159, 0.2), RobotState(0.75, 3.4, 3.14159, 0.2)),
        Robot("r3", spec, RobotState(0.75, 3.4, 0.0, 0.2), RobotState(7.5, 0.75, 0.0, 0.2)),
    )
    pedestrian = DynamicObstacle("worker", 0.24, ((0.0, 4.0, 0.5), (4.0, 4.0, 3.5), (8.0, 4.0, 0.5)))
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=10.0),
        warehouse=WarehouseMap(8.5, 4.5, 0.25, obstacles),
        robots=robots,
        dynamic_obstacles=(pedestrian,),
    )


def make_random_warehouse(seed: int) -> ScenarioConfig:
    """Return a deterministic random-map scenario catalog entry."""
    rng = np.random.default_rng(seed)
    spec = RobotSpec(wheelbase=0.7, max_speed=1.0, max_steering_angle=0.5)
    obstacles = tuple(
        RectangleObstacle(
            f"random_{idx}",
            float(rng.uniform(1.5, 6.5)),
            float(rng.uniform(0.8, 3.5)),
            float(rng.uniform(0.35, 0.65)),
            float(rng.uniform(0.7, 1.3)),
        )
        for idx in range(6)
    )
    robots = (
        Robot("r1", spec, RobotState(0.75, 0.75, 0.0, 0.2), RobotState(7.5, 3.8, 0.0, 0.2)),
        Robot("r2", spec, RobotState(7.5, 0.75, 3.14159, 0.2), RobotState(0.75, 3.8, 3.14159, 0.2)),
        Robot("r3", spec, RobotState(0.75, 2.4, 0.0, 0.2), RobotState(7.5, 1.7, 0.0, 0.2)),
        Robot("r4", spec, RobotState(7.5, 2.4, 3.14159, 0.2), RobotState(0.75, 1.7, 3.14159, 0.2)),
    )
    pedestrians = (
        DynamicObstacle("p1", 0.24, ((0.0, 2.0, 0.5), (5.0, 6.2, 3.9), (10.0, 2.0, 0.5))),
        DynamicObstacle("p2", 0.24, ((0.0, 6.2, 0.5), (5.0, 2.0, 3.9), (10.0, 6.2, 0.5))),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=10.0),
        warehouse=WarehouseMap(8.5, 4.5, 0.25, obstacles),
        robots=robots,
        dynamic_obstacles=pedestrians,
    )


def write_experiment_notes() -> None:
    """Write paper-facing experiment notes and suggested extensions."""
    notes = [
        "# CVPR-style experiment notes",
        "",
        "Implemented method:",
        "- Proposed Windowed Risk-Aware uses risk-aware kinodynamic lattice A*, inflated space-time reservations, and bounded conflict-window replanning.",
        "- This is a practical approximation to windowed MAPF, not full optimal CBS/ECBS.",
        "- The demo video projects pedestrian waypoint intents into valid free space and rolls them out with static-obstacle clearance before planning and rendering.",
        "",
        "Dynamic-obstacle baseline:",
        "- ORCA/RVO is not included because this repository has no RVO/ORCA dependency and the robot model is a heading-constrained lattice planner rather than holonomic velocity control.",
        "- A fair future ORCA baseline should either add a dedicated RVO library or implement a nonholonomic velocity projection layer.",
        "",
        "Ablation ideas beyond the compact runs here:",
        "- Full factorial risk_weight x safety_radius x horizon on fixed seeds.",
        "- Robot-count sweep: 2, 3, 4, 6 robots.",
        "- Pedestrian-count sweep: 0, 1, 3, 5 pedestrians.",
        "- Obstacle-count sweep on random maps with fixed start-goal templates.",
        "- Scenario families: narrow aisle, human crossing, bottleneck, cluttered warehouse, and random dense warehouse.",
    ]
    (OUTPUT_DIR / "experiment_notes.md").write_text("\n".join(notes), encoding="utf-8")


def save_figure(fig, name: str) -> None:
    """Save a figure as PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
