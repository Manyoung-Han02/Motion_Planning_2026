"""Run the final seeded algorithm comparison for the warehouse project."""

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

from warehouse_planning.config import ScenarioConfig, load_scenario_config
from warehouse_planning.maps.warehouse_map import WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose, KinodynamicAStarPlanner
from warehouse_planning.planning.prioritized import (
    ConcurrentLocalWaitPlanner,
    IndependentPlanner,
    MultiRobotPlanResult,
    PrioritizedPlanner,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "final_evaluation"
SCENARIO_PATH = PROJECT_ROOT / "configs" / "warehouse_clean_demo.yaml"
TRIAL_SEEDS = tuple(range(6))
NEAR_MISS_DISTANCE = 0.75


def main() -> None:
    """Run all trials and write the final table and comparison figure."""
    set_style()
    prepare_output_dir(OUTPUT_DIR)
    base_scenario = load_scenario_config(SCENARIO_PATH)
    trial_rows = run_trials(base_scenario, TRIAL_SEEDS)
    trial_frame = pd.DataFrame(trial_rows)
    trial_frame.to_csv(OUTPUT_DIR / "final_trial_metrics.csv", index=False)

    summary = summarize_trials(trial_frame)
    summary.to_csv(OUTPUT_DIR / "final_metrics.csv", index=False)
    save_table_figure(summary)
    save_comparison_plot(summary)
    write_notes(summary)
    print(f"Saved final evaluation to {OUTPUT_DIR}")


def set_style() -> None:
    """Configure a compact academic plotting style."""
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
    """Recreate the final evaluation output directory."""
    resolved_output = output_dir.resolve()
    resolved_results = (PROJECT_ROOT / "results").resolve()
    if not str(resolved_output).startswith(str(resolved_results)):
        raise ValueError(f"Refusing to clean outside results/: {resolved_output}")
    if resolved_output.exists():
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def run_trials(
    base_scenario: ScenarioConfig,
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    """Run all seeded scenario variants for each algorithm."""
    rows: list[dict[str, object]] = []
    for seed in seeds:
        scenario = perturb_pedestrians(base_scenario, seed)
        for method_name, result in run_methods(scenario):
            metrics = evaluate_trial(
                scenario,
                result,
                near_miss_distance=NEAR_MISS_DISTANCE,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    **metrics,
                }
            )
    return rows


def run_methods(scenario: ScenarioConfig) -> list[tuple[str, MultiRobotPlanResult]]:
    """Run the four final comparison methods on one scenario."""
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
            "CBS-style planner",
            PrioritizedPlanner(make_planner(scenario, risk_weight=0.0, reservation_padding=2)),
        ),
        (
            "Proposed risk-aware planner",
            ConcurrentLocalWaitPlanner(
                make_planner(scenario, risk_weight=8.0, reservation_padding=3),
                time_step=scenario.simulation.dt,
                wait_step=0.4,
                max_total_wait=6.0,
                clearance_margin=0.14,
            ),
        ),
    )

    results: list[tuple[str, MultiRobotPlanResult]] = []
    for method_name, planner in methods:
        start = perf_counter()
        result = planner.plan(scenario.robots)
        # Keep planner timing from planner implementations, but make sure any
        # future planner object without timing still has a useful duration.
        if result.planning_time <= 0.0:
            result = replace(result, planning_time=perf_counter() - start)
        results.append((method_name, result))
    return results


def make_planner(
    scenario: ScenarioConfig,
    risk_weight: float,
    reservation_padding: int,
) -> KinodynamicAStarPlanner:
    """Create a single-robot planner for the final comparison."""
    return KinodynamicAStarPlanner(
        collision_checker=CollisionChecker(
            warehouse=scenario.warehouse,
            dynamic_obstacles=scenario.dynamic_obstacles,
        ),
        dt=scenario.simulation.dt,
        theta_bins=16,
        step_distance=0.5,
        goal_tolerance=0.65,
        max_time_steps=min(70, int(scenario.simulation.horizon / scenario.simulation.dt)),
        risk_weight=risk_weight,
        safety_distance=2.4 if risk_weight > 0.0 else 1.0,
        risk_sigma=1.0 if risk_weight > 0.0 else 0.5,
        risk_time_offsets=(-0.6, 0.0, 0.6, 1.2) if risk_weight > 0.0 else (0.0,),
        risk_time_decay=0.55,
        wait_cost=3.0 if risk_weight > 0.0 else 0.5,
        heuristic_weight=3.0 if risk_weight > 0.0 else 1.0,
        reservation_padding=reservation_padding,
        allow_partial=risk_weight > 0.0,
    )


def perturb_pedestrians(scenario: ScenarioConfig, seed: int) -> ScenarioConfig:
    """Create a deterministic pedestrian-track variant for one trial."""
    rng = np.random.default_rng(seed)
    checker = CollisionChecker(warehouse=scenario.warehouse)
    perturbed_obstacles: list[DynamicObstacle] = []
    for pedestrian in scenario.dynamic_obstacles:
        trajectory: list[tuple[float, float, float]] = []
        for time, x, y in pedestrian.trajectory:
            candidate = (x, y)
            for _ in range(20):
                dx, dy = rng.normal(0.0, 0.22, size=2)
                candidate_x = min(
                    scenario.warehouse.width - pedestrian.radius,
                    max(pedestrian.radius, x + float(dx)),
                )
                candidate_y = min(
                    scenario.warehouse.height - pedestrian.radius,
                    max(pedestrian.radius, y + float(dy)),
                )
                state = RobotState(candidate_x, candidate_y, 0.0, pedestrian.radius)
                if not checker.collides_with_static_obstacle(state):
                    candidate = (candidate_x, candidate_y)
                    break
            trajectory.append((time, candidate[0], candidate[1]))
        perturbed_obstacles.append(
            DynamicObstacle(
                id=pedestrian.id,
                radius=pedestrian.radius,
                trajectory=tuple(trajectory),
            )
        )

    return replace(scenario, dynamic_obstacles=tuple(perturbed_obstacles))


def evaluate_trial(
    scenario: ScenarioConfig,
    result: MultiRobotPlanResult,
    near_miss_distance: float,
) -> dict[str, float | int | bool]:
    """Compute final safety and efficiency metrics for one run."""
    robot_robot_collision = has_robot_robot_collision(
        scenario.robots,
        result.paths,
        dt=scenario.simulation.dt,
    )
    wall_collision = has_wall_collision(
        scenario.robots,
        result.paths,
        scenario.warehouse,
        dt=scenario.simulation.dt,
    )
    human_near_miss = has_human_near_miss(
        scenario.robots,
        result.paths,
        scenario.dynamic_obstacles,
        dt=scenario.simulation.dt,
        near_miss_distance=near_miss_distance,
    )
    return {
        "robot_or_wall_collision": int(robot_robot_collision or wall_collision),
        "robot_human_near_miss": int(human_near_miss),
        "makespan": makespan(result.paths),
        "total_travel_distance": total_path_length(result.paths),
        "success": bool(result.success and len(result.paths) == len(scenario.robots)),
        "planning_time": result.planning_time,
    }


def has_robot_robot_collision(
    robots: tuple[Robot, ...],
    paths: dict[str, list[ContinuousPose]],
    dt: float,
) -> bool:
    """Return whether any pair of robot discs overlaps."""
    robot_by_id = {robot.id: robot for robot in robots}
    robot_ids = [robot.id for robot in robots if paths.get(robot.id)]
    for time in sample_times(paths, dt):
        for index, first_id in enumerate(robot_ids):
            first_pose = pose_at_time(paths[first_id], time)
            if first_pose is None:
                continue
            for second_id in robot_ids[index + 1 :]:
                second_pose = pose_at_time(paths[second_id], time)
                if second_pose is None:
                    continue
                clearance = robot_by_id[first_id].radius + robot_by_id[second_id].radius
                if hypot(first_pose[0] - second_pose[0], first_pose[1] - second_pose[1]) <= clearance:
                    return True
    return False


def has_wall_collision(
    robots: tuple[Robot, ...],
    paths: dict[str, list[ContinuousPose]],
    warehouse: WarehouseMap,
    dt: float,
) -> bool:
    """Return whether any robot footprint intersects shelves or boundaries."""
    checker = CollisionChecker(warehouse=warehouse)
    robot_by_id = {robot.id: robot for robot in robots}
    for robot_id, path in paths.items():
        robot = robot_by_id[robot_id]
        for time in sample_times({robot_id: path}, dt):
            pose = pose_at_time(path, time)
            if pose is None:
                continue
            state = RobotState(pose[0], pose[1], pose[2], robot.radius)
            if checker.collides_with_static_obstacle(state):
                return True
    return False


def has_human_near_miss(
    robots: tuple[Robot, ...],
    paths: dict[str, list[ContinuousPose]],
    pedestrians: tuple[DynamicObstacle, ...],
    dt: float,
    near_miss_distance: float,
) -> bool:
    """Return whether any robot enters the pedestrian safety threshold."""
    robot_by_id = {robot.id: robot for robot in robots}
    for time in sample_times(paths, dt):
        for robot_id, path in paths.items():
            pose = pose_at_time(path, time)
            if pose is None:
                continue
            robot = robot_by_id[robot_id]
            for pedestrian in pedestrians:
                px, py = pedestrian.predicted_position(time)
                threshold = robot.radius + pedestrian.radius + near_miss_distance
                if hypot(pose[0] - px, pose[1] - py) <= threshold:
                    return True
    return False


def sample_times(paths: dict[str, list[ContinuousPose]], dt: float) -> np.ndarray:
    """Return evaluation sample times at the scenario time step."""
    horizon = makespan(paths)
    if horizon <= 0.0:
        return np.array([0.0])
    return np.arange(0.0, horizon + dt, dt)


def pose_at_time(path: list[ContinuousPose], time: float) -> ContinuousPose | None:
    """Return linearly interpolated pose with final-pose hold."""
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


def makespan(paths: dict[str, list[ContinuousPose]]) -> float:
    """Return the maximum final path time."""
    return max((path[-1][3] for path in paths.values() if path), default=0.0)


def total_path_length(paths: dict[str, list[ContinuousPose]]) -> float:
    """Return total geometric path length over all robots."""
    total = 0.0
    for path in paths.values():
        for previous, current in zip(path, path[1:]):
            total += hypot(current[0] - previous[0], current[1] - previous[1])
    return total


def summarize_trials(trial_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial rows into the requested final metrics table."""
    rows = []
    order = [
        "Independent A*",
        "Prioritized Planning",
        "CBS-style planner",
        "Proposed risk-aware planner",
    ]
    for method in order:
        group = trial_frame[trial_frame["method"] == method]
        rows.append(
            {
                "algorithm": method,
                "robot_or_wall_collision_pct": 100.0 * group["robot_or_wall_collision"].mean(),
                "robot_human_near_miss_pct": 100.0 * group["robot_human_near_miss"].mean(),
                "makespan": group["makespan"].mean(),
                "total_travel_distance": group["total_travel_distance"].mean(),
                "success_pct": 100.0 * group["success"].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_table_figure(summary: pd.DataFrame) -> None:
    """Save the final metrics table as PNG and PDF."""
    display = summary.copy()
    display["robot_or_wall_collision_pct"] = display["robot_or_wall_collision_pct"].map("{:.1f}%".format)
    display["robot_human_near_miss_pct"] = display["robot_human_near_miss_pct"].map("{:.1f}%".format)
    display["makespan"] = display["makespan"].map("{:.2f}".format)
    display["total_travel_distance"] = display["total_travel_distance"].map("{:.2f}".format)
    display["success_pct"] = display["success_pct"].map("{:.1f}%".format)
    display.columns = [
        "Algorithm",
        "Robot/Wall Collision",
        "Human Near-Miss",
        "Makespan [s]",
        "Travel Distance [m]",
        "Success",
    ]

    fig, ax = plt.subplots(figsize=(11.8, 2.7))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white")
    ax.set_title("Final Evaluation Summary (6 seeded trials)", pad=12, weight="bold")
    save_figure(fig, "final_metrics_table")


def save_comparison_plot(summary: pd.DataFrame) -> None:
    """Save a compact multi-panel bar plot for the final comparison."""
    metrics = [
        ("robot_or_wall_collision_pct", "Collision Trials [%]"),
        ("robot_human_near_miss_pct", "Human Near-Miss Trials [%]"),
        ("makespan", "Makespan [s]"),
        ("total_travel_distance", "Travel Distance [m]"),
    ]
    colors = ["#6b7280", "#4b5563", "#64748b", "#2563eb"]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.6))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.bar(summary["algorithm"], summary[metric], color=colors, width=0.66)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.35, alpha=0.3)
        ax.tick_params(axis="x", rotation=18)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Final Algorithm Comparison", y=1.02, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, "final_comparison_plot")


def save_figure(fig, name: str) -> None:
    """Save a Matplotlib figure as PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_notes(summary: pd.DataFrame) -> None:
    """Write a small provenance note for the final evaluation."""
    notes = [
        "# Final evaluation notes",
        "",
        f"- Trials: {len(TRIAL_SEEDS)} seeded pedestrian-track perturbations.",
        "- Warehouse, robot starts, robot goals, and obstacle layout are fixed.",
        "- Pedestrian waypoints are lightly perturbed per seed and checked against static obstacles.",
        "- Robot/wall collision percentage is the share of trials with any robot-robot overlap or static-map collision.",
        f"- Human near-miss percentage uses robot radius + pedestrian radius + {NEAR_MISS_DISTANCE:.2f} m.",
        "- Makespan and total travel distance are averaged across trials.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "final_evaluation_notes.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
