"""Generate publication-style figures and video for the warehouse demo."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from warehouse_planning.config import ScenarioConfig, load_scenario_config
from warehouse_planning.evaluation.metrics import evaluate_multi_robot_plan
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose, KinodynamicAStarPlanner
from warehouse_planning.planning.prioritized import (
    IndependentPlanner,
    MultiRobotPlanResult,
    PrioritizedPlanner,
)
from warehouse_planning.planning.windowed import WindowedConflictReplanner
from warehouse_planning.visualization.clean_demo import (
    generate_pedestrian_paths,
    pedestrian_positions_at_time,
    plan_and_smooth_demo_paths,
)
from warehouse_planning.visualization.plotting import ROBOT_COLORS, WarehousePlotter
from warehouse_planning.visualization.smoothing import interpolate_path, path_between, path_until, pose_at_time


OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
SCENARIO_PATH = PROJECT_ROOT / "configs" / "warehouse_clean_demo.yaml"
PSEUDOCODE_TEXT = """Algorithm: Risk-Aware Windowed Conflict Replanner
Input: warehouse map M, robots R, pedestrian tracks H, horizon T
1. Observe current robot poses and pedestrian positions.
2. Predict pedestrian positions over the local planning horizon.
3. Build a smooth Gaussian risk field around predicted pedestrians.
4. Plan kinodynamic nonholonomic paths with A* motion primitives.
5. Reserve inflated robot footprints for priority-space separation.
6. Scan a bounded future window for robot-robot conflicts.
7. Replan the lower-priority robot around the conflict region when needed.
Output: collision-aware smooth robot trajectories
"""


@dataclass(frozen=True)
class PaperData:
    """Reusable scenario products for paper outputs."""

    scenario: ScenarioConfig
    robot_paths: dict[str, list[ContinuousPose]]
    pedestrian_paths: dict[str, list[ContinuousPose]]


def main() -> None:
    """Generate all paper figures, tables, pseudocode notes, and demo video."""
    set_paper_style()
    prepare_output_dir(OUTPUT_DIR)
    scenario = load_scenario_config(SCENARIO_PATH)
    data = PaperData(
        scenario=scenario,
        robot_paths=plan_and_smooth_demo_paths(scenario),
        pedestrian_paths=generate_pedestrian_paths(
            scenario,
            duration=scenario.simulation.horizon,
            dt=min(0.1, scenario.simulation.dt),
        ),
    )

    save_qualitative_overview(data)
    save_risk_comparison(data)
    metrics_frame = save_metric_comparison(scenario)
    save_pipeline_figure()
    save_pseudocode_artifacts()
    save_demo_video(data)
    write_generation_notes(metrics_frame)


def set_paper_style() -> None:
    """Configure Matplotlib for clean paper-style output."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2933",
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#d0d0d0",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def prepare_output_dir(output_dir: Path) -> None:
    """Recreate the paper figure output directory."""
    resolved_output = output_dir.resolve()
    resolved_results = (PROJECT_ROOT / "results").resolve()
    if not str(resolved_output).startswith(str(resolved_results)):
        raise ValueError(f"Refusing to clean outside results/: {resolved_output}")
    if resolved_output.exists():
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)


def save_figure(fig: Figure, name: str) -> None:
    """Save a figure as both PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_qualitative_overview(data: PaperData) -> None:
    """Figure 1: full qualitative trajectory overview."""
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    draw_paper_base_map(ax, data.scenario)
    draw_pedestrian_tracks(ax, data.scenario)
    draw_robot_trajectories(ax, data.scenario, data.robot_paths, time=data.scenario.simulation.horizon * 0.5)
    ax.set_title("Figure 1. Qualitative multi-robot trajectories")
    add_compact_legend(ax, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.015))
    save_figure(fig, "figure1_qualitative_trajectory_overview")


def save_risk_comparison(data: PaperData) -> None:
    """Figure 2: path comparison with and without human risk."""
    scenario = data.scenario
    robot = scenario.robots[0]
    comparison_time = min(6.0, scenario.simulation.horizon * 0.45)
    pedestrian_positions = pedestrian_positions_at_time(data.pedestrian_paths, comparison_time)
    baseline_path = plan_single_robot_path(scenario, robot_index=0, risk_weight=0.0)
    risk_path = data.robot_paths[robot.id]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    panels = (
        ("Without human risk field", baseline_path, False),
        ("With human risk field (same path as video)", risk_path, True),
    )
    for ax, (title, path, show_risk) in zip(axes, panels):
        draw_paper_base_map(ax, scenario)
        if show_risk:
            WarehousePlotter(style="clean").draw_risk_field(
                ax,
                scenario,
                time=comparison_time,
                sigma=0.95,
                resolution=0.08,
                pedestrian_positions=pedestrian_positions,
            )
            WarehousePlotter(style="clean").draw_clean_obstacles(ax, scenario)
        draw_pedestrians_at(ax, scenario, pedestrian_positions)
        draw_start_goal(ax, robot, ROBOT_COLORS[0], label_prefix="R1")
        draw_path_line(ax, path, ROBOT_COLORS[0], label="planned path", linewidth=2.4)
        pose = pose_at_time(path, comparison_time)
        if pose is not None:
            WarehousePlotter(style="clean").draw_robot_rectangle(
                ax,
                pose[0],
                pose[1],
                pose[2],
                robot.radius,
                ROBOT_COLORS[0],
                alpha=0.96,
            )
        ax.set_title(title)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015))
    save_figure(fig, "figure2_risk_aware_path_comparison")


def save_metric_comparison(scenario: ScenarioConfig) -> pd.DataFrame:
    """Figure 3: baseline metric comparison using real project runs."""
    rows = []
    methods = [
        ("Independent A*", "independent", 0.0),
        ("Prioritized Planning", "prioritized", 0.0),
        ("CBS-style", "prioritized", 0.0),
        ("Proposed", "concurrent", 8.0),
    ]
    for label, planner_kind, risk_weight in methods:
        result = run_metric_method(scenario, planner_kind=planner_kind, risk_weight=risk_weight)
        metrics = evaluate_multi_robot_plan(
            scenario.robots,
            result,
            dt=scenario.simulation.dt,
            warehouse=scenario.warehouse,
            dynamic_obstacles=scenario.dynamic_obstacles,
        )
        rows.append(
            {
                "method": label,
                "robot_robot_collision_count": metrics.robot_robot_collision_count,
                "human_near_miss_count": metrics.dynamic_obstacle_near_miss_count,
                "path_length": metrics.total_path_length,
                "computation_time": metrics.planning_time,
                "success": result.success,
                "failed_robot_id": result.failed_robot_id or "",
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "figure3_metric_data.csv", index=False)
    save_metric_bar_figure(frame)
    save_metric_table_figure(frame)
    return frame


def run_metric_method(
    scenario: ScenarioConfig,
    planner_kind: str,
    risk_weight: float,
) -> MultiRobotPlanResult:
    """Run one metric method on the final demo scenario."""
    planner = make_planner(scenario, risk_weight=risk_weight)
    if planner_kind == "independent":
        return IndependentPlanner(planner).plan(scenario.robots)
    if planner_kind == "prioritized":
        return PrioritizedPlanner(planner).plan(scenario.robots)
    if planner_kind == "concurrent":
        return WindowedConflictReplanner(
            planner,
            window_steps=32,
            repair_iterations=2,
            lookback_steps=4,
            clearance_margin=0.18,
        ).plan(scenario.robots)
    raise ValueError(f"Unsupported planner kind: {planner_kind}")


def make_planner(scenario: ScenarioConfig, risk_weight: float) -> KinodynamicAStarPlanner:
    """Create the kinodynamic planner used by paper metrics."""
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
        max_time_steps=min(70, int(scenario.simulation.horizon / scenario.simulation.dt)),
        risk_weight=risk_weight,
        safety_distance=2.4,
        risk_sigma=1.0,
        risk_time_offsets=(-0.6, 0.0, 0.6, 1.2),
        risk_time_decay=0.55,
        wait_cost=3.0 if risk_weight > 0.0 else 0.5,
        heuristic_weight=3.0 if risk_weight > 0.0 else 1.0,
        reservation_padding=1 if risk_weight > 0.0 else 0,
        allow_partial=risk_weight > 0.0,
    )


def save_metric_bar_figure(frame: pd.DataFrame) -> None:
    """Save a 2x2 metric bar chart."""
    metrics = [
        ("robot_robot_collision_count", "Robot-robot collisions"),
        ("human_near_miss_count", "Human near misses"),
        ("path_length", "Total path length [m]"),
        ("computation_time", "Computation time [s]"),
    ]
    colors = ["#6b7280", "#4b5563", "#374151", "#2563eb"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        ax.bar(frame["method"], frame[metric], color=colors, width=0.68)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.35, alpha=0.32)
        ax.tick_params(axis="x", rotation=18)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.suptitle("Figure 3. Baseline metric comparison", y=1.02, fontsize=12)
    fig.tight_layout()
    save_figure(fig, "figure3_baseline_metric_comparison")


def save_metric_table_figure(frame: pd.DataFrame) -> None:
    """Save a clean table view of the metric data."""
    display_frame = frame[
        [
            "method",
            "robot_robot_collision_count",
            "human_near_miss_count",
            "path_length",
            "computation_time",
        ]
    ].copy()
    display_frame["path_length"] = display_frame["path_length"].map("{:.2f}".format)
    display_frame["computation_time"] = display_frame["computation_time"].map("{:.3f}".format)
    display_frame.columns = [
        "Method",
        "Robot collisions",
        "Human near misses",
        "Path length",
        "Time",
    ]

    fig, ax = plt.subplots(figsize=(9.8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white")
    ax.set_title("Metric table for Figure 3", pad=12)
    save_figure(fig, "figure3_metric_table")


def save_pipeline_figure() -> None:
    """Figure 4: clean pipeline/algorithm visualization."""
    steps = [
        ("Observe", "states"),
        ("Predict", "pedestrians"),
        ("Risk field", "human risk"),
        ("Plan", "nonholonomic paths"),
        ("Resolve", "conflicts"),
        ("Execute", "replan"),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 2.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    x_positions = np.linspace(0.095, 0.905, len(steps))
    box_width = 0.105
    half_box = box_width * 0.5
    for index, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):
        box = FancyBboxPatch(
            (x - half_box, 0.36),
            box_width,
            0.28,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="#f8fafc",
            edgecolor="#94a3b8",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x, 0.535, title, ha="center", va="center", fontsize=10, weight="bold", color="#111827")
        ax.text(x, 0.455, subtitle, ha="center", va="center", fontsize=7.6, color="#475569")
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + half_box + 0.006, 0.5),
                (x_positions[index + 1] - half_box - 0.006, 0.5),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.2,
                color="#64748b",
            )
            ax.add_patch(arrow)

    ax.text(0.5, 0.86, "Figure 4. Planning and execution pipeline", ha="center", fontsize=12, weight="bold")
    save_figure(fig, "figure4_pipeline_algorithm")


def save_pseudocode_artifacts() -> None:
    """Save pseudocode as text and as a simple figure."""
    (OUTPUT_DIR / "algorithm_pseudocode.txt").write_text(PSEUDOCODE_TEXT, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.axis("off")
    box = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.9,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        facecolor="#fbfbfb",
        edgecolor="#cbd5e1",
        linewidth=0.8,
    )
    ax.add_patch(box)
    ax.text(
        0.06,
        0.9,
        PSEUDOCODE_TEXT,
        ha="left",
        va="top",
        family="DejaVu Sans Mono",
        fontsize=8.6,
        color="#111827",
        linespacing=1.38,
    )
    save_figure(fig, "algorithm_pseudocode")


def save_demo_video(data: PaperData, fps: int = 10) -> None:
    """Save the presentation-style MP4 demo video."""
    duration = min(12.0, data.scenario.simulation.horizon)
    frame_times = np.linspace(0.0, duration, max(2, int(duration * fps)))
    frames = [
        render_paper_frame(data, float(time))
        for time in frame_times
    ]
    imageio.mimsave(OUTPUT_DIR / "demo_video.mp4", frames, fps=fps)


def render_paper_frame(data: PaperData, time: float) -> np.ndarray:
    """Render one video frame with paper-style motion elements."""
    fig, ax = plt.subplots(figsize=(12.0, 7.68))
    draw_paper_base_map(ax, data.scenario)
    pedestrian_positions = pedestrian_positions_at_time(data.pedestrian_paths, time)
    draw_pedestrians_at(ax, data.scenario, pedestrian_positions)

    plotter = WarehousePlotter(style="clean")
    for robot_index, robot in enumerate(data.scenario.robots):
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        path = data.robot_paths[robot.id]
        history = path_until(path, time)
        future = path_between(path, time, time + 2.0)
        if len(history) > 1:
            draw_path_line(ax, history, color, label=None, linewidth=2.2, alpha=0.88)
        if len(future) > 1:
            draw_path_line(
                ax,
                future,
                color,
                label=None,
                linewidth=1.6,
                alpha=0.42,
                linestyle=(0, (4, 4)),
            )
        pose = pose_at_time(path, time)
        if pose is not None:
            plotter.draw_robot_rectangle(ax, pose[0], pose[1], pose[2], robot.radius, color)
    ax.text(
        0.015,
        0.97,
        f"t = {time:04.1f} s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#334155",
    )
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = rgba.reshape((height, width, 4))[:, :, :3].copy()
    plt.close(fig)
    return frame


def draw_paper_base_map(ax: Axes, scenario: ScenarioConfig) -> None:
    """Draw the shared minimal warehouse base map."""
    plotter = WarehousePlotter(style="clean")
    plotter.configure_clean_axis(ax, scenario)
    plotter.draw_clean_obstacles(ax, scenario)


def draw_robot_trajectories(
    ax: Axes,
    scenario: ScenarioConfig,
    robot_paths: dict[str, list[ContinuousPose]],
    time: float,
) -> None:
    """Draw all smooth robot trajectories, start/goal markers, and mid-poses."""
    plotter = WarehousePlotter(style="clean")
    for robot_index, robot in enumerate(scenario.robots):
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        draw_start_goal(
            ax,
            robot,
            color,
            label_prefix=f"R{robot_index + 1}",
            show_labels=robot_index == 0,
        )
        path = robot_paths[robot.id]
        draw_path_line(ax, path, color, label=f"R{robot_index + 1}", linewidth=2.3, alpha=0.9)
        pose = pose_at_time(path, time)
        if pose is not None:
            plotter.draw_robot_rectangle(ax, pose[0], pose[1], pose[2], robot.radius, color, alpha=0.94)


def draw_start_goal(
    ax: Axes,
    robot,
    color: str,
    label_prefix: str,
    show_labels: bool = True,
) -> None:
    """Draw start and goal markers for one robot."""
    ax.scatter(
        robot.start.x,
        robot.start.y,
        s=46,
        facecolors="white",
        edgecolors=color,
        linewidths=1.4,
        marker="o",
        zorder=12,
        label=f"{label_prefix} start" if show_labels else None,
    )
    ax.scatter(
        robot.goal.x,
        robot.goal.y,
        s=58,
        color=color,
        marker="x",
        linewidths=1.8,
        zorder=12,
        label=f"{label_prefix} goal" if show_labels else None,
    )


def draw_path_line(
    ax: Axes,
    path: list[ContinuousPose],
    color: str,
    label: str | None,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str | tuple[int, tuple[int, ...]] = "-",
) -> None:
    """Draw a path with rounded caps."""
    if not path:
        return
    ax.plot(
        [pose[0] for pose in path],
        [pose[1] for pose in path],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        zorder=5,
    )


def draw_pedestrian_tracks(ax: Axes, scenario: ScenarioConfig) -> None:
    """Draw pedestrian waypoint tracks and initial positions."""
    for index, pedestrian in enumerate(scenario.dynamic_obstacles):
        xs = [point[1] for point in pedestrian.trajectory]
        ys = [point[2] for point in pedestrian.trajectory]
        ax.plot(
            xs,
            ys,
            color="#111827",
            linewidth=1.2,
            linestyle=(0, (2, 3)),
            alpha=0.45,
            zorder=3,
            label="pedestrian track" if index == 0 else None,
        )
        ax.scatter(xs[0], ys[0], s=42, color="#111827", edgecolors="white", linewidths=0.8, zorder=9)


def draw_pedestrians_at(
    ax: Axes,
    scenario: ScenarioConfig,
    pedestrian_positions: dict[str, tuple[float, float]],
) -> None:
    """Draw pedestrian discs at supplied positions."""
    plotter = WarehousePlotter(style="clean")
    plotter.draw_pedestrians(
        ax,
        scenario,
        time=0.0,
        pedestrian_positions=pedestrian_positions,
    )


def add_compact_legend(
    ax: Axes,
    loc: str,
    ncol: int = 2,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    """Add a de-duplicated compact legend."""
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc=loc,
            fontsize=7.4,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
        )


def plan_single_robot_path(
    scenario: ScenarioConfig,
    robot_index: int,
    risk_weight: float,
) -> list[ContinuousPose]:
    """Plan and smooth one path for the comparison figure."""
    planner = make_planner(scenario, risk_weight=risk_weight)
    path = planner.plan(scenario.robots[robot_index])
    return interpolate_path(path, samples_per_segment=12)


def write_generation_notes(metrics_frame: pd.DataFrame) -> None:
    """Write notes describing data provenance for paper artifacts."""
    metric_table = metrics_frame.to_string(index=False)
    notes = [
        "# Paper figure generation notes",
        "",
        "- Figures 1, 2, and the video use the final YAML scenario with 4 robots and 3 pedestrians.",
        "- Figure 3 is generated from live project planner runs on the same final demo scenario.",
        "- Independent A*, Prioritized Planning, and Proposed are real runs from the implemented planners.",
        "- Proposed uses stronger pedestrian risk plus bounded windowed conflict replanning over time-expanded reservations.",
        "- The CBS-style column is a transparent proxy using the project's reservation-constrained prioritized planner; a full CBS search tree is not implemented in this repository.",
        "- Metric CSV values are saved in figure3_metric_data.csv.",
        "",
        "## Metric data",
        "```text",
        metric_table,
        "```",
    ]
    (OUTPUT_DIR / "generation_notes.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    start = perf_counter()
    main()
    print(f"Saved paper figures to {OUTPUT_DIR}")
    print(f"Done in {perf_counter() - start:.1f}s")
