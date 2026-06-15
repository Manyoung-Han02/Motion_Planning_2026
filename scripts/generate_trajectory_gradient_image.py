"""Generate a gradient trajectory overview from the 0615 experiment setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from warehouse_planning.config import load_scenario_config
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose
from warehouse_planning.planning.windowed import WindowedConflictReplanner
from warehouse_planning.visualization.plotting import ROBOT_COLORS, WarehousePlotter
from warehouse_planning.visualization.smoothing import interpolate_path, pose_at_time

from run_0615_experiments import (
    OUTPUT_DIR,
    SCENARIO_PATH,
    make_planner,
    obstacle_aware_pedestrian_scenario,
    set_style,
)


def main() -> None:
    """Save PNG/PDF gradient trajectory figures."""
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario = obstacle_aware_pedestrian_scenario(load_scenario_config(SCENARIO_PATH))
    planner = WindowedConflictReplanner(
        make_planner(scenario, risk_weight=8.0, reservation_padding=3),
        window_steps=32,
        repair_iterations=2,
        lookback_steps=4,
        clearance_margin=0.14,
    )
    result = planner.plan(scenario.robots)
    final_time = min(12.0, scenario.simulation.horizon)
    robot_paths = {
        robot_id: path_until_time(interpolate_path(path, samples_per_segment=12), final_time)
        for robot_id, path in result.paths.items()
    }
    save_gradient_overview(scenario, robot_paths, final_time)
    print(OUTPUT_DIR / "trajectory_gradient_overview.png")


def save_gradient_overview(scenario, robot_paths, final_time: float) -> None:
    """Draw all robot and pedestrian trajectories with time-darkening colors."""
    fig, ax = plt.subplots(figsize=(12.0, 7.68))
    plotter = WarehousePlotter(style="clean")
    plotter.configure_clean_axis(ax, scenario)
    plotter.draw_clean_obstacles(ax, scenario)

    for robot_index, robot in enumerate(scenario.robots):
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        path = robot_paths[robot.id]
        draw_gradient_path(
            ax,
            path,
            color=color,
            linewidth=2.9,
            alpha_start=0.18,
            alpha_end=0.98,
            zorder=6,
        )
        ax.scatter(
            robot.start.x,
            robot.start.y,
            s=42,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=9,
        )
        ax.scatter(
            robot.goal.x,
            robot.goal.y,
            s=54,
            marker="x",
            color=color,
            linewidths=1.7,
            zorder=9,
        )
        final_pose = path[-1]
        plotter.draw_robot_rectangle(
            ax,
            final_pose[0],
            final_pose[1],
            final_pose[2],
            robot.radius,
            color,
            alpha=0.96,
        )

    for pedestrian in scenario.dynamic_obstacles:
        ped_path = sample_pedestrian_path(pedestrian, final_time)
        draw_gradient_path(
            ax,
            ped_path,
            color="#111827",
            linewidth=2.2,
            alpha_start=0.16,
            alpha_end=0.94,
            zorder=5,
        )
        final_x, final_y = pedestrian.position_at(final_time)
        ax.add_patch(
            Circle(
                (final_x, final_y),
                radius=pedestrian.radius,
                facecolor="#111827",
                edgecolor="white",
                linewidth=1.0,
                alpha=0.94,
                zorder=10,
            )
        )
        start_x, start_y = pedestrian.position_at(0.0)
        ax.add_patch(
            Circle(
                (start_x, start_y),
                radius=pedestrian.radius * 0.72,
                facecolor="white",
                edgecolor="#111827",
                linewidth=0.9,
                alpha=0.86,
                zorder=8,
            )
        )

    fig.savefig(OUTPUT_DIR / "trajectory_gradient_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / "trajectory_gradient_overview.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_gradient_path(
    ax,
    path,
    color: str,
    linewidth: float,
    alpha_start: float,
    alpha_end: float,
    zorder: int,
) -> None:
    """Draw a path whose color opacity and saturation increase over time."""
    if len(path) < 2:
        return
    points = np.array([[pose[0], pose[1]] for pose in path], dtype=float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    progress = np.linspace(0.0, 1.0, len(segments))
    base = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    colors = []
    for value in progress:
        rgb = (1.0 - value) * (0.70 * white + 0.30 * base) + value * base
        alpha = alpha_start + (alpha_end - alpha_start) * value
        colors.append((*rgb, alpha))
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_collection(collection)


def path_until_time(path: list[ContinuousPose], final_time: float) -> list[ContinuousPose]:
    """Return path samples up to final_time with an interpolated endpoint."""
    if not path:
        return []
    clipped = [pose for pose in path if pose[3] <= final_time]
    endpoint = pose_at_time(path, final_time)
    if endpoint is not None and (not clipped or abs(clipped[-1][3] - final_time) > 1e-9):
        clipped.append(endpoint)
    return clipped or [path[0]]


def sample_pedestrian_path(pedestrian, final_time: float) -> list[ContinuousPose]:
    """Densely sample a pedestrian trajectory for smooth gradient rendering."""
    sample_count = max(2, int(final_time / 0.05) + 1)
    times = np.linspace(0.0, final_time, sample_count)
    samples: list[ContinuousPose] = []
    previous_heading = 0.0
    previous_xy: tuple[float, float] | None = None
    for time in times:
        x, y = pedestrian.position_at(float(time))
        if previous_xy is not None:
            dx = x - previous_xy[0]
            dy = y - previous_xy[1]
            if abs(dx) + abs(dy) > 1e-9:
                previous_heading = float(np.arctan2(dy, dx))
        samples.append((x, y, previous_heading, float(time)))
        previous_xy = (x, y)
    return samples


if __name__ == "__main__":
    main()
