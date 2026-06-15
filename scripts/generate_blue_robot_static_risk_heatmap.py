"""Generate a static-pedestrian risk heatmap for the main blue robot path."""

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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from warehouse_planning.config import load_scenario_config
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
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
    """Save a main-scenario risk heatmap focused on the blue robot."""
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario = obstacle_aware_pedestrian_scenario(load_scenario_config(SCENARIO_PATH))
    robot = scenario.robots[0]
    freeze_time = 4.8
    pedestrian_positions = {
        pedestrian.id: pedestrian.position_at(freeze_time)
        for pedestrian in scenario.dynamic_obstacles
    }
    frozen_scenario = freeze_pedestrians(scenario, pedestrian_positions)
    risk_paths = {}
    for risk_weight in (0.0, 2.0, 4.0, 8.0, 12.0):
        planner = make_planner(
            frozen_scenario,
            risk_weight=risk_weight,
            reservation_padding=0,
            safety_distance=2.4,
            risk_sigma=1.0,
        )
        risk_paths[risk_weight] = interpolate_path(
            planner.plan(robot),
            samples_per_segment=12,
        )
    save_static_risk_heatmap(frozen_scenario, robot, risk_paths, pedestrian_positions)
    print(OUTPUT_DIR / "risk_heatmap_blue_robot_static_pedestrians.png")


def save_static_risk_heatmap(
    scenario,
    robot,
    risk_paths: dict[float, list],
    pedestrian_positions: dict[str, tuple[float, float]],
) -> None:
    """Draw one blue robot path over a static pedestrian heatmap."""
    fig, ax = plt.subplots(figsize=(12.0, 7.68))
    plotter = WarehousePlotter(style="clean")
    plotter.configure_clean_axis(ax, scenario)
    draw_static_heatmap(ax, scenario, pedestrian_positions, sigma=1.0, resolution=0.08)
    plotter.draw_clean_obstacles(ax, scenario)

    cmap = plt.get_cmap("Blues")
    weights = sorted(risk_paths)
    for index, risk_weight in enumerate(weights):
        color = cmap(0.34 + 0.58 * index / max(len(weights) - 1, 1))
        path = risk_paths[risk_weight]
        ax.plot(
            [pose[0] for pose in path],
            [pose[1] for pose in path],
            color=color,
            linewidth=2.0 + 0.28 * index,
            alpha=0.92,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"$\\lambda_h={risk_weight:g}$",
            zorder=7 + index,
        )
    blue = ROBOT_COLORS[0]
    ax.scatter(
        robot.start.x,
        robot.start.y,
        s=58,
        facecolors="white",
        edgecolors=blue,
        linewidths=1.5,
        zorder=10,
    )
    ax.scatter(
        robot.goal.x,
        robot.goal.y,
        s=70,
        marker="x",
        color=blue,
        linewidths=2.0,
        zorder=10,
    )
    final_pose = risk_paths[max(weights)][-1]
    plotter.draw_robot_rectangle(
        ax,
        final_pose[0],
        final_pose[1],
        final_pose[2],
        robot.radius,
        blue,
        alpha=0.96,
    )

    for pedestrian in scenario.dynamic_obstacles:
        x, y = pedestrian_positions[pedestrian.id]
        ax.add_patch(
            Circle(
                (x, y),
                radius=pedestrian.radius,
                facecolor="#111827",
                edgecolor="white",
                linewidth=1.0,
                alpha=0.95,
                zorder=11,
            )
        )

    ax.legend(
        loc="lower right",
        ncol=1,
        fontsize=13,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cbd5e1",
        borderpad=1.1,
        labelspacing=0.75,
        handlelength=4.0,
        handletextpad=1.0,
    )

    fig.savefig(
        OUTPUT_DIR / "risk_heatmap_blue_robot_static_pedestrians.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        OUTPUT_DIR / "risk_heatmap_blue_robot_static_pedestrians.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def draw_static_heatmap(
    ax,
    scenario,
    pedestrian_positions: dict[str, tuple[float, float]],
    sigma: float,
    resolution: float,
) -> None:
    """Draw a normalized static Gaussian pedestrian-risk heatmap."""
    warehouse = scenario.warehouse
    xs = np.arange(0.0, warehouse.width + resolution, resolution)
    ys = np.arange(0.0, warehouse.height + resolution, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    risk = np.zeros_like(grid_x)

    for x, y in pedestrian_positions.values():
        squared_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        risk += np.exp(-0.5 * squared_distance / (sigma * sigma))

    if risk.max() > 0.0:
        risk = risk / risk.max()
    ax.imshow(
        risk,
        extent=(0.0, warehouse.width, 0.0, warehouse.height),
        origin="lower",
        cmap="YlOrRd",
        interpolation="bilinear",
        alpha=0.48,
        zorder=1,
    )


def freeze_pedestrians(
    scenario,
    pedestrian_positions: dict[str, tuple[float, float]],
):
    """Return a scenario where pedestrians stay fixed at selected positions."""
    frozen = []
    for pedestrian in scenario.dynamic_obstacles:
        x, y = pedestrian_positions[pedestrian.id]
        frozen.append(
            DynamicObstacle(
                id=pedestrian.id,
                radius=pedestrian.radius,
                trajectory=((0.0, x, y), (scenario.simulation.horizon, x, y)),
            )
        )
    return type(scenario)(
        simulation=scenario.simulation,
        warehouse=scenario.warehouse,
        robots=scenario.robots,
        dynamic_obstacles=tuple(frozen),
        visualization=scenario.visualization,
    )


if __name__ == "__main__":
    main()
