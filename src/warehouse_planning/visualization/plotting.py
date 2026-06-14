"""Matplotlib visualization for warehouse planning scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

from warehouse_planning.config import ScenarioConfig
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose
from warehouse_planning.visualization.smoothing import (
    display_path,
    path_between,
    path_until,
    pose_at_time,
)


@dataclass
class WarehousePlotter:
    """Draw warehouse maps, robots, goals, and dynamic obstacles."""

    style: str = "clean"
    robot_color: str = "#1f77b4"
    goal_color: str = "#2ca02c"
    obstacle_color: str = "#d62728"
    shelf_color: str = "#7f7f7f"
    path_color: str = "#ff7f0e"
    pedestrian_color: str = "#252525"

    def draw_scenario(
        self,
        scenario: ScenarioConfig,
        time: float = 0.0,
    ) -> tuple[Figure, Axes]:
        """Draw a complete scenario at one instant in time."""
        if self.style == "clean":
            return self.draw_clean_scenario(scenario, time=time)

        fig, ax = plt.subplots(figsize=(10, 6))
        warehouse = scenario.warehouse

        ax.set_xlim(0, warehouse.width)
        ax.set_ylim(0, warehouse.height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Warehouse planning scenario at t={time:.1f}s")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.imshow(
            warehouse.occupancy_grid,
            cmap="Greys",
            extent=warehouse.extent,
            origin="lower",
            interpolation="nearest",
            alpha=0.25,
        )

        for obstacle in warehouse.static_obstacles:
            ax.add_patch(
                Rectangle(
                    (obstacle.x, obstacle.y),
                    obstacle.width,
                    obstacle.height,
                    facecolor=self.shelf_color,
                    edgecolor="black",
                    alpha=0.55,
                )
            )
            ax.text(
                obstacle.x + obstacle.width / 2.0,
                obstacle.y + obstacle.height / 2.0,
                obstacle.id,
                ha="center",
                va="center",
                fontsize=8,
            )

        for robot in scenario.robots:
            ax.add_patch(
                Circle(
                    (robot.start.x, robot.start.y),
                    radius=robot.start.radius,
                    facecolor=self.robot_color,
                    edgecolor="black",
                    alpha=0.8,
                )
            )
            ax.plot(robot.goal.x, robot.goal.y, marker="*", color=self.goal_color, ms=12)
            ax.plot(
                [robot.start.x, robot.goal.x],
                [robot.start.y, robot.goal.y],
                linestyle="--",
                color=self.robot_color,
                linewidth=1.0,
                alpha=0.45,
            )
            ax.text(robot.start.x, robot.start.y + 0.55, robot.id, ha="center", fontsize=8)

        for obstacle in scenario.dynamic_obstacles:
            x, y = obstacle.position_at(time)
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=obstacle.radius,
                    facecolor=self.obstacle_color,
                    edgecolor="black",
                    alpha=0.75,
                )
            )
            if obstacle.trajectory:
                xs = [point[1] for point in obstacle.trajectory]
                ys = [point[2] for point in obstacle.trajectory]
                ax.plot(xs, ys, color=self.obstacle_color, linestyle="-.", linewidth=1.2)
            ax.text(x, y + 0.65, obstacle.id, ha="center", fontsize=8)

        return fig, ax

    def draw_clean_scenario(
        self,
        scenario: ScenarioConfig,
        time: float = 0.0,
        show_pedestrians: bool = True,
    ) -> tuple[Figure, Axes]:
        """Draw a minimal, presentation-ready warehouse scene."""
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
        self.configure_clean_axis(ax, scenario)
        self.draw_clean_obstacles(ax, scenario)

        for robot_index, robot in enumerate(scenario.robots):
            color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
            ax.plot(
                robot.goal.x,
                robot.goal.y,
                marker="x",
                color=color,
                markersize=7,
                markeredgewidth=1.8,
                alpha=0.7,
            )
            self.draw_robot_rectangle(
                ax,
                robot.start.x,
                robot.start.y,
                robot.start.theta,
                robot.start.radius,
                color,
            )

        if show_pedestrians:
            self.draw_pedestrians(ax, scenario, time=time)

        return fig, ax

    def configure_clean_axis(self, ax: Axes, scenario: ScenarioConfig) -> None:
        """Apply clean map styling with no ticks, labels, or grid."""
        warehouse = scenario.warehouse
        ax.set_facecolor("white")
        ax.set_xlim(0.0, warehouse.width)
        ax.set_ylim(0.0, warehouse.height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                warehouse.width,
                warehouse.height,
                facecolor="none",
                edgecolor="#2f2f2f",
                linewidth=1.1,
                zorder=20,
            )
        )

    def draw_clean_obstacles(self, ax: Axes, scenario: ScenarioConfig) -> None:
        """Draw static obstacles as simple minimal rectangles."""
        for obstacle in scenario.warehouse.static_obstacles:
            ax.add_patch(
                Rectangle(
                    (obstacle.x, obstacle.y),
                    obstacle.width,
                    obstacle.height,
                    facecolor="#d7d7d2",
                    edgecolor="#9f9f99",
                    linewidth=1.0,
                    alpha=1.0,
                    zorder=2,
                )
            )

    def draw_pedestrians(
        self,
        ax: Axes,
        scenario: ScenarioConfig,
        time: float,
        pedestrian_positions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Draw pedestrians at their current positions only."""
        for obstacle in scenario.dynamic_obstacles:
            if pedestrian_positions is None:
                x, y = obstacle.predicted_position(time)
            else:
                x, y = pedestrian_positions[obstacle.id]
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=obstacle.radius,
                    facecolor=self.pedestrian_color,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=8,
                )
            )

    def draw_robot_triangle(
        self,
        ax: Axes,
        x: float,
        y: float,
        theta: float,
        radius: float,
        color: str,
        alpha: float = 0.95,
    ) -> None:
        """Draw a triangular robot with its front vertex along heading."""
        length = radius * 2.8
        width = radius * 2.0
        local_points = (
            (length * 0.58, 0.0),
            (-length * 0.42, width * 0.5),
            (-length * 0.42, -width * 0.5),
        )

    def draw_robot_rectangle(
        self,
        ax: Axes,
        x: float,
        y: float,
        theta: float,
        radius: float,
        color: str,
        alpha: float = 0.96,
    ) -> None:
        """Draw an oriented rectangular robot with a clear front direction."""
        length = radius * 3.1
        width = radius * 1.85
        half_length = length * 0.5
        half_width = width * 0.5
        local_points = (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        )
        points = [_rotate_point(x, y, theta, point) for point in local_points]
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                alpha=alpha,
                zorder=10,
            )
        )

        front_center = _rotate_point(x, y, theta, (half_length * 0.72, 0.0))
        front_left = _rotate_point(x, y, theta, (half_length * 0.35, half_width * 0.58))
        front_right = _rotate_point(x, y, theta, (half_length * 0.35, -half_width * 0.58))
        ax.add_patch(
            Polygon(
                (front_center, front_left, front_right),
                closed=True,
                facecolor="white",
                edgecolor="none",
                alpha=0.82,
                zorder=11,
            )
        )
        points = []
        for local_x, local_y in local_points:
            world_x = x + local_x * cos(theta) - local_y * sin(theta)
            world_y = y + local_x * sin(theta) + local_y * cos(theta)
            points.append((world_x, world_y))

        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                alpha=alpha,
                zorder=10,
            )
        )

    def draw_robot_motion(
        self,
        ax: Axes,
        path: list[ContinuousPose],
        time: float,
        color: str,
        radius: float,
        future_horizon: float = 1.8,
    ) -> None:
        """Draw robot history, current triangular body, and predicted future path."""
        smooth_path = display_path(path)
        pose = pose_at_time(smooth_path, time)
        if pose is None:
            return

        history = path_until(smooth_path, time)
        if len(history) > 1:
            ax.plot(
                [sample[0] for sample in history],
                [sample[1] for sample in history],
                color=color,
                linewidth=2.0,
                alpha=0.78,
                zorder=5,
            )

        future = path_between(smooth_path, time, time + future_horizon)
        if len(future) > 1:
            ax.plot(
                [sample[0] for sample in future],
                [sample[1] for sample in future],
                color=color,
                linewidth=1.4,
                linestyle=(0, (4, 4)),
                alpha=0.35,
                zorder=4,
            )

        self.draw_robot_rectangle(
            ax,
            pose[0],
            pose[1],
            pose[2],
            radius,
            color,
        )

    def draw_risk_field(
        self,
        ax: Axes,
        scenario: ScenarioConfig,
        time: float,
        sigma: float = 0.85,
        resolution: float = 0.12,
        pedestrian_positions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Render a smooth pedestrian-related Gaussian risk heatmap."""
        warehouse = scenario.warehouse
        xs = np.arange(0.0, warehouse.width + resolution, resolution)
        ys = np.arange(0.0, warehouse.height + resolution, resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        risk = np.zeros_like(grid_x)

        for pedestrian in scenario.dynamic_obstacles:
            if pedestrian_positions is None:
                px, py = pedestrian.predicted_position(time)
            else:
                px, py = pedestrian_positions[pedestrian.id]
            squared_distance = (grid_x - px) ** 2 + (grid_y - py) ** 2
            risk += np.exp(-0.5 * squared_distance / (sigma * sigma))

        if risk.max() > 0.0:
            risk = risk / risk.max()
        ax.imshow(
            risk,
            extent=(0.0, warehouse.width, 0.0, warehouse.height),
            origin="lower",
            cmap="YlOrRd",
            interpolation="bilinear",
            alpha=0.42,
            zorder=1,
        )

    def draw_path(
        self,
        ax: Axes,
        path: list[ContinuousPose],
        label: str = "planned path",
        color: str | None = None,
        smooth: bool = True,
    ) -> None:
        """Draw a continuous kinodynamic path on an existing scenario axis."""
        if not path:
            return

        display_samples = display_path(path) if smooth else path
        xs = [pose[0] for pose in display_samples]
        ys = [pose[1] for pose in display_samples]
        ax.plot(
            xs,
            ys,
            color=color or self.path_color,
            linewidth=2.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=label,
        )
        ax.legend(loc="upper right")

    def show(self) -> None:
        """Open the Matplotlib figure window."""
        plt.show()


ROBOT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
)


def _rotate_point(
    x: float,
    y: float,
    theta: float,
    local_point: tuple[float, float],
) -> tuple[float, float]:
    """Rotate a local body point into world coordinates."""
    local_x, local_y = local_point
    return (
        x + local_x * cos(theta) - local_y * sin(theta),
        y + local_x * sin(theta) + local_y * cos(theta),
    )
