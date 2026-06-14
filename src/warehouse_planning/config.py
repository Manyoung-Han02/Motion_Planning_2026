"""Configuration loading for warehouse planning scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from warehouse_planning.maps.warehouse_map import RectangleObstacle, WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState


@dataclass(frozen=True)
class SimulationConfig:
    """Time discretization settings for a planning or simulation run."""

    dt: float
    horizon: float


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization settings loaded from scenario configuration."""

    style: str = "clean"


@dataclass(frozen=True)
class ScenarioConfig:
    """Complete scenario assembled from a YAML configuration file."""

    simulation: SimulationConfig
    warehouse: WarehouseMap
    robots: tuple[Robot, ...]
    dynamic_obstacles: tuple[DynamicObstacle, ...]
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    """Load a scenario configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return scenario_from_dict(raw)


def scenario_from_dict(raw: dict[str, Any]) -> ScenarioConfig:
    """Build a typed scenario from a raw configuration mapping."""
    simulation = raw.get("simulation", {})
    warehouse = raw.get("warehouse", {})
    visualization = raw.get("visualization", {})

    return ScenarioConfig(
        simulation=SimulationConfig(
            dt=float(simulation.get("dt", 0.1)),
            horizon=float(simulation.get("horizon", 10.0)),
        ),
        warehouse=_parse_warehouse(warehouse),
        robots=tuple(_parse_robot(item) for item in raw.get("robots", [])),
        dynamic_obstacles=tuple(
            _parse_dynamic_obstacle(item) for item in raw.get("dynamic_obstacles", [])
        ),
        visualization=VisualizationConfig(
            style=str(visualization.get("style", "clean")),
        ),
    )


def _parse_warehouse(raw: dict[str, Any]) -> WarehouseMap:
    obstacles = []
    for item in raw.get("static_obstacles", []):
        if item.get("type", "rectangle") != "rectangle":
            raise ValueError(f"Unsupported obstacle type: {item.get('type')}")
        obstacles.append(
            RectangleObstacle(
                id=str(item["id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item["width"]),
                height=float(item["height"]),
            )
        )

    return WarehouseMap(
        width=float(raw["width"]),
        height=float(raw["height"]),
        resolution=float(raw.get("resolution", 0.5)),
        static_obstacles=tuple(obstacles),
    )


def _parse_robot(raw: dict[str, Any]) -> Robot:
    start = raw["start"]
    goal = raw["goal"]
    radius = float(raw.get("radius", 0.3))
    spec = RobotSpec(
        wheelbase=float(raw.get("wheelbase", 0.7)),
        max_speed=float(raw.get("max_speed", 1.0)),
        max_steering_angle=float(raw.get("max_steering_angle", 0.5)),
    )
    return Robot(
        id=str(raw["id"]),
        spec=spec,
        start=RobotState(float(start[0]), float(start[1]), float(start[2]), radius),
        goal=RobotState(float(goal[0]), float(goal[1]), float(goal[2]), radius),
    )


def _parse_dynamic_obstacle(raw: dict[str, Any]) -> DynamicObstacle:
    return DynamicObstacle(
        id=str(raw["id"]),
        radius=float(raw.get("radius", 0.3)),
        trajectory=tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in raw.get("trajectory", [])
        ),
    )
