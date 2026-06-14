from pathlib import Path

from warehouse_planning.config import load_scenario_config
from warehouse_planning.models.robot import RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import ContinuousPose
from warehouse_planning.visualization.clean_demo import (
    generate_pedestrian_paths,
    path_conflicts_with_scheduled,
    plan_and_smooth_demo_paths,
)
from warehouse_planning.visualization.smoothing import interpolate_path, pose_at_time


def test_clean_demo_config_has_expected_scene_size() -> None:
    scenario = load_scenario_config(Path("configs/warehouse_clean_demo.yaml"))

    assert scenario.visualization.style == "clean"
    assert scenario.warehouse.width == 18.0
    assert len(scenario.warehouse.static_obstacles) == 10
    assert len(scenario.robots) == 4
    assert len(scenario.dynamic_obstacles) == 3


def test_interpolate_path_adds_smooth_samples() -> None:
    path: list[ContinuousPose] = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.5, 1.0),
    ]

    smoothed = interpolate_path(path, samples_per_segment=5)
    midpoint = pose_at_time(smoothed, 0.5)

    assert len(smoothed) == 6
    assert midpoint is not None
    assert 0.0 < midpoint[0] < 1.0
    assert 0.0 < midpoint[1] < 1.0


def test_clean_demo_robot_paths_are_priority_scheduled() -> None:
    scenario = load_scenario_config(Path("configs/warehouse_clean_demo.yaml"))
    paths = plan_and_smooth_demo_paths(scenario)
    robot_by_id = {robot.id: robot for robot in scenario.robots}
    scheduled: dict[str, list[ContinuousPose]] = {}

    for robot in scenario.robots:
        assert not path_conflicts_with_scheduled(
            paths[robot.id],
            robot,
            scheduled,
            robot_by_id,
            time_step=0.2,
            clearance_margin=0.12,
        )
        scheduled[robot.id] = paths[robot.id]


def test_clean_demo_robot_paths_avoid_static_obstacles() -> None:
    scenario = load_scenario_config(Path("configs/warehouse_clean_demo.yaml"))
    checker = CollisionChecker(warehouse=scenario.warehouse)
    paths = plan_and_smooth_demo_paths(scenario)

    for robot in scenario.robots:
        for x, y, theta, _ in paths[robot.id]:
            state = RobotState(x, y, theta, robot.radius)
            assert not checker.collides_with_static_obstacle(state)


def test_clean_demo_pedestrians_avoid_obstacles() -> None:
    scenario = load_scenario_config(Path("configs/warehouse_clean_demo.yaml"))
    checker = CollisionChecker(warehouse=scenario.warehouse)
    pedestrian_paths = generate_pedestrian_paths(scenario, duration=4.0, dt=0.2)
    radius_by_id = {
        pedestrian.id: pedestrian.radius for pedestrian in scenario.dynamic_obstacles
    }

    for pedestrian_id, path in pedestrian_paths.items():
        for x, y, theta, _ in path:
            state = RobotState(x, y, theta, radius_by_id[pedestrian_id])
            assert not checker.collides_with_static_obstacle(state)
