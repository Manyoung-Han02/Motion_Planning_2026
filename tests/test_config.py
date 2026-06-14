from pathlib import Path

from warehouse_planning.config import load_scenario_config


def test_load_default_config() -> None:
    scenario = load_scenario_config(Path("configs/default.yaml"))

    assert scenario.warehouse.width == 20.0
    assert scenario.warehouse.occupancy_grid.dtype == bool
    assert len(scenario.robots) == 2
    assert len(scenario.dynamic_obstacles) == 1


def test_load_small_warehouse_demo_config() -> None:
    scenario = load_scenario_config(Path("configs/warehouse_small.yaml"))

    assert scenario.warehouse.width == 12.0
    assert scenario.robots[0].start.radius == 0.3
    assert len(scenario.warehouse.static_obstacles) >= 4
