from pathlib import Path

from warehouse_planning.config import load_scenario_config


def test_load_clean_demo_config() -> None:
    scenario = load_scenario_config(Path("configs/warehouse_clean_demo.yaml"))

    assert scenario.visualization.style == "clean"
    assert scenario.warehouse.width == 18.0
    assert scenario.warehouse.occupancy_grid.dtype == bool
    assert len(scenario.robots) == 4
    assert len(scenario.dynamic_obstacles) == 3
    assert len(scenario.warehouse.static_obstacles) == 10
    assert scenario.robots[0].start.radius == 0.28
