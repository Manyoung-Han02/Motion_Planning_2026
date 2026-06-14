from pathlib import Path

from warehouse_planning.config import ScenarioConfig, SimulationConfig
from warehouse_planning.maps.warehouse_map import WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState
from warehouse_planning.simulation.simulator import RecedingHorizonConfig, Simulator


def make_scenario() -> ScenarioConfig:
    robot = Robot(
        id="r1",
        spec=RobotSpec(
            wheelbase=0.7,
            max_speed=1.0,
            max_steering_angle=0.5,
        ),
        start=RobotState(0.75, 0.75, 0.0, 0.2),
        goal=RobotState(4.25, 0.75, 0.0, 0.2),
    )
    obstacle = DynamicObstacle(
        id="cart",
        radius=0.25,
        trajectory=((0.0, 2.5, 3.5), (10.0, 2.5, 3.5)),
    )
    return ScenarioConfig(
        simulation=SimulationConfig(dt=0.2, horizon=3.0),
        warehouse=WarehouseMap(width=5.0, height=5.0, resolution=0.25),
        robots=(robot,),
        dynamic_obstacles=(obstacle,),
    )


def test_receding_horizon_executes_short_segments() -> None:
    simulator = Simulator(make_scenario())
    result = simulator.run_receding_horizon_single_robot(
        RecedingHorizonConfig(
            horizon_steps=8,
            execute_steps=2,
            max_cycles=3,
            risk_weight=1.0,
            safety_distance=0.8,
        )
    )

    assert result.robot_id == "r1"
    assert len(result.executed_path) > 1
    assert len(result.steps) >= 1
    assert all(len(step.executed_path) <= 2 for step in result.steps)
    assert "cart" in result.steps[0].obstacle_predictions


def test_receding_horizon_exports_gif(tmp_path: Path) -> None:
    simulator = Simulator(make_scenario())
    result = simulator.run_receding_horizon_single_robot(
        RecedingHorizonConfig(
            horizon_steps=6,
            execute_steps=2,
            max_cycles=1,
            risk_weight=1.0,
            safety_distance=0.8,
        )
    )
    output_path = tmp_path / "receding.gif"

    simulator.export_animation(result, output_path, fps=4)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
