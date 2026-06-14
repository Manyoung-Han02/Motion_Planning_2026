from math import pi

from warehouse_planning.evaluation.metrics import evaluate_multi_robot_plan
from warehouse_planning.maps.warehouse_map import WarehouseMap
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import KinodynamicAStarPlanner
from warehouse_planning.planning.prioritized import IndependentPlanner, PrioritizedPlanner


def make_swap_robots() -> tuple[Robot, ...]:
    spec = RobotSpec(wheelbase=0.7, max_speed=1.0, max_steering_angle=0.5)
    return (
        Robot(
            id="left_to_right",
            spec=spec,
            start=RobotState(0.75, 1.5, 0.0, 0.2),
            goal=RobotState(4.25, 1.5, 0.0, 0.2),
        ),
        Robot(
            id="right_to_left",
            spec=spec,
            start=RobotState(4.25, 1.5, pi, 0.2),
            goal=RobotState(0.75, 1.5, pi, 0.2),
        ),
    )


def make_base_planner() -> tuple[WarehouseMap, KinodynamicAStarPlanner]:
    warehouse = WarehouseMap(width=5.0, height=3.0, resolution=0.25)
    checker = CollisionChecker(warehouse=warehouse)
    planner = KinodynamicAStarPlanner(
        collision_checker=checker,
        dt=0.2,
        theta_bins=8,
        step_distance=0.5,
        goal_tolerance=0.5,
        max_time_steps=35,
    )
    return warehouse, planner


def test_independent_planning_detects_posthoc_robot_conflicts() -> None:
    warehouse, base_planner = make_base_planner()
    robots = make_swap_robots()

    result = IndependentPlanner(base_planner).plan(robots)
    metrics = evaluate_multi_robot_plan(
        robots,
        result,
        dt=base_planner.dt,
        warehouse=warehouse,
    )

    assert result.success
    assert metrics.success_rate == 1.0
    assert metrics.same_cell_conflict_count + metrics.edge_swap_conflict_count > 0
    assert metrics.total_path_length > 0.0
    assert metrics.makespan > 0.0
    assert metrics.planning_time >= 0.0


def test_prioritized_planning_avoids_reserved_cell_and_edge_conflicts() -> None:
    warehouse, base_planner = make_base_planner()
    robots = make_swap_robots()

    result = PrioritizedPlanner(base_planner).plan(robots)
    metrics = evaluate_multi_robot_plan(
        robots,
        result,
        dt=base_planner.dt,
        warehouse=warehouse,
    )

    assert result.success
    assert metrics.success_rate == 1.0
    assert metrics.same_cell_conflict_count == 0
    assert metrics.edge_swap_conflict_count == 0
    assert metrics.total_path_length > 0.0
