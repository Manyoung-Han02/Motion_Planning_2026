from math import hypot

from warehouse_planning.maps.warehouse_map import RectangleObstacle, WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState
from warehouse_planning.planning.collision import CollisionChecker
from warehouse_planning.planning.kinodynamic_astar import KinodynamicAStarPlanner


def make_robot() -> Robot:
    return Robot(
        id="r1",
        spec=RobotSpec(
            wheelbase=0.7,
            max_speed=1.0,
            max_steering_angle=0.5,
        ),
        start=RobotState(0.75, 0.75, 0.0, 0.2),
        goal=RobotState(4.25, 4.25, 0.0, 0.2),
    )


def make_test_map() -> WarehouseMap:
    return WarehouseMap(
        width=5.0,
        height=5.0,
        resolution=0.25,
        static_obstacles=(
            RectangleObstacle(id="wall", x=2.0, y=0.0, width=0.4, height=3.6),
        ),
    )


def test_kinodynamic_astar_finds_path() -> None:
    robot = make_robot()
    checker = CollisionChecker(warehouse=make_test_map())
    planner = KinodynamicAStarPlanner(
        collision_checker=checker,
        dt=0.2,
        step_distance=0.5,
        goal_tolerance=0.5,
        max_time_steps=80,
    )

    path = planner.plan(robot)

    assert len(path) > 2
    assert path[0] == (robot.start.x, robot.start.y, robot.start.theta, 0.0)
    assert hypot(path[-1][0] - robot.goal.x, path[-1][1] - robot.goal.y) <= 0.5


def test_kinodynamic_astar_path_is_static_collision_free() -> None:
    robot = make_robot()
    checker = CollisionChecker(warehouse=make_test_map())
    planner = KinodynamicAStarPlanner(
        collision_checker=checker,
        dt=0.2,
        step_distance=0.5,
        goal_tolerance=0.5,
        max_time_steps=80,
    )

    path = planner.plan(robot)

    for x, y, theta, _ in path:
        state = RobotState(x=x, y=y, theta=theta, radius=robot.radius)
        assert not checker.collides_with_static_obstacle(state)


def test_dynamic_risk_cost_soft_penalizes_inside_safety_distance() -> None:
    obstacle = DynamicObstacle(
        id="worker",
        radius=0.2,
        trajectory=((0.0, 2.0, 2.0), (10.0, 2.0, 2.0)),
    )
    checker = CollisionChecker(
        warehouse=WarehouseMap(width=5.0, height=5.0, resolution=0.25),
        dynamic_obstacles=(obstacle,),
    )
    planner = KinodynamicAStarPlanner(
        collision_checker=checker,
        dt=0.2,
        risk_weight=5.0,
        safety_distance=1.0,
        risk_sigma=0.5,
    )

    near = planner.dynamic_risk_cost(RobotState(2.8, 2.0, 0.0, 0.2), time=0.0)
    far = planner.dynamic_risk_cost(RobotState(4.5, 4.5, 0.0, 0.2), time=0.0)

    assert near > 0.0
    assert far == 0.0


def test_kinodynamic_astar_path_avoids_dynamic_collision_distance() -> None:
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
        radius=0.3,
        trajectory=((0.0, 1.75, 0.75), (20.0, 1.75, 0.75)),
    )
    checker = CollisionChecker(
        warehouse=WarehouseMap(width=5.0, height=5.0, resolution=0.25),
        dynamic_obstacles=(obstacle,),
    )
    planner = KinodynamicAStarPlanner(
        collision_checker=checker,
        dt=0.2,
        step_distance=0.5,
        goal_tolerance=0.5,
        max_time_steps=80,
        risk_weight=2.0,
    )

    path = planner.plan(robot)

    for x, y, theta, time in path:
        state = RobotState(x=x, y=y, theta=theta, radius=robot.radius)
        assert not checker.collides_with_dynamic_obstacle(state, time)
