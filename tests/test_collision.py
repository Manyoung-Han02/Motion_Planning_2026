from warehouse_planning.maps.warehouse_map import RectangleObstacle, WarehouseMap
from warehouse_planning.models.dynamic_obstacle import DynamicObstacle
from warehouse_planning.models.robot import Robot, RobotSpec, RobotState
from warehouse_planning.planning.collision import CollisionChecker


def make_robot(
    start: RobotState = RobotState(1.0, 1.0, 0.0, 0.25),
) -> Robot:
    return Robot(
        id="r1",
        spec=RobotSpec(
            wheelbase=0.7,
            max_speed=1.0,
            max_steering_angle=0.5,
        ),
        start=start,
        goal=RobotState(8.0, 8.0, 0.0, start.radius),
    )


def test_collision_checker_accepts_collision_free_state() -> None:
    warehouse = WarehouseMap(
        width=10.0,
        height=10.0,
        resolution=0.5,
        static_obstacles=(
            RectangleObstacle(id="shelf", x=2.0, y=2.0, width=2.0, height=2.0),
        ),
    )
    robot = make_robot()
    checker = CollisionChecker(warehouse=warehouse)

    assert checker.is_state_valid(robot, RobotState(1.0, 1.0, 0.0, 0.25))


def test_collision_checker_rejects_static_obstacle() -> None:
    warehouse = WarehouseMap(
        width=10.0,
        height=10.0,
        resolution=0.5,
        static_obstacles=(
            RectangleObstacle(id="shelf", x=2.0, y=2.0, width=2.0, height=2.0),
        ),
    )
    robot = make_robot()
    checker = CollisionChecker(warehouse=warehouse)

    assert not checker.is_state_valid(robot, RobotState(2.5, 2.5, 0.0, 0.25))


def test_collision_checker_rejects_inflated_static_obstacle() -> None:
    warehouse = WarehouseMap(
        width=10.0,
        height=10.0,
        resolution=0.5,
        static_obstacles=(
            RectangleObstacle(id="shelf", x=2.0, y=2.0, width=2.0, height=2.0),
        ),
    )
    robot = make_robot()
    checker = CollisionChecker(warehouse=warehouse)

    assert not checker.is_state_valid(robot, RobotState(1.5, 2.5, 0.0, 0.5))


def test_collision_checker_rejects_robot_robot_overlap() -> None:
    warehouse = WarehouseMap(width=10.0, height=10.0, resolution=0.5)
    robot = make_robot()
    checker = CollisionChecker(warehouse=warehouse)
    state = RobotState(1.0, 1.0, 0.0, 0.4)
    other_state = RobotState(1.7, 1.0, 0.0, 0.4)

    assert not checker.is_state_valid(
        robot,
        state,
        other_robot_states=(other_state,),
    )
    assert checker.collides_with_robot(state, other_state)


def test_collision_checker_accepts_separated_robots() -> None:
    warehouse = WarehouseMap(width=10.0, height=10.0, resolution=0.5)
    checker = CollisionChecker(warehouse=warehouse)

    assert not checker.collides_with_robot(
        RobotState(1.0, 1.0, 0.0, 0.3),
        RobotState(2.0, 1.0, 0.0, 0.3),
    )


def test_collision_checker_rejects_dynamic_obstacle_overlap() -> None:
    warehouse = WarehouseMap(width=10.0, height=10.0, resolution=0.5)
    dynamic_obstacle = DynamicObstacle(
        id="forklift",
        radius=0.4,
        trajectory=((0.0, 2.0, 2.0), (1.0, 3.0, 2.0)),
    )
    robot = make_robot()
    checker = CollisionChecker(
        warehouse=warehouse,
        dynamic_obstacles=(dynamic_obstacle,),
    )

    assert not checker.is_state_valid(robot, RobotState(2.2, 2.0, 0.0, 0.3), time=0.0)


def test_collision_checker_accepts_state_clear_of_dynamic_obstacles() -> None:
    warehouse = WarehouseMap(width=10.0, height=10.0, resolution=0.5)
    dynamic_obstacle = DynamicObstacle(
        id="forklift",
        radius=0.4,
        trajectory=((0.0, 2.0, 2.0), (1.0, 3.0, 2.0)),
    )
    robot = make_robot()
    checker = CollisionChecker(
        warehouse=warehouse,
        dynamic_obstacles=(dynamic_obstacle,),
    )

    assert checker.is_state_valid(robot, RobotState(5.0, 5.0, 0.0, 0.3), time=0.0)
