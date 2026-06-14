from warehouse_planning.models.dynamic_obstacle import DynamicObstacle


def test_dynamic_obstacle_interpolates_position() -> None:
    obstacle = DynamicObstacle(
        id="moving_box",
        radius=0.5,
        trajectory=((0.0, 0.0, 0.0), (10.0, 10.0, 5.0)),
    )

    assert obstacle.position_at(5.0) == (5.0, 2.5)
    assert obstacle.predicted_position(5.0) == (5.0, 2.5)
