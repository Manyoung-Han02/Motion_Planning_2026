from warehouse_planning.maps.warehouse_map import RectangleObstacle, WarehouseMap


def test_warehouse_map_rasterizes_rectangular_obstacles() -> None:
    warehouse = WarehouseMap(
        width=5.0,
        height=4.0,
        resolution=1.0,
        static_obstacles=(
            RectangleObstacle(id="shelf", x=1.0, y=1.0, width=2.0, height=1.0),
        ),
    )

    assert warehouse.occupancy_grid.shape == (4, 5)
    assert warehouse.is_occupied(1.5, 1.5)
    assert warehouse.is_occupied(2.5, 1.5)
    assert not warehouse.is_occupied(4.5, 3.5)


def test_warehouse_map_reports_distance_to_nearest_obstacle() -> None:
    warehouse = WarehouseMap(
        width=5.0,
        height=4.0,
        resolution=1.0,
        static_obstacles=(
            RectangleObstacle(id="shelf", x=1.0, y=1.0, width=1.0, height=1.0),
        ),
    )

    assert warehouse.distance_to_nearest_obstacle(1.5, 1.5) == 0.0
    assert warehouse.distance_to_nearest_obstacle(3.5, 1.5) == 2.0
