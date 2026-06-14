# Dynamic Obstacle-Aware Planning for Nonholonomic Multi-Robot Systems

Initial Python 3.11 scaffold for a 2D warehouse multi-robot planning simulator.

The project currently provides clean interfaces and lightweight placeholder behavior for:

- Warehouse map representation
- Nonholonomic robot model
- Dynamic obstacle model
- Collision checking
- Kinodynamic A*
- Prioritized planning
- CBS-style planning
- Simulation
- Visualization
- Evaluation metrics

The multi-robot baseline layer includes independent planning and prioritized
planning with time-indexed reservations for same-cell and edge-swap conflicts.
Metrics report success rate, robot-robot collisions, path length, makespan, and
planning time.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The default demo loads `configs/warehouse_small.yaml`, plans a single-robot
kinodynamic A* path for the first robot, and overlays it on the warehouse map.
It also saves dynamic-risk comparison figures to `results/`.

To save the first scene instead of opening an interactive window:

```bash
python main.py --config configs/warehouse_small.yaml --save outputs/warehouse_scene.png --no-show
```

To compare paths with and without dynamic obstacle risk:

```bash
python main.py --risk-weight 10.0 --no-show
```

To run receding-horizon simulation and export an animation:

```bash
python main.py --simulate-receding-horizon --animation results/receding_horizon.gif --no-show
```

To run the benchmark suite and save CSV/bar plots:

```bash
python main.py --benchmark --results-dir results/benchmark --no-show
```

To run the clean presentation demo with 8 triangular robots, pedestrians,
smooth motion, and saved PNG/MP4 outputs:

```bash
python main.py --clean-demo
```

For headless export:

```bash
python main.py --clean-demo --no-show --save-gif
```

## Test

```bash
pytest
```

## Project Layout

```text
configs/                 YAML experiment configs
src/warehouse_planning/  Research simulator package
tests/                   Basic pytest coverage
main.py                  Runnable visualization entry point
```

The planner classes intentionally do not implement full research algorithms yet. They define typed inputs and outputs so experiments can be added incrementally without reshaping the package.
