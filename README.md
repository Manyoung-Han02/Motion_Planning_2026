# Multi-Robot Warehouse Motion Planning

This project simulates dynamic-obstacle-aware warehouse motion planning for a
small fleet of nonholonomic robots. The final demo uses 4 robots, 3 pedestrian
obstacles, static shelf geometry, prioritized multi-robot scheduling, and
smooth larger-radius trajectory visualization.

The robot paths are planned on a kinodynamic lattice, then rounded with a
Chaikin-style path smoother and dense time sampling so the rendered motion
looks closer to a Dubins-car-like vehicle than a grid polyline.

## File Structure

```text
configs/
  warehouse_clean_demo.yaml       Final 4-robot, 3-pedestrian scenario
src/warehouse_planning/
  config.py                       YAML loading into typed scenario objects
  maps/warehouse_map.py           Static warehouse geometry and occupancy grid
  models/robot.py                 Robot state, controls, and bicycle step model
  models/dynamic_obstacle.py      Time-indexed pedestrian obstacle model
  planning/collision.py           Static and dynamic collision checks
  planning/kinodynamic_astar.py   Single-robot kinodynamic A* lattice planner
  planning/prioritized.py         Independent and prioritized multi-robot planners
  simulation/simulator.py         Receding-horizon simulation/export utilities
  visualization/smoothing.py      Larger-radius path rounding and interpolation
  visualization/plotting.py       Matplotlib warehouse, robot, and path drawing
  visualization/clean_demo.py     Final presentation demo pipeline
  evaluation/metrics.py           Path and multi-robot metrics
  evaluation/benchmark.py         Benchmark runner and plots
tests/                            Pytest coverage for planning and visualization
main.py                           Command-line entry point
requirements.txt                  Python dependencies
pytest.ini                        Test configuration
```

Generated files are written to `results/` and are ignored by Git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Main Commands

Run the default risk-comparison visualization:

```powershell
python main.py --no-show
```

Run the final clean multi-robot demo and export PNG/MP4 outputs:

```powershell
python main.py --clean-demo --no-show
```

Also export a GIF:

```powershell
python main.py --clean-demo --no-show --save-gif
```

Run receding-horizon simulation:

```powershell
python main.py --simulate-receding-horizon --animation results/receding_horizon.gif --no-show
```

Run benchmark plots and CSV output:

```powershell
python main.py --benchmark --results-dir results/benchmark --no-show
```

Run tests:

```powershell
python -m pytest
```

If using the local virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Generated Outputs

The clean demo writes:

- `results/clean_map_demo.png`
- `results/risk_field_demo.png`
- `results/multi_robot_demo.mp4`
- `results/multi_robot_demo.gif` when `--save-gif` is used

The default visualization writes:

- `results/risk_comparison.png`
- `results/risk_weight_0.png`
- `results/risk_weight_positive.png`

The benchmark command writes:

- `results/benchmark/benchmark_results.csv`
- grouped bar plots for success rate, path length, makespan, computation time,
  robot-robot collisions, and pedestrian near misses

## Motion Model Notes

- The final clean demo uses 32 heading bins in the kinodynamic A* planner,
  producing smaller heading changes and a larger effective turning radius than
  the older 16-bin demo path.
- Planner transition validation samples at quarter-cell spacing to avoid
  slipping through shelf corners between coarse states.
- `visualization/smoothing.py` applies open Chaikin corner rounding and dense
  time sampling. This keeps the path continuous-looking while preserving the
  planned route shape.
- Prioritized timing delays lower-priority robots until their smoothed paths do
  not overlap earlier robots.
