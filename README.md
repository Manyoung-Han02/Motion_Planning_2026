# Multi-Robot Warehouse Motion Planning

This project simulates dynamic-obstacle-aware warehouse motion planning for a
small fleet of nonholonomic robots. The final demo uses 4 robots, 3 pedestrian
obstacles, static shelf geometry, risk-aware kinodynamic planning, inflated
space-time reservations, and concurrent local-wait coordination.

The robot paths are planned on a kinodynamic lattice, then rounded with a
Chaikin-style path smoother and dense time sampling so the rendered motion
looks closer to a Dubins-car-like vehicle than a grid polyline.
Pedestrian predictions are treated as soft Gaussian risk fields over a short
time tube, so the risk-aware planner visibly bends away from human motion.

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
  planning/coordination.py        Local-wait coordination utilities
  planning/prioritized.py         Independent, prioritized, and proposed planners
  simulation/simulator.py         Receding-horizon simulation/export utilities
  visualization/smoothing.py      Larger-radius path rounding and interpolation
  visualization/plotting.py       Matplotlib warehouse, robot, and path drawing
  visualization/clean_demo.py     Final presentation demo pipeline
  evaluation/metrics.py           Path and multi-robot metrics
  evaluation/benchmark.py         Benchmark runner and plots
scripts/
  generate_paper_figures.py       Publication-style figures and demo video
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

Generate publication-style paper figures and the presentation video:

```powershell
python scripts/generate_paper_figures.py
```

Run the final seeded algorithm evaluation:

```powershell
python scripts/run_final_evaluation.py
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

The paper-figure script writes to `results/paper_figures/`:

- `figure1_qualitative_trajectory_overview.{png,pdf}`
- `figure2_risk_aware_path_comparison.{png,pdf}`
- `figure3_baseline_metric_comparison.{png,pdf}`
- `figure3_metric_table.{png,pdf}`
- `figure3_metric_data.csv`
- `figure4_pipeline_algorithm.{png,pdf}`
- `algorithm_pseudocode.{png,pdf}`
- `algorithm_pseudocode.txt`
- `demo_video.mp4`
- `generation_notes.md`

The final evaluation script writes to `results/final_evaluation/`:

- `final_metrics.csv`
- `final_metrics_table.{png,pdf}`
- `final_comparison_plot.{png,pdf}`
- `final_trial_metrics.csv`
- `final_evaluation_notes.md`

## Motion Model Notes

- The final clean demo uses 16 heading bins, small forward primitives, and
  Chaikin smoothing to create continuous, larger-radius visual trajectories.
- Planner transition validation samples at quarter-cell spacing to avoid
  slipping through shelf corners between coarse states.
- Dynamic pedestrian risk is evaluated at multiple nearby time offsets, making
  predicted human motion affect route choice instead of only exact-time
  collisions.
- Risk-aware prioritized planning reserves a small inflated grid halo around
  earlier robot paths.
- `visualization/smoothing.py` applies open Chaikin corner rounding and dense
  time sampling. This keeps the path continuous-looking while preserving the
  planned route shape.
- `planning/coordination.py` inserts local waits only near sampled space-time
  conflicts, avoiding artificial start delays when routes are already clear.

## Final Scenario

- Robots: 4
- Pedestrians: 3
- Scenario file: `configs/warehouse_clean_demo.yaml`
- Pedestrian trajectories intentionally cross main robot aisles so human-aware
  risk has a visible effect in the risk-comparison figure and video.
