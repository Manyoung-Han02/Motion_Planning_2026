# Dynamic Obstacle-Aware Planning for Nonholonomic Multi-Robot Systems

**Author:** Manyoung Han  
**Class:** Motion planning and decision making for autonomous system (H.J.Kim.)

This project studies motion planning for nonholonomic multi-robot warehouse
systems operating around dynamic pedestrians. The planner generates smooth
robot trajectories while avoiding static shelves, other robots, and human-risk
regions.

## Demo Video

<video src="./results/demo_video.mp4" controls width="100%"></video>

[Open demo video](./results/demo_video.mp4)

## Risk Heatmap Figure

<p align="center">
  <img src="./results/0615_results/risk_heatmap_blue_robot_static_pedestrians.png" width="850" alt="Risk heatmap with blue robot path ablation">
</p>

The heatmap shows pedestrian-risk regions. Blue paths show how the robot
trajectory changes as the human-risk weight increases.

## Project Overview

The goal is to build a practical planner for warehouse robots that must move
efficiently while maintaining safety around both robots and pedestrians.

The final setup includes:

- 4 warehouse robots
- 3 pedestrians
- shelf-like static obstacles
- nonholonomic robot motion
- pedestrian-aware risk costs
- multi-robot conflict handling

## Algorithm / Method Summary

The planner is a risk-aware windowed multi-robot planner. Each robot is planned
with kinodynamic lattice A*, pedestrian proximity is added as a Gaussian risk
cost, and robot-robot conflicts are handled with time-expanded reservations and
bounded conflict-window replanning.

In short:

- **Nonholonomic planning:** robots follow heading-aware motion primitives.
- **Human-risk field:** predicted pedestrian positions create soft risk costs.
- **Robot coordination:** planned robot paths reserve space-time cells.
- **Windowed replanning:** near-future conflicts are repaired locally.
- **Smoothing:** paths are rounded for clean presentation figures and videos.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

Generate the main experiment results:

```powershell
python scripts/run_0615_experiments.py
```

Generate the risk heatmap figure:

```powershell
python scripts/generate_blue_robot_static_risk_heatmap.py
```

Generate the presentation demo video:

```powershell
python main.py --clean-demo --no-show
```

Run tests:

```powershell
python -m pytest
```
