"""Entry point for visualizing a small warehouse planning scenario."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from warehouse_planning.config import load_scenario_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "warehouse_clean_demo.yaml",
        help="Path to a YAML scenario configuration.",
    )
    parser.add_argument(
        "--style",
        choices=("clean", "technical"),
        default="clean",
        help="Visualization style.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Simulation time used to draw dynamic obstacles.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path where the visualization image will be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening a Matplotlib window.",
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Only render the scenario without running kinodynamic A*.",
    )
    parser.add_argument(
        "--risk-weight",
        type=float,
        default=8.0,
        help="Soft dynamic-obstacle risk weight for the risk-aware path.",
    )
    parser.add_argument(
        "--safety-distance",
        type=float,
        default=1.2,
        help="Extra clearance distance where dynamic obstacles add soft risk.",
    )
    parser.add_argument(
        "--risk-sigma",
        type=float,
        default=None,
        help="Gaussian risk field sigma. Defaults to half the safety distance.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory where risk comparison demo figures are saved.",
    )
    parser.add_argument(
        "--simulate-receding-horizon",
        action="store_true",
        help="Run closed-loop receding-horizon simulation for one robot.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=24,
        help="Number of planner time steps in each receding horizon.",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=3,
        help="Number of planned steps executed before replanning.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=25,
        help="Maximum receding-horizon replanning cycles.",
    )
    parser.add_argument(
        "--animation",
        type=Path,
        default=PROJECT_ROOT / "results" / "receding_horizon.gif",
        help="GIF or MP4 path for receding-horizon animation export.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Animation frames per second.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the multi-robot benchmark suite and save CSV/bar plots.",
    )
    parser.add_argument(
        "--clean-demo",
        action="store_true",
        help="Run the presentation-ready clean multi-robot demo.",
    )
    parser.add_argument(
        "--demo-duration",
        type=float,
        default=None,
        help="Duration in seconds for the clean demo animation.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Also save results/multi_robot_demo.gif for the clean demo.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a YAML scenario and render the initial warehouse scene."""
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    if (
        args.no_show
        or args.save is not None
        or args.simulate_receding_horizon
        or args.benchmark
        or (args.clean_demo and args.no_show)
    ):
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    scenario = load_scenario_config(args.config)
    if args.benchmark:
        from warehouse_planning.evaluation.benchmark import run_benchmark

        run_benchmark(args.results_dir, risk_weight=args.risk_weight)
        return

    if args.clean_demo:
        from warehouse_planning.visualization.clean_demo import run_clean_multi_robot_demo

        run_clean_multi_robot_demo(
            scenario,
            results_dir=args.results_dir,
            show=not args.no_show,
            fps=args.fps,
            duration=args.demo_duration,
            save_gif=args.save_gif,
        )
        return

    if args.simulate_receding_horizon:
        from warehouse_planning.simulation.simulator import (
            RecedingHorizonConfig,
            Simulator,
        )

        simulator = Simulator(scenario)
        sim_config = RecedingHorizonConfig(
            horizon_steps=args.horizon_steps,
            execute_steps=args.execute_steps,
            max_cycles=args.max_cycles,
            risk_weight=args.risk_weight,
            safety_distance=args.safety_distance,
            risk_sigma=args.risk_sigma,
        )
        result = simulator.run_receding_horizon_single_robot(sim_config)
        simulator.export_animation(result, args.animation, fps=args.fps)
        final_frame = simulator.render_receding_horizon_frame(
            result,
            len(result.executed_path) - 1,
        )
        args.results_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(args.results_dir / "receding_horizon_final.png", final_frame)
        return

    from warehouse_planning.planning.collision import CollisionChecker
    from warehouse_planning.planning.kinodynamic_astar import KinodynamicAStarPlanner
    from warehouse_planning.visualization.plotting import WarehousePlotter

    plotter = WarehousePlotter(style=args.style)
    fig, ax = plotter.draw_scenario(scenario, time=args.time)

    if scenario.robots and not args.no_plan:
        robot = scenario.robots[0]
        collision_checker = CollisionChecker(
            warehouse=scenario.warehouse,
            dynamic_obstacles=scenario.dynamic_obstacles,
        )
        max_time_steps = int(scenario.simulation.horizon / scenario.simulation.dt)

        baseline_planner = KinodynamicAStarPlanner(
            collision_checker=collision_checker,
            dt=scenario.simulation.dt,
            max_time_steps=max_time_steps,
            risk_weight=0.0,
        )
        baseline_path = baseline_planner.plan(robot)
        plotter.draw_path(ax, baseline_path, label="risk_weight=0", color="#ff7f0e")

        planned_paths = [("risk_weight_0", baseline_path, "#ff7f0e")]
        if scenario.dynamic_obstacles and args.risk_weight > 0.0:
            risk_planner = KinodynamicAStarPlanner(
                collision_checker=collision_checker,
                dt=scenario.simulation.dt,
                max_time_steps=max_time_steps,
                risk_weight=args.risk_weight,
                safety_distance=args.safety_distance,
                risk_sigma=args.risk_sigma,
            )
            risk_path = risk_planner.plan(robot)
            plotter.draw_path(
                ax,
                risk_path,
                label=f"risk_weight={args.risk_weight:g}",
                color="#2ca02c",
            )
            planned_paths.append(("risk_weight_positive", risk_path, "#2ca02c"))

        args.results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.results_dir / "risk_comparison.png",
            dpi=160,
            bbox_inches="tight",
        )
        for name, path, color in planned_paths:
            path_fig, path_ax = plotter.draw_scenario(scenario, time=args.time)
            plotter.draw_path(path_ax, path, label=name, color=color)
            path_fig.savefig(
                args.results_dir / f"{name}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(path_fig)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160, bbox_inches="tight")

    if not args.no_show:
        plotter.show()


if __name__ == "__main__":
    main()
