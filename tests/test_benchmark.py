from pathlib import Path

import pandas as pd

from warehouse_planning.evaluation.benchmark import run_benchmark


def test_run_benchmark_saves_csv_and_bar_plots(tmp_path: Path) -> None:
    frame = run_benchmark(tmp_path, risk_weight=4.0)

    csv_path = tmp_path / "benchmark_results.csv"
    assert csv_path.exists()
    assert len(frame) == 16
    assert set(frame["scenario"]) == {
        "narrow_aisle_crossing",
        "human_crossing_path",
        "bottleneck_warehouse",
        "random_start_goal_tasks",
    }
    assert set(frame["method"]) == {
        "Independent A*",
        "Prioritized Planning",
        "CBS-style Planner",
        "Proposed risk-aware CBS-style Planner",
    }
    assert (tmp_path / "success_rate.png").exists()
    assert (tmp_path / "computation_time.png").exists()

    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == list(frame.columns)
