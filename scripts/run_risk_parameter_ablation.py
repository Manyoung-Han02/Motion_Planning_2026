"""Run only the fine risk-parameter ablation without cleaning old results."""

from __future__ import annotations

from pathlib import Path

from run_0615_experiments import (
    OUTPUT_DIR,
    make_human_crossing_ablation_scenario,
    run_risk_ablation,
    save_risk_ablation_plot,
    save_risk_heatmap_overlay,
    set_style,
)


def main() -> None:
    """Generate fine risk-weight and risk-sigma ablation data."""
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario = make_human_crossing_ablation_scenario()
    frame = run_risk_ablation(scenario)

    frame.to_csv(OUTPUT_DIR / "risk_ablation_metrics.csv", index=False)
    frame.to_csv(OUTPUT_DIR / "risk_parameter_ablation_fine.csv", index=False)
    frame[frame["ablation"] == "risk_weight"].to_csv(
        OUTPUT_DIR / "risk_weight_0_4_ablation.csv",
        index=False,
    )
    frame[frame["ablation"] == "risk_sigma"].to_csv(
        OUTPUT_DIR / "risk_sigma_0_1_ablation.csv",
        index=False,
    )

    save_risk_ablation_plot(frame)
    save_risk_heatmap_overlay(scenario)

    print(f"Saved fine risk-parameter ablation to {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
