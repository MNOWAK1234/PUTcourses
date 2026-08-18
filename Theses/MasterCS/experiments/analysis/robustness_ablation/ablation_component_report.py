#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, read_csv_required


COMPONENTS = {
    "provisional_without_pool": "pool component",
    "provisional_without_dynamic_exposure": "dynamic exposure",
    "provisional_without_performance_entry": "performance-based entry",
    "provisional_fixed_1500_entry": "non-1500 entry correction",
    "provisional_without_pair_interaction": "pool-pair interaction",
    "provisional_without_uncertainty_prediction": "uncertainty-aware prediction",
    "provisional_without_white_advantage": "white advantage",
    "provisional_without_event_normalization": "event normalization",
    "provisional_without_tail_scale": "tail-scale correction",
    "provisional_without_form": "short-term form",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create component contribution report from final_test_comparison.csv.")
    parser.add_argument("--comparison", default="experiments/results/final_test_comparison.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    args = parser.parse_args()

    results_dir, plots_dir = ensure_dirs(args.out)
    df = read_csv_required(args.comparison)

    test = df[df["scope"] == "test"].copy()
    full = test[test["model"] == "provisional_full"].iloc[0]
    classic = test[test["model"] == "classic_elo"].iloc[0]

    rows = []
    for model, component in COMPONENTS.items():
        subset = test[test["model"] == model]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append({
            "ablated_model": model,
            "component_removed_or_changed": component,
            "full_model_mse": float(full["mse_all"]),
            "ablated_mse": float(row["mse_all"]),
            "classic_mse": float(classic["mse_all"]),
            "mse_loss_vs_full": float(row["mse_all"] - full["mse_all"]),
            "relative_loss_vs_full_percent": float((row["mse_all"] - full["mse_all"]) / full["mse_all"] * 100.0),
            "improvement_vs_classic_percent": float(row["improvement_vs_classic_percent"]),
            "full_improvement_vs_classic_percent": float(full["improvement_vs_classic_percent"]),
            "lost_improvement_points": float(full["improvement_vs_classic_percent"] - row["improvement_vs_classic_percent"]),
        })

    out = pd.DataFrame(rows).sort_values("mse_loss_vs_full", ascending=False)
    out_csv = results_dir / "ablation_component_contribution.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plot = out.sort_values("lost_improvement_points", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot["component_removed_or_changed"], plot["lost_improvement_points"])
    ax.set_title("Loss of improvement after removing model components")
    ax.set_xlabel("lost improvement over classic Elo [percentage points]")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "ablation_component_contribution.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved plot")


if __name__ == "__main__":
    main()
