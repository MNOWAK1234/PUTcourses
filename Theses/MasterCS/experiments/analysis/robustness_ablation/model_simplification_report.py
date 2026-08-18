#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from analysis_common import ensure_dirs, read_csv_required


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify simpler variants that are close to the best model.")
    parser.add_argument("--comparison", default="experiments/results/final_test_comparison.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    parser.add_argument("--tolerance-percent-points", type=float, default=0.10)
    args = parser.parse_args()

    results_dir, _ = ensure_dirs(args.out)
    df = read_csv_required(args.comparison)
    test = df[df["scope"] == "test"].copy()
    best = test.sort_values("mse_all").iloc[0]
    full = test[test["model"] == "provisional_full"].iloc[0]

    test["mse_gap_vs_best"] = test["mse_all"] - float(best["mse_all"])
    test["mse_gap_vs_full"] = test["mse_all"] - float(full["mse_all"])
    test["improvement_gap_vs_full_points"] = float(full["improvement_vs_classic_percent"]) - test["improvement_vs_classic_percent"]
    test["within_tolerance_of_full"] = test["improvement_gap_vs_full_points"].abs() <= args.tolerance_percent_points
    test = test.sort_values(["within_tolerance_of_full", "mse_all"], ascending=[False, True])

    test.to_csv(results_dir / "model_simplification_report.csv", index=False, encoding="utf-8-sig")

    close = test[test["within_tolerance_of_full"]].copy()
    close.to_csv(results_dir / "model_simplification_close_variants.csv", index=False, encoding="utf-8-sig")

    print("[OK] saved model simplification outputs")
    print(f"[INFO] best model: {best['model']}, MSE={best['mse_all']}")
    print(f"[INFO] close variants within {args.tolerance_percent_points} percentage points of full improvement: {len(close)}")


if __name__ == "__main__":
    main()
