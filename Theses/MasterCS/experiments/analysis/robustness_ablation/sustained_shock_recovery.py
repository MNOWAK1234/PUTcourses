#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, read_csv_required


def sustained_recovery_month(group: pd.DataFrame, threshold: float, window: int, shock_month: str) -> str | None:
    group = group[group["month"] >= shock_month].sort_values("month").copy()
    values = group["excess_mse_vs_baseline"].abs().to_numpy()
    months = group["month"].to_list()
    for i in range(0, len(values) - window + 1):
        if (values[i:i + window] <= threshold).all():
            return months[i]
    return None


def half_life_month(group: pd.DataFrame, shock_month: str) -> str | None:
    group = group[group["month"] >= shock_month].sort_values("month").copy()
    if group.empty:
        return None
    values = group["excess_mse_vs_baseline"].abs().to_numpy()
    months = group["month"].to_list()
    start = values[0]
    if start <= 0:
        return months[0]
    target = start / 2.0
    for month, value in zip(months, values):
        if value <= target:
            return month
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sustained recovery metrics from shock monthly results.")
    parser.add_argument("--monthly", default="outputs/prediction_sensitivity/results/rating_shock_recovery_monthly.csv")
    parser.add_argument("--pool-monthly", default="outputs/robustness_ablation/results/pool_specific_shock_monthly.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    parser.add_argument("--shock-month", default="2022-01")
    parser.add_argument("--threshold", type=float, default=0.0005)
    parser.add_argument("--window", type=int, default=6)
    args = parser.parse_args()

    results_dir, plots_dir = ensure_dirs(args.out)

    frames = []
    main_path = Path(args.monthly)
    if main_path.exists():
        df = read_csv_required(main_path)
        df["experiment"] = "global_shock"
        df["pool_id"] = df.get("shock_pool_id", -1)
        frames.append(df)

    pool_path = Path(args.pool_monthly)
    if pool_path.exists():
        df = read_csv_required(pool_path)
        df["experiment"] = "pool_specific_shock"
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No shock monthly files found.")

    data = pd.concat(frames, ignore_index=True)
    rows = []
    keys = ["experiment", "pool_id", "shock_delta"]
    for key, group in data.groupby(keys, dropna=False):
        experiment, pool_id, delta = key
        after = group[group["month"] >= args.shock_month].sort_values("month")
        rows.append({
            "experiment": experiment,
            "pool_id": pool_id,
            "shock_delta": delta,
            "shock_month": args.shock_month,
            "threshold": args.threshold,
            "sustained_window_months": args.window,
            "sustained_recovery_month": sustained_recovery_month(group, args.threshold, args.window, args.shock_month),
            "half_life_month": half_life_month(group, args.shock_month),
            "start_abs_excess_mse": float(after["excess_mse_vs_baseline"].abs().iloc[0]) if not after.empty else float("nan"),
            "max_abs_excess_mse": float(after["excess_mse_vs_baseline"].abs().max()) if not after.empty else float("nan"),
            "mean_abs_excess_mse": float(after["excess_mse_vs_baseline"].abs().mean()) if not after.empty else float("nan"),
            "last_12_month_mean_abs_excess_mse": float(after.tail(12)["excess_mse_vs_baseline"].abs().mean()) if not after.empty else float("nan"),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "sustained_recovery_summary.csv", index=False, encoding="utf-8-sig")

    plot = summary[summary["experiment"] == "global_shock"].copy()
    if not plot.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(plot["shock_delta"].astype(str), plot["last_12_month_mean_abs_excess_mse"])
        ax.set_title("Remaining shock effect in the final 12 months")
        ax.set_xlabel("global shock delta")
        ax.set_ylabel("mean absolute excess MSE")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "global_shock_remaining_effect.png", dpi=180)
        plt.close(fig)

    print("[OK] saved sustained recovery outputs")


if __name__ == "__main__":
    main()
