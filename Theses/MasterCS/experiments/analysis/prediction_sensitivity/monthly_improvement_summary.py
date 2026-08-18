#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_extra_dirs


def first_sustained_positive(group: pd.DataFrame, window: int) -> str | None:
    group = group.sort_values("month").copy()
    values = group["improvement_vs_classic_percent"].to_numpy()
    months = group["month"].to_list()
    for i in range(0, len(values) - window + 1):
        if (values[i:i + window] > 0).all():
            return months[i]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise monthly improvement stability using final_monthly_metrics.csv."
    )
    parser.add_argument("--monthly", default="experiments/results/final_monthly_metrics.csv")
    parser.add_argument("--out", default="outputs/prediction_sensitivity")
    parser.add_argument("--window", type=int, default=6, help="Rolling window in months.")
    args = parser.parse_args()

    monthly_path = Path(args.monthly)
    if not monthly_path.exists():
        raise FileNotFoundError(monthly_path)

    results_dir, plots_dir = ensure_extra_dirs(args.out)

    df = pd.read_csv(monthly_path)
    df["date"] = pd.to_datetime(df["month"] + "-01")
    baseline = df[df["model"] == "classic_elo"][["month", "mse_all"]].rename(columns={"mse_all": "classic_mse"})
    df = df.merge(baseline, on="month", how="left")
    df["improvement_vs_classic_percent"] = (df["classic_mse"] - df["mse_all"]) / df["classic_mse"] * 100.0

    rows = []
    for model, group in df.groupby("model"):
        if model == "classic_elo":
            continue
        group = group.sort_values("date")
        rolling = group["improvement_vs_classic_percent"].rolling(args.window, min_periods=max(2, args.window // 2)).mean()
        rows.append({
            "model": model,
            "months": int(len(group)),
            "mean_improvement_percent": float(group["improvement_vs_classic_percent"].mean()),
            "median_improvement_percent": float(group["improvement_vs_classic_percent"].median()),
            "worst_month_improvement_percent": float(group["improvement_vs_classic_percent"].min()),
            "best_month_improvement_percent": float(group["improvement_vs_classic_percent"].max()),
            "positive_month_share": float((group["improvement_vs_classic_percent"] > 0).mean()),
            "rolling_window_months": args.window,
            "mean_rolling_improvement_percent": float(rolling.mean()),
            "first_sustained_positive_month": first_sustained_positive(group, args.window),
        })

    summary = pd.DataFrame(rows).sort_values("mean_improvement_percent", ascending=False)
    out_csv = results_dir / "monthly_improvement_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    selected_models = [
        "provisional_full",
        "provisional_without_form",
        "provisional_without_pool",
        "provisional_without_dynamic_exposure",
        "classic_elo",
    ]
    plot_df = df[df["model"].isin(selected_models)].copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    for model, group in plot_df.groupby("model"):
        if model == "classic_elo":
            continue
        group = group.sort_values("date")
        rolling = group["improvement_vs_classic_percent"].rolling(args.window, min_periods=max(2, args.window // 2)).mean()
        ax.plot(group["date"], rolling, linewidth=1.4, label=model)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(f"Rolling {args.window}-month improvement over classical Elo")
    ax.set_xlabel("month")
    ax.set_ylabel("improvement [%]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "rolling_monthly_improvement.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved plot in {plots_dir}")


if __name__ == "__main__":
    main()
