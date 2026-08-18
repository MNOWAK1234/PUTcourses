#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, read_csv_required


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse monthly rating scale drift using monthly_rating_distributions.csv.")
    parser.add_argument("--monthly-ratings", default="experiments/results/monthly_rating_distributions.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    parser.add_argument("--models", default="classic_elo,provisional_full,provisional_without_pool,provisional_without_form")
    args = parser.parse_args()

    results_dir, plots_dir = ensure_dirs(args.out)
    df = read_csv_required(args.monthly_ratings)
    df["date"] = pd.to_datetime(df["month"] + "-01")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    df = df[df["model"].isin(models)].copy()

    for col in ["effective_p99", "effective_p01", "effective_p90", "effective_p10", "effective_p50"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column {col} in {args.monthly_ratings}")

    df["p99_p01_width"] = df["effective_p99"] - df["effective_p01"]
    df["p90_p10_width"] = df["effective_p90"] - df["effective_p10"]

    summary = df.groupby("model", as_index=False).agg(
        months=("month", "count"),
        median_rating_mean=("effective_p50", "mean"),
        median_rating_min=("effective_p50", "min"),
        median_rating_max=("effective_p50", "max"),
        p99_p01_width_mean=("p99_p01_width", "mean"),
        p99_p01_width_min=("p99_p01_width", "min"),
        p99_p01_width_max=("p99_p01_width", "max"),
        p90_p10_width_mean=("p90_p10_width", "mean"),
        p90_p10_width_min=("p90_p10_width", "min"),
        p90_p10_width_max=("p90_p10_width", "max"),
    )
    summary.to_csv(results_dir / "rating_scale_drift_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(results_dir / "rating_scale_drift_monthly.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(13, 6))
    for model, group in df.groupby("model"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["effective_p50"], linewidth=1.4, label=model)
    ax.set_title("Monthly median effective rating")
    ax.set_xlabel("month")
    ax.set_ylabel("median effective rating")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "monthly_median_rating.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for model, group in df.groupby("model"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["p99_p01_width"], linewidth=1.4, label=model)
    ax.set_title("Monthly effective rating range (p99 - p01)")
    ax.set_xlabel("month")
    ax.set_ylabel("p99 - p01 rating range")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "monthly_rating_width_p99_p01.png", dpi=180)
    plt.close(fig)

    print("[OK] saved rating scale drift outputs")


if __name__ == "__main__":
    main()
