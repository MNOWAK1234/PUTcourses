#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import (
    add_previous_extra_to_path,
    ensure_dirs,
    import_experiment_module,
    load_best_parameters,
    load_cache,
    month_index,
    output_scope_records,
)

add_previous_extra_to_path()
from replay_kernels import run_rating_shock_replay  # noqa: E402


def monthly_records(exp, cache, model_name, output):
    rows = []
    monthly_squared_error = output[9]
    monthly_count = output[10]
    for i in range(len(monthly_count)):
        count = int(monthly_count[i])
        if count <= 0:
            continue
        rows.append({
            "model": model_name,
            "month": exp.index_to_month(cache.first_month_index + i),
            "games_all": count,
            "mse_all": float(monthly_squared_error[i] / count),
        })
    return rows


def parse_pools(text: str, summary_path: str, n: int) -> list[int]:
    if text.strip():
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    summary = pd.read_csv(summary_path)
    summary = summary[summary["pool_id"] > 0].copy()
    if "players" in summary.columns:
        summary = summary.sort_values("players", ascending=False)
    return [int(x) for x in summary["pool_id"].head(n).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run artificial rating shocks separately for selected latent pools.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default="experiments/results/best_model_parameters.json")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--summary", default="experiments/results/latent_pool_summary.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    parser.add_argument("--shock-month", default="2022-01")
    parser.add_argument("--deltas", default="-300,-150,150,300")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-pools", default="", help="Comma-separated pool ids. Empty = use largest pools from summary.")
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)

    pools = parse_pools(args.top_pools, args.summary, args.top_n)
    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    shock_idx = month_index(args.shock_month)

    print(f"[INFO] pools: {pools}")
    print(f"[INFO] deltas: {deltas}")

    print("[RUN] baseline")
    baseline = exp.evaluate_candidate(cache, best, collect_monthly=True, collect_rating_ranges=False)
    baseline_monthly = pd.DataFrame(monthly_records(exp, cache, "baseline", baseline))
    base_lookup = baseline_monthly[["month", "mse_all"]].rename(columns={"mse_all": "baseline_mse"})

    all_rows = []
    summary_rows = []

    for pool_id in pools:
        for delta in deltas:
            print(f"[RUN] pool={pool_id}, delta={delta:g}")
            output = run_rating_shock_replay(
                exp,
                cache,
                best,
                shock_month_index=shock_idx,
                shock_pool_id=int(pool_id),
                shock_rating_delta=float(delta),
            )
            monthly = pd.DataFrame(monthly_records(exp, cache, f"pool_{pool_id}_shock_{delta:g}", output))
            monthly["pool_id"] = int(pool_id)
            monthly["shock_delta"] = float(delta)
            monthly = monthly.merge(base_lookup, on="month", how="left")
            monthly["excess_mse_vs_baseline"] = monthly["mse_all"] - monthly["baseline_mse"]
            all_rows.append(monthly)

            after = monthly[monthly["month"] >= args.shock_month]
            summary_rows.append({
                "pool_id": int(pool_id),
                "shock_delta": float(delta),
                "shock_month": args.shock_month,
                "max_abs_excess_mse": float(after["excess_mse_vs_baseline"].abs().max()),
                "mean_abs_excess_mse": float(after["excess_mse_vs_baseline"].abs().mean()),
                "final_12_month_mean_abs_excess_mse": float(after.tail(12)["excess_mse_vs_baseline"].abs().mean()),
            })

    monthly_all = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    monthly_all.to_csv(results_dir / "pool_specific_shock_monthly.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results_dir / "pool_specific_shock_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 7))
    for delta, group in summary.groupby("shock_delta"):
        ordered = group.sort_values("pool_id")
        ax.plot(ordered["pool_id"], ordered["mean_abs_excess_mse"], marker="o", label=f"delta {delta:g}")
    ax.set_title("Pool-specific shock sensitivity")
    ax.set_xlabel("pool id")
    ax.set_ylabel("mean absolute excess MSE after shock")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_specific_shock_summary.png", dpi=180)
    plt.close(fig)

    print("[OK] saved pool-specific shock outputs")


if __name__ == "__main__":
    main()
