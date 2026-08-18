#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import (
    ensure_extra_dirs,
    import_experiment_module,
    load_best_parameters,
    load_cache,
    month_to_index,
    output_monthly_records,
    parse_float_list,
)
from replay_kernels import run_rating_shock_replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Artificial rating shock experiment: perturb internal ratings and measure recovery."
    )
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default="outputs/prediction_sensitivity")
    parser.add_argument("--shock-month", default="2022-01", help="Month when the artificial shock is applied.")
    parser.add_argument(
        "--pool-id",
        type=int,
        default=-1,
        help="-1 = all already-seen players; otherwise affect only players from this static latent pool.",
    )
    parser.add_argument("--deltas", default="-300,-150,150,300")
    parser.add_argument(
        "--recovery-threshold",
        type=float,
        default=0.0005,
        help="Absolute excess MSE threshold used to define recovery.",
    )
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_extra_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)

    shock_month_index = month_to_index(args.shock_month)
    all_monthly = []

    # Baseline without shock.
    print("[RUN] baseline")
    baseline_output = exp.evaluate_candidate(cache, best, collect_monthly=True, collect_rating_ranges=False)
    baseline_monthly = pd.DataFrame(output_monthly_records(exp, cache, "baseline_no_shock", baseline_output))
    baseline_monthly["shock_delta"] = 0.0
    baseline_monthly["shock_pool_id"] = args.pool_id
    all_monthly.append(baseline_monthly)

    for delta in parse_float_list(args.deltas):
        print(f"[RUN] shock_delta={delta:g}, pool_id={args.pool_id}, shock_month={args.shock_month}")
        output = run_rating_shock_replay(
            exp,
            cache,
            best,
            shock_month_index=shock_month_index,
            shock_pool_id=args.pool_id,
            shock_rating_delta=float(delta),
        )
        monthly = pd.DataFrame(output_monthly_records(exp, cache, f"shock_{delta:g}", output))
        monthly["shock_delta"] = float(delta)
        monthly["shock_pool_id"] = args.pool_id
        all_monthly.append(monthly)

    df = pd.concat(all_monthly, ignore_index=True)
    baseline = baseline_monthly[["month", "mse_all"]].rename(columns={"mse_all": "baseline_mse"})
    df = df.merge(baseline, on="month", how="left")
    df["excess_mse_vs_baseline"] = df["mse_all"] - df["baseline_mse"]
    df["month_date"] = pd.to_datetime(df["month"] + "-01")

    out_csv = results_dir / "rating_shock_recovery_monthly.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Recovery summary: first month after the shock when excess MSE becomes small.
    summary_rows = []
    for delta, group in df[df["shock_delta"] != 0.0].groupby("shock_delta"):
        after = group[group["month"] >= args.shock_month].sort_values("month")
        recovered = after[after["excess_mse_vs_baseline"].abs() <= args.recovery_threshold]
        recovery_month = recovered["month"].iloc[0] if not recovered.empty else None
        summary_rows.append({
            "shock_delta": delta,
            "shock_pool_id": args.pool_id,
            "shock_month": args.shock_month,
            "max_abs_excess_mse": float(after["excess_mse_vs_baseline"].abs().max()) if not after.empty else float("nan"),
            "mean_abs_excess_mse_after_shock": float(after["excess_mse_vs_baseline"].abs().mean()) if not after.empty else float("nan"),
            "recovery_threshold": args.recovery_threshold,
            "recovery_month": recovery_month,
        })
    summary = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "rating_shock_recovery_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(13, 6))
    for delta, group in df.groupby("shock_delta"):
        if float(delta) == 0.0:
            continue
        group = group.sort_values("month_date")
        ax.plot(group["month_date"], group["excess_mse_vs_baseline"], linewidth=1.3, label=f"shock {delta:g}")
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(pd.to_datetime(args.shock_month + "-01"), linewidth=1.0, linestyle="--")
    ax.set_title("Artificial rating shock: excess monthly MSE over baseline")
    ax.set_xlabel("month")
    ax.set_ylabel("excess MSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "rating_shock_recovery_excess_mse.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved {summary_csv}")
    print(f"[OK] saved plot in {plots_dir}")


if __name__ == "__main__":
    main()
