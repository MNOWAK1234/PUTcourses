#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import ensure_extra_dirs, import_experiment_module, load_best_parameters, load_cache
from replay_kernels import run_calibration_replay


def calibration_frame(output, bins: int) -> pd.DataFrame:
    sum_pred = output[-3]
    sum_actual = output[-2]
    count = output[-1]
    rows = []
    for i in range(bins):
        c = int(count[i])
        lower = i / bins
        upper = (i + 1) / bins
        rows.append({
            "bucket": i,
            "lower": lower,
            "upper": upper,
            "center": 0.5 * (lower + upper),
            "games": c,
            "mean_predicted_score": float(sum_pred[i] / c) if c else float("nan"),
            "observed_score": float(sum_actual[i] / c) if c else float("nan"),
            "calibration_error": float(sum_actual[i] / c - sum_pred[i] / c) if c else float("nan"),
            "absolute_calibration_error": abs(float(sum_actual[i] / c - sum_pred[i] / c)) if c else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute calibration bins for final-model expected scores on the test period."
    )
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default="outputs/prediction_sensitivity")
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_extra_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)

    print("[RUN] calibration replay")
    output = run_calibration_replay(exp, cache, best, bins=args.bins)
    df = calibration_frame(output, args.bins)

    out_csv = results_dir / "calibration_bins.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    weighted_abs = np.average(
        df.dropna(subset=["absolute_calibration_error"])["absolute_calibration_error"],
        weights=df.dropna(subset=["absolute_calibration_error"])["games"],
    )
    summary = pd.DataFrame([{
        "bins": args.bins,
        "games": int(df["games"].sum()),
        "weighted_mean_absolute_calibration_error": float(weighted_abs),
    }])
    summary.to_csv(results_dir / "calibration_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 7))
    visible = df[df["games"] > 0]
    ax.plot([0, 1], [0, 1], linewidth=1.0, label="perfect calibration")
    ax.scatter(visible["mean_predicted_score"], visible["observed_score"], s=np.maximum(20, np.sqrt(visible["games"]) / 10.0), label="model buckets")
    ax.set_title("Calibration of predicted expected scores")
    ax.set_xlabel("mean predicted expected score")
    ax.set_ylabel("observed average score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "calibration_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(visible["center"], visible["games"], width=1.0 / args.bins * 0.9)
    ax.set_title("Distribution of predictions across probability buckets")
    ax.set_xlabel("predicted expected score bucket")
    ax.set_ylabel("games")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "calibration_bucket_counts.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved calibration_summary.csv")
    print(f"[OK] saved plots in {plots_dir}")


if __name__ == "__main__":
    main()
