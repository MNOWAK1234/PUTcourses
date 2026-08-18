#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import ensure_extra_dirs, expected_points


def load_params(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in payload.get("parameters", payload).items()}


def model_scale(params: dict[str, float], diff: np.ndarray, average_level: float, uncertainty: float) -> np.ndarray:
    initial_rating = 1500.0
    scale = (
        params["scale_base"]
        + params["scale_level_slope"] * ((average_level - initial_rating) / 100.0)
        + params["scale_abs_diff_slope"] * np.minimum(np.abs(diff) / 400.0, 3.0)
    )
    scale = np.clip(scale, 180.0, 1000.0)
    scale *= 1.0 + params["prediction_uncertainty_scale_weight"] * uncertainty
    return np.clip(scale, 180.0, 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create expected-score probability curves for the final model and classical Elo."
    )
    parser.add_argument("--best", default="experiments/results/best_model_parameters.json")
    parser.add_argument("--out", default="outputs/prediction_sensitivity")
    parser.add_argument("--average-levels", default="1200,1500,2000,2500")
    parser.add_argument("--uncertainties", default="0,0.5,1.0")
    args = parser.parse_args()

    results_dir, plots_dir = ensure_extra_dirs(args.out)
    params = load_params(args.best)

    diffs = np.arange(-800, 801, 10, dtype=float)
    rows = []

    classic = expected_points(diffs, 400.0)
    for diff, p in zip(diffs, classic):
        rows.append({
            "curve": "classic_elo_scale_400",
            "rating_difference": diff,
            "average_level": np.nan,
            "uncertainty": np.nan,
            "expected_score": float(p),
        })

    for level in [float(x) for x in args.average_levels.split(",") if x.strip()]:
        for uncertainty in [float(x) for x in args.uncertainties.split(",") if x.strip()]:
            scale = model_scale(params, diffs, level, uncertainty)
            probabilities = expected_points(diffs, scale)
            curve = f"final_level_{level:g}_unc_{uncertainty:g}"
            for diff, p, s in zip(diffs, probabilities, scale):
                rows.append({
                    "curve": curve,
                    "rating_difference": diff,
                    "average_level": level,
                    "uncertainty": uncertainty,
                    "expected_score": float(p),
                    "scale": float(s),
                })

    df = pd.DataFrame(rows)
    out_csv = results_dir / "expected_score_curves.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(diffs, classic, linewidth=2.0, label="classic Elo")
    for level in [1500.0, 2000.0, 2500.0]:
        subset = df[(df["curve"] == f"final_level_{level:g}_unc_0")]
        ax.plot(subset["rating_difference"], subset["expected_score"], linewidth=1.5, label=f"final, level={level:g}, uncertainty=0")
    ax.axhline(0.5, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Expected score as a function of rating difference")
    ax.set_xlabel("rating difference")
    ax.set_ylabel("expected score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "expected_score_curves_by_level.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(diffs, classic, linewidth=2.0, label="classic Elo")
    for uncertainty in [0.0, 0.5, 1.0]:
        subset = df[(df["curve"] == f"final_level_1500_unc_{uncertainty:g}")]
        ax.plot(subset["rating_difference"], subset["expected_score"], linewidth=1.5, label=f"final, uncertainty={uncertainty:g}")
    ax.axhline(0.5, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Effect of uncertainty on expected score")
    ax.set_xlabel("rating difference")
    ax.set_ylabel("expected score")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "expected_score_curves_by_uncertainty.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved plots in {plots_dir}")


if __name__ == "__main__":
    main()
