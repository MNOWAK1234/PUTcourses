#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import add_classic_improvement, ensure_dirs, evaluate_model, import_experiment_module, load_best_parameters, load_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity to the learned white advantage parameter.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default="experiments/results/best_model_parameters.json")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default="outputs/robustness_ablation")
    parser.add_argument("--values", default="0,10,20,30,36.106,45,60,70")
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)
    values = [float(x.strip()) for x in args.values.split(",") if x.strip()]

    rows = []
    rows.append(evaluate_model(exp, cache, "classic_elo", exp.params_dict_to_array(exp.CLASSIC_SEED)))
    rows.append(evaluate_model(exp, cache, "best_original", best))
    for value in values:
        candidate = best.copy()
        candidate[exp.PARAMETER_INDEX["white_advantage"]] = value
        rows.append(evaluate_model(exp, cache, f"white_advantage_{value:g}", candidate).assign(white_advantage=value))

    df = pd.concat(rows, ignore_index=True)
    df = add_classic_improvement(df)
    df.to_csv(results_dir / "white_advantage_sensitivity.csv", index=False, encoding="utf-8-sig")

    test = df[(df["scope"] == "test") & df["white_advantage"].notna()].copy().sort_values("white_advantage")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(test["white_advantage"], test["mse_all"], marker="o")
    ax.axvline(36.106, linewidth=1.0, linestyle="--")
    ax.set_title("Sensitivity to white advantage")
    ax.set_xlabel("white advantage [rating points]")
    ax.set_ylabel("test MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "white_advantage_sensitivity_mse.png", dpi=180)
    plt.close(fig)

    print("[OK] saved white advantage sensitivity outputs")


if __name__ == "__main__":
    main()
