#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import add_classic_improvement, ensure_dirs, evaluate_model, import_experiment_module, load_best_parameters, load_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity to rating update speed (K) and expected-score scale.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default="experiments/results/best_model_parameters.json")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default="outputs/robustness_ablation")
    parser.add_argument("--multipliers", default="0.5,0.75,1,1.25,1.5,2")
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)
    mults = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]

    rows = []
    rows.append(evaluate_model(exp, cache, "classic_elo", exp.params_dict_to_array(exp.CLASSIC_SEED)))
    rows.append(evaluate_model(exp, cache, "best_original", best))

    for m in mults:
        candidate = best.copy()
        candidate[exp.PARAMETER_INDEX["base_k"]] *= m
        rows.append(evaluate_model(exp, cache, f"base_k_x_{m:g}", candidate).assign(parameter="base_k", multiplier=m))

    for m in mults:
        candidate = best.copy()
        candidate[exp.PARAMETER_INDEX["scale_base"]] *= m
        rows.append(evaluate_model(exp, cache, f"scale_base_x_{m:g}", candidate).assign(parameter="scale_base", multiplier=m))

    df = pd.concat(rows, ignore_index=True)
    df = add_classic_improvement(df)
    df.to_csv(results_dir / "k_scale_sensitivity.csv", index=False, encoding="utf-8-sig")

    test = df[df["scope"] == "test"].copy()
    for parameter in ["base_k", "scale_base"]:
        subset = test[test["parameter"] == parameter].dropna(subset=["multiplier"]).sort_values("multiplier")
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(subset["multiplier"], subset["mse_all"], marker="o")
        ax.axvline(1.0, linewidth=1.0, linestyle="--")
        ax.set_title(f"Sensitivity to {parameter}")
        ax.set_xlabel("multiplier")
        ax.set_ylabel("test MSE")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{parameter}_sensitivity_mse.png", dpi=180)
        plt.close(fig)

    print("[OK] saved K/scale sensitivity outputs")


if __name__ == "__main__":
    main()
