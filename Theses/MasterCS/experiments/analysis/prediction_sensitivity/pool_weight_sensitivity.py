#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import (
    add_improvement_vs_baseline,
    ensure_extra_dirs,
    import_experiment_module,
    load_best_parameters,
    load_cache,
    parse_float_list,
    evaluate_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how sensitive the final model is to the learned pool weight."
    )
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument(
        "--multipliers",
        default="0,0.1,0.25,0.5,0.75,1,1.25,1.5,2,3",
        help="Comma-separated multipliers applied to pool_weight. 0 reproduces the no-pool contribution.",
    )
    parser.add_argument("--out", default="outputs/prediction_sensitivity")
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_extra_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)

    all_scope = []

    classic = exp.params_dict_to_array(exp.CLASSIC_SEED)
    scope, _, _ = evaluate_model(exp, cache, "classic_elo", classic, monthly=False)
    all_scope.append(scope)

    for multiplier in parse_float_list(args.multipliers):
        candidate = best.copy()
        candidate[exp.PARAMETER_INDEX["pool_weight"]] *= float(multiplier)
        name = f"pool_weight_x_{multiplier:g}"
        print(f"[RUN] {name}")
        scope, _, _ = evaluate_model(exp, cache, name, candidate, monthly=False)
        scope["pool_weight_multiplier"] = multiplier
        scope["pool_weight_value"] = float(candidate[exp.PARAMETER_INDEX["pool_weight"]])
        all_scope.append(scope)

    df = pd.concat(all_scope, ignore_index=True)
    df = add_improvement_vs_baseline(df, "classic_elo")
    out_csv = results_dir / "pool_weight_sensitivity.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    test = df[(df["scope"] == "test") & (df["model"] != "classic_elo")].copy()
    test = test.sort_values("pool_weight_multiplier")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(test["pool_weight_multiplier"], test["mse_all"], marker="o")
    ax.set_title("Sensitivity to pool weight")
    ax.set_xlabel("multiplier applied to pool_weight")
    ax.set_ylabel("test MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_weight_sensitivity_mse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(test["pool_weight_multiplier"], test["improvement_vs_baseline_percent"], marker="o")
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(1.0, linewidth=1.0, linestyle="--")
    ax.set_title("Sensitivity to pool weight")
    ax.set_xlabel("multiplier applied to pool_weight")
    ax.set_ylabel("improvement over classic Elo [%]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_weight_sensitivity_improvement.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved plots in {plots_dir}")


if __name__ == "__main__":
    main()
