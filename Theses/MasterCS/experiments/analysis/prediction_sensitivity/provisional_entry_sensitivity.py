#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import (
    add_improvement_vs_baseline,
    ensure_extra_dirs,
    evaluate_model,
    import_experiment_module,
    load_best_parameters,
    load_cache,
    parse_float_list,
    parse_int_list,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the provisional player entry mechanism."
    )
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--best", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default="outputs/prediction_sensitivity")
    parser.add_argument("--targets", default="3,6,9,12,15")
    parser.add_argument("--blends", default="0,0.25,0.5,0.72,1.0")
    parser.add_argument("--k-multipliers", default="1,1.5,1.8,2.5,3.5")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run a full grid over target x blend x K multiplier. Default is one-parameter-at-a-time and faster.",
    )
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_extra_dirs(args.out)
    cache = load_cache(exp, args.cache)
    best = load_best_parameters(exp, args.best)

    records = []

    classic = exp.params_dict_to_array(exp.CLASSIC_SEED)
    scope, _, _ = evaluate_model(exp, cache, "classic_elo", classic, monthly=False)
    records.append(scope)

    base_scope, _, _ = evaluate_model(exp, cache, "best_original", best, monthly=False)
    records.append(base_scope)

    targets = parse_int_list(args.targets)
    blends = parse_float_list(args.blends)
    k_multipliers = parse_float_list(args.k_multipliers)

    if args.grid:
        for target in targets:
            for blend in blends:
                for k_mult in k_multipliers:
                    candidate = best.copy()
                    candidate[exp.PARAMETER_INDEX["provisional_games_target"]] = float(target)
                    candidate[exp.PARAMETER_INDEX["provisional_blend"]] = float(blend)
                    candidate[exp.PARAMETER_INDEX["provisional_k_multiplier"]] = float(k_mult)
                    name = f"entry_grid_target_{target}_blend_{blend:g}_k_{k_mult:g}"
                    print(f"[RUN] {name}")
                    scope, _, _ = evaluate_model(exp, cache, name, candidate, monthly=False)
                    scope["variant_type"] = "grid"
                    scope["provisional_games_target"] = target
                    scope["provisional_blend"] = blend
                    scope["provisional_k_multiplier"] = k_mult
                    records.append(scope)
    else:
        for target in targets:
            candidate = best.copy()
            candidate[exp.PARAMETER_INDEX["provisional_games_target"]] = float(target)
            name = f"entry_target_{target}"
            print(f"[RUN] {name}")
            scope, _, _ = evaluate_model(exp, cache, name, candidate, monthly=False)
            scope["variant_type"] = "target"
            scope["provisional_games_target"] = target
            records.append(scope)

        for blend in blends:
            candidate = best.copy()
            candidate[exp.PARAMETER_INDEX["provisional_blend"]] = float(blend)
            name = f"entry_blend_{blend:g}"
            print(f"[RUN] {name}")
            scope, _, _ = evaluate_model(exp, cache, name, candidate, monthly=False)
            scope["variant_type"] = "blend"
            scope["provisional_blend"] = blend
            records.append(scope)

        for k_mult in k_multipliers:
            candidate = best.copy()
            candidate[exp.PARAMETER_INDEX["provisional_k_multiplier"]] = float(k_mult)
            name = f"entry_k_multiplier_{k_mult:g}"
            print(f"[RUN] {name}")
            scope, _, _ = evaluate_model(exp, cache, name, candidate, monthly=False)
            scope["variant_type"] = "k_multiplier"
            scope["provisional_k_multiplier"] = k_mult
            records.append(scope)

    df = pd.concat(records, ignore_index=True)
    df = add_improvement_vs_baseline(df, "classic_elo")
    out_csv = results_dir / "provisional_entry_sensitivity.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    test = df[df["scope"] == "test"].copy()
    for variant_type, x_column in [
        ("target", "provisional_games_target"),
        ("blend", "provisional_blend"),
        ("k_multiplier", "provisional_k_multiplier"),
    ]:
        subset = test[test["variant_type"] == variant_type].dropna(subset=[x_column])
        if subset.empty:
            continue
        subset = subset.sort_values(x_column)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(subset[x_column], subset["mse_all"], marker="o")
        ax.set_title(f"Provisional entry sensitivity: {variant_type}")
        ax.set_xlabel(x_column)
        ax.set_ylabel("test MSE")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"provisional_entry_{variant_type}_mse.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(subset[x_column], subset["improvement_vs_baseline_percent"], marker="o")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"Provisional entry sensitivity: {variant_type}")
        ax.set_xlabel(x_column)
        ax.set_ylabel("improvement over classic Elo [%]")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"provisional_entry_{variant_type}_improvement.png", dpi=180)
        plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved plots in {plots_dir}")


if __name__ == "__main__":
    main()
