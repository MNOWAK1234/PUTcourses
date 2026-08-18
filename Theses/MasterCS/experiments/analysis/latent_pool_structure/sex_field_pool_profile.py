#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import attach_metadata, ensure_dirs, import_experiment_module, load_cache, pool_membership_frame, read_or_build_player_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional diagnostic: distribution of the FIDE sex field across pools.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--games", default="games.parquet")
    parser.add_argument("--out", default="outputs/latent_pool_structure")
    parser.add_argument("--reference-year", type=int, default=2025)
    parser.add_argument("--top-pools", type=int, default=20)
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    membership = pool_membership_frame(cache)
    metadata = read_or_build_player_metadata(args.games, results_dir / "player_metadata_latest.csv")
    players = attach_metadata(membership, metadata, reference_year=args.reference_year)

    counts = players.groupby(["pool_id", "sex"], as_index=False).agg(players=("player_key", "count"))
    counts["pool_total"] = counts.groupby("pool_id")["players"].transform("sum")
    counts["share"] = counts["players"] / counts["pool_total"]
    counts.to_csv(results_dir / "pool_sex_field_counts.csv", index=False, encoding="utf-8-sig")

    summary = counts.pivot_table(index="pool_id", columns="sex", values="share", fill_value=0.0).reset_index()
    pool_sizes = players.groupby("pool_id", as_index=False).agg(total_players=("player_key", "count"))
    summary = summary.merge(pool_sizes, on="pool_id", how="left").sort_values("total_players", ascending=False)
    summary.to_csv(results_dir / "pool_sex_field_summary.csv", index=False, encoding="utf-8-sig")

    # Plot most common non-unknown female-coded share if column exists.
    female_cols = [c for c in summary.columns if str(c).upper() in {"F", "W", "FEMALE"}]
    if female_cols:
        col = female_cols[0]
        plot = summary.head(args.top_pools).sort_values(col)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot))))
        ax.barh(["pool " + str(x) for x in plot["pool_id"]], plot[col])
        ax.set_title("Share of FIDE sex field value F by pool")
        ax.set_xlabel("share")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "pool_fide_sex_f_share.png", dpi=180)
        plt.close(fig)

    print("[OK] saved sex-field diagnostic outputs")


if __name__ == "__main__":
    main()
