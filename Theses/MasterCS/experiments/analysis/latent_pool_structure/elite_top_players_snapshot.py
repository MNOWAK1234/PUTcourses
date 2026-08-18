#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import attach_metadata, ensure_dirs, import_experiment_module, load_cache, pool_membership_frame, read_or_build_player_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a top-player snapshot with pool, federation and junior labels.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--games", default="games.parquet")
    parser.add_argument("--out", default="outputs/latent_pool_structure")
    parser.add_argument("--reference-year", type=int, default=2025)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--min-rating", type=float, default=2400.0)
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    membership = pool_membership_frame(cache)
    metadata = read_or_build_player_metadata(args.games, results_dir / "player_metadata_latest.csv")
    players = attach_metadata(membership, metadata, reference_year=args.reference_year)

    ranked = players.dropna(subset=["current_or_max_rating"]).copy()
    ranked = ranked.sort_values("current_or_max_rating", ascending=False)
    elite = ranked.head(args.top_n).copy()
    elite.to_csv(results_dir / "elite_top_players_snapshot.csv", index=False, encoding="utf-8-sig")

    threshold = ranked[ranked["current_or_max_rating"] >= args.min_rating].copy()
    threshold.to_csv(results_dir / "elite_players_above_rating_threshold.csv", index=False, encoding="utf-8-sig")

    pool_summary = elite.groupby("pool_id", as_index=False).agg(
        elite_players=("player_key", "count"),
        mean_rating=("current_or_max_rating", "mean"),
        max_rating=("current_or_max_rating", "max"),
        junior_u21=("is_junior_u21", "sum"),
    )
    pool_summary["elite_share"] = pool_summary["elite_players"] / len(elite) if len(elite) else 0
    pool_summary = pool_summary.sort_values("elite_players", ascending=False)
    pool_summary.to_csv(results_dir / "elite_top_players_by_pool.csv", index=False, encoding="utf-8-sig")

    fed_summary = elite.groupby("fed", as_index=False).agg(
        elite_players=("player_key", "count"),
        mean_rating=("current_or_max_rating", "mean"),
        max_rating=("current_or_max_rating", "max"),
        junior_u21=("is_junior_u21", "sum"),
    )
    fed_summary["elite_share"] = fed_summary["elite_players"] / len(elite) if len(elite) else 0
    fed_summary = fed_summary.sort_values("elite_players", ascending=False)
    fed_summary.to_csv(results_dir / "elite_top_players_by_federation.csv", index=False, encoding="utf-8-sig")

    plot = pool_summary.head(20).sort_values("elite_players")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot))))
    ax.barh(["pool " + str(x) for x in plot["pool_id"]], plot["elite_players"])
    ax.set_title(f"Top {args.top_n} players by latent pool")
    ax.set_xlabel("players")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "elite_top_players_by_pool.png", dpi=180)
    plt.close(fig)

    plot = fed_summary.head(20).sort_values("elite_players")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot))))
    ax.barh(plot["fed"], plot["elite_players"])
    ax.set_title(f"Top {args.top_n} players by federation")
    ax.set_xlabel("players")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "elite_top_players_by_federation.png", dpi=180)
    plt.close(fig)

    print("[OK] saved elite snapshot outputs")


if __name__ == "__main__":
    main()
