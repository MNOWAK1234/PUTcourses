#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import attach_metadata, ensure_dirs, import_experiment_module, load_cache, pool_membership_frame, read_or_build_player_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain latent pools using age and junior representation.")
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

    summary = players.groupby("pool_id", as_index=False).agg(
        players=("player_key", "count"),
        known_birth_year=("birth_year", lambda x: int(x.notna().sum())),
        mean_age=("age_at_reference", "mean"),
        median_age=("age_at_reference", "median"),
        junior_u18=("is_junior_u18", "sum"),
        junior_u21=("is_junior_u21", "sum"),
        young_u25=("is_young_u25", "sum"),
        known_rating=("current_or_max_rating", lambda x: int(x.notna().sum())),
        mean_rating=("current_or_max_rating", "mean"),
        median_rating=("current_or_max_rating", "median"),
    )
    summary["known_birth_year_share"] = summary["known_birth_year"] / summary["players"]
    summary["junior_u18_share_all"] = summary["junior_u18"] / summary["players"]
    summary["junior_u21_share_all"] = summary["junior_u21"] / summary["players"]
    summary["young_u25_share_all"] = summary["young_u25"] / summary["players"]
    summary["junior_u18_share_known_birth"] = summary["junior_u18"] / summary["known_birth_year"].replace(0, pd.NA)
    summary["junior_u21_share_known_birth"] = summary["junior_u21"] / summary["known_birth_year"].replace(0, pd.NA)
    summary["young_u25_share_known_birth"] = summary["young_u25"] / summary["known_birth_year"].replace(0, pd.NA)
    summary = summary.sort_values("players", ascending=False)
    summary.to_csv(results_dir / "pool_age_junior_summary.csv", index=False, encoding="utf-8-sig")

    players[["player_key", "pool_id", "player_name", "fed", "sex", "birth_year", "age_at_reference", "is_junior_u18", "is_junior_u21", "is_young_u25", "current_or_max_rating"]].to_csv(
        results_dir / "player_age_junior_metadata.csv", index=False, encoding="utf-8-sig"
    )

    plot = summary.head(args.top_pools).sort_values("junior_u21_share_known_birth")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot))))
    ax.barh(["pool " + str(x) for x in plot["pool_id"]], plot["junior_u21_share_known_birth"])
    ax.set_title(f"Junior U21 share by pool, reference year {args.reference_year}")
    ax.set_xlabel("U21 share among players with known birth year")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_junior_u21_share.png", dpi=180)
    plt.close(fig)

    plot = summary.head(args.top_pools).sort_values("median_age")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot))))
    ax.barh(["pool " + str(x) for x in plot["pool_id"]], plot["median_age"])
    ax.set_title(f"Median age by pool, reference year {args.reference_year}")
    ax.set_xlabel("median age")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_median_age.png", dpi=180)
    plt.close(fig)

    print("[OK] saved age/junior latent-pool structure outputs")


if __name__ == "__main__":
    main()
