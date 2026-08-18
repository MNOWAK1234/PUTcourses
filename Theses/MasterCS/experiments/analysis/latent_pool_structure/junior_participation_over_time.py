#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, read_or_build_player_metadata, require_duckdb, sql_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse junior participation over time in unique games.")
    parser.add_argument("--games", default="games.parquet")
    parser.add_argument("--games-unique", default="games_unique.parquet")
    parser.add_argument("--out", default="outputs/latent_pool_structure")
    parser.add_argument("--junior-age", type=int, default=21)
    args = parser.parse_args()

    results_dir, plots_dir = ensure_dirs(args.out)
    metadata = read_or_build_player_metadata(args.games, results_dir / "player_metadata_latest.csv")
    meta_csv = results_dir / "player_metadata_for_duckdb.csv"
    metadata[["player_key", "birth_year", "fed", "sex", "player_name"]].to_csv(meta_csv, index=False, encoding="utf-8-sig")

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE meta AS SELECT * FROM read_csv_auto('{sql_path(meta_csv)}')")
    gu = Path(args.games_unique)
    if not gu.exists():
        raise FileNotFoundError(gu)

    query = f"""
    WITH participants AS (
        SELECT
            DATE_TRUNC('year', month)::DATE AS year_date,
            EXTRACT(year FROM month)::INTEGER AS game_year,
            player_a_key AS player_key,
            player_a_fide_rating AS fide_rating,
            score_a AS score
        FROM read_parquet('{sql_path(gu)}')
        UNION ALL
        SELECT
            DATE_TRUNC('year', month)::DATE AS year_date,
            EXTRACT(year FROM month)::INTEGER AS game_year,
            player_b_key AS player_key,
            player_b_fide_rating AS fide_rating,
            1.0 - score_a AS score
        FROM read_parquet('{sql_path(gu)}')
    ),
    joined AS (
        SELECT
            participants.*,
            TRY_CAST(meta.birth_year AS INTEGER) AS birth_year,
            UPPER(TRIM(CAST(meta.fed AS VARCHAR))) AS fed,
            UPPER(TRIM(CAST(meta.sex AS VARCHAR))) AS sex,
            participants.game_year - TRY_CAST(meta.birth_year AS INTEGER) AS age_in_game_year
        FROM participants
        LEFT JOIN meta ON meta.player_key = participants.player_key
    )
    SELECT
        game_year,
        COUNT(*)::BIGINT AS player_game_entries,
        COUNT(DISTINCT player_key)::BIGINT AS active_players,
        SUM(CASE WHEN age_in_game_year < {args.junior_age} THEN 1 ELSE 0 END)::BIGINT AS junior_entries,
        COUNT(DISTINCT CASE WHEN age_in_game_year < {args.junior_age} THEN player_key ELSE NULL END)::BIGINT AS active_juniors,
        AVG(CASE WHEN age_in_game_year < {args.junior_age} THEN score ELSE NULL END) AS junior_average_score,
        AVG(CASE WHEN age_in_game_year >= {args.junior_age} THEN score ELSE NULL END) AS non_junior_average_score,
        AVG(CASE WHEN age_in_game_year < {args.junior_age} THEN fide_rating ELSE NULL END) AS junior_average_rating,
        AVG(CASE WHEN age_in_game_year >= {args.junior_age} THEN fide_rating ELSE NULL END) AS non_junior_average_rating,
        SUM(CASE WHEN birth_year IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS known_birth_entries
    FROM joined
    GROUP BY game_year
    ORDER BY game_year
    """
    yearly = con.execute(query).fetchdf()
    yearly["junior_entry_share"] = yearly["junior_entries"] / yearly["player_game_entries"]
    yearly["active_junior_share"] = yearly["active_juniors"] / yearly["active_players"]
    yearly["known_birth_entry_share"] = yearly["known_birth_entries"] / yearly["player_game_entries"]
    yearly.to_csv(results_dir / "junior_participation_over_time.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly["game_year"], yearly["junior_entry_share"], marker="o")
    ax.set_title(f"Junior U{args.junior_age} participation over time")
    ax.set_xlabel("year")
    ax.set_ylabel("share of player-game entries")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "junior_entry_share_over_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly["game_year"], yearly["junior_average_rating"], marker="o", label="juniors")
    ax.plot(yearly["game_year"], yearly["non_junior_average_rating"], marker="o", label="non-juniors")
    ax.set_title(f"Average FIDE rating by junior status, U{args.junior_age}")
    ax.set_xlabel("year")
    ax.set_ylabel("average FIDE rating")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "junior_vs_non_junior_average_rating.png", dpi=180)
    plt.close(fig)

    print("[OK] saved junior participation outputs")


if __name__ == "__main__":
    main()
