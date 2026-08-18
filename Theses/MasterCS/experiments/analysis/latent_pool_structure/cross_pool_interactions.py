#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, import_experiment_module, load_cache, pool_membership_frame, require_duckdb, sql_path


def mirror_matrix(frame: pd.DataFrame, pools: list[int], value_column: str) -> pd.DataFrame:
    pivot = frame.pivot_table(index="pool_low", columns="pool_high", values=value_column, fill_value=0.0)
    for i in pools:
        for j in pools:
            a, b = min(i, j), max(i, j)
            match = frame[(frame["pool_low"] == a) & (frame["pool_high"] == b)][value_column]
            if not match.empty:
                pivot.loc[i, j] = float(match.iloc[0])
    return pivot.reindex(index=pools, columns=pools, fill_value=0.0)


def save_heatmap(pivot: pd.DataFrame, path: Path, title: str, colorbar_label: str) -> None:
    labels = [str(x) for x in pivot.index.tolist()]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("pool")
    ax.set_ylabel("pool")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse cross-pool games and relative interaction volume.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--games-unique", default="games_unique.parquet")
    parser.add_argument("--out", default="outputs/latent_pool_structure")
    parser.add_argument("--from-month", default="2015-01")
    parser.add_argument("--top-pools", type=int, default=15)
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    membership = pool_membership_frame(cache)
    map_csv = results_dir / "player_pool_map.csv"
    membership.to_csv(map_csv, index=False, encoding="utf-8-sig")

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE player_pool AS SELECT * FROM read_csv_auto('{sql_path(map_csv)}')")
    gu = Path(args.games_unique)
    if not gu.exists():
        raise FileNotFoundError(gu)

    query = f"""
    WITH games AS (
        SELECT
            month,
            player_a_key,
            player_b_key,
            score_a,
            player_a_fide_rating,
            player_b_fide_rating
        FROM read_parquet('{sql_path(gu)}')
        WHERE month >= DATE '{args.from_month}-01'
    ),
    joined AS (
        SELECT
            COALESCE(pa.pool_id, 0)::INTEGER AS pool_a,
            COALESCE(pb.pool_id, 0)::INTEGER AS pool_b,
            score_a,
            player_a_fide_rating,
            player_b_fide_rating
        FROM games
        LEFT JOIN player_pool pa ON pa.player_key = games.player_a_key
        LEFT JOIN player_pool pb ON pb.player_key = games.player_b_key
    ),
    oriented AS (
        SELECT
            LEAST(pool_a, pool_b) AS pool_low,
            GREATEST(pool_a, pool_b) AS pool_high,
            CASE WHEN pool_a <= pool_b THEN score_a ELSE 1.0 - score_a END AS score_low,
            CASE WHEN pool_a <= pool_b THEN player_a_fide_rating ELSE player_b_fide_rating END AS rating_low,
            CASE WHEN pool_a <= pool_b THEN player_b_fide_rating ELSE player_a_fide_rating END AS rating_high,
            CASE WHEN pool_a = pool_b THEN 0 ELSE 1 END AS cross_pool
        FROM joined
    )
    SELECT
        pool_low,
        pool_high,
        COUNT(*)::BIGINT AS games,
        AVG(score_low) AS average_score_for_low_pool,
        AVG(rating_low) AS average_rating_low_pool,
        AVG(rating_high) AS average_rating_high_pool,
        AVG(rating_low - rating_high) AS average_rating_difference_low_minus_high,
        SUM(cross_pool)::BIGINT AS cross_pool_games
    FROM oriented
    GROUP BY pool_low, pool_high
    ORDER BY games DESC
    """
    pair = con.execute(query).fetchdf()
    pair.to_csv(results_dir / "cross_pool_interactions.csv", index=False, encoding="utf-8-sig")

    pool_games = []
    for _, row in pair.iterrows():
        pool_games.append({"pool_id": int(row["pool_low"]), "games": int(row["games"])})
        if int(row["pool_high"]) != int(row["pool_low"]):
            pool_games.append({"pool_id": int(row["pool_high"]), "games": int(row["games"])})
    pool_games_df = (
        pd.DataFrame(pool_games)
        .groupby("pool_id", as_index=False)["games"]
        .sum()
        .sort_values("games", ascending=False)
    )
    pool_games_df.to_csv(results_dir / "pool_game_volume.csv", index=False, encoding="utf-8-sig")

    totals = dict(zip(pool_games_df["pool_id"].astype(int), pool_games_df["games"].astype(float)))
    pair["relative_volume_per_1000"] = pair.apply(
        lambda r: 1000.0 * float(r["games"]) / ((totals.get(int(r["pool_low"]), 0.0) * totals.get(int(r["pool_high"]), 0.0)) ** 0.5)
        if totals.get(int(r["pool_low"]), 0.0) > 0.0 and totals.get(int(r["pool_high"]), 0.0) > 0.0
        else 0.0,
        axis=1,
    )

    cross_only = pair[(pair["pool_low"] != pair["pool_high"]) & (pair["pool_low"] != 0) & (pair["pool_high"] != 0)].copy()
    cross_only = cross_only.sort_values(["games", "relative_volume_per_1000"], ascending=[False, False])
    cross_only.to_csv(results_dir / "cross_pool_interactions_cross_only.csv", index=False, encoding="utf-8-sig")

    table = cross_only.head(30).copy()
    table.to_csv(results_dir / "cross_pool_interactions_for_table.csv", index=False, encoding="utf-8-sig")

    top = pool_games_df[pool_games_df["pool_id"] != 0].head(args.top_pools)["pool_id"].astype(int).tolist()
    mat = pair[pair["pool_low"].isin(top) & pair["pool_high"].isin(top)].copy()
    rel_pivot = mirror_matrix(mat, top, "relative_volume_per_1000")
    abs_pivot = mirror_matrix(mat, top, "games")

    save_heatmap(
        rel_pivot,
        plots_dir / "cross_pool_game_volume_heatmap.png",
        f"Relative cross-pool interaction volume since {args.from_month}",
        "relative volume per 1000",
    )
    save_heatmap(
        rel_pivot,
        plots_dir / "cross_pool_relative_volume_heatmap.png",
        f"Relative cross-pool interaction volume since {args.from_month}",
        "relative volume per 1000",
    )
    save_heatmap(
        abs_pivot,
        plots_dir / "cross_pool_absolute_volume_heatmap_no_pool0.png",
        f"Absolute cross-pool game volume since {args.from_month}",
        "games",
    )

    print("[OK] saved cross-pool interaction outputs")


if __name__ == "__main__":
    main()
