from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def import_experiment_module(path: str | Path = "run_experiments.py"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("run_experiments_imported_pool_structure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs(out: str | Path = "outputs/latent_pool_structure") -> tuple[Path, Path]:
    out = Path(out)
    results = out / "results"
    plots = out / "plots"
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return results, plots


def find_final_cache(path: str | Path | None = None) -> Path:
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path("experiments/cache").glob("final_*to_*.npz"))
    full = [p for p in candidates if "fullhistory" in p.name]
    usable = full if full else candidates
    if not usable:
        raise FileNotFoundError("No final replay cache found in experiments/cache")
    usable.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    return usable[-1]


def load_cache(exp, path: str | Path | None = None):
    cache_path = find_final_cache(path)
    cache = exp.load_cache(cache_path)
    print(f"[CACHE] using {cache_path}")
    print(f"[CACHE] games={cache.games:,}, players={cache.players:,}, pools={cache.pools:,}")
    return cache


def pool_membership_frame(cache) -> pd.DataFrame:
    return pd.DataFrame({
        "player_key": cache.player_keys,
        "pool_id": cache.static_pool.astype(int),
    })


def sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def normalized_text_sql(column: str) -> str:
    return (
        "LOWER(REGEXP_REPLACE("
        f"STRIP_ACCENTS(TRIM(COALESCE(CAST({column} AS VARCHAR), ''))), "
        "'[^a-zA-Z0-9]+', ' ', 'g'))"
    )


def require_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("Missing dependency duckdb. Install with: pip install duckdb pyarrow") from exc
    return duckdb


def read_or_build_player_metadata(
    games_parquet: str | Path = "games.parquet",
    cache_csv: str | Path = "outputs/latent_pool_structure/results/player_metadata_latest.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Build latest player metadata from raw directed FIDE rows.

    The metadata is approximate: it is based on rows where the person appears as
    the scraped `player`, not merely as the opponent. This is usually enough when
    the scraping covered the player pool broadly.
    """
    cache_csv = Path(cache_csv)
    if cache_csv.exists() and not force:
        return pd.read_csv(cache_csv, dtype={"player_key": str})

    games_parquet = Path(games_parquet)
    if not games_parquet.exists():
        raise FileNotFoundError(games_parquet)

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    p = sql_path(games_parquet)
    key_expr = (
        "CASE WHEN player_id IS NOT NULL "
        "THEN 'id:' || CAST(CAST(player_id AS BIGINT) AS VARCHAR) "
        f"ELSE 'name:' || {normalized_text_sql('player_name')} END"
    )
    # Use the last available row per player_key. The scraper stores standard_rating_from_list
    # and standard_games_from_list copied from the rating list header when available.
    query = f"""
    WITH raw AS (
        SELECT
            {key_expr} AS player_key,
            CAST(player_id AS BIGINT) AS fide_id,
            CAST(player_name AS VARCHAR) AS player_name,
            UPPER(TRIM(CAST(fed AS VARCHAR))) AS fed,
            UPPER(TRIM(CAST(sex AS VARCHAR))) AS sex,
            TRY_CAST(birth_year AS INTEGER) AS birth_year,
            TRY_CAST(standard_rating_from_list AS DOUBLE) AS standard_rating_from_list,
            TRY_CAST(standard_games_from_list AS INTEGER) AS standard_games_from_list,
            TRY_CAST(player_rating AS DOUBLE) AS player_rating,
            TRY_CAST(period AS VARCHAR) AS period,
            COALESCE(
                TRY_CAST(period AS DATE),
                TRY_STRPTIME(period, '%Y-%m'),
                TRY_STRPTIME(period, '%Y%m'),
                TRY_STRPTIME(date_from, '%Y-%m-%d'),
                TRY_STRPTIME(date_from, '%d.%m.%Y'),
                TRY_STRPTIME(date_from, '%Y.%m.%d')
            ) AS row_date
        FROM read_parquet('{p}')
        WHERE player_name IS NOT NULL
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY player_key
                ORDER BY row_date DESC NULLS LAST, standard_rating_from_list DESC NULLS LAST, player_rating DESC NULLS LAST
            ) AS rn,
            COUNT(*) OVER (PARTITION BY player_key) AS directed_rows_as_player,
            MAX(player_rating) OVER (PARTITION BY player_key) AS max_observed_player_rating,
            MAX(standard_rating_from_list) OVER (PARTITION BY player_key) AS max_list_rating
        FROM raw
        WHERE player_key IS NOT NULL AND player_key <> ''
    )
    SELECT
        player_key,
        fide_id,
        player_name,
        NULLIF(fed, '') AS fed,
        NULLIF(sex, '') AS sex,
        birth_year,
        standard_rating_from_list,
        standard_games_from_list,
        player_rating AS latest_player_rating,
        max_observed_player_rating,
        max_list_rating,
        period AS latest_period,
        row_date AS latest_row_date,
        directed_rows_as_player
    FROM ranked
    WHERE rn = 1
    """
    df = con.execute(query).fetchdf()
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    return df


def attach_metadata(membership: pd.DataFrame, metadata: pd.DataFrame, reference_year: int = 2025) -> pd.DataFrame:
    df = membership.merge(metadata, on="player_key", how="left")
    df["fed"] = df["fed"].fillna("UNK").replace("", "UNK")
    df["sex"] = df["sex"].fillna("UNK").replace("", "UNK")
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age_at_reference"] = reference_year - df["birth_year"]
    df.loc[(df["age_at_reference"] < 0) | (df["age_at_reference"] > 110), "age_at_reference"] = np.nan
    df["is_junior_u18"] = df["age_at_reference"].notna() & (df["age_at_reference"] < 18)
    df["is_junior_u21"] = df["age_at_reference"].notna() & (df["age_at_reference"] < 21)
    df["is_young_u25"] = df["age_at_reference"].notna() & (df["age_at_reference"] < 25)
    rating = pd.to_numeric(df["standard_rating_from_list"], errors="coerce")
    fallback = pd.to_numeric(df["max_list_rating"], errors="coerce")
    fallback2 = pd.to_numeric(df["max_observed_player_rating"], errors="coerce")
    df["current_or_max_rating"] = rating.fillna(fallback).fillna(fallback2)
    return df


def entropy_from_counts(counts: pd.Series) -> float:
    values = counts.astype(float).to_numpy()
    total = values.sum()
    if total <= 0:
        return float("nan")
    p = values[values > 0] / total
    return float(-(p * np.log2(p)).sum())


def hhi_from_counts(counts: pd.Series) -> float:
    values = counts.astype(float).to_numpy()
    total = values.sum()
    if total <= 0:
        return float("nan")
    p = values / total
    return float((p * p).sum())


def top_share(counts: pd.Series) -> float:
    total = counts.sum()
    return float(counts.max() / total) if total else float("nan")


def top_values_text(df: pd.DataFrame, value_col: str, count_col: str, n: int = 5) -> str:
    if df.empty:
        return ""
    rows = []
    for _, row in df.sort_values(count_col, ascending=False).head(n).iterrows():
        rows.append(f"{row[value_col]}:{int(row[count_col])}")
    return "; ".join(rows)
