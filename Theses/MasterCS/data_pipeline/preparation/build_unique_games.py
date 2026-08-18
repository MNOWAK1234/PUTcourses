#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import time

try:
    import duckdb
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Brakuje bibliotek. Zainstaluj:\n"
        "        pip install duckdb pyarrow pandas tqdm"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

INPUT_PARQUET = Path("games_resolved.parquet")
OUTPUT_PARQUET = Path("games_unique.parquet")
AUDIT_DIR = Path("games_unique_audit")
WORK_DB = Path("games_unique_work.duckdb")

OVERWRITE_OUTPUT = True
DELETE_WORK_DB_AFTER_SUCCESS = True

THREADS = 8
MEMORY_LIMIT = "8GB"
EXPORT_BATCH_ROWS = 250_000
PARQUET_COMPRESSION = "zstd"


# ============================================================
# SQL HELPERS
# ============================================================

DATE_EXPR = r"""
COALESCE(
    TRY_CAST(period AS DATE),
    TRY_STRPTIME(period, '%Y-%m'),
    TRY_STRPTIME(period, '%Y%m'),
    TRY_STRPTIME(date_from, '%Y-%m-%d'),
    TRY_STRPTIME(date_from, '%d.%m.%Y'),
    TRY_STRPTIME(date_from, '%Y.%m.%d')
)
""".strip()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def normalized_text(column: str) -> str:
    return (
        "LOWER(REGEXP_REPLACE("
        f"STRIP_ACCENTS(TRIM(COALESCE(CAST({column} AS VARCHAR), ''))), "
        "'[^a-zA-Z0-9]+', ' ', 'g'))"
    )


def remove_work_db_files() -> None:
    for suffix in ["", ".wal", ".tmp"]:
        path = Path(str(WORK_DB) + suffix)
        if path.exists():
            path.unlink()


# ============================================================
# EXPORT
# ============================================================

def export_query_to_parquet_with_tqdm(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
    total_rows: int,
) -> None:
    if output_path.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(f"Plik już istnieje: {output_path}")
        output_path.unlink()

    reader = con.execute(query).fetch_record_batch(EXPORT_BATCH_ROWS)
    writer: pq.ParquetWriter | None = None

    progress = tqdm(
        total=total_rows,
        desc="[7/7] exporting games_unique.parquet",
        unit="row",
        unit_scale=True,
    )

    try:
        for batch in reader:
            if batch.num_rows == 0:
                continue

            table = pa.Table.from_batches([batch])

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression=PARQUET_COMPRESSION,
                )

            writer.write_table(table)
            progress.update(batch.num_rows)
    finally:
        progress.close()
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("Eksport nie zwrócił żadnych rekordów.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Nie ma pliku: {INPUT_PARQUET}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if WORK_DB.exists():
        remove_work_db_files()

    print("========== BUILD UNIQUE GAMES PARQUET ==========")
    print(f"input:                 {INPUT_PARQUET}")
    print(f"output:                {OUTPUT_PARQUET}")
    print(f"work db:               {WORK_DB}")
    print(f"threads:               {THREADS}")
    print(f"memory limit:          {MEMORY_LIMIT}")
    print()

    started = time.perf_counter()

    con = duckdb.connect(str(WORK_DB))
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA enable_progress_bar=false")

    parquet = sql_path(INPUT_PARQUET)

    available_columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet}')"
        ).fetchall()
    }

    required = {
        "player_id",
        "player_name",
        "opponent_name",
        "player_rating",
        "score",
    }
    missing = sorted(required - available_columns)
    if missing:
        raise RuntimeError(f"Brakuje kolumn w parquet: {missing}")

    resolved_opponent_id_expr = (
        "CAST(resolved_opponent_fide_id AS BIGINT)"
        if "resolved_opponent_fide_id" in available_columns
        else "NULL::BIGINT"
    )

    actual_opponent_rating_expr = (
        "CAST(actual_opponent_rating AS DOUBLE)"
        if "actual_opponent_rating" in available_columns
        else "NULL::DOUBLE"
    )

    display_opponent_rating_expr = (
        "CAST(display_opponent_rating AS DOUBLE)"
        if "display_opponent_rating" in available_columns
        else (
            "CAST(opponent_rating AS DOUBLE)"
            if "opponent_rating" in available_columns
            else "NULL::DOUBLE"
        )
    )

    event_id_expr = (
        "CAST(event_id AS VARCHAR)"
        if "event_id" in available_columns
        else "NULL::VARCHAR"
    )

    event_name_expr = (
        "CAST(event_name AS VARCHAR)"
        if "event_name" in available_columns
        else "NULL::VARCHAR"
    )

    date_from_expr = (
        "CAST(date_from AS VARCHAR)"
        if "date_from" in available_columns
        else "NULL::VARCHAR"
    )

    date_to_expr = (
        "CAST(date_to AS VARCHAR)"
        if "date_to" in available_columns
        else "NULL::VARCHAR"
    )

    period_expr = (
        "CAST(period AS VARCHAR)"
        if "period" in available_columns
        else "NULL::VARCHAR"
    )

    color_expr = (
        "LOWER(TRIM(COALESCE(CAST(color AS VARCHAR), '')))"
        if "color" in available_columns
        else "''"
    )

    star_expr = (
        "COALESCE(CAST(star_400_rule AS BOOLEAN), FALSE)"
        if "star_400_rule" in available_columns
        else "FALSE"
    )

    recovery_method_expr = (
        "CAST(rating_recovery_method AS VARCHAR)"
        if "rating_recovery_method" in available_columns
        else "NULL::VARCHAR"
    )

    player_norm_expr = normalized_text("player_name")
    opponent_norm_expr = normalized_text("opponent_name")
    event_norm_expr = normalized_text("event_name") if "event_name" in available_columns else "''"

    print("[1/7] Building unique normalized-name -> player_id map...")
    con.execute("DROP TABLE IF EXISTS unique_name_to_id")
    con.execute(
        f"""
        CREATE TABLE unique_name_to_id AS
        SELECT
            player_name_norm,
            MIN(player_id)::BIGINT AS player_id
        FROM (
            SELECT
                CAST(player_id AS BIGINT) AS player_id,
                {player_norm_expr} AS player_name_norm
            FROM read_parquet('{parquet}')
            WHERE player_id IS NOT NULL
              AND {player_norm_expr} <> ''
        )
        GROUP BY player_name_norm
        HAVING COUNT(DISTINCT player_id) = 1
        """
    )

    print("[2/7] Canonicalizing every directed row...")
    con.execute("DROP TABLE IF EXISTS canonical_rows")
    con.execute(
        f"""
        CREATE TABLE canonical_rows AS
        WITH raw AS (
            SELECT
                ROW_NUMBER() OVER ()::BIGINT AS source_row_id,
                CAST(DATE_TRUNC('month', {DATE_EXPR}) AS DATE) AS month,
                {period_expr} AS period,
                {event_id_expr} AS event_id,
                {event_name_expr} AS event_name,
                {event_norm_expr} AS event_name_norm,
                {date_from_expr} AS date_from,
                {date_to_expr} AS date_to,

                CAST(player_id AS BIGINT) AS player_id,
                CAST(player_name AS VARCHAR) AS player_name,
                {player_norm_expr} AS player_name_norm,

                {resolved_opponent_id_expr} AS resolved_opponent_fide_id,
                CAST(opponent_name AS VARCHAR) AS opponent_name,
                {opponent_norm_expr} AS opponent_name_norm,

                CAST(player_rating AS DOUBLE) AS player_rating,
                {actual_opponent_rating_expr} AS actual_opponent_rating,
                {display_opponent_rating_expr} AS display_opponent_rating,

                CAST(score AS DOUBLE) AS score,
                {color_expr} AS color,
                {star_expr} AS star_400_rule,
                {recovery_method_expr} AS rating_recovery_method
            FROM read_parquet('{parquet}')
            WHERE {DATE_EXPR} IS NOT NULL
              AND score IN (0.0, 0.5, 1.0)
              AND player_name IS NOT NULL
              AND opponent_name IS NOT NULL
        ),
        keyed AS (
            SELECT
                raw.*,
                COALESCE(
                    raw.resolved_opponent_fide_id,
                    opponent_map.player_id
                ) AS opponent_player_id,
                CASE
                    WHEN raw.player_id IS NOT NULL
                        THEN 'id:' || CAST(raw.player_id AS VARCHAR)
                    ELSE 'name:' || raw.player_name_norm
                END AS player_key,
                CASE
                    WHEN COALESCE(
                        raw.resolved_opponent_fide_id,
                        opponent_map.player_id
                    ) IS NOT NULL
                        THEN 'id:' || CAST(
                            COALESCE(
                                raw.resolved_opponent_fide_id,
                                opponent_map.player_id
                            ) AS VARCHAR
                        )
                    ELSE 'name:' || raw.opponent_name_norm
                END AS opponent_key
            FROM raw
            LEFT JOIN unique_name_to_id opponent_map
              ON opponent_map.player_name_norm = raw.opponent_name_norm
        ),
        oriented AS (
            SELECT
                *,
                player_key <= opponent_key AS player_is_a
            FROM keyed
            WHERE player_key <> ''
              AND opponent_key <> ''
              AND player_key <> opponent_key
        )
        SELECT
            source_row_id,
            month,
            period,
            event_id,
            event_name,
            event_name_norm,
            date_from,
            date_to,

            CASE WHEN player_is_a THEN player_key ELSE opponent_key END AS player_a_key,
            CASE WHEN player_is_a THEN opponent_key ELSE player_key END AS player_b_key,

            CASE WHEN player_is_a THEN player_id ELSE opponent_player_id END AS player_a_id,
            CASE WHEN player_is_a THEN opponent_player_id ELSE player_id END AS player_b_id,

            CASE WHEN player_is_a THEN player_name ELSE opponent_name END AS player_a_name,
            CASE WHEN player_is_a THEN opponent_name ELSE player_name END AS player_b_name,

            CASE WHEN player_is_a THEN player_rating ELSE actual_opponent_rating END AS player_a_fide_rating,
            CASE WHEN player_is_a THEN actual_opponent_rating ELSE player_rating END AS player_b_fide_rating,

            CASE
                WHEN player_is_a THEN display_opponent_rating
                ELSE player_rating
            END AS player_b_display_rating_from_source,

            CASE WHEN player_is_a THEN score ELSE 1.0 - score END AS score_a,

            CASE
                WHEN player_is_a THEN color
                WHEN color = 'white' THEN 'black'
                WHEN color = 'black' THEN 'white'
                WHEN color = 'w' THEN 'b'
                WHEN color = 'b' THEN 'w'
                ELSE color
            END AS color_a,

            CASE WHEN player_is_a THEN 0 ELSE 1 END AS source_side,
            star_400_rule,
            rating_recovery_method,

            MD5(
                CONCAT_WS(
                    '||',
                    COALESCE(CAST(month AS VARCHAR), ''),
                    COALESCE(period, ''),
                    COALESCE(event_id, ''),
                    COALESCE(event_name_norm, ''),
                    COALESCE(date_from, ''),
                    COALESCE(date_to, ''),
                    CASE WHEN player_is_a THEN player_key ELSE opponent_key END,
                    CASE WHEN player_is_a THEN opponent_key ELSE player_key END,
                    CAST(CASE WHEN player_is_a THEN score ELSE 1.0 - score END AS VARCHAR),
                    CASE
                        WHEN player_is_a THEN color
                        WHEN color = 'white' THEN 'black'
                        WHEN color = 'black' THEN 'white'
                        WHEN color = 'w' THEN 'b'
                        WHEN color = 'b' THEN 'w'
                        ELSE color
                    END
                )
            ) AS canonical_signature
        FROM oriented
        """
    )

    print("[3/7] Aggregating reciprocal perspectives...")
    con.execute("DROP TABLE IF EXISTS grouped_games")
    con.execute(
        """
        CREATE TABLE grouped_games AS
        WITH aggregated AS (
            SELECT
                canonical_signature,
                month,
                MIN(period) AS period,
                MIN(event_id) AS event_id,
                MIN(event_name) AS event_name,
                MIN(date_from) AS date_from,
                MIN(date_to) AS date_to,

                player_a_key,
                player_b_key,

                MIN(player_a_id) AS player_a_id,
                MIN(player_b_id) AS player_b_id,
                MIN(player_a_name) AS player_a_name,
                MIN(player_b_name) AS player_b_name,

                MEDIAN(player_a_fide_rating)::DOUBLE AS player_a_fide_rating,
                MEDIAN(player_b_fide_rating)::DOUBLE AS player_b_fide_rating,

                score_a,
                MIN(color_a) AS color_a,

                COUNT(*)::BIGINT AS source_rows,
                SUM(CASE WHEN source_side = 0 THEN 1 ELSE 0 END)::BIGINT AS side_a_rows,
                SUM(CASE WHEN source_side = 1 THEN 1 ELSE 0 END)::BIGINT AS side_b_rows,

                GREATEST(
                    SUM(CASE WHEN source_side = 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN source_side = 1 THEN 1 ELSE 0 END)
                )::BIGINT AS physical_game_instances,

                LEAST(
                    SUM(CASE WHEN source_side = 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN source_side = 1 THEN 1 ELSE 0 END)
                )::BIGINT AS reciprocal_pairs,

                ABS(
                    SUM(CASE WHEN source_side = 0 THEN 1 ELSE 0 END)
                    -
                    SUM(CASE WHEN source_side = 1 THEN 1 ELSE 0 END)
                )::BIGINT AS unpaired_source_rows,

                BOOL_OR(star_400_rule) AS any_star_400_rule
            FROM canonical_rows
            GROUP BY
                canonical_signature,
                month,
                player_a_key,
                player_b_key,
                score_a
        )
        SELECT
            aggregated.*,
            CASE
                WHEN player_a_fide_rating IS NOT NULL
                 AND player_b_fide_rating IS NOT NULL
                THEN TRUE
                ELSE FALSE
            END AS fide_ratings_resolved
        FROM aggregated
        """
    )

    print("[4/7] Counting output rows...")
    total_unique_games = int(
        con.execute(
            "SELECT COALESCE(SUM(physical_game_instances), 0) FROM grouped_games"
        ).fetchone()[0]
    )

    total_source_rows = int(
        con.execute("SELECT COUNT(*) FROM canonical_rows").fetchone()[0]
    )

    print("[5/7] Writing audit CSV files...")
    summary = con.execute(
        """
        SELECT
            COUNT(*)::BIGINT AS grouped_signatures,
            SUM(source_rows)::BIGINT AS canonical_source_rows,
            SUM(physical_game_instances)::BIGINT AS unique_physical_games,
            SUM(reciprocal_pairs)::BIGINT AS reciprocal_pairs,
            SUM(unpaired_source_rows)::BIGINT AS unpaired_source_rows,
            SUM(CASE WHEN side_a_rows > 0 AND side_b_rows > 0 THEN physical_game_instances ELSE 0 END)::BIGINT
                AS games_with_both_perspectives,
            SUM(CASE WHEN side_a_rows = 0 OR side_b_rows = 0 THEN physical_game_instances ELSE 0 END)::BIGINT
                AS games_with_one_perspective_only,
            SUM(CASE WHEN fide_ratings_resolved THEN physical_game_instances ELSE 0 END)::BIGINT
                AS games_with_resolved_fide_ratings,
            SUM(CASE WHEN any_star_400_rule THEN physical_game_instances ELSE 0 END)::BIGINT
                AS games_touching_star_400_rule
        FROM grouped_games
        """
    ).fetchdf()

    summary.to_csv(
        AUDIT_DIR / "games_unique_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    unmatched = con.execute(
        """
        SELECT *
        FROM grouped_games
        WHERE side_a_rows = 0
           OR side_b_rows = 0
           OR side_a_rows <> side_b_rows
        ORDER BY unpaired_source_rows DESC, source_rows DESC
        LIMIT 100000
        """
    ).fetchdf()

    unmatched.to_csv(
        AUDIT_DIR / "games_unique_unmatched_or_imbalanced_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[6/7] Preparing streaming export...")
    export_query = """
        SELECT
            canonical_signature || ':' || CAST(instance_index AS VARCHAR) AS game_uid,
            canonical_signature,
            instance_index::BIGINT AS duplicate_instance_index,

            month,
            period,
            event_id,
            event_name,
            date_from,
            date_to,

            player_a_key,
            player_b_key,
            player_a_id,
            player_b_id,
            player_a_name,
            player_b_name,

            player_a_fide_rating,
            player_b_fide_rating,
            score_a,
            color_a,

            source_rows,
            side_a_rows,
            side_b_rows,
            reciprocal_pairs,
            unpaired_source_rows,
            any_star_400_rule,
            fide_ratings_resolved
        FROM grouped_games,
             RANGE(physical_game_instances) AS generated(instance_index)
        ORDER BY month, date_from, event_id, canonical_signature, instance_index
    """

    export_started = time.perf_counter()
    export_query_to_parquet_with_tqdm(
        con,
        export_query,
        OUTPUT_PARQUET,
        total_unique_games,
    )
    export_seconds = time.perf_counter() - export_started

    con.close()

    output_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    total_seconds = time.perf_counter() - started

    print()
    print("========== DONE ==========")
    print(f"canonical source rows:      {total_source_rows:,}")
    print(f"unique physical games:      {total_unique_games:,}")
    print(f"output size:                {output_mb:.2f} MB")
    print(f"export seconds:             {export_seconds:.2f}")
    print(f"export speed:               {total_unique_games / max(export_seconds, 1e-9):,.0f} games/s")
    print(f"total time:                 {total_seconds / 60.0:.2f} min")
    print(f"output parquet:             {OUTPUT_PARQUET}")
    print(f"summary CSV:                {AUDIT_DIR / 'games_unique_summary.csv'}")
    print(
        "unmatched sample CSV:       "
        f"{AUDIT_DIR / 'games_unique_unmatched_or_imbalanced_sample.csv'}"
    )

    if DELETE_WORK_DB_AFTER_SUCCESS:
        remove_work_db_files()
        print(f"work db deleted:            {WORK_DB}")
    else:
        print(f"work db kept:               {WORK_DB}")

    print("==========================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
