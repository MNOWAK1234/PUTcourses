#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Brakuje duckdb. Zainstaluj zależności:\n"
        "        pip install duckdb pyarrow tqdm"
    ) from exc

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Brakuje pyarrow. Zainstaluj zależności:\n"
        "        pip install duckdb pyarrow tqdm"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Brakuje tqdm. Zainstaluj zależności:\n"
        "        pip install duckdb pyarrow tqdm"
    ) from exc


# ============================================================================
# CONFIG — zmieniaj tylko tutaj
# ============================================================================

INPUT_PARQUET = Path("games.parquet")
OUTPUT_PARQUET = Path("games_resolved.parquet")
WORK_DB = Path("games_resolver_work.duckdb")
AUDIT_DIR = Path("games_resolved_audit")

# Finalny eksport jest wykonywany partiami, dzięki czemu widać rows/s w tqdm.
EXPORT_BATCH_ROWS = 250_000

# DuckDB zwykle dobrze dobiera liczbę wątków. Możesz ustawić np. 8 ręcznie.
DUCKDB_THREADS = max(1, os.cpu_count() or 4)

# Przykład ograniczenia RAM: "8GB". None = zostaw automatyczne ustawienie DuckDB.
DUCKDB_MEMORY_LIMIT: str | None = None

# True = usuń stary output i techniczną bazę roboczą przy nowym uruchomieniu.
OVERWRITE_OUTPUT = True
RESET_WORK_DB_ON_START = True

# True = po udanym zakończeniu usuń dużą techniczną bazę DuckDB.
DELETE_WORK_DB_AFTER_SUCCESS = True

# Kompresja finalnego Parquet.
PARQUET_COMPRESSION = "zstd"


# ============================================================================
# HELPERS
# ============================================================================

INTERNAL_COLUMNS = {
    "__row_id",
    "__player_name_exact",
    "__opponent_name_exact",
    "__player_name_canonical",
    "__opponent_name_canonical",
    "__player_name_loose",
    "__opponent_name_loose",
    "__period_key",
    "__event_key",
    "__date_from_key",
    "__date_to_key",
    "__score_code",
    "__key_exact_strict",
    "__key_canonical_strict",
    "__key_loose_strict",
    "__key_canonical_relaxed",
    "__key_loose_relaxed",
}

ADDED_OUTPUT_COLUMNS = {
    "actual_opponent_rating",
    "actual_rating_diff",
    "resolved_opponent_fide_id",
    "rating_recovery_method",
    "rating_recovery_confidence",
    "reciprocal_candidate_rows",
    "reciprocal_candidate_rating_count",
}


def log(message: str = "") -> None:
    print(message, flush=True)


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def human_bytes(size: int | float) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def remove_work_db_files() -> None:
    remove_if_exists(WORK_DB)
    remove_if_exists(Path(str(WORK_DB) + ".wal"))


def fetch_scalar(con: duckdb.DuckDBPyConnection, query: str, params: Sequence | None = None):
    return con.execute(query, params or []).fetchone()[0]


def execute_with_tqdm(
    con: duckdb.DuckDBPyConnection,
    query: str,
    *,
    desc: str,
    params: Sequence | None = None,
) -> float:
    """
    DuckDB execute() jest blokujące. Uruchamiamy je w osobnym wątku i pokazujemy
    pasek tqdm z czasem działania. Po fazie drukujemy właściwe rows/s na podstawie
    liczby przetworzonych rekordów.
    """
    result: dict[str, BaseException] = {}

    def worker() -> None:
        try:
            con.execute(query, params or [])
        except BaseException as exc:  # noqa: BLE001 - przekazujemy błąd do głównego wątku
            result["error"] = exc

    started = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Nie wszystkie operatory DuckDB udostępniają dokładny procent. TQDM pokazuje
    # więc czas fazy i aktywność, zamiast udawać precyzyjny postęp procentowy.
    with tqdm(desc=desc, unit="tick", leave=True) as bar:
        while thread.is_alive():
            elapsed = time.perf_counter() - started
            bar.set_postfix_str(f"elapsed={elapsed:.1f}s | DuckDB pracuje")
            bar.update(1)
            time.sleep(0.5)

    thread.join()
    elapsed = time.perf_counter() - started

    if "error" in result:
        raise result["error"]

    return elapsed


def configure_duckdb(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"PRAGMA threads={int(DUCKDB_THREADS)}")
    con.execute("PRAGMA preserve_insertion_order=false")
    if DUCKDB_MEMORY_LIMIT:
        escaped = DUCKDB_MEMORY_LIMIT.replace("'", "''")
        con.execute(f"PRAGMA memory_limit='{escaped}'")


def create_macros(con: duckdb.DuckDBPyConnection) -> None:
    # exact: tylko casing + nadmiarowe spacje
    con.execute(
        r"""
        CREATE OR REPLACE MACRO norm_exact(x) AS
            lower(regexp_replace(trim(coalesce(cast(x AS VARCHAR), '')), '\s+', ' ', 'g'));
        """
    )

    # canonical: usuń interpunkcję/akcenty, posortuj tokeny; kolejność imię/nazwisko
    # przestaje mieć znaczenie, np. "Fernandez, Michael" == "Michael Fernandez".
    con.execute(
        r"""
        CREATE OR REPLACE MACRO norm_canonical(x) AS
            array_to_string(
                list_sort(
                    list_filter(
                        string_split(
                            trim(regexp_replace(
                                lower(strip_accents(coalesce(cast(x AS VARCHAR), ''))),
                                '[^a-z0-9]+', ' ', 'g'
                            )),
                            ' '
                        ),
                        token -> length(token) > 0
                    )
                ),
                ' '
            );
        """
    )

    # loose: dodatkowo usuń liczby i jednoliterowe inicjały. Używamy dopiero po
    # bezpieczniejszych metodach i tylko wtedy, gdy ranking kandydata jest jednoznaczny.
    con.execute(
        r"""
        CREATE OR REPLACE MACRO norm_loose(x) AS
            array_to_string(
                list_sort(
                    list_filter(
                        string_split(
                            trim(regexp_replace(
                                lower(strip_accents(coalesce(cast(x AS VARCHAR), ''))),
                                '[^a-z0-9]+', ' ', 'g'
                            )),
                            ' '
                        ),
                        token -> length(token) > 1
                                 AND NOT regexp_matches(token, '^[0-9]+$')
                    )
                ),
                ' '
            );
        """
    )

    con.execute(
        r"""
        CREATE OR REPLACE MACRO make_event_key(event_id_value, event_name_value) AS
            CASE
                WHEN event_id_value IS NOT NULL
                    THEN 'id:' || cast(event_id_value AS VARCHAR)
                ELSE 'name:' || norm_canonical(event_name_value)
            END;
        """
    )


def source_columns(con: duckdb.DuckDBPyConnection, input_path: str) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{input_path}')"
    ).fetchall()
    return [str(row[0]) for row in rows]


def check_source_columns(columns: Iterable[str]) -> None:
    columns_set = set(columns)
    required = {
        "player_id",
        "player_name",
        "opponent_name",
        "period",
        "event_id",
        "event_name",
        "date_from",
        "date_to",
        "player_rating",
        "opponent_rating",
        "score",
        "star_400_rule",
    }
    missing = sorted(required - columns_set)
    if missing:
        raise RuntimeError(f"Brak wymaganych kolumn w games.parquet: {missing}")

    collisions = sorted((INTERNAL_COLUMNS | ADDED_OUTPUT_COLUMNS) & columns_set)
    if collisions:
        raise RuntimeError(
            "Input wygląda jak już przetworzony plik albo ma kolizję nazw kolumn: "
            + ", ".join(collisions)
        )


# ============================================================================
# SQL BUILD STEPS
# ============================================================================


def create_base_table(con: duckdb.DuckDBPyConnection, input_path: str) -> float:
    query = f"""
        CREATE TABLE base AS
        WITH normalized AS (
            SELECT
                row_number() OVER ()::BIGINT AS __row_id,
                src.*,
                norm_exact(player_name) AS __player_name_exact,
                norm_exact(opponent_name) AS __opponent_name_exact,
                norm_canonical(player_name) AS __player_name_canonical,
                norm_canonical(opponent_name) AS __opponent_name_canonical,
                norm_loose(player_name) AS __player_name_loose,
                norm_loose(opponent_name) AS __opponent_name_loose,
                coalesce(cast(period AS VARCHAR), '') AS __period_key,
                make_event_key(event_id, event_name) AS __event_key,
                coalesce(cast(date_from AS VARCHAR), '') AS __date_from_key,
                coalesce(cast(date_to AS VARCHAR), '') AS __date_to_key,
                cast(round(cast(score AS DOUBLE) * 2.0) AS INTEGER) AS __score_code
            FROM read_parquet('{input_path}') AS src
        )
        SELECT
            normalized.*,
            hash(
                __period_key, __event_key, __date_from_key, __date_to_key,
                __score_code, __player_name_exact, __opponent_name_exact
            ) AS __key_exact_strict,
            hash(
                __period_key, __event_key, __date_from_key, __date_to_key,
                __score_code, __player_name_canonical, __opponent_name_canonical
            ) AS __key_canonical_strict,
            hash(
                __period_key, __event_key, __date_from_key, __date_to_key,
                __score_code, __player_name_loose, __opponent_name_loose
            ) AS __key_loose_strict,
            hash(
                __period_key, __event_key,
                __score_code, __player_name_canonical, __opponent_name_canonical
            ) AS __key_canonical_relaxed,
            hash(
                __period_key, __event_key,
                __score_code, __player_name_loose, __opponent_name_loose
            ) AS __key_loose_relaxed
        FROM normalized
    """
    return execute_with_tqdm(con, query, desc="[1/9] staging games.parquet")


def create_star_rows(con: duckdb.DuckDBPyConnection) -> float:
    query = """
        CREATE TABLE star_rows AS
        SELECT
            __row_id,
            player_id,
            player_name,
            opponent_name,
            period,
            event_id,
            event_name,
            date_from,
            date_to,
            player_rating,
            opponent_rating,
            score,
            __player_name_exact,
            __opponent_name_exact,
            __player_name_canonical,
            __opponent_name_canonical,
            __player_name_loose,
            __opponent_name_loose,
            __period_key,
            __event_key,
            __date_from_key,
            __date_to_key,
            __score_code,
            hash(
                __period_key, __event_key, __date_from_key, __date_to_key,
                2 - __score_code, __opponent_name_exact, __player_name_exact
            ) AS __reverse_key_exact_strict,
            hash(
                __period_key, __event_key, __date_from_key, __date_to_key,
                2 - __score_code, __opponent_name_canonical, __player_name_canonical
            ) AS __reverse_key_canonical_strict,
            hash(
                __period_key, __event_key, __date_from_key, __date_to_key,
                2 - __score_code, __opponent_name_loose, __player_name_loose
            ) AS __reverse_key_loose_strict,
            hash(
                __period_key, __event_key,
                2 - __score_code, __opponent_name_canonical, __player_name_canonical
            ) AS __reverse_key_canonical_relaxed,
            hash(
                __period_key, __event_key,
                2 - __score_code, __opponent_name_loose, __player_name_loose
            ) AS __reverse_key_loose_relaxed
        FROM base
        WHERE coalesce(cast(star_400_rule AS BOOLEAN), false) = true
          AND player_rating IS NOT NULL
          AND opponent_rating IS NOT NULL
          AND __score_code IN (0, 1, 2)
    """
    return execute_with_tqdm(con, query, desc="[2/9] extracting star_400 rows")


def create_resolution_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE resolutions (
            __row_id BIGINT,
            actual_opponent_rating DOUBLE,
            resolved_opponent_fide_id BIGINT,
            rating_recovery_method VARCHAR,
            rating_recovery_confidence VARCHAR,
            reciprocal_candidate_rows BIGINT,
            reciprocal_candidate_rating_count BIGINT
        )
        """
    )


def insert_resolution_tier(
    con: duckdb.DuckDBPyConnection,
    *,
    desc: str,
    method: str,
    confidence: str,
    reverse_key_column: str,
    base_key_column: str,
    player_name_column: str,
    opponent_name_column: str,
    strict_dates: bool,
) -> float:
    date_conditions = """
        AND b.__date_from_key = a.__date_from_key
        AND b.__date_to_key = a.__date_to_key
    """ if strict_dates else ""

    query = f"""
        INSERT INTO resolutions
        WITH candidates AS (
            SELECT
                a.__row_id,
                count(*)::BIGINT AS reciprocal_candidate_rows,
                count(DISTINCT cast(b.player_rating AS DOUBLE))::BIGINT
                    AS reciprocal_candidate_rating_count,
                min(cast(b.player_rating AS DOUBLE)) AS actual_opponent_rating,
                count(DISTINCT cast(b.player_id AS BIGINT))::BIGINT
                    AS reciprocal_candidate_player_id_count,
                min(cast(b.player_id AS BIGINT)) AS resolved_opponent_fide_id
            FROM star_rows AS a
            JOIN base AS b
              ON b.{qident(base_key_column)} = a.{qident(reverse_key_column)}
             AND b.__row_id <> a.__row_id
             AND b.__period_key = a.__period_key
             AND b.__event_key = a.__event_key
             AND b.__score_code = 2 - a.__score_code
             AND b.{qident(player_name_column)} = a.{qident(opponent_name_column)}
             AND b.{qident(opponent_name_column)} = a.{qident(player_name_column)}
             {date_conditions}
             AND b.player_rating IS NOT NULL
            LEFT JOIN resolutions AS already
              ON already.__row_id = a.__row_id
            WHERE already.__row_id IS NULL
            GROUP BY a.__row_id
        )
        SELECT
            __row_id,
            actual_opponent_rating,
            CASE
                WHEN reciprocal_candidate_player_id_count = 1
                    THEN resolved_opponent_fide_id
                ELSE NULL
            END AS resolved_opponent_fide_id,
            '{method}' AS rating_recovery_method,
            '{confidence}' AS rating_recovery_confidence,
            reciprocal_candidate_rows,
            reciprocal_candidate_rating_count
        FROM candidates
        WHERE reciprocal_candidate_rating_count = 1
    """

    return execute_with_tqdm(con, query, desc=desc)


# ============================================================================
# OUTPUT
# ============================================================================


def export_unresolved_csv(con: duckdb.DuckDBPyConnection, output_path: Path) -> float:
    escaped = sql_path(output_path)
    query = f"""
        COPY (
            SELECT
                b.__row_id AS source_row_id,
                b.player_id,
                b.player_name,
                b.opponent_fide_id,
                b.opponent_name,
                b.period,
                b.event_id,
                b.event_name,
                b.date_from,
                b.date_to,
                b.player_rating,
                b.opponent_rating AS displayed_or_capped_opponent_rating,
                b.score,
                b.star_400_rule,
                'unresolved' AS rating_recovery_method
            FROM base AS b
            LEFT JOIN resolutions AS r USING (__row_id)
            WHERE coalesce(cast(b.star_400_rule AS BOOLEAN), false) = true
              AND r.__row_id IS NULL
            ORDER BY b.__row_id
        ) TO '{escaped}' (HEADER, DELIMITER ',')
    """
    return execute_with_tqdm(con, query, desc="[8/9] exporting unresolved audit CSV")


def export_resolved_parquet(
    con: duckdb.DuckDBPyConnection,
    *,
    columns: list[str],
    total_rows: int,
) -> float:
    temp_output = Path(str(OUTPUT_PARQUET) + ".tmp")
    remove_if_exists(temp_output)

    original_select = ",\n                ".join(
        f"b.{qident(column)}" for column in columns
    )

    query = f"""
        SELECT
            {original_select},
            CASE
                WHEN coalesce(cast(b.star_400_rule AS BOOLEAN), false) = false
                    THEN cast(b.opponent_rating AS DOUBLE)
                ELSE r.actual_opponent_rating
            END AS actual_opponent_rating,
            CASE
                WHEN coalesce(cast(b.star_400_rule AS BOOLEAN), false) = false
                     AND b.player_rating IS NOT NULL
                     AND b.opponent_rating IS NOT NULL
                    THEN cast(b.player_rating AS DOUBLE) - cast(b.opponent_rating AS DOUBLE)
                WHEN r.actual_opponent_rating IS NOT NULL
                     AND b.player_rating IS NOT NULL
                    THEN cast(b.player_rating AS DOUBLE) - r.actual_opponent_rating
                ELSE NULL
            END AS actual_rating_diff,
            coalesce(cast(b.opponent_fide_id AS BIGINT), r.resolved_opponent_fide_id)
                AS resolved_opponent_fide_id,
            CASE
                WHEN coalesce(cast(b.star_400_rule AS BOOLEAN), false) = false
                    THEN 'non_star_original'
                ELSE coalesce(r.rating_recovery_method, 'unresolved')
            END AS rating_recovery_method,
            CASE
                WHEN coalesce(cast(b.star_400_rule AS BOOLEAN), false) = false
                    THEN 'direct'
                ELSE coalesce(r.rating_recovery_confidence, 'none')
            END AS rating_recovery_confidence,
            coalesce(r.reciprocal_candidate_rows, 0)::BIGINT
                AS reciprocal_candidate_rows,
            coalesce(r.reciprocal_candidate_rating_count, 0)::BIGINT
                AS reciprocal_candidate_rating_count
        FROM base AS b
        LEFT JOIN resolutions AS r USING (__row_id)
        WHERE b.__row_id >= ? AND b.__row_id < ?
        ORDER BY b.__row_id
    """

    writer: pq.ParquetWriter | None = None
    started = time.perf_counter()

    try:
        with tqdm(
            total=total_rows,
            desc="[9/9] exporting games_resolved.parquet",
            unit="row",
            unit_scale=True,
            smoothing=0.05,
        ) as bar:
            start_row = 1
            while start_row <= total_rows:
                end_row = min(total_rows + 1, start_row + EXPORT_BATCH_ROWS)
                table = con.execute(query, [start_row, end_row]).to_arrow_table()

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(temp_output),
                        table.schema,
                        compression=PARQUET_COMPRESSION,
                        use_dictionary=True,
                    )

                writer.write_table(table)
                count = table.num_rows
                bar.update(count)
                start_row = end_row

        if writer is None:
            raise RuntimeError("Nie zapisano żadnych wierszy do finalnego Parquet")

    finally:
        if writer is not None:
            writer.close()

    os.replace(temp_output, OUTPUT_PARQUET)
    return time.perf_counter() - started


def write_summary_csv(
    con: duckdb.DuckDBPyConnection,
    *,
    total_rows: int,
    star_rows: int,
    unresolved_rows: int,
    stage_seconds: float,
    export_seconds: float,
    total_seconds: float,
) -> Path:
    summary_path = AUDIT_DIR / "games_resolved_summary.csv"

    method_rows = con.execute(
        """
        SELECT rating_recovery_method, count(*)::BIGINT AS rows
        FROM resolutions
        GROUP BY rating_recovery_method
        ORDER BY rating_recovery_method
        """
    ).fetchall()

    resolved_rows = sum(int(row[1]) for row in method_rows)

    rows = [
        ("input_parquet", str(INPUT_PARQUET)),
        ("output_parquet", str(OUTPUT_PARQUET)),
        ("rows_total", total_rows),
        ("rows_non_star", total_rows - star_rows),
        ("rows_star_400", star_rows),
        ("rows_star_resolved", resolved_rows),
        ("rows_star_unresolved", unresolved_rows),
        ("star_recovery_percent", f"{(resolved_rows / star_rows * 100.0) if star_rows else 100.0:.4f}"),
        ("stage_seconds", f"{stage_seconds:.3f}"),
        ("export_seconds", f"{export_seconds:.3f}"),
        ("total_seconds", f"{total_seconds:.3f}"),
        ("output_size_bytes", OUTPUT_PARQUET.stat().st_size if OUTPUT_PARQUET.exists() else 0),
    ]

    for method, count in method_rows:
        rows.append((f"method__{method}", int(count)))

    with open(summary_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    return summary_path


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    started_total = time.perf_counter()

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Nie ma pliku wejściowego: {INPUT_PARQUET}")
    if EXPORT_BATCH_ROWS < 1:
        raise ValueError("EXPORT_BATCH_ROWS musi być >= 1")

    if OUTPUT_PARQUET.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"Plik już istnieje: {OUTPUT_PARQUET}. Ustaw OVERWRITE_OUTPUT=True, aby nadpisać."
        )

    if RESET_WORK_DB_ON_START:
        remove_work_db_files()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    remove_if_exists(Path(str(OUTPUT_PARQUET) + ".tmp"))

    disk = shutil.disk_usage(Path.cwd())

    log("========== RESOLVE FIDE STAR-400 PARQUET ==========")
    log(f"input:                  {INPUT_PARQUET}")
    log(f"input size:             {human_bytes(INPUT_PARQUET.stat().st_size)}")
    log(f"output:                 {OUTPUT_PARQUET}")
    log(f"work db:                {WORK_DB}")
    log(f"free disk before:       {human_bytes(disk.free)}")
    log(f"duckdb threads:         {DUCKDB_THREADS}")
    log(f"export batch rows:      {EXPORT_BATCH_ROWS}")
    log()

    con = duckdb.connect(str(WORK_DB))
    configure_duckdb(con)
    create_macros(con)

    input_path = sql_path(INPUT_PARQUET)
    columns = source_columns(con, input_path)
    check_source_columns(columns)

    stage_seconds = create_base_table(con, input_path)
    total_rows = int(fetch_scalar(con, "SELECT count(*) FROM base"))
    log(f"[INFO] staged rows:      {total_rows:,}")
    log(f"[INFO] staging speed:    {total_rows / max(stage_seconds, 1e-9):,.0f} rows/s")
    log()

    star_extract_seconds = create_star_rows(con)
    star_rows = int(fetch_scalar(con, "SELECT count(*) FROM star_rows"))
    log(f"[INFO] star_400 rows:    {star_rows:,}")
    log(f"[INFO] extraction speed: {total_rows / max(star_extract_seconds, 1e-9):,.0f} input rows/s")
    log()

    create_resolution_table(con)

    tiers = [
        {
            "desc": "[3/9] resolve exact names + strict dates",
            "method": "reverse_exact_strict_dates",
            "confidence": "very_high",
            "reverse_key_column": "__reverse_key_exact_strict",
            "base_key_column": "__key_exact_strict",
            "player_name_column": "__player_name_exact",
            "opponent_name_column": "__opponent_name_exact",
            "strict_dates": True,
        },
        {
            "desc": "[4/9] resolve canonical names + strict dates",
            "method": "reverse_canonical_strict_dates",
            "confidence": "high",
            "reverse_key_column": "__reverse_key_canonical_strict",
            "base_key_column": "__key_canonical_strict",
            "player_name_column": "__player_name_canonical",
            "opponent_name_column": "__opponent_name_canonical",
            "strict_dates": True,
        },
        {
            "desc": "[5/9] resolve loose names + strict dates",
            "method": "reverse_loose_name_strict_dates",
            "confidence": "medium_high",
            "reverse_key_column": "__reverse_key_loose_strict",
            "base_key_column": "__key_loose_strict",
            "player_name_column": "__player_name_loose",
            "opponent_name_column": "__opponent_name_loose",
            "strict_dates": True,
        },
        {
            "desc": "[6/9] resolve canonical names + relaxed dates",
            "method": "reverse_canonical_relaxed_dates",
            "confidence": "medium",
            "reverse_key_column": "__reverse_key_canonical_relaxed",
            "base_key_column": "__key_canonical_relaxed",
            "player_name_column": "__player_name_canonical",
            "opponent_name_column": "__opponent_name_canonical",
            "strict_dates": False,
        },
        {
            "desc": "[7/9] resolve loose names + relaxed dates",
            "method": "reverse_loose_name_relaxed_dates",
            "confidence": "medium_low",
            "reverse_key_column": "__reverse_key_loose_relaxed",
            "base_key_column": "__key_loose_relaxed",
            "player_name_column": "__player_name_loose",
            "opponent_name_column": "__opponent_name_loose",
            "strict_dates": False,
        },
    ]

    previous_resolved = 0
    for tier in tiers:
        elapsed = insert_resolution_tier(con, **tier)
        resolved = int(fetch_scalar(con, "SELECT count(*) FROM resolutions"))
        delta = resolved - previous_resolved
        previous_resolved = resolved
        left = star_rows - resolved
        log(
            f"[INFO] {tier['method']}: +{delta:,} | total resolved={resolved:,} "
            f"| unresolved={left:,} | phase={elapsed:.1f}s"
        )
        log()

    unresolved_rows = star_rows - previous_resolved

    unresolved_path = AUDIT_DIR / "star_400_unresolved.csv"
    export_unresolved_csv(con, unresolved_path)

    export_seconds = export_resolved_parquet(
        con,
        columns=columns,
        total_rows=total_rows,
    )

    total_seconds = time.perf_counter() - started_total
    summary_path = write_summary_csv(
        con,
        total_rows=total_rows,
        star_rows=star_rows,
        unresolved_rows=unresolved_rows,
        stage_seconds=stage_seconds,
        export_seconds=export_seconds,
        total_seconds=total_seconds,
    )

    con.close()

    output_size = OUTPUT_PARQUET.stat().st_size
    resolved_rows = star_rows - unresolved_rows
    recovery_percent = resolved_rows / star_rows * 100.0 if star_rows else 100.0

    log()
    log("========== DONE ==========")
    log(f"rows total:              {total_rows:,}")
    log(f"star_400 rows:           {star_rows:,}")
    log(f"star resolved:           {resolved_rows:,}")
    log(f"star unresolved:         {unresolved_rows:,}")
    log(f"star recovery:           {recovery_percent:.2f}%")
    log(f"output size:             {human_bytes(output_size)}")
    log(f"export speed:            {total_rows / max(export_seconds, 1e-9):,.0f} rows/s")
    log(f"total time:              {total_seconds / 60.0:.2f} min")
    log(f"output parquet:          {OUTPUT_PARQUET}")
    log(f"summary CSV:             {summary_path}")
    log(f"unresolved audit CSV:    {unresolved_path}")

    if DELETE_WORK_DB_AFTER_SUCCESS:
        remove_work_db_files()
        log(f"work db deleted:         {WORK_DB}")
    else:
        log(f"work db kept:            {WORK_DB}")

    log("==========================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Przerwano przez użytkownika.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - czytelny komunikat CLI
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
