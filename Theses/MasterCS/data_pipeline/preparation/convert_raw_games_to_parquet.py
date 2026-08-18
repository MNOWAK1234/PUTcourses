#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit(
        "[ERROR] Brakuje biblioteki pyarrow. Zainstaluj ją komendą:\n"
        "        pip install pyarrow tqdm"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit(
        "[ERROR] Brakuje biblioteki tqdm. Zainstaluj ją komendą:\n"
        "        pip install pyarrow tqdm"
    ) from exc


DEFAULT_INPUT_DIR = Path("fide_standard_games_by_id")
DEFAULT_OUTPUT_FILE = Path("games.parquet")
DEFAULT_ERROR_LOG = Path("gamesToParquet_errors.csv")
DEFAULT_BATCH_ROWS = 100_000

GAMES_COLUMNS: List[str] = [
    "player_id",
    "player_name",
    "fed",
    "sex",
    "standard_rating_from_list",
    "standard_games_from_list",
    "birth_year",
    "period",
    "rating_type",
    "rating_type_name",
    "event_id",
    "event_name",
    "date_from",
    "date_to",
    "player_rating",
    "event_rc",
    "event_w",
    "event_n",
    "event_chg",
    "event_k",
    "event_k_chg",
    "opponent_fide_id",
    "opponent_name",
    "opponent_rating",
    "display_opponent_rating",
    "score",
    "color",
    "star_400_rule",
]

INT_COLUMNS = {
    "player_id",
    "standard_rating_from_list",
    "standard_games_from_list",
    "birth_year",
    "rating_type",
    "event_id",
    "player_rating",
    "event_rc",
    "event_n",
    "event_k",
    "opponent_fide_id",
    "opponent_rating",
    "display_opponent_rating",
}

FLOAT_COLUMNS = {
    "event_w",
    "event_chg",
    "event_k_chg",
    "score",
}

BOOL_COLUMNS = {"star_400_rule"}

STRING_COLUMNS = set(GAMES_COLUMNS) - INT_COLUMNS - FLOAT_COLUMNS - BOOL_COLUMNS

SCHEMA = pa.schema(
    [
        pa.field(column, pa.int64())
        if column in INT_COLUMNS
        else pa.field(column, pa.float64())
        if column in FLOAT_COLUMNS
        else pa.field(column, pa.bool_())
        if column in BOOL_COLUMNS
        else pa.field(column, pa.string())
        for column in GAMES_COLUMNS
    ]
)


@dataclass
class Stats:
    real_csv_seen: int = 0
    files_read_ok: int = 0
    files_with_no_games: int = 0
    files_failed: int = 0
    rows_written: int = 0
    batches_written: int = 0
    ignored_error_csv: int = 0
    ignored_tmp_csv: int = 0
    ignored_run_errors_csv: int = 0
    ignored_other_csv: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scal fide_standard_games_by_id/<pid>.csv do pojedynczego games.parquet "
            "bez wczytywania całego datasetu do RAM."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder z plikami <pid>.csv. Domyślnie: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Docelowy plik Parquet. Domyślnie: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help=f"CSV z błędami walidacji/odczytu. Domyślnie: {DEFAULT_ERROR_LOG}",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"Ile wierszy trzymać maksymalnie przed flush do Parquet. Domyślnie: {DEFAULT_BATCH_ROWS}",
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy", "gzip", "brotli", "none"),
        default="zstd",
        help="Kompresja Parquet. Domyślnie: zstd",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pozwól nadpisać istniejący docelowy plik Parquet.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Opcjonalny limit realnych CSV do testu, np. --max-files 10000.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Przerwij przy pierwszym błędzie CSV zamiast logować błąd i iść dalej.",
    )
    return parser.parse_args()


def classify_csv(path: Path) -> str:
    name = path.name.lower()
    if not name.endswith(".csv"):
        return "other"
    if name.endswith(".tmp.csv"):
        return "tmp"
    if name == "run_errors.csv" or name.startswith("run_errors__"):
        return "run_errors"
    if name.endswith("_errors.csv"):
        return "error"
    return "real" if name[:-4].isdigit() else "other"


def collect_real_csv_files(input_dir: Path, max_files: Optional[int], stats: Stats) -> List[Path]:
    real_files: List[Path] = []

    for path in input_dir.glob("*.csv"):
        kind = classify_csv(path)
        if kind == "real":
            real_files.append(path)
        elif kind == "error":
            stats.ignored_error_csv += 1
        elif kind == "tmp":
            stats.ignored_tmp_csv += 1
        elif kind == "run_errors":
            stats.ignored_run_errors_csv += 1
        else:
            stats.ignored_other_csv += 1

    # Stable numeric ordering is helpful for repeatability and debugging.
    real_files.sort(key=lambda path: int(path.stem))
    stats.real_csv_seen = len(real_files)

    if max_files is not None:
        if max_files < 0:
            raise ValueError("--max-files musi być >= 0")
        real_files = real_files[:max_files]

    return real_files


def empty_to_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


def parse_int(value: Any) -> Optional[int]:
    text = empty_to_none(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        number = float(text.replace(",", "."))
        if not number.is_integer():
            raise ValueError(f"expected integer, got {text!r}")
        return int(number)


def parse_float(value: Any) -> Optional[float]:
    text = empty_to_none(value)
    if text is None:
        return None
    return float(text.replace(",", "."))


def parse_bool(value: Any) -> Optional[bool]:
    text = empty_to_none(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"expected boolean, got {text!r}")


def convert_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for column in GAMES_COLUMNS:
        value = raw.get(column)
        if column in INT_COLUMNS:
            converted[column] = parse_int(value)
        elif column in FLOAT_COLUMNS:
            converted[column] = parse_float(value)
        elif column in BOOL_COLUMNS:
            converted[column] = parse_bool(value)
        else:
            converted[column] = empty_to_none(value)
    return converted


def write_error_log(error_log: Path, errors: Sequence[Tuple[str, str, str]]) -> None:
    if not errors:
        if error_log.exists():
            error_log.unlink()
        return

    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "stage", "error"])
        writer.writerows(errors)


def flush_batch(
    writer: pq.ParquetWriter,
    batch_rows: List[Dict[str, Any]],
    stats: Stats,
) -> None:
    if not batch_rows:
        return
    table = pa.Table.from_pylist(batch_rows, schema=SCHEMA)
    writer.write_table(table, row_group_size=len(batch_rows))
    stats.rows_written += len(batch_rows)
    stats.batches_written += 1
    batch_rows.clear()


def create_parquet(args: argparse.Namespace) -> Stats:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Folder wejściowy nie istnieje: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"To nie jest folder: {args.input_dir}")
    if args.batch_rows < 1:
        raise ValueError("--batch-rows musi być >= 1")

    output = args.output.resolve()
    temp_output = output.with_name(output.name + ".tmp")

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Plik już istnieje: {output}\n"
            "Uruchom ponownie z --overwrite, jeśli chcesz go zastąpić."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if temp_output.exists():
        temp_output.unlink()

    stats = Stats()
    errors: List[Tuple[str, str, str]] = []

    print("========== CSV -> PARQUET ==========")
    print(f"input_dir:         {args.input_dir.resolve()}")
    print(f"output:            {output}")
    print(f"batch_rows:        {args.batch_rows}")
    print(f"compression:       {args.compression}")
    print(f"max_files:         {args.max_files if args.max_files is not None else 'ALL'}")
    print()
    print("[1/2] Skanowanie folderu...")

    scan_started = time.perf_counter()
    real_files = collect_real_csv_files(args.input_dir, args.max_files, stats)
    scan_seconds = time.perf_counter() - scan_started

    print(f"real_csv_total:    {stats.real_csv_seen}")
    print(f"files_to_process:  {len(real_files)}")
    print(f"ignored_errors:    {stats.ignored_error_csv}")
    print(f"ignored_tmp:       {stats.ignored_tmp_csv}")
    print(f"ignored_run_error: {stats.ignored_run_errors_csv}")
    print(f"ignored_other:     {stats.ignored_other_csv}")
    print(f"scan_seconds:      {scan_seconds:.3f}")
    print()
    print("[2/2] Strumieniowy zapis Parquet...")

    compression = None if args.compression == "none" else args.compression
    batch_rows: List[Dict[str, Any]] = []
    processing_started = time.perf_counter()

    try:
        with pq.ParquetWriter(
            str(temp_output),
            SCHEMA,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        ) as writer:
            for csv_path in tqdm(real_files, desc="CSV -> Parquet", unit="file"):
                try:
                    filename_player_id = int(csv_path.stem)
                    file_rows = 0

                    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                        reader = csv.DictReader(file)
                        header = reader.fieldnames or []

                        if header != GAMES_COLUMNS:
                            raise ValueError(
                                "unexpected header; expected exact GAMES_COLUMNS, got: "
                                + repr(header)
                            )

                        for row_number, raw_row in enumerate(reader, start=2):
                            try:
                                converted = convert_row(raw_row)
                            except Exception as exc:
                                raise ValueError(f"row {row_number}: {exc}") from exc

                            row_player_id = converted["player_id"]
                            if row_player_id is not None and row_player_id != filename_player_id:
                                raise ValueError(
                                    f"row {row_number}: player_id={row_player_id} "
                                    f"does not match filename pid={filename_player_id}"
                                )

                            batch_rows.append(converted)
                            file_rows += 1

                            if len(batch_rows) >= args.batch_rows:
                                flush_batch(writer, batch_rows, stats)

                    stats.files_read_ok += 1
                    if file_rows == 0:
                        stats.files_with_no_games += 1

                except Exception as exc:
                    stats.files_failed += 1
                    errors.append((str(csv_path), "read_or_validate", repr(exc)))
                    if args.strict:
                        raise

            flush_batch(writer, batch_rows, stats)

        os.replace(temp_output, output)

    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        write_error_log(args.error_log, errors)
        raise

    write_error_log(args.error_log, errors)

    processing_seconds = time.perf_counter() - processing_started
    output_size_mb = output.stat().st_size / (1024 * 1024)
    files_per_second = stats.files_read_ok / processing_seconds if processing_seconds else 0.0
    rows_per_second = stats.rows_written / processing_seconds if processing_seconds else 0.0

    print()
    print("========== DONE ==========")
    print(f"output:                 {output}")
    print(f"output_size_MB:         {output_size_mb:.1f}")
    print(f"files_read_ok:          {stats.files_read_ok}")
    print(f"files_with_no_games:    {stats.files_with_no_games}")
    print(f"files_failed:           {stats.files_failed}")
    print(f"rows_written:           {stats.rows_written}")
    print(f"row_groups_written:     {stats.batches_written}")
    print(f"processing_seconds:     {processing_seconds:.2f}")
    print(f"files_per_second:       {files_per_second:.1f}")
    print(f"rows_per_second:        {rows_per_second:.1f}")
    print(f"error_log:              {args.error_log if errors else 'none'}")
    print("==========================")

    return stats


def main() -> int:
    args = parse_args()
    try:
        stats = create_parquet(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if stats.files_failed:
        print(
            "[WARNING] Parquet utworzony, ale część plików pominięto. "
            f"Sprawdź: {args.error_log}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
