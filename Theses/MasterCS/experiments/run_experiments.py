#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

try:
    import duckdb
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from numba import njit
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Brakuje bibliotek. Zainstaluj:\n"
        "        pip install duckdb numpy pandas matplotlib tqdm pyarrow numba igraph"
    ) from exc


# =============================================================================
# CONFIG
# =============================================================================

INPUT_PARQUET = Path("games_unique.parquet")
OUTPUT_DIR = Path("experiments")
CACHE_DIR = OUTPUT_DIR / "cache"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

THREADS = 8
MEMORY_LIMIT = "8GB"

# Neutralny podział czasowy. Model nie zna reform FIDE.
REPLAY_START_MONTH = "2008-01"
POOL_DISCOVERY_END_MONTH = "2014-12"
SEARCH_TRAIN_START_MONTH = "2015-01"
SEARCH_TRAIN_END_MONTH = "2018-12"
SEARCH_VALID_START_MONTH = "2019-01"
SEARCH_VALID_END_MONTH = "2021-12"
FINAL_TEST_START_MONTH = "2022-01"
FINAL_TEST_END_MONTH = None  # None = ostatni miesiąc w danych

INITIAL_RATING = 1500.0

# Graf początkowych pul.
POOL_MIN_EDGE_GAMES = 2
POOL_MAX_GRAPH_EDGES = 1_100_000
POOL_MAX_KEPT_COMMUNITIES = 72
POOL_MIN_COMMUNITY_PLAYERS = 30
POOL_NEW_PLAYER_ASSIGNMENT_VOTES = 3

# Fitness: walidacja ogólna pozostaje najważniejsza. Część wagi pilnuje
# wspólnego podzbioru z benchmarkiem FIDE i stabilności pomiędzy latami.
FITNESS_SHARED_WEIGHT = 0.24
FITNESS_YEAR_BALANCED_WEIGHT = 0.20
FITNESS_WORST_YEAR_WEIGHT = 0.06
OVERFIT_PENALTY_WEIGHT = 0.025

# Granice numeryczne.
MIN_EXPECTED = 1e-9
MAX_EXPECTED = 1.0 - 1e-9
MIN_SCALE = 180.0
MAX_SCALE = 1000.0
RANDOM_SEED = 20260608

# Wykresy miesięczne: ograniczamy liczbę linii do czytelnych wariantów.
PLOT_SELECTED_MODELS = (
    "classic_elo",
    "advanced_seed",
    "provisional_9_seed",
    "provisional_full",
    "provisional_fixed_1500_entry",
    "provisional_without_pool",
    "provisional_without_pair_interaction",
)

# Dla tych modeli zapisujemy również dokładne miesięczne min/max ratingu
# efektywnego. To jest sanity-check, nie część fitnessu.
RATING_RANGE_PLOT_MODELS = (
    "classic_elo",
    "advanced_seed",
    "provisional_9_seed",
    "provisional_full",
    "provisional_fixed_1500_entry",
    "provisional_without_pool",
    "provisional_without_pair_interaction",
    "provisional_without_form",
)

# Miesięczne percentyle są liczone przybliżeniem histogramowym co 5 punktów.
# Końcowe percentyle w final_rating_distributions.csv są dokładne.
RATING_HISTOGRAM_MIN = -1000.0
RATING_HISTOGRAM_MAX = 5000.0
RATING_HISTOGRAM_BIN_WIDTH = 5.0
RATING_HISTOGRAM_BINS = int((RATING_HISTOGRAM_MAX - RATING_HISTOGRAM_MIN) / RATING_HISTOGRAM_BIN_WIDTH) + 1


# =============================================================================
# PROFILE WYSZUKIWANIA
# =============================================================================

@dataclass(frozen=True)
class SearchProfile:
    fast_sample_percent: int
    population: int
    generations: int
    elites: int
    workers: int
    full_finalists: int
    full_refine_rounds: int
    full_refine_candidates_per_round: int
    coordinate_refine_rounds: int
    coordinate_step_fraction: float


def search_profile(name: str) -> SearchProfile:
    cpu = max(1, os.cpu_count() or 4)
    if name == "quick":
        return SearchProfile(
            fast_sample_percent=6,
            population=14,
            generations=5,
            elites=4,
            workers=min(2, cpu),
            full_finalists=7,
            full_refine_rounds=1,
            full_refine_candidates_per_round=8,
            coordinate_refine_rounds=0,
            coordinate_step_fraction=0.04,
        )
    if name == "max":
        return SearchProfile(
            fast_sample_percent=58,
            population=144,
            generations=120,
            elites=32,
            workers=min(8, max(1, cpu // 2)),
            full_finalists=56,
            full_refine_rounds=16,
            full_refine_candidates_per_round=92,
            coordinate_refine_rounds=4,
            coordinate_step_fraction=0.028,
        )
    if name == "ultra":
        return SearchProfile(
            fast_sample_percent=48,
            population=124,
            generations=100,
            elites=28,
            workers=min(8, max(1, cpu // 2)),
            full_finalists=48,
            full_refine_rounds=13,
            full_refine_candidates_per_round=78,
            coordinate_refine_rounds=3,
            coordinate_step_fraction=0.032,
        )
    if name == "deep":
        return SearchProfile(
            fast_sample_percent=32,
            population=84,
            generations=55,
            elites=18,
            workers=min(6, max(1, cpu // 2)),
            full_finalists=30,
            full_refine_rounds=8,
            full_refine_candidates_per_round=52,
            coordinate_refine_rounds=2,
            coordinate_step_fraction=0.045,
        )
    return SearchProfile(
        fast_sample_percent=22,
        population=56,
        generations=30,
        elites=12,
        workers=min(5, max(1, cpu // 2)),
        full_finalists=20,
        full_refine_rounds=5,
        full_refine_candidates_per_round=32,
        coordinate_refine_rounds=1,
        coordinate_step_fraction=0.055,
    )


# =============================================================================
# PARAMETRY MODELU
# =============================================================================

@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    mutation_sigma_fraction: float


PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("base_k", 8.0, 42.0, 0.10),
    ParameterSpec("scale_base", 250.0, 640.0, 0.10),
    ParameterSpec("scale_level_slope", -35.0, 25.0, 0.13),
    ParameterSpec("scale_abs_diff_slope", -70.0, 110.0, 0.13),
    ParameterSpec("white_advantage", 0.0, 70.0, 0.12),
    ParameterSpec("white_advantage_level_slope", -12.0, 12.0, 0.14),
    ParameterSpec("initial_uncertainty", 0.10, 2.00, 0.13),
    ParameterSpec("uncertainty_floor", 0.005, 0.30, 0.14),
    ParameterSpec("uncertainty_cap", 0.80, 2.80, 0.13),
    ParameterSpec("uncertainty_k_weight", 0.0, 3.00, 0.15),
    ParameterSpec("opponent_uncertainty_k_damping", 0.0, 1.80, 0.15),
    ParameterSpec("prediction_uncertainty_scale_weight", 0.0, 1.80, 0.15),
    ParameterSpec("uncertainty_game_decay", 0.935, 0.9997, 0.09),
    ParameterSpec("inactivity_uncertainty_growth", 0.0, 0.160, 0.15),
    ParameterSpec("form_gain", 0.0, 55.0, 0.15),
    ParameterSpec("form_event_decay", 0.70, 0.9995, 0.12),
    ParameterSpec("form_month_decay", 0.70, 1.0000, 0.12),
    ParameterSpec("form_limit", 20.0, 280.0, 0.14),
    ParameterSpec("event_residual_norm_power", 0.0, 0.90, 0.16),
    ParameterSpec("event_residual_clip", 0.70, 6.00, 0.14),
    ParameterSpec("rating_month_reversion", 0.9970, 1.0000, 0.10),
    ParameterSpec("pool_k", 0.0, 5.00, 0.17),
    ParameterSpec("pool_weight", 0.0, 2.80, 0.16),
    ParameterSpec("pool_month_decay", 0.940, 1.000, 0.12),
    ParameterSpec("pool_limit", 20.0, 300.0, 0.14),
    ParameterSpec("pool_event_norm_power", 0.0, 0.90, 0.16),
    ParameterSpec("pool_residual_clip", 0.70, 10.0, 0.15),
    ParameterSpec("pool_exposure_gain", 0.0, 0.48, 0.18),
    ParameterSpec("pool_exposure_month_decay", 0.900, 1.000, 0.12),
    ParameterSpec("initial_home_pool_exposure", 0.10, 1.000, 0.14),
    ParameterSpec("pool_dominant_threshold", 0.05, 0.60, 0.15),
    ParameterSpec("pool_pair_k", 0.0, 4.00, 0.18),
    ParameterSpec("pool_pair_weight", 0.0, 2.40, 0.17),
    ParameterSpec("pool_pair_month_decay", 0.940, 1.000, 0.12),
    ParameterSpec("pool_pair_limit", 0.0, 180.0, 0.16),
    # Bezpieczna inicjalizacja nowych zawodników. Predykcja bieżącej partii
    # nigdy nie korzysta z jej wyniku; performance jest aktualizowany dopiero
    # po całym wydarzeniu na podstawie wyłącznie wcześniejszych obserwacji.
    ParameterSpec("provisional_games_target", 3.0, 15.0, 0.16),
    ParameterSpec("provisional_prior_games", 0.5, 18.0, 0.16),
    ParameterSpec("provisional_performance_clip", 120.0, 1000.0, 0.15),
    ParameterSpec("provisional_blend", 0.0, 1.0, 0.16),
    ParameterSpec("provisional_k_multiplier", 1.0, 4.5, 0.16),
    ParameterSpec("provisional_pool_adjustment", 0.0, 1.0, 0.15),
    # Shrinkage zabezpiecza rzadkie pary pul przed przeuczeniem.
    ParameterSpec("pool_pair_shrinkage_games", 0.0, 260.0, 0.18),
    # Centrowanie nie powinno zmieniać relatywnej informacji o pulach, ale
    # stabilizuje i ułatwia interpretację skali wyświetlanego ratingu.
    ParameterSpec("pool_centering_strength", 0.0, 1.0, 0.16),
    # Liczba jednocześnie przechowywanych ekspozycji zawodnika na pule.
    # W kernelu wartość jest zaokrąglana do 1, 2 albo 3.
    ParameterSpec("max_pool_exposures", 1.0, 3.0, 0.20),
)

PARAMETER_NAMES = tuple(spec.name for spec in PARAMETER_SPECS)
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}


def with_defaults(**updates: float) -> dict[str, float]:
    values = {
        "base_k": 20.0,
        "scale_base": 400.0,
        "scale_level_slope": 0.0,
        "scale_abs_diff_slope": 0.0,
        "white_advantage": 0.0,
        "white_advantage_level_slope": 0.0,
        "initial_uncertainty": 1.0,
        "uncertainty_floor": 0.02,
        "uncertainty_cap": 1.80,
        "uncertainty_k_weight": 0.0,
        "opponent_uncertainty_k_damping": 0.0,
        "prediction_uncertainty_scale_weight": 0.0,
        "uncertainty_game_decay": 0.98,
        "inactivity_uncertainty_growth": 0.0,
        "form_gain": 0.0,
        "form_event_decay": 0.90,
        "form_month_decay": 0.85,
        "form_limit": 180.0,
        "event_residual_norm_power": 0.0,
        "event_residual_clip": 5.0,
        "rating_month_reversion": 1.0,
        "pool_k": 0.0,
        "pool_weight": 0.0,
        "pool_month_decay": 1.0,
        "pool_limit": 200.0,
        "pool_event_norm_power": 0.0,
        "pool_residual_clip": 8.0,
        "pool_exposure_gain": 0.0,
        "pool_exposure_month_decay": 1.0,
        "initial_home_pool_exposure": 1.0,
        "pool_dominant_threshold": 0.20,
        "pool_pair_k": 0.0,
        "pool_pair_weight": 0.0,
        "pool_pair_month_decay": 1.0,
        "pool_pair_limit": 100.0,
        "provisional_games_target": 9.0,
        "provisional_prior_games": 5.0,
        "provisional_performance_clip": 650.0,
        "provisional_blend": 0.0,
        "provisional_k_multiplier": 1.0,
        "provisional_pool_adjustment": 1.0,
        "pool_pair_shrinkage_games": 0.0,
        "pool_centering_strength": 0.0,
        "max_pool_exposures": 3.0,
    }
    values.update(updates)
    return values


CLASSIC_SEED = with_defaults()

# Najlepszy model z poprzedniego eksperymentu dynamicznych pul. Dzięki temu
# nowy eksperyment startuje co najmniej z już sensownego obszaru parametrów.
PREVIOUS_DYNAMIC_SEED = with_defaults(
    base_k=30.53146140873925,
    scale_base=350.19477096636166,
    scale_level_slope=-17.55073837470563,
    white_advantage=16.88145501054188,
    initial_uncertainty=0.6306592338977994,
    uncertainty_k_weight=0.2134615039124634,
    uncertainty_game_decay=0.9511926519441497,
    inactivity_uncertainty_growth=0.08777502501743159,
    form_gain=0.0,
    form_event_decay=0.995,
    form_month_decay=0.995,
    form_limit=120.46705097708299,
    rating_month_reversion=0.998990164621496,
    pool_k=1.6221190221669999,
    pool_weight=1.0912838414813066,
    pool_month_decay=0.9980561670376281,
    pool_limit=203.64062022717206,
    pool_exposure_gain=0.2588464390523543,
    pool_exposure_month_decay=1.0,
)

PREVIOUS_DYNAMIC_SIMPLE_SEED = {
    **PREVIOUS_DYNAMIC_SEED,
    "scale_level_slope": 0.0,
    "form_gain": 0.0,
}

PAIR_POOL_SEED = {
    **PREVIOUS_DYNAMIC_SIMPLE_SEED,
    "pool_pair_k": 0.70,
    "pool_pair_weight": 0.70,
    "pool_pair_month_decay": 0.995,
    "pool_pair_limit": 70.0,
    "prediction_uncertainty_scale_weight": 0.10,
    "event_residual_norm_power": 0.10,
    "pool_event_norm_power": 0.15,
}

# Najlepszy model z poprzedniego eksperymentu advanced. Jest seedem, a nie
# wynikiem narzuconym z góry: nowe wyszukiwanie może go poprawić albo odrzucić.
ADVANCED_BEST_SEED = with_defaults(
    base_k=25.10913113579727,
    scale_base=489.1729410815662,
    scale_level_slope=-9.608107910524586,
    scale_abs_diff_slope=-55.62621839697571,
    white_advantage=15.302373303285671,
    white_advantage_level_slope=7.466195007257888,
    initial_uncertainty=1.1328954932180242,
    uncertainty_floor=0.09631443543632673,
    uncertainty_cap=1.1206204674901439,
    uncertainty_k_weight=1.9943975611259215,
    opponent_uncertainty_k_damping=0.12765364101410687,
    prediction_uncertainty_scale_weight=0.0,
    uncertainty_game_decay=0.959177112215485,
    inactivity_uncertainty_growth=0.02085960039235841,
    form_gain=0.0,  # ablation z poprzedniego eksperymentu wskazało, że forma nie pomaga
    form_event_decay=0.9258761167794987,
    form_month_decay=0.976490718466771,
    form_limit=61.7231277337726,
    event_residual_norm_power=0.0,
    event_residual_clip=4.200117704979297,
    rating_month_reversion=0.9987963316389782,
    pool_k=3.962696142295539,
    pool_weight=2.5306411760049894,
    pool_month_decay=0.9997999628204848,
    pool_limit=175.719309608425,
    pool_event_norm_power=0.4864551174210738,
    pool_residual_clip=8.273548185762142,
    pool_exposure_gain=0.014675640907683508,
    pool_exposure_month_decay=1.0,
    initial_home_pool_exposure=0.7577310295471933,
    pool_dominant_threshold=0.17713265712129422,
    pool_pair_k=3.26213369864184,
    pool_pair_weight=0.8697041073405941,
    pool_pair_month_decay=0.9934114420218301,
    pool_pair_limit=97.72821810679169,
)

# Startowe warianty nowego mechanizmu wejścia. Wyszukiwarka nie musi ich
# zachować; są tylko sensownymi punktami początkowymi.
PROVISIONAL_9_SEED = {
    **ADVANCED_BEST_SEED,
    "provisional_games_target": 9.0,
    "provisional_prior_games": 5.0,
    "provisional_performance_clip": 650.0,
    "provisional_blend": 0.72,
    "provisional_k_multiplier": 2.10,
    "provisional_pool_adjustment": 1.0,
    "pool_pair_shrinkage_games": 28.0,
    "pool_centering_strength": 1.0,
    "max_pool_exposures": 3.0,
}

PROVISIONAL_CONSERVATIVE_SEED = {
    **ADVANCED_BEST_SEED,
    "provisional_games_target": 12.0,
    "provisional_prior_games": 8.0,
    "provisional_performance_clip": 500.0,
    "provisional_blend": 0.48,
    "provisional_k_multiplier": 1.65,
    "provisional_pool_adjustment": 1.0,
    "pool_pair_shrinkage_games": 55.0,
    "pool_centering_strength": 1.0,
    "max_pool_exposures": 3.0,
}

LOCAL_BEST_PATH = RESULTS_DIR / "best_model_parameters.json"


def load_optional_local_best_seed() -> dict[str, float] | None:
    """Wczytuje najlepszy wynik poprzedniego uruchomienia tego samego skryptu.

    Dzięki temu kolejne wykonanie może kontynuować poszukiwanie bez zależności
    od starych folderów eksperymentalnych. Brak pliku podczas pierwszego startu
    jest poprawnym zachowaniem.
    """
    if not LOCAL_BEST_PATH.exists():
        return None
    try:
        payload = json.loads(LOCAL_BEST_PATH.read_text(encoding="utf-8"))
        incoming = payload.get("parameters", payload)
        values = with_defaults()
        for name in PARAMETER_NAMES:
            if name in incoming:
                values[name] = float(incoming[name])
        return values
    except Exception as exc:
        print(f"[WARN] Nie udało się wczytać lokalnego seedu {LOCAL_BEST_PATH}: {exc}")
        return None

# =============================================================================
# CLI I HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maksymalne wyszukiwanie modelu z dynamicznymi pulami, prowizorycznym performance ratingiem nowych graczy i pełnym ablation study."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "pools", "tune", "evaluate"),
        default="all",
        help="all = pule + tuning + finalna ewaluacja; domyślnie: all",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "deep", "ultra", "max"),
        default="ultra",
        help="quick = test techniczny; standard = pełny; deep = długi; ultra = domyślne 100 generacji; max = najbardziej kosztowne wyszukiwanie.",
    )
    parser.add_argument("--rebuild-pools", action="store_true", help="Wymuś ponowne wykrycie pul.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Wymuś ponowne zbudowanie cache replayu.")
    parser.add_argument("--no-pools", action="store_true", help="Wyłącz pule jako kontrolę techniczną.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Usuń folder experiments przed startem i policz wszystko od zera.",
    )
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def month_to_index(value: str | pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    return int(timestamp.year * 12 + timestamp.month - 1)


def index_to_month(value: int) -> str:
    year = value // 12
    month = value % 12 + 1
    return f"{year:04d}-{month:02d}"


def next_month(value: str) -> str:
    return (pd.Timestamp(value + "-01") + pd.offsets.MonthBegin(1)).strftime("%Y-%m")


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


CACHE_SCHEMA_VERSION = "standalone_dynamic_pool_v1"


def input_fingerprint() -> dict[str, object]:
    stat = INPUT_PARQUET.stat()
    return {
        "path": str(INPUT_PARQUET.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def load_json_or_none(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pool_cache_metadata(*, quick: bool) -> dict[str, object]:
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "type": "latent_pool_membership",
        "input": input_fingerprint(),
        "quick": bool(quick),
        "pool_discovery_end_month": POOL_DISCOVERY_END_MONTH,
        "pool_min_edge_games": int(POOL_MIN_EDGE_GAMES),
        "pool_max_graph_edges": int(180_000 if quick else POOL_MAX_GRAPH_EDGES),
        "pool_max_kept_communities": int(POOL_MAX_KEPT_COMMUNITIES),
        "pool_min_community_players": int(POOL_MIN_COMMUNITY_PLAYERS),
        "random_seed": int(RANDOM_SEED),
    }


def replay_cache_metadata(*, end_month: str, sample_percent: int | None, quick: bool) -> dict[str, object]:
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "type": "replay_cache",
        "input": input_fingerprint(),
        "quick": bool(quick),
        "replay_start_month": max(REPLAY_START_MONTH, "2016-01") if quick else REPLAY_START_MONTH,
        "end_month": str(end_month),
        "sample_percent": int(sample_percent) if sample_percent is not None else 100,
        "pool_cache": pool_cache_metadata(quick=quick),
    }


def validate_input(con: duckdb.DuckDBPyConnection, parquet: str) -> None:
    available = {
        str(row[0])
        for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet}')").fetchall()
    }
    required = {
        "game_uid",
        "month",
        "event_id",
        "event_name",
        "date_from",
        "canonical_signature",
        "duplicate_instance_index",
        "player_a_key",
        "player_b_key",
        "player_a_fide_rating",
        "player_b_fide_rating",
        "score_a",
        "color_a",
        "fide_ratings_resolved",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Brakuje kolumn w games_unique.parquet: {missing}")


def find_latest_month(con: duckdb.DuckDBPyConnection, parquet: str) -> str:
    value = con.execute(
        f"SELECT MAX(CAST(month AS DATE)) FROM read_parquet('{parquet}') WHERE month IS NOT NULL"
    ).fetchone()[0]
    if value is None:
        raise RuntimeError("Brak miesięcy w wejściowym Parquet.")
    return pd.Timestamp(value).strftime("%Y-%m")


def params_dict_to_array(values: dict[str, float]) -> np.ndarray:
    return np.asarray([float(values[name]) for name in PARAMETER_NAMES], dtype=np.float64)


def params_array_to_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(PARAMETER_NAMES)}


def clip_candidate(values: np.ndarray) -> np.ndarray:
    clipped = values.astype(np.float64, copy=True)
    for index, spec in enumerate(PARAMETER_SPECS):
        clipped[index] = min(spec.high, max(spec.low, float(clipped[index])))
    # cap nie może być niższy od floor
    floor_i = PARAMETER_INDEX["uncertainty_floor"]
    cap_i = PARAMETER_INDEX["uncertainty_cap"]
    if clipped[cap_i] < clipped[floor_i] + 0.05:
        clipped[cap_i] = min(PARAMETER_SPECS[cap_i].high, clipped[floor_i] + 0.05)

    # Dwa parametry architektoniczne są dyskretne. Algorytm ewolucyjny nadal
    # może je optymalizować, ale replay otrzymuje czytelne wartości całkowite.
    clipped[PARAMETER_INDEX["provisional_games_target"]] = float(
        int(round(clipped[PARAMETER_INDEX["provisional_games_target"]]))
    )
    clipped[PARAMETER_INDEX["max_pool_exposures"]] = float(
        int(round(clipped[PARAMETER_INDEX["max_pool_exposures"]]))
    )
    return clipped


def random_candidate(rng: np.random.Generator) -> np.ndarray:
    return clip_candidate(
        np.asarray([rng.uniform(spec.low, spec.high) for spec in PARAMETER_SPECS], dtype=np.float64)
    )


def arrow_reader(con: duckdb.DuckDBPyConnection, query: str, params: list[object], batch_size: int):
    """Preferuje nowe API DuckDB, ale zachowuje kompatybilność ze starszym klientem."""
    try:
        relation = con.sql(query, params=params)
        return relation.to_arrow_reader(batch_size)
    except (AttributeError, TypeError):
        return con.execute(query, params).fetch_record_batch(batch_size)


# =============================================================================
# LATENT POOLS
# =============================================================================

POOL_MAP_PATH = CACHE_DIR / "latent_pool_membership.csv"
POOL_MAP_META_PATH = CACHE_DIR / "latent_pool_membership.meta.json"
POOL_SUMMARY_PATH = RESULTS_DIR / "latent_pool_summary.csv"


def build_latent_pools(
    con: duckdb.DuckDBPyConnection,
    parquet: str,
    *,
    force: bool,
    quick: bool,
    disabled: bool,
) -> pd.DataFrame:
    if disabled:
        print("[POOLS] Wyłączone przez --no-pools. Każdy zawodnik trafia do pool_id=0.")
        return pd.DataFrame(columns=["player_key", "pool_id", "community_players"])

    expected_metadata = pool_cache_metadata(quick=quick)
    if POOL_MAP_PATH.exists() and not force:
        if load_json_or_none(POOL_MAP_META_PATH) == expected_metadata:
            print(f"[POOLS] Loading cached membership: {POOL_MAP_PATH}")
            return pd.read_csv(POOL_MAP_PATH, dtype={"player_key": str, "pool_id": int})
        print("[POOLS] Cache pul jest nieaktualny albo pochodzi z innej konfiguracji. Przeliczam.")

    try:
        import igraph as ig
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] Do pul potrzebna jest biblioteka igraph.\n"
            "        Zainstaluj: pip install igraph"
        ) from exc

    # Louvain może korzystać z losowości. Ustawienie generatora zwiększa
    # odtwarzalność kolejnych uruchomień.
    try:
        ig.set_random_number_generator(random.Random(RANDOM_SEED))
    except Exception:
        pass

    max_edges = 180_000 if quick else POOL_MAX_GRAPH_EDGES
    print("[POOLS] Building historical interaction graph...")
    print(f"[POOLS] discovery end:      {POOL_DISCOVERY_END_MONTH}")
    print(f"[POOLS] edge limit:         {max_edges:,}")
    started = time.perf_counter()

    edges = con.execute(
        f"""
        WITH aggregated AS (
            SELECT
                CASE WHEN player_a_key <= player_b_key THEN CAST(player_a_key AS VARCHAR)
                     ELSE CAST(player_b_key AS VARCHAR) END AS source,
                CASE WHEN player_a_key <= player_b_key THEN CAST(player_b_key AS VARCHAR)
                     ELSE CAST(player_a_key AS VARCHAR) END AS target,
                COUNT(*)::BIGINT AS games
            FROM read_parquet('{parquet}')
            WHERE month IS NOT NULL
              AND CAST(month AS DATE) < CAST(? AS DATE)
              AND player_a_key IS NOT NULL
              AND player_b_key IS NOT NULL
              AND player_a_key <> player_b_key
            GROUP BY source, target
            HAVING COUNT(*) >= ?
        )
        SELECT source, target, games
        FROM aggregated
        ORDER BY games DESC, source, target
        LIMIT ?
        """,
        [next_month(POOL_DISCOVERY_END_MONTH) + "-01", POOL_MIN_EDGE_GAMES, max_edges],
    ).fetchdf()

    if edges.empty:
        raise RuntimeError("Nie udało się zbudować grafu pul: brak krawędzi.")

    print(f"[POOLS] aggregated edges:   {len(edges):,}")
    print("[POOLS] Running Louvain community detection...")

    graph = ig.Graph.TupleList(
        edges[["source", "target", "games"]].itertuples(index=False, name=None),
        directed=False,
        weights=True,
        vertex_name_attr="name",
    )
    communities = graph.community_multilevel(weights="weight")
    raw_sizes = communities.sizes()
    ranked_raw_ids = sorted(range(len(raw_sizes)), key=lambda raw_id: raw_sizes[raw_id], reverse=True)
    kept_raw_ids = [
        raw_id for raw_id in ranked_raw_ids if raw_sizes[raw_id] >= POOL_MIN_COMMUNITY_PLAYERS
    ][:POOL_MAX_KEPT_COMMUNITIES]
    raw_to_pool = {raw_id: index + 1 for index, raw_id in enumerate(kept_raw_ids)}

    rows: list[dict[str, object]] = []
    for vertex_id, raw_community in enumerate(communities.membership):
        pool_id = raw_to_pool.get(int(raw_community), 0)
        rows.append(
            {
                "player_key": str(graph.vs[vertex_id]["name"]),
                "pool_id": int(pool_id),
                "community_players": int(raw_sizes[int(raw_community)]),
            }
        )

    membership = pd.DataFrame(rows)
    membership.to_csv(POOL_MAP_PATH, index=False, encoding="utf-8-sig")
    save_json(POOL_MAP_META_PATH, expected_metadata)
    summary = (
        membership.groupby("pool_id", as_index=False)
        .agg(players=("player_key", "count"), source_community_players=("community_players", "max"))
        .sort_values("pool_id")
    )
    summary.to_csv(POOL_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - started
    print(f"[POOLS] vertices:           {graph.vcount():,}")
    print(f"[POOLS] raw communities:    {len(raw_sizes):,}")
    print(f"[POOLS] kept communities:   {len(kept_raw_ids):,} + fallback pool 0")
    print(f"[POOLS] done in:            {elapsed / 60.0:.2f} min")
    print(f"[POOLS] map:                {POOL_MAP_PATH}")
    return membership


# =============================================================================
# REPLAY CACHE
# =============================================================================

@dataclass
class ReplayCache:
    player_a: np.ndarray
    player_b: np.ndarray
    score_a: np.ndarray
    white_sign: np.ndarray
    event_group: np.ndarray
    month_index: np.ndarray
    scope: np.ndarray
    validation_fold: np.ndarray
    fide_resolved: np.ndarray
    fide_rating_a: np.ndarray
    fide_rating_b: np.ndarray
    static_pool: np.ndarray
    player_keys: list[str]
    first_month_index: int
    last_month_index: int

    @property
    def games(self) -> int:
        return int(len(self.score_a))

    @property
    def players(self) -> int:
        return int(len(self.player_keys))

    @property
    def pools(self) -> int:
        return int(self.static_pool.max()) + 1 if len(self.static_pool) else 1


def fast_cache_path(profile_name: str, sample_percent: int) -> Path:
    history = "quick" if profile_name == "quick" else "fullhistory"
    return CACHE_DIR / f"tuning_fast_{history}_{sample_percent:03d}pct.npz"


def full_tune_cache_path(profile_name: str) -> Path:
    history = "quick" if profile_name == "quick" else "fullhistory"
    return CACHE_DIR / f"tuning_full_{history}.npz"


def final_cache_path(profile_name: str, end_month: str) -> Path:
    history = "quick" if profile_name == "quick" else "fullhistory"
    return CACHE_DIR / f"final_{history}_to_{end_month}.npz"


def scope_code(month: str) -> int:
    if SEARCH_TRAIN_START_MONTH <= month <= SEARCH_TRAIN_END_MONTH:
        return 1
    if SEARCH_VALID_START_MONTH <= month <= SEARCH_VALID_END_MONTH:
        return 2
    if month >= FINAL_TEST_START_MONTH:
        return 3
    return 0


def validation_fold_code(month: str) -> int:
    if SEARCH_VALID_START_MONTH <= month <= SEARCH_VALID_END_MONTH:
        return int(month[:4]) - int(SEARCH_VALID_START_MONTH[:4]) + 1
    return 0


def color_to_sign(value: object) -> int:
    normalized = str(value or "").strip().lower()
    if normalized in {"white", "w", "biale", "biały", "bialy"}:
        return 1
    if normalized in {"black", "b", "czarne", "czarny"}:
        return -1
    return 0


def save_cache(path: Path, cache: ReplayCache) -> None:
    np.savez_compressed(
        path,
        player_a=cache.player_a,
        player_b=cache.player_b,
        score_a=cache.score_a,
        white_sign=cache.white_sign,
        event_group=cache.event_group,
        month_index=cache.month_index,
        scope=cache.scope,
        validation_fold=cache.validation_fold,
        fide_resolved=cache.fide_resolved,
        fide_rating_a=cache.fide_rating_a,
        fide_rating_b=cache.fide_rating_b,
        static_pool=cache.static_pool,
        player_keys=np.asarray(cache.player_keys, dtype=object),
        first_month_index=np.asarray([cache.first_month_index], dtype=np.int32),
        last_month_index=np.asarray([cache.last_month_index], dtype=np.int32),
    )


def load_cache(path: Path) -> ReplayCache:
    with np.load(path, allow_pickle=True) as loaded:
        return ReplayCache(
            player_a=loaded["player_a"],
            player_b=loaded["player_b"],
            score_a=loaded["score_a"],
            white_sign=loaded["white_sign"],
            event_group=loaded["event_group"],
            month_index=loaded["month_index"],
            scope=loaded["scope"],
            validation_fold=loaded["validation_fold"],
            fide_resolved=loaded["fide_resolved"],
            fide_rating_a=loaded["fide_rating_a"],
            fide_rating_b=loaded["fide_rating_b"],
            static_pool=loaded["static_pool"],
            player_keys=[str(value) for value in loaded["player_keys"].tolist()],
            first_month_index=int(loaded["first_month_index"][0]),
            last_month_index=int(loaded["last_month_index"][0]),
        )


def build_replay_cache(
    con: duckdb.DuckDBPyConnection,
    parquet: str,
    membership: pd.DataFrame,
    *,
    output_path: Path,
    end_month: str,
    sample_percent: int | None,
    force: bool,
    quick: bool,
) -> ReplayCache:
    expected_metadata = replay_cache_metadata(
        end_month=end_month,
        sample_percent=sample_percent,
        quick=quick,
    )
    cache_metadata_path = metadata_path(output_path)
    if output_path.exists() and not force:
        if load_json_or_none(cache_metadata_path) == expected_metadata:
            print(f"[CACHE] Loading cached replay: {output_path}")
            cache = load_cache(output_path)
            print(f"[CACHE] games={cache.games:,}, players={cache.players:,}, pools={cache.pools:,}")
            return cache
        print(f"[CACHE] Cache {output_path.name} jest nieaktualny albo pochodzi z innej konfiguracji. Przeliczam.")

    print(f"[CACHE] Building:           {output_path.name}")
    print(f"[CACHE] range:              {REPLAY_START_MONTH} .. {end_month}")
    print(f"[CACHE] sample percent:     {sample_percent if sample_percent is not None else 100}%")
    started = time.perf_counter()

    start_month = REPLAY_START_MONTH
    if quick:
        start_month = max(start_month, "2016-01")

    sample_sql = ""
    params: list[object] = [start_month + "-01", next_month(end_month) + "-01"]
    if sample_percent is not None and sample_percent < 100:
        sample_sql = "AND ABS(HASH(game_uid)) % 100 < ?"
        params.append(int(sample_percent))

    where_sql = f"""
        month IS NOT NULL
        AND CAST(month AS DATE) >= CAST(? AS DATE)
        AND CAST(month AS DATE) < CAST(? AS DATE)
        {sample_sql}
    """

    total_rows = int(
        con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet}') WHERE {where_sql}", params).fetchone()[0]
    )
    if total_rows <= 0:
        raise RuntimeError("Replay cache jest pusty.")

    query = f"""
        SELECT
            CAST(month AS DATE) AS month,
            COALESCE(CAST(date_from AS VARCHAR), '') AS date_from,
            COALESCE(
                'id:' || CAST(event_id AS VARCHAR),
                'name:' || COALESCE(CAST(event_name AS VARCHAR), ''),
                'unknown'
            ) AS event_key,
            CAST(player_a_key AS VARCHAR) AS player_a_key,
            CAST(player_b_key AS VARCHAR) AS player_b_key,
            CAST(score_a AS DOUBLE) AS score_a,
            COALESCE(CAST(color_a AS VARCHAR), '') AS color_a,
            COALESCE(CAST(fide_ratings_resolved AS BOOLEAN), FALSE) AS fide_resolved,
            CAST(player_a_fide_rating AS DOUBLE) AS fide_rating_a,
            CAST(player_b_fide_rating AS DOUBLE) AS fide_rating_b
        FROM read_parquet('{parquet}')
        WHERE {where_sql}
        ORDER BY month, date_from, event_key, canonical_signature, duplicate_instance_index
    """

    pool_lookup: dict[str, int] = {}
    if not membership.empty:
        pool_lookup = {
            str(row.player_key): int(row.pool_id)
            for row in membership[["player_key", "pool_id"]].itertuples(index=False)
        }

    player_to_id: dict[str, int] = {}
    player_keys: list[str] = []

    def encode_player(key: object) -> int:
        normalized = str(key)
        existing = player_to_id.get(normalized)
        if existing is not None:
            return existing
        value = len(player_keys)
        player_to_id[normalized] = value
        player_keys.append(normalized)
        return value

    player_a = np.empty(total_rows, dtype=np.int32)
    player_b = np.empty(total_rows, dtype=np.int32)
    score_values = np.empty(total_rows, dtype=np.float64)
    white_values = np.empty(total_rows, dtype=np.int8)
    event_values = np.empty(total_rows, dtype=np.int32)
    month_values = np.empty(total_rows, dtype=np.int32)
    scope_values = np.empty(total_rows, dtype=np.int8)
    fold_values = np.empty(total_rows, dtype=np.int8)
    fide_resolved_values = np.empty(total_rows, dtype=np.bool_)
    fide_a_values = np.empty(total_rows, dtype=np.float64)
    fide_b_values = np.empty(total_rows, dtype=np.float64)

    reader = arrow_reader(con, query, params, 250_000)
    position = 0
    current_group = -1
    previous_event: tuple[int, str, str] | None = None

    progress = tqdm(total=total_rows, desc=f"building {output_path.name}", unit="game", unit_scale=True)
    try:
        for batch in reader:
            columns = batch.to_pydict()
            batch_rows = int(batch.num_rows)
            for local_index in range(batch_rows):
                global_index = position + local_index
                month_object = columns["month"][local_index]
                month_id = int(month_object.year * 12 + month_object.month - 1)
                month_string = f"{month_object.year:04d}-{month_object.month:02d}"
                date_from_value = str(columns["date_from"][local_index] or "")
                event_key_value = str(columns["event_key"][local_index] or "")
                identity = (month_id, date_from_value, event_key_value)
                if identity != previous_event:
                    current_group += 1
                    previous_event = identity

                player_a[global_index] = encode_player(columns["player_a_key"][local_index])
                player_b[global_index] = encode_player(columns["player_b_key"][local_index])
                score_values[global_index] = float(columns["score_a"][local_index])
                white_values[global_index] = color_to_sign(columns["color_a"][local_index])
                event_values[global_index] = current_group
                month_values[global_index] = month_id
                scope_values[global_index] = scope_code(month_string)
                fold_values[global_index] = validation_fold_code(month_string)
                fide_resolved_values[global_index] = bool(columns["fide_resolved"][local_index])
                fide_a = columns["fide_rating_a"][local_index]
                fide_b = columns["fide_rating_b"][local_index]
                fide_a_values[global_index] = float(fide_a) if fide_a is not None else np.nan
                fide_b_values[global_index] = float(fide_b) if fide_b is not None else np.nan

            position += batch_rows
            progress.update(batch_rows)
    finally:
        progress.close()

    if position != total_rows:
        raise RuntimeError(f"Niespójna liczba rekordów cache: oczekiwano {total_rows}, zapisano {position}")

    static_pool = np.zeros(len(player_keys), dtype=np.int32)
    for key, player_id in player_to_id.items():
        static_pool[player_id] = int(pool_lookup.get(key, 0))

    cache = ReplayCache(
        player_a=player_a,
        player_b=player_b,
        score_a=score_values,
        white_sign=white_values,
        event_group=event_values,
        month_index=month_values,
        scope=scope_values,
        validation_fold=fold_values,
        fide_resolved=fide_resolved_values,
        fide_rating_a=fide_a_values,
        fide_rating_b=fide_b_values,
        static_pool=static_pool,
        player_keys=player_keys,
        first_month_index=int(month_values.min()),
        last_month_index=int(month_values.max()),
    )
    save_cache(output_path, cache)
    save_json(cache_metadata_path, expected_metadata)

    elapsed = time.perf_counter() - started
    print(f"[CACHE] games:              {cache.games:,}")
    print(f"[CACHE] players:            {cache.players:,}")
    print(f"[CACHE] pools:              {cache.pools:,}")
    print(f"[CACHE] events:             {int(cache.event_group.max()) + 1:,}")
    print(f"[CACHE] done in:            {elapsed / 60.0:.2f} min")
    return cache


# =============================================================================
# NUMBA REPLAY KERNEL
# =============================================================================

# Indeksy parametrów.
P_BASE_K = PARAMETER_INDEX["base_k"]
P_SCALE_BASE = PARAMETER_INDEX["scale_base"]
P_SCALE_LEVEL_SLOPE = PARAMETER_INDEX["scale_level_slope"]
P_SCALE_ABS_DIFF_SLOPE = PARAMETER_INDEX["scale_abs_diff_slope"]
P_WHITE_ADVANTAGE = PARAMETER_INDEX["white_advantage"]
P_WHITE_ADVANTAGE_LEVEL_SLOPE = PARAMETER_INDEX["white_advantage_level_slope"]
P_INITIAL_UNCERTAINTY = PARAMETER_INDEX["initial_uncertainty"]
P_UNCERTAINTY_FLOOR = PARAMETER_INDEX["uncertainty_floor"]
P_UNCERTAINTY_CAP = PARAMETER_INDEX["uncertainty_cap"]
P_UNCERTAINTY_K_WEIGHT = PARAMETER_INDEX["uncertainty_k_weight"]
P_OPPONENT_UNCERTAINTY_K_DAMPING = PARAMETER_INDEX["opponent_uncertainty_k_damping"]
P_PREDICTION_UNCERTAINTY_SCALE_WEIGHT = PARAMETER_INDEX["prediction_uncertainty_scale_weight"]
P_UNCERTAINTY_GAME_DECAY = PARAMETER_INDEX["uncertainty_game_decay"]
P_INACTIVITY_UNCERTAINTY_GROWTH = PARAMETER_INDEX["inactivity_uncertainty_growth"]
P_FORM_GAIN = PARAMETER_INDEX["form_gain"]
P_FORM_EVENT_DECAY = PARAMETER_INDEX["form_event_decay"]
P_FORM_MONTH_DECAY = PARAMETER_INDEX["form_month_decay"]
P_FORM_LIMIT = PARAMETER_INDEX["form_limit"]
P_EVENT_RESIDUAL_NORM_POWER = PARAMETER_INDEX["event_residual_norm_power"]
P_EVENT_RESIDUAL_CLIP = PARAMETER_INDEX["event_residual_clip"]
P_RATING_MONTH_REVERSION = PARAMETER_INDEX["rating_month_reversion"]
P_POOL_K = PARAMETER_INDEX["pool_k"]
P_POOL_WEIGHT = PARAMETER_INDEX["pool_weight"]
P_POOL_MONTH_DECAY = PARAMETER_INDEX["pool_month_decay"]
P_POOL_LIMIT = PARAMETER_INDEX["pool_limit"]
P_POOL_EVENT_NORM_POWER = PARAMETER_INDEX["pool_event_norm_power"]
P_POOL_RESIDUAL_CLIP = PARAMETER_INDEX["pool_residual_clip"]
P_POOL_EXPOSURE_GAIN = PARAMETER_INDEX["pool_exposure_gain"]
P_POOL_EXPOSURE_MONTH_DECAY = PARAMETER_INDEX["pool_exposure_month_decay"]
P_INITIAL_HOME_POOL_EXPOSURE = PARAMETER_INDEX["initial_home_pool_exposure"]
P_POOL_DOMINANT_THRESHOLD = PARAMETER_INDEX["pool_dominant_threshold"]
P_POOL_PAIR_K = PARAMETER_INDEX["pool_pair_k"]
P_POOL_PAIR_WEIGHT = PARAMETER_INDEX["pool_pair_weight"]
P_POOL_PAIR_MONTH_DECAY = PARAMETER_INDEX["pool_pair_month_decay"]
P_POOL_PAIR_LIMIT = PARAMETER_INDEX["pool_pair_limit"]
P_PROVISIONAL_GAMES_TARGET = PARAMETER_INDEX["provisional_games_target"]
P_PROVISIONAL_PRIOR_GAMES = PARAMETER_INDEX["provisional_prior_games"]
P_PROVISIONAL_PERFORMANCE_CLIP = PARAMETER_INDEX["provisional_performance_clip"]
P_PROVISIONAL_BLEND = PARAMETER_INDEX["provisional_blend"]
P_PROVISIONAL_K_MULTIPLIER = PARAMETER_INDEX["provisional_k_multiplier"]
P_PROVISIONAL_POOL_ADJUSTMENT = PARAMETER_INDEX["provisional_pool_adjustment"]
P_POOL_PAIR_SHRINKAGE_GAMES = PARAMETER_INDEX["pool_pair_shrinkage_games"]
P_POOL_CENTERING_STRENGTH = PARAMETER_INDEX["pool_centering_strength"]
P_MAX_POOL_EXPOSURES = PARAMETER_INDEX["max_pool_exposures"]


@njit(cache=True, nogil=True)
def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


@njit(cache=True, nogil=True)
def _expected_points(diff: float, scale: float) -> float:
    scale = _clip(scale, MIN_SCALE, MAX_SCALE)
    exponent = _clip(-diff / scale, -20.0, 20.0)
    predicted = 1.0 / (1.0 + 10.0 ** exponent)
    return _clip(predicted, MIN_EXPECTED, MAX_EXPECTED)


@njit(cache=True, nogil=True)
def _performance_diff_from_score(score_fraction: float, scale: float, limit: float) -> float:
    """Logistyczny performance rating z bezpiecznym clippingiem.

    score_fraction jest wcześniej wygładzony priorem, więc nie osiąga dokładnie
    zera ani jedynki. Funkcja działa wyłącznie po zakończeniu wydarzenia.
    """
    p = _clip(score_fraction, 1e-6, 1.0 - 1e-6)
    diff = scale * math.log10(p / (1.0 - p))
    return _clip(diff, -limit, limit)


@njit(cache=True, nogil=True)
def _center_pool_offsets(pool_offset: np.ndarray, strength: float, limit: float) -> None:
    """Częściowo centruje offsety pul wokół zera dla stabilnej skali."""
    if strength <= 0.0 or len(pool_offset) <= 1:
        return
    total = 0.0
    count = 0
    for pool_id in range(1, len(pool_offset)):
        total += pool_offset[pool_id]
        count += 1
    if count <= 0:
        return
    shift = strength * (total / count)
    for pool_id in range(1, len(pool_offset)):
        pool_offset[pool_id] = _clip(pool_offset[pool_id] - shift, -limit, limit)


@njit(cache=True, nogil=True)
def _pool_value(
    player: int,
    exposure_pool_1: np.ndarray,
    exposure_pool_2: np.ndarray,
    exposure_pool_3: np.ndarray,
    exposure_weight_1: np.ndarray,
    exposure_weight_2: np.ndarray,
    exposure_weight_3: np.ndarray,
    pool_offset: np.ndarray,
) -> float:
    value = 0.0
    weight_sum = 0.0
    p1, p2, p3 = exposure_pool_1[player], exposure_pool_2[player], exposure_pool_3[player]
    w1, w2, w3 = exposure_weight_1[player], exposure_weight_2[player], exposure_weight_3[player]
    if p1 > 0 and w1 > 0.0:
        value += w1 * pool_offset[p1]
        weight_sum += w1
    if p2 > 0 and p2 != p1 and w2 > 0.0:
        value += w2 * pool_offset[p2]
        weight_sum += w2
    if p3 > 0 and p3 != p1 and p3 != p2 and w3 > 0.0:
        value += w3 * pool_offset[p3]
        weight_sum += w3
    if weight_sum <= 1e-12:
        return 0.0
    if weight_sum > 1.0:
        return value / weight_sum
    return value


@njit(cache=True, nogil=True)
def _dominant_pool(
    player: int,
    player_pool: np.ndarray,
    exposure_pool_1: np.ndarray,
    exposure_pool_2: np.ndarray,
    exposure_pool_3: np.ndarray,
    exposure_weight_1: np.ndarray,
    exposure_weight_2: np.ndarray,
    exposure_weight_3: np.ndarray,
    threshold: float,
) -> int:
    p1, p2, p3 = exposure_pool_1[player], exposure_pool_2[player], exposure_pool_3[player]
    w1, w2, w3 = exposure_weight_1[player], exposure_weight_2[player], exposure_weight_3[player]
    best_pool = player_pool[player]
    best_weight = threshold
    if p1 > 0 and w1 >= best_weight:
        best_pool, best_weight = p1, w1
    if p2 > 0 and w2 >= best_weight:
        best_pool, best_weight = p2, w2
    if p3 > 0 and w3 >= best_weight:
        best_pool = p3
    return best_pool


@njit(cache=True, nogil=True)
def _add_vote(
    player: int,
    target_pool: int,
    vote_pool_1: np.ndarray,
    vote_pool_2: np.ndarray,
    vote_pool_3: np.ndarray,
    vote_count_1: np.ndarray,
    vote_count_2: np.ndarray,
    vote_count_3: np.ndarray,
) -> None:
    if target_pool <= 0:
        return
    if vote_pool_1[player] == target_pool:
        vote_count_1[player] += 1
        return
    if vote_pool_2[player] == target_pool:
        vote_count_2[player] += 1
        return
    if vote_pool_3[player] == target_pool:
        vote_count_3[player] += 1
        return
    if vote_pool_1[player] == 0:
        vote_pool_1[player] = target_pool
        vote_count_1[player] = 1
        return
    if vote_pool_2[player] == 0:
        vote_pool_2[player] = target_pool
        vote_count_2[player] = 1
        return
    if vote_pool_3[player] == 0:
        vote_pool_3[player] = target_pool
        vote_count_3[player] = 1
        return
    # Gdy wydarzenie zawiera kontakt z >3 pulami, zachowujemy trzy najczęstsze.
    if vote_count_1[player] <= vote_count_2[player] and vote_count_1[player] <= vote_count_3[player]:
        vote_pool_1[player] = target_pool
        vote_count_1[player] = 1
    elif vote_count_2[player] <= vote_count_3[player]:
        vote_pool_2[player] = target_pool
        vote_count_2[player] = 1
    else:
        vote_pool_3[player] = target_pool
        vote_count_3[player] = 1


@njit(cache=True, nogil=True)
def _sort_exposures(
    player: int,
    exposure_pool_1: np.ndarray,
    exposure_pool_2: np.ndarray,
    exposure_pool_3: np.ndarray,
    exposure_weight_1: np.ndarray,
    exposure_weight_2: np.ndarray,
    exposure_weight_3: np.ndarray,
) -> None:
    # Niewielki sort trzech slotów malejąco po wadze.
    if exposure_weight_2[player] > exposure_weight_1[player]:
        exposure_pool_1[player], exposure_pool_2[player] = exposure_pool_2[player], exposure_pool_1[player]
        exposure_weight_1[player], exposure_weight_2[player] = exposure_weight_2[player], exposure_weight_1[player]
    if exposure_weight_3[player] > exposure_weight_2[player]:
        exposure_pool_2[player], exposure_pool_3[player] = exposure_pool_3[player], exposure_pool_2[player]
        exposure_weight_2[player], exposure_weight_3[player] = exposure_weight_3[player], exposure_weight_2[player]
    if exposure_weight_2[player] > exposure_weight_1[player]:
        exposure_pool_1[player], exposure_pool_2[player] = exposure_pool_2[player], exposure_pool_1[player]
        exposure_weight_1[player], exposure_weight_2[player] = exposure_weight_2[player], exposure_weight_1[player]


@njit(cache=True, nogil=True)
def _add_or_increase_exposure(
    player: int,
    target_pool: int,
    amount: float,
    exposure_pool_1: np.ndarray,
    exposure_pool_2: np.ndarray,
    exposure_pool_3: np.ndarray,
    exposure_weight_1: np.ndarray,
    exposure_weight_2: np.ndarray,
    exposure_weight_3: np.ndarray,
) -> None:
    if target_pool <= 0 or amount <= 0.0:
        return
    if exposure_pool_1[player] == target_pool:
        exposure_weight_1[player] += amount
    elif exposure_pool_2[player] == target_pool:
        exposure_weight_2[player] += amount
    elif exposure_pool_3[player] == target_pool:
        exposure_weight_3[player] += amount
    elif exposure_pool_1[player] == 0 or exposure_weight_1[player] <= exposure_weight_2[player] and exposure_weight_1[player] <= exposure_weight_3[player]:
        exposure_pool_1[player] = target_pool
        exposure_weight_1[player] = amount
    elif exposure_pool_2[player] == 0 or exposure_weight_2[player] <= exposure_weight_3[player]:
        exposure_pool_2[player] = target_pool
        exposure_weight_2[player] = amount
    else:
        exposure_pool_3[player] = target_pool
        exposure_weight_3[player] = amount


@njit(cache=True, nogil=True)
def _apply_exposure_votes(
    player: int,
    gain: float,
    max_exposures: int,
    vote_pool_1: np.ndarray,
    vote_pool_2: np.ndarray,
    vote_pool_3: np.ndarray,
    vote_count_1: np.ndarray,
    vote_count_2: np.ndarray,
    vote_count_3: np.ndarray,
    exposure_pool_1: np.ndarray,
    exposure_pool_2: np.ndarray,
    exposure_pool_3: np.ndarray,
    exposure_weight_1: np.ndarray,
    exposure_weight_2: np.ndarray,
    exposure_weight_3: np.ndarray,
) -> int:
    total = vote_count_1[player] + vote_count_2[player] + vote_count_3[player]
    if total <= 0:
        return 0
    if gain > 0.0:
        gain = min(gain, 0.60)
        keep = (1.0 - gain) ** total
        exposure_weight_1[player] *= keep
        exposure_weight_2[player] *= keep
        exposure_weight_3[player] *= keep
        _add_or_increase_exposure(player, vote_pool_1[player], gain * vote_count_1[player], exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3)
        _add_or_increase_exposure(player, vote_pool_2[player], gain * vote_count_2[player], exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3)
        _add_or_increase_exposure(player, vote_pool_3[player], gain * vote_count_3[player], exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3)
        _sort_exposures(player, exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3)
        if max_exposures <= 1:
            exposure_pool_2[player] = 0
            exposure_pool_3[player] = 0
            exposure_weight_2[player] = 0.0
            exposure_weight_3[player] = 0.0
        elif max_exposures == 2:
            exposure_pool_3[player] = 0
            exposure_weight_3[player] = 0.0
        weight_sum = exposure_weight_1[player] + exposure_weight_2[player] + exposure_weight_3[player]
        if weight_sum > 1.0:
            exposure_weight_1[player] /= weight_sum
            exposure_weight_2[player] /= weight_sum
            exposure_weight_3[player] /= weight_sum

    # Zwracamy najczęściej spotkaną pulę wydarzenia do propozycji home pool.
    best_pool = vote_pool_1[player]
    best_count = vote_count_1[player]
    if vote_count_2[player] > best_count:
        best_pool, best_count = vote_pool_2[player], vote_count_2[player]
    if vote_count_3[player] > best_count:
        best_pool = vote_pool_3[player]

    vote_pool_1[player] = 0
    vote_pool_2[player] = 0
    vote_pool_3[player] = 0
    vote_count_1[player] = 0
    vote_count_2[player] = 0
    vote_count_3[player] = 0
    return best_pool


@njit(cache=True, nogil=True)
def _histogram_quantile(histogram: np.ndarray, count: int, quantile: float) -> float:
    if count <= 0:
        return np.nan
    target = quantile * max(0, count - 1)
    cumulative = 0
    for index in range(len(histogram)):
        cumulative += histogram[index]
        if cumulative > target:
            return RATING_HISTOGRAM_MIN + (index + 0.5) * RATING_HISTOGRAM_BIN_WIDTH
    return RATING_HISTOGRAM_MAX


@njit(cache=True, nogil=True)
def _record_monthly_rating_distribution(
    slot: int,
    rating: np.ndarray,
    form: np.ndarray,
    seen: np.ndarray,
    exposure_pool_1: np.ndarray,
    exposure_pool_2: np.ndarray,
    exposure_pool_3: np.ndarray,
    exposure_weight_1: np.ndarray,
    exposure_weight_2: np.ndarray,
    exposure_weight_3: np.ndarray,
    pool_offset: np.ndarray,
    pool_weight: float,
    monthly_rating_min: np.ndarray,
    monthly_rating_p01: np.ndarray,
    monthly_rating_p10: np.ndarray,
    monthly_rating_p25: np.ndarray,
    monthly_rating_p50: np.ndarray,
    monthly_rating_p75: np.ndarray,
    monthly_rating_p90: np.ndarray,
    monthly_rating_p99: np.ndarray,
    monthly_rating_max: np.ndarray,
    monthly_rating_seen_players: np.ndarray,
) -> None:
    if slot < 0 or slot >= len(monthly_rating_min):
        return
    histogram = np.zeros(RATING_HISTOGRAM_BINS, dtype=np.int64)
    found = False
    minimum = 0.0
    maximum = 0.0
    count = 0
    for player in range(len(rating)):
        if seen[player] == 0:
            continue
        value = (
            rating[player]
            + form[player]
            + pool_weight * _pool_value(
                player,
                exposure_pool_1, exposure_pool_2, exposure_pool_3,
                exposure_weight_1, exposure_weight_2, exposure_weight_3,
                pool_offset,
            )
        )
        if not found:
            minimum = value
            maximum = value
            found = True
        else:
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        index = int((value - RATING_HISTOGRAM_MIN) / RATING_HISTOGRAM_BIN_WIDTH)
        if index < 0:
            index = 0
        elif index >= RATING_HISTOGRAM_BINS:
            index = RATING_HISTOGRAM_BINS - 1
        histogram[index] += 1
        count += 1
    if found:
        monthly_rating_min[slot] = minimum
        monthly_rating_p01[slot] = _histogram_quantile(histogram, count, 0.01)
        monthly_rating_p10[slot] = _histogram_quantile(histogram, count, 0.10)
        monthly_rating_p25[slot] = _histogram_quantile(histogram, count, 0.25)
        monthly_rating_p50[slot] = _histogram_quantile(histogram, count, 0.50)
        monthly_rating_p75[slot] = _histogram_quantile(histogram, count, 0.75)
        monthly_rating_p90[slot] = _histogram_quantile(histogram, count, 0.90)
        monthly_rating_p99[slot] = _histogram_quantile(histogram, count, 0.99)
        monthly_rating_max[slot] = maximum
        monthly_rating_seen_players[slot] = count


@njit(cache=True, nogil=True)
def replay_candidate(
    player_a: np.ndarray,
    player_b: np.ndarray,
    score_a: np.ndarray,
    white_sign: np.ndarray,
    event_group: np.ndarray,
    month_index_values: np.ndarray,
    scopes: np.ndarray,
    validation_folds: np.ndarray,
    fide_resolved: np.ndarray,
    fide_rating_a: np.ndarray,
    fide_rating_b: np.ndarray,
    static_pool: np.ndarray,
    params: np.ndarray,
    pool_assignment_votes_required: int,
    collect_monthly: bool,
    collect_rating_ranges: bool,
    first_month_index: int,
    last_month_index: int,
):
    games_count = len(score_a)
    players_count = len(static_pool)
    pools_count = int(static_pool.max()) + 1 if players_count else 1
    if pools_count < 1:
        pools_count = 1

    rating = np.full(players_count, INITIAL_RATING, dtype=np.float64)
    form = np.zeros(players_count, dtype=np.float64)
    uncertainty = np.full(players_count, params[P_INITIAL_UNCERTAINTY], dtype=np.float64)
    seen = np.zeros(players_count, dtype=np.uint8)
    last_decay_month = np.full(players_count, -1, dtype=np.int32)
    last_active_month = np.full(players_count, -1, dtype=np.int32)

    # Stan prowizoryczny nowych zawodników. Rating performance jest liczony
    # online: bieżąca partia korzysta wyłącznie ze stanu sprzed wydarzenia,
    # a jej wynik trafia do estymacji dopiero po wydarzeniu.
    provisional_games = np.zeros(players_count, dtype=np.int32)
    provisional_score_sum = np.zeros(players_count, dtype=np.float64)
    provisional_opponent_effective_sum = np.zeros(players_count, dtype=np.float64)
    provisional_score_acc = np.zeros(players_count, dtype=np.float64)
    provisional_opponent_effective_acc = np.zeros(players_count, dtype=np.float64)
    provisional_target = max(1, int(round(params[P_PROVISIONAL_GAMES_TARGET])))
    max_pool_exposures = max(1, min(3, int(round(params[P_MAX_POOL_EXPOSURES]))))

    player_pool = static_pool.copy()
    proposed_pool = np.zeros(players_count, dtype=np.int32)
    proposed_votes = np.zeros(players_count, dtype=np.int16)

    exposure_pool_1 = static_pool.copy()
    exposure_pool_2 = np.zeros(players_count, dtype=np.int32)
    exposure_pool_3 = np.zeros(players_count, dtype=np.int32)
    exposure_weight_1 = np.zeros(players_count, dtype=np.float64)
    exposure_weight_2 = np.zeros(players_count, dtype=np.float64)
    exposure_weight_3 = np.zeros(players_count, dtype=np.float64)
    for player in range(players_count):
        if exposure_pool_1[player] > 0:
            exposure_weight_1[player] = params[P_INITIAL_HOME_POOL_EXPOSURE]

    pool_offset = np.zeros(pools_count, dtype=np.float64)
    pair_offset = np.zeros((pools_count, pools_count), dtype=np.float64)

    residual_acc = np.zeros(players_count, dtype=np.float64)
    opponent_uncertainty_acc = np.zeros(players_count, dtype=np.float64)
    event_games_acc = np.zeros(players_count, dtype=np.int32)
    touched_players = np.empty(players_count, dtype=np.int32)

    vote_pool_1 = np.zeros(players_count, dtype=np.int32)
    vote_pool_2 = np.zeros(players_count, dtype=np.int32)
    vote_pool_3 = np.zeros(players_count, dtype=np.int32)
    vote_count_1 = np.zeros(players_count, dtype=np.int16)
    vote_count_2 = np.zeros(players_count, dtype=np.int16)
    vote_count_3 = np.zeros(players_count, dtype=np.int16)

    pool_residual_acc = np.zeros(pools_count, dtype=np.float64)
    pool_games_acc = np.zeros(pools_count, dtype=np.int32)
    touched_pool_flag = np.zeros(pools_count, dtype=np.uint8)
    touched_pools = np.empty(pools_count, dtype=np.int32)

    pair_slots = pools_count * pools_count
    pair_residual_acc = np.zeros(pair_slots, dtype=np.float64)
    pair_games_acc = np.zeros(pair_slots, dtype=np.int32)
    pair_lifetime_games = np.zeros(pair_slots, dtype=np.int64)
    touched_pair_flag = np.zeros(pair_slots, dtype=np.uint8)
    touched_pairs = np.empty(pair_slots, dtype=np.int32)

    squared_error = np.zeros(4, dtype=np.float64)
    absolute_error = np.zeros(4, dtype=np.float64)
    metric_count = np.zeros(4, dtype=np.int64)
    shared_squared_error = np.zeros(4, dtype=np.float64)
    shared_count = np.zeros(4, dtype=np.int64)
    fide_squared_error = np.zeros(4, dtype=np.float64)
    fide_count = np.zeros(4, dtype=np.int64)

    # Walidacja 2019/2020/2021 jako osobne foldy czasowe.
    fold_squared_error = np.zeros(4, dtype=np.float64)
    fold_count = np.zeros(4, dtype=np.int64)

    months_count = max(1, last_month_index - first_month_index + 1)
    monthly_squared_error = np.zeros(months_count, dtype=np.float64)
    monthly_count = np.zeros(months_count, dtype=np.int64)
    monthly_shared_squared_error = np.zeros(months_count, dtype=np.float64)
    monthly_shared_count = np.zeros(months_count, dtype=np.int64)
    monthly_fide_squared_error = np.zeros(months_count, dtype=np.float64)
    monthly_fide_count = np.zeros(months_count, dtype=np.int64)
    monthly_rating_min = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p01 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p10 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p25 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p50 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p75 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p90 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_p99 = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_max = np.full(months_count, np.nan, dtype=np.float64)
    monthly_rating_seen_players = np.zeros(months_count, dtype=np.int64)

    last_pool_decay_month = -1
    last_rating_range_month = -1
    i = 0
    while i < games_count:
        group = event_group[i]
        current_month = month_index_values[i]
        j = i + 1
        while j < games_count and event_group[j] == group:
            j += 1

        if last_rating_range_month < 0:
            last_rating_range_month = current_month
        elif current_month > last_rating_range_month:
            if collect_rating_ranges:
                _record_monthly_rating_distribution(
                    last_rating_range_month - first_month_index,
                    rating, form, seen,
                    exposure_pool_1, exposure_pool_2, exposure_pool_3,
                    exposure_weight_1, exposure_weight_2, exposure_weight_3,
                    pool_offset, params[P_POOL_WEIGHT],
                    monthly_rating_min, monthly_rating_p01, monthly_rating_p10,
                    monthly_rating_p25, monthly_rating_p50, monthly_rating_p75,
                    monthly_rating_p90, monthly_rating_p99, monthly_rating_max,
                    monthly_rating_seen_players,
                )
            last_rating_range_month = current_month

        if last_pool_decay_month < 0:
            last_pool_decay_month = current_month
        elif current_month > last_pool_decay_month:
            gap_months = current_month - last_pool_decay_month
            pool_decay = params[P_POOL_MONTH_DECAY] ** gap_months
            pair_decay = params[P_POOL_PAIR_MONTH_DECAY] ** gap_months
            for pool_id in range(1, pools_count):
                pool_offset[pool_id] *= pool_decay
            for pool_a_id in range(1, pools_count):
                for pool_b_id in range(1, pools_count):
                    pair_offset[pool_a_id, pool_b_id] *= pair_decay
            _center_pool_offsets(pool_offset, params[P_POOL_CENTERING_STRENGTH], params[P_POOL_LIMIT])
            last_pool_decay_month = current_month

        touched_players_count = 0
        touched_pools_count = 0
        touched_pairs_count = 0

        # Wszystkie predykcje wydarzenia korzystają ze snapshotu sprzed wydarzenia.
        for row in range(i, j):
            a = player_a[row]
            b = player_b[row]

            for player in (a, b):
                if seen[player] == 0:
                    seen[player] = 1
                    last_decay_month[player] = current_month
                    last_active_month[player] = current_month
                elif last_decay_month[player] < current_month:
                    gap = current_month - last_decay_month[player]
                    rating[player] = INITIAL_RATING + (rating[player] - INITIAL_RATING) * (params[P_RATING_MONTH_REVERSION] ** gap)
                    form[player] *= params[P_FORM_MONTH_DECAY] ** gap
                    exposure_decay = params[P_POOL_EXPOSURE_MONTH_DECAY] ** gap
                    exposure_weight_1[player] *= exposure_decay
                    exposure_weight_2[player] *= exposure_decay
                    exposure_weight_3[player] *= exposure_decay
                    if exposure_weight_1[player] + exposure_weight_2[player] + exposure_weight_3[player] < 0.03 and static_pool[player] > 0:
                        exposure_pool_1[player] = static_pool[player]
                        exposure_pool_2[player] = 0
                        exposure_pool_3[player] = 0
                        exposure_weight_1[player] = params[P_INITIAL_HOME_POOL_EXPOSURE]
                        exposure_weight_2[player] = 0.0
                        exposure_weight_3[player] = 0.0
                    inactive_gap = current_month - last_active_month[player] - 1
                    if inactive_gap > 0:
                        uncertainty[player] = min(
                            params[P_UNCERTAINTY_CAP],
                            uncertainty[player] + params[P_INACTIVITY_UNCERTAINTY_GROWTH] * inactive_gap,
                        )
                    last_decay_month[player] = current_month

            pool_a_id = _dominant_pool(a, player_pool, exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3, params[P_POOL_DOMINANT_THRESHOLD])
            pool_b_id = _dominant_pool(b, player_pool, exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3, params[P_POOL_DOMINANT_THRESHOLD])

            effective_a = rating[a] + form[a] + params[P_POOL_WEIGHT] * _pool_value(a, exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3, pool_offset)
            effective_b = rating[b] + form[b] + params[P_POOL_WEIGHT] * _pool_value(b, exposure_pool_1, exposure_pool_2, exposure_pool_3, exposure_weight_1, exposure_weight_2, exposure_weight_3, pool_offset)

            raw_diff = effective_a - effective_b
            if pool_a_id > 0 and pool_b_id > 0 and pool_a_id != pool_b_id:
                if pool_a_id < pool_b_id:
                    shrink_left, shrink_right = pool_a_id, pool_b_id
                else:
                    shrink_left, shrink_right = pool_b_id, pool_a_id
                shrink_slot = shrink_left * pools_count + shrink_right
                pair_seen = pair_lifetime_games[shrink_slot]
                if pair_seen > 0:
                    shrinkage = pair_seen / (pair_seen + params[P_POOL_PAIR_SHRINKAGE_GAMES])
                    raw_diff += params[P_POOL_PAIR_WEIGHT] * shrinkage * pair_offset[pool_a_id, pool_b_id]

            average_level = 0.5 * (effective_a + effective_b)
            scale = (
                params[P_SCALE_BASE]
                + params[P_SCALE_LEVEL_SLOPE] * ((average_level - INITIAL_RATING) / 100.0)
                + params[P_SCALE_ABS_DIFF_SLOPE] * min(abs(raw_diff) / 400.0, 3.0)
            )
            scale *= 1.0 + params[P_PREDICTION_UNCERTAINTY_SCALE_WEIGHT] * 0.5 * (uncertainty[a] + uncertainty[b])
            white_bonus = params[P_WHITE_ADVANTAGE] + params[P_WHITE_ADVANTAGE_LEVEL_SLOPE] * ((average_level - INITIAL_RATING) / 100.0)
            white_bonus = _clip(white_bonus, -30.0, 120.0)
            predicted = _expected_points(raw_diff + white_bonus * white_sign[row], scale)
            actual = score_a[row]
            residual = actual - predicted

            scope = scopes[row]
            if scope > 0:
                squared_error[scope] += residual * residual
                absolute_error[scope] += abs(residual)
                metric_count[scope] += 1
                fold = validation_folds[row]
                if fold > 0:
                    fold_squared_error[fold] += residual * residual
                    fold_count[fold] += 1
                if fide_resolved[row]:
                    shared_squared_error[scope] += residual * residual
                    shared_count[scope] += 1
                    fide_predicted = _expected_points(fide_rating_a[row] - fide_rating_b[row], 400.0)
                    fide_residual = actual - fide_predicted
                    fide_squared_error[scope] += fide_residual * fide_residual
                    fide_count[scope] += 1

            if collect_monthly:
                slot = current_month - first_month_index
                if 0 <= slot < months_count:
                    monthly_squared_error[slot] += residual * residual
                    monthly_count[slot] += 1
                    if fide_resolved[row]:
                        monthly_shared_squared_error[slot] += residual * residual
                        monthly_shared_count[slot] += 1
                        fide_predicted = _expected_points(fide_rating_a[row] - fide_rating_b[row], 400.0)
                        fide_residual = actual - fide_predicted
                        monthly_fide_squared_error[slot] += fide_residual * fide_residual
                        monthly_fide_count[slot] += 1

            if event_games_acc[a] == 0:
                touched_players[touched_players_count] = a
                touched_players_count += 1
            if event_games_acc[b] == 0:
                touched_players[touched_players_count] = b
                touched_players_count += 1
            residual_acc[a] += residual
            residual_acc[b] -= residual
            opponent_uncertainty_acc[a] += uncertainty[b]
            opponent_uncertainty_acc[b] += uncertainty[a]
            event_games_acc[a] += 1
            event_games_acc[b] += 1

            # Performance wejściowy jest aktualizowany dopiero po wydarzeniu.
            # Dzięki temu predykcja tej partii nie zna jeszcze jej wyniku.
            if provisional_games[a] < provisional_target:
                provisional_score_acc[a] += actual
                provisional_opponent_effective_acc[a] += effective_b
            if provisional_games[b] < provisional_target:
                provisional_score_acc[b] += 1.0 - actual
                provisional_opponent_effective_acc[b] += effective_a

            _add_vote(a, pool_b_id, vote_pool_1, vote_pool_2, vote_pool_3, vote_count_1, vote_count_2, vote_count_3)
            _add_vote(b, pool_a_id, vote_pool_1, vote_pool_2, vote_pool_3, vote_count_1, vote_count_2, vote_count_3)

            if pool_a_id > 0 and pool_b_id > 0 and pool_a_id != pool_b_id:
                if touched_pool_flag[pool_a_id] == 0:
                    touched_pool_flag[pool_a_id] = 1
                    touched_pools[touched_pools_count] = pool_a_id
                    touched_pools_count += 1
                if touched_pool_flag[pool_b_id] == 0:
                    touched_pool_flag[pool_b_id] = 1
                    touched_pools[touched_pools_count] = pool_b_id
                    touched_pools_count += 1
                pool_residual_acc[pool_a_id] += residual
                pool_residual_acc[pool_b_id] -= residual
                pool_games_acc[pool_a_id] += 1
                pool_games_acc[pool_b_id] += 1

                if pool_a_id < pool_b_id:
                    left, right, signed_residual = pool_a_id, pool_b_id, residual
                else:
                    left, right, signed_residual = pool_b_id, pool_a_id, -residual
                pair_slot = left * pools_count + right
                if touched_pair_flag[pair_slot] == 0:
                    touched_pair_flag[pair_slot] = 1
                    touched_pairs[touched_pairs_count] = pair_slot
                    touched_pairs_count += 1
                pair_residual_acc[pair_slot] += signed_residual
                pair_games_acc[pair_slot] += 1

        # Aktualizacja graczy po wydarzeniu.
        for touched_index in range(touched_players_count):
            player = touched_players[touched_index]
            count = event_games_acc[player]
            normalized_residual = residual_acc[player] / (count ** params[P_EVENT_RESIDUAL_NORM_POWER])
            normalized_residual = _clip(normalized_residual, -params[P_EVENT_RESIDUAL_CLIP], params[P_EVENT_RESIDUAL_CLIP])
            average_opponent_uncertainty = opponent_uncertainty_acc[player] / count
            k = params[P_BASE_K] * (1.0 + params[P_UNCERTAINTY_K_WEIGHT] * uncertainty[player])
            k /= 1.0 + params[P_OPPONENT_UNCERTAINTY_K_DAMPING] * average_opponent_uncertainty
            if provisional_games[player] < provisional_target:
                remaining_share = 1.0 - min(1.0, provisional_games[player] / provisional_target)
                k *= 1.0 + (params[P_PROVISIONAL_K_MULTIPLIER] - 1.0) * remaining_share
            rating[player] += k * normalized_residual
            form[player] = _clip(
                form[player] * (params[P_FORM_EVENT_DECAY] ** count) + params[P_FORM_GAIN] * normalized_residual,
                -params[P_FORM_LIMIT],
                params[P_FORM_LIMIT],
            )
            uncertainty[player] = max(
                params[P_UNCERTAINTY_FLOOR],
                uncertainty[player] * (params[P_UNCERTAINTY_GAME_DECAY] ** count),
            )
            last_active_month[player] = current_month

            event_vote_pool = _apply_exposure_votes(
                player,
                params[P_POOL_EXPOSURE_GAIN],
                max_pool_exposures,
                vote_pool_1, vote_pool_2, vote_pool_3,
                vote_count_1, vote_count_2, vote_count_3,
                exposure_pool_1, exposure_pool_2, exposure_pool_3,
                exposure_weight_1, exposure_weight_2, exposure_weight_3,
            )
            if player_pool[player] == 0 and event_vote_pool > 0 and pool_assignment_votes_required > 0:
                if proposed_pool[player] == event_vote_pool:
                    proposed_votes[player] += 1
                elif proposed_votes[player] <= 0:
                    proposed_pool[player] = event_vote_pool
                    proposed_votes[player] = 1
                else:
                    proposed_votes[player] -= 1
                if proposed_votes[player] >= pool_assignment_votes_required:
                    player_pool[player] = proposed_pool[player]

            # Prowizoryczny ranking wejściowy: po kilku pierwszych partiach
            # estymujemy performance względem ratingów przeciwników. Prior 50%
            # ogranicza skrajności dla wyniku 0/N albo N/N. Korekta własnej
            # puli zapobiega podwójnemu naliczaniu informacji środowiskowej.
            if provisional_games[player] < provisional_target:
                provisional_score_sum[player] += provisional_score_acc[player]
                provisional_opponent_effective_sum[player] += provisional_opponent_effective_acc[player]
                provisional_games[player] += count
                total_games = provisional_games[player]
                if total_games > 0 and params[P_PROVISIONAL_BLEND] > 0.0:
                    prior_games = params[P_PROVISIONAL_PRIOR_GAMES]
                    score_fraction = (provisional_score_sum[player] + 0.5 * prior_games) / (total_games + prior_games)
                    performance_diff = _performance_diff_from_score(
                        score_fraction,
                        params[P_SCALE_BASE],
                        params[P_PROVISIONAL_PERFORMANCE_CLIP],
                    )
                    opponent_average = provisional_opponent_effective_sum[player] / total_games
                    own_pool_component = params[P_POOL_WEIGHT] * _pool_value(
                        player,
                        exposure_pool_1, exposure_pool_2, exposure_pool_3,
                        exposure_weight_1, exposure_weight_2, exposure_weight_3,
                        pool_offset,
                    )
                    performance_target = opponent_average + performance_diff - params[P_PROVISIONAL_POOL_ADJUSTMENT] * own_pool_component
                    progress = min(1.0, total_games / provisional_target)
                    blend = params[P_PROVISIONAL_BLEND] * progress
                    rating[player] = (1.0 - blend) * rating[player] + blend * performance_target

            provisional_score_acc[player] = 0.0
            provisional_opponent_effective_acc[player] = 0.0
            residual_acc[player] = 0.0
            opponent_uncertainty_acc[player] = 0.0
            event_games_acc[player] = 0

        # Aktualizacja ogólnych offsetów pul.
        for touched_index in range(touched_pools_count):
            pool_id = touched_pools[touched_index]
            count = pool_games_acc[pool_id]
            normalized_residual = pool_residual_acc[pool_id] / (count ** params[P_POOL_EVENT_NORM_POWER])
            normalized_residual = _clip(normalized_residual, -params[P_POOL_RESIDUAL_CLIP], params[P_POOL_RESIDUAL_CLIP])
            pool_offset[pool_id] = _clip(
                pool_offset[pool_id] + params[P_POOL_K] * normalized_residual,
                -params[P_POOL_LIMIT],
                params[P_POOL_LIMIT],
            )
            pool_residual_acc[pool_id] = 0.0
            pool_games_acc[pool_id] = 0
            touched_pool_flag[pool_id] = 0

        # Aktualizacja macierzy par pul.
        for touched_index in range(touched_pairs_count):
            pair_slot = touched_pairs[touched_index]
            left = pair_slot // pools_count
            right = pair_slot % pools_count
            count = pair_games_acc[pair_slot]
            normalized_residual = pair_residual_acc[pair_slot] / (count ** params[P_POOL_EVENT_NORM_POWER])
            normalized_residual = _clip(normalized_residual, -params[P_POOL_RESIDUAL_CLIP], params[P_POOL_RESIDUAL_CLIP])
            value = _clip(
                pair_offset[left, right] + params[P_POOL_PAIR_K] * normalized_residual,
                -params[P_POOL_PAIR_LIMIT],
                params[P_POOL_PAIR_LIMIT],
            )
            pair_offset[left, right] = value
            pair_offset[right, left] = -value
            pair_lifetime_games[pair_slot] += count
            pair_residual_acc[pair_slot] = 0.0
            pair_games_acc[pair_slot] = 0
            touched_pair_flag[pair_slot] = 0

        i = j

    if collect_rating_ranges and last_rating_range_month >= 0:
        _record_monthly_rating_distribution(
            last_rating_range_month - first_month_index,
            rating, form, seen,
            exposure_pool_1, exposure_pool_2, exposure_pool_3,
            exposure_weight_1, exposure_weight_2, exposure_weight_3,
            pool_offset, params[P_POOL_WEIGHT],
            monthly_rating_min, monthly_rating_p01, monthly_rating_p10,
            monthly_rating_p25, monthly_rating_p50, monthly_rating_p75,
            monthly_rating_p90, monthly_rating_p99, monthly_rating_max,
            monthly_rating_seen_players,
        )

    # Rating wyświetlany zawodnikowi: indywidualny slow rating + forma +
    # ważona korekta pulowa. Offset konkretnej pary pul nie jest częścią tej
    # liczby, bo zależy od przeciwnika i działa wyłącznie przy predykcji meczu.
    final_effective_rating = np.full(players_count, np.nan, dtype=np.float64)
    for player in range(players_count):
        if seen[player] != 0:
            final_effective_rating[player] = (
                rating[player]
                + form[player]
                + params[P_POOL_WEIGHT] * _pool_value(
                    player,
                    exposure_pool_1, exposure_pool_2, exposure_pool_3,
                    exposure_weight_1, exposure_weight_2, exposure_weight_3,
                    pool_offset,
                )
            )

    return (
        squared_error,
        absolute_error,
        metric_count,
        shared_squared_error,
        shared_count,
        fide_squared_error,
        fide_count,
        fold_squared_error,
        fold_count,
        monthly_squared_error,
        monthly_count,
        monthly_shared_squared_error,
        monthly_shared_count,
        monthly_fide_squared_error,
        monthly_fide_count,
        pool_offset,
        pair_offset,
        final_effective_rating,
        rating,
        seen,
        monthly_rating_min,
        monthly_rating_p01,
        monthly_rating_p10,
        monthly_rating_p25,
        monthly_rating_p50,
        monthly_rating_p75,
        monthly_rating_p90,
        monthly_rating_p99,
        monthly_rating_max,
        monthly_rating_seen_players,
    )


# =============================================================================
# OCENA KANDYDATÓW
# =============================================================================

@dataclass
class CandidateResult:
    parameters: np.ndarray
    train_mse: float
    validation_mse: float
    test_mse: float
    validation_shared_mse: float
    validation_year_balanced_mse: float
    validation_worst_year_mse: float
    fitness: float


def safe_mse(error_sum: float, count: int) -> float:
    return float(error_sum / count) if count else float("nan")


def evaluate_candidate(
    cache: ReplayCache,
    parameters: np.ndarray,
    collect_monthly: bool = False,
    collect_rating_ranges: bool = False,
):
    return replay_candidate(
        cache.player_a,
        cache.player_b,
        cache.score_a,
        cache.white_sign,
        cache.event_group,
        cache.month_index,
        cache.scope,
        cache.validation_fold,
        cache.fide_resolved,
        cache.fide_rating_a,
        cache.fide_rating_b,
        cache.static_pool,
        parameters,
        POOL_NEW_PLAYER_ASSIGNMENT_VOTES,
        collect_monthly,
        collect_rating_ranges,
        cache.first_month_index,
        cache.last_month_index,
    )


def summarize_candidate(cache: ReplayCache, parameters: np.ndarray) -> CandidateResult:
    output = evaluate_candidate(cache, parameters, collect_monthly=False)
    squared_error, _, counts, shared_squared_error, shared_counts, _, _, fold_error, fold_counts, *_ = output
    train_mse = safe_mse(squared_error[1], int(counts[1]))
    validation_mse = safe_mse(squared_error[2], int(counts[2]))
    test_mse = safe_mse(squared_error[3], int(counts[3]))
    validation_shared_mse = safe_mse(shared_squared_error[2], int(shared_counts[2]))

    fold_mses: list[float] = []
    for fold in range(1, len(fold_counts)):
        if int(fold_counts[fold]) > 0:
            fold_mses.append(safe_mse(fold_error[fold], int(fold_counts[fold])))
    validation_year_balanced_mse = float(np.mean(fold_mses)) if fold_mses else validation_mse
    validation_worst_year_mse = float(np.max(fold_mses)) if fold_mses else validation_mse

    fitness = (1.0 - FITNESS_SHARED_WEIGHT - FITNESS_YEAR_BALANCED_WEIGHT - FITNESS_WORST_YEAR_WEIGHT) * validation_mse
    if math.isfinite(validation_shared_mse):
        fitness += FITNESS_SHARED_WEIGHT * validation_shared_mse
    else:
        fitness += FITNESS_SHARED_WEIGHT * validation_mse
    fitness += FITNESS_YEAR_BALANCED_WEIGHT * validation_year_balanced_mse
    fitness += FITNESS_WORST_YEAR_WEIGHT * validation_worst_year_mse
    if math.isfinite(train_mse) and math.isfinite(validation_mse):
        fitness += OVERFIT_PENALTY_WEIGHT * max(0.0, validation_mse - train_mse)

    return CandidateResult(
        parameters=parameters,
        train_mse=train_mse,
        validation_mse=validation_mse,
        test_mse=test_mse,
        validation_shared_mse=validation_shared_mse,
        validation_year_balanced_mse=validation_year_balanced_mse,
        validation_worst_year_mse=validation_worst_year_mse,
        fitness=fitness,
    )


def evaluate_many(cache: ReplayCache, candidates: Iterable[np.ndarray], workers: int) -> list[CandidateResult]:
    candidate_list = [clip_candidate(candidate) for candidate in candidates]
    if workers <= 1:
        return [summarize_candidate(cache, candidate) for candidate in candidate_list]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda candidate: summarize_candidate(cache, candidate), candidate_list))


# =============================================================================
# WYSZUKIWANIE EWOLUCYJNE
# =============================================================================

SEARCH_HISTORY_PATH = RESULTS_DIR / "evolution_history.csv"
FAST_LEADERBOARD_PATH = RESULTS_DIR / "fast_search_leaderboard.csv"
FULL_LEADERBOARD_PATH = RESULTS_DIR / "full_finalists_leaderboard.csv"
REFINEMENT_HISTORY_PATH = RESULTS_DIR / "full_refinement_history.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "best_model_parameters.json"


def candidate_signature(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(round(value, 10)) for value in values)


def mutate_candidate(parent: np.ndarray, rng: np.random.Generator, sigma_multiplier: float = 1.0) -> np.ndarray:
    child = parent.copy()
    for index, spec in enumerate(PARAMETER_SPECS):
        if rng.random() < 0.68:
            span = spec.high - spec.low
            child[index] += rng.normal(0.0, span * spec.mutation_sigma_fraction * sigma_multiplier)
    return clip_candidate(child)


def crossover_candidate(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(PARAMETER_SPECS)) < 0.5
    return mutate_candidate(np.where(mask, a, b), rng)


def differential_candidate(
    base: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    factor: float | None = None,
) -> np.ndarray:
    """Mutacja typu differential evolution; pomaga przeskakiwać między dobrymi regionami."""
    if factor is None:
        factor = float(rng.uniform(0.25, 0.90))
    proposal = base + factor * (a - b)
    mask = rng.random(len(PARAMETER_SPECS)) < 0.68
    mixed = np.where(mask, proposal, base)
    return clip_candidate(mixed)


def coordinate_candidates(parent: np.ndarray, step_fraction: float) -> list[np.ndarray]:
    """Lokalne sąsiedztwo +/- jednego parametru na pełnym cache."""
    candidates: list[np.ndarray] = []
    for index, spec in enumerate(PARAMETER_SPECS):
        step = (spec.high - spec.low) * step_fraction
        for sign in (-1.0, 1.0):
            child = parent.copy()
            child[index] += sign * step
            candidates.append(clip_candidate(child))
    return candidates


def results_to_frame(results: list[CandidateResult], generation: int | None = None, stage: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.fitness), start=1):
        row: dict[str, object] = {
            "rank": rank,
            "generation": generation,
            "stage": stage,
            "fitness": result.fitness,
            "train_mse": result.train_mse,
            "validation_mse": result.validation_mse,
            "validation_shared_mse": result.validation_shared_mse,
            "validation_year_balanced_mse": result.validation_year_balanced_mse,
            "validation_worst_year_mse": result.validation_worst_year_mse,
            "test_mse": result.test_mse,
        }
        row.update(params_array_to_dict(result.parameters))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_population() -> list[np.ndarray]:
    seeds = [
        params_dict_to_array(CLASSIC_SEED),
        params_dict_to_array(PREVIOUS_DYNAMIC_SEED),
        params_dict_to_array(PREVIOUS_DYNAMIC_SIMPLE_SEED),
        params_dict_to_array(PAIR_POOL_SEED),
        params_dict_to_array(ADVANCED_BEST_SEED),
        params_dict_to_array(PROVISIONAL_9_SEED),
        params_dict_to_array(PROVISIONAL_CONSERVATIVE_SEED),
    ]
    previous_local = load_optional_local_best_seed()
    if previous_local is not None:
        seeds.append(params_dict_to_array(previous_local))
        previous_local_provisional = {
            **previous_local,
            "provisional_games_target": 9.0,
            "provisional_prior_games": 5.0,
            "provisional_performance_clip": 650.0,
            "provisional_blend": 0.65,
            "provisional_k_multiplier": 1.90,
            "provisional_pool_adjustment": 1.0,
            "pool_pair_shrinkage_games": 35.0,
            "pool_centering_strength": 1.0,
        }
        seeds.append(params_dict_to_array(previous_local_provisional))
    return [clip_candidate(seed) for seed in seeds]


def evolve_parameters(cache: ReplayCache, *, profile: SearchProfile) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    population = seed_population()
    while len(population) < profile.population:
        if rng.random() < 0.50:
            population.append(mutate_candidate(population[int(rng.integers(1, len(population)))], rng, sigma_multiplier=1.5))
        else:
            population.append(random_candidate(rng))

    history_frames: list[pd.DataFrame] = []
    all_seen: dict[tuple[float, ...], CandidateResult] = {}

    print("[TUNE] Compiling Numba replay kernel...")
    summarize_candidate(cache, population[0])
    print("[TUNE] Evolutionary search started.")
    print(f"[TUNE] population:          {profile.population}")
    print(f"[TUNE] generations:         {profile.generations}")
    print(f"[TUNE] workers:             {profile.workers}")

    for generation in range(1, profile.generations + 1):
        started = time.perf_counter()
        unique_population: list[np.ndarray] = []
        signatures: set[tuple[float, ...]] = set()
        for candidate in population:
            signature = candidate_signature(candidate)
            if signature not in signatures:
                signatures.add(signature)
                unique_population.append(candidate)
        missing = [candidate for candidate in unique_population if candidate_signature(candidate) not in all_seen]
        if missing:
            for result in evaluate_many(cache, missing, workers=profile.workers):
                all_seen[candidate_signature(result.parameters)] = result
        generation_results = [all_seen[candidate_signature(candidate)] for candidate in unique_population]
        generation_results.sort(key=lambda item: item.fitness)
        best = generation_results[0]
        elapsed = time.perf_counter() - started
        print(
            f"[TUNE] generation {generation:02d}/{profile.generations:02d} | "
            f"fitness={best.fitness:.8f} | validation={best.validation_mse:.8f} | "
            f"shared={best.validation_shared_mse:.8f} | elapsed={elapsed:.1f}s"
        )
        history_frames.append(results_to_frame(generation_results, generation=generation, stage="fast_evolution"))
        pd.concat(history_frames, ignore_index=True).to_csv(SEARCH_HISTORY_PATH, index=False, encoding="utf-8-sig")

        elites = [result.parameters for result in generation_results[: profile.elites]]
        next_population = [candidate.copy() for candidate in elites]
        while len(next_population) < profile.population:
            draw = rng.random()
            if draw < 0.25 and len(elites) >= 3:
                base = elites[int(rng.integers(0, len(elites)))]
                parent_a = elites[int(rng.integers(0, len(elites)))]
                parent_b = elites[int(rng.integers(0, len(elites)))]
                next_population.append(differential_candidate(base, parent_a, parent_b, rng))
            elif draw < 0.54 and len(elites) >= 2:
                parent_a = elites[int(rng.integers(0, len(elites)))]
                parent_b = elites[int(rng.integers(0, len(elites)))]
                next_population.append(crossover_candidate(parent_a, parent_b, rng))
            elif draw < 0.92:
                parent = elites[int(rng.integers(0, len(elites)))]
                next_population.append(mutate_candidate(parent, rng))
            else:
                next_population.append(random_candidate(rng))
        population = next_population

    ranked = sorted(all_seen.values(), key=lambda item: item.fitness)
    results_to_frame(ranked, stage="fast_evolution").to_csv(FAST_LEADERBOARD_PATH, index=False, encoding="utf-8-sig")
    print(f"[TUNE] Fast leaderboard:    {FAST_LEADERBOARD_PATH}")
    return [result.parameters for result in ranked[: profile.full_finalists]]


def refine_on_full_cache(
    cache: ReplayCache,
    initial_candidates: list[np.ndarray],
    *,
    profile: SearchProfile,
) -> list[CandidateResult]:
    rng = np.random.default_rng(RANDOM_SEED + 19)
    deduplicated: dict[tuple[float, ...], np.ndarray] = {}
    for candidate in initial_candidates + seed_population():
        deduplicated[candidate_signature(candidate)] = clip_candidate(candidate)

    print("[TUNE] Re-ranking finalists on complete TRAIN+VALIDATION replay...")
    results = evaluate_many(cache, deduplicated.values(), workers=min(profile.workers, 4))
    all_seen: dict[tuple[float, ...], CandidateResult] = {
        candidate_signature(result.parameters): result for result in results
    }
    history_frames = [results_to_frame(results, generation=0, stage="full_rerank")]

    for round_index in range(1, profile.full_refine_rounds + 1):
        ranked = sorted(all_seen.values(), key=lambda item: item.fitness)
        parents = [result.parameters for result in ranked[: min(6, len(ranked))]]
        sigma = 0.45 * (0.62 ** (round_index - 1))
        candidates: list[np.ndarray] = []
        while len(candidates) < profile.full_refine_candidates_per_round:
            parent = parents[int(rng.integers(0, len(parents)))]
            candidates.append(mutate_candidate(parent, rng, sigma_multiplier=sigma))
        missing = [candidate for candidate in candidates if candidate_signature(candidate) not in all_seen]
        started = time.perf_counter()
        for result in evaluate_many(cache, missing, workers=min(profile.workers, 4)):
            all_seen[candidate_signature(result.parameters)] = result
        ranked = sorted(all_seen.values(), key=lambda item: item.fitness)
        best = ranked[0]
        elapsed = time.perf_counter() - started
        print(
            f"[REFINE] round {round_index:02d}/{profile.full_refine_rounds:02d} | "
            f"fitness={best.fitness:.8f} | validation={best.validation_mse:.8f} | "
            f"shared={best.validation_shared_mse:.8f} | elapsed={elapsed:.1f}s"
        )
        round_results = [all_seen[candidate_signature(candidate)] for candidate in missing]
        history_frames.append(results_to_frame(round_results, generation=round_index, stage="full_refinement"))

    # Końcowy coordinate refinement: każdy parametr najlepszego kandydata jest
    # osobno przesuwany w dół i w górę na pełnym cache. To dobrze dopracowuje
    # ostatnie procenty po szerokim wyszukiwaniu ewolucyjnym.
    for coordinate_round in range(1, profile.coordinate_refine_rounds + 1):
        ranked = sorted(all_seen.values(), key=lambda item: item.fitness)
        parent = ranked[0].parameters
        step = profile.coordinate_step_fraction * (0.55 ** (coordinate_round - 1))
        candidates = coordinate_candidates(parent, step)
        missing = [candidate for candidate in candidates if candidate_signature(candidate) not in all_seen]
        started = time.perf_counter()
        for result in evaluate_many(cache, missing, workers=min(profile.workers, 4)):
            all_seen[candidate_signature(result.parameters)] = result
        ranked = sorted(all_seen.values(), key=lambda item: item.fitness)
        best = ranked[0]
        elapsed = time.perf_counter() - started
        print(
            f"[COORD] round {coordinate_round:02d}/{profile.coordinate_refine_rounds:02d} | "
            f"fitness={best.fitness:.8f} | validation={best.validation_mse:.8f} | "
            f"shared={best.validation_shared_mse:.8f} | elapsed={elapsed:.1f}s"
        )
        coord_results = [all_seen[candidate_signature(candidate)] for candidate in missing]
        history_frames.append(results_to_frame(coord_results, generation=coordinate_round, stage="coordinate_refinement"))

    ranked = sorted(all_seen.values(), key=lambda item: item.fitness)
    results_to_frame(ranked, stage="full_final").to_csv(FULL_LEADERBOARD_PATH, index=False, encoding="utf-8-sig")
    pd.concat(history_frames, ignore_index=True).to_csv(REFINEMENT_HISTORY_PATH, index=False, encoding="utf-8-sig")
    return ranked


# =============================================================================
# FINALNA EWALUACJA, RAPORTY I WYKRESY
# =============================================================================

FINAL_SCOPE_PATH = RESULTS_DIR / "final_scope_metrics.csv"
FINAL_MONTHLY_PATH = RESULTS_DIR / "final_monthly_metrics.csv"
FINAL_TEST_COMPARISON_PATH = RESULTS_DIR / "final_test_comparison.csv"
FINAL_REPORT_PATH = RESULTS_DIR / "final_report.txt"
FINAL_POOL_OFFSETS_PATH = RESULTS_DIR / "final_pool_offsets.csv"
FINAL_PAIR_OFFSETS_PATH = RESULTS_DIR / "final_pool_pair_offsets.csv"
FINAL_RATING_RANGES_PATH = RESULTS_DIR / "final_rating_distributions.csv"
FINAL_EXTREME_PLAYERS_PATH = RESULTS_DIR / "final_rating_extreme_players.csv"
MONTHLY_RATING_RANGES_PATH = RESULTS_DIR / "monthly_rating_distributions.csv"

# Diagnostyka sanity-check. To nie ogranicza modelu; tylko raportuje potencjalnie
# podejrzane skrajności skali ratingowej.
SANITY_MIN_RATING = -500.0
SANITY_MAX_RATING = 4000.0
TOP_BOTTOM_PLAYERS_PER_MODEL = 20


def set_zero(values: np.ndarray, *names: str) -> np.ndarray:
    result = values.copy()
    for name in names:
        result[PARAMETER_INDEX[name]] = 0.0
    return result


def named_candidate_variants(best: np.ndarray) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {
        "classic_elo": params_dict_to_array(CLASSIC_SEED),
        "previous_dynamic_seed": params_dict_to_array(PREVIOUS_DYNAMIC_SIMPLE_SEED),
        "advanced_seed": params_dict_to_array(ADVANCED_BEST_SEED),
        "provisional_9_seed": params_dict_to_array(PROVISIONAL_9_SEED),
        "provisional_full": best.copy(),
    }

    # Ablacje mechanizmu wejścia nowych zawodników.
    fixed_1500 = best.copy()
    fixed_1500[PARAMETER_INDEX["provisional_blend"]] = 0.0
    fixed_1500[PARAMETER_INDEX["provisional_k_multiplier"]] = 1.0
    variants["provisional_fixed_1500_entry"] = fixed_1500

    variants["provisional_without_performance_entry"] = set_zero(best, "provisional_blend")
    no_entry_acceleration = best.copy()
    no_entry_acceleration[PARAMETER_INDEX["provisional_k_multiplier"]] = 1.0
    variants["provisional_without_entry_acceleration"] = no_entry_acceleration

    fixed_nine = best.copy()
    fixed_nine[PARAMETER_INDEX["provisional_games_target"]] = 9.0
    variants["provisional_exactly_9_games"] = fixed_nine

    # Ablacje głównej hipotezy pulowej.
    variants["provisional_without_pool"] = set_zero(
        best,
        "pool_k", "pool_weight", "pool_exposure_gain", "pool_pair_k", "pool_pair_weight",
    )
    variants["provisional_without_dynamic_exposure"] = set_zero(best, "pool_exposure_gain")
    variants["provisional_without_pair_interaction"] = set_zero(best, "pool_pair_k", "pool_pair_weight")
    variants["provisional_without_pair_shrinkage"] = set_zero(best, "pool_pair_shrinkage_games")
    variants["provisional_without_pool_centering"] = set_zero(best, "pool_centering_strength")

    # Ablacje liczby ekspozycji na pule. Pozwalają pokazać, czy 1, 2 czy 3
    # środowiska jednocześnie są najlepszym kompromisem.
    for exposures in (1.0, 2.0, 3.0):
        candidate = best.copy()
        candidate[PARAMETER_INDEX["max_pool_exposures"]] = exposures
        variants[f"provisional_max_exposures_{int(exposures)}"] = candidate

    # Klasyczne ablacje pozostałych komponentów.
    variants["provisional_without_form"] = set_zero(best, "form_gain")
    variants["provisional_without_white_advantage"] = set_zero(best, "white_advantage", "white_advantage_level_slope")
    variants["provisional_without_uncertainty_prediction"] = set_zero(best, "prediction_uncertainty_scale_weight")
    variants["provisional_without_event_normalization"] = set_zero(best, "event_residual_norm_power")
    variants["provisional_without_tail_scale"] = set_zero(best, "scale_abs_diff_slope")

    simplified = set_zero(
        best,
        "form_gain",
        "scale_level_slope",
        "scale_abs_diff_slope",
        "white_advantage_level_slope",
        "prediction_uncertainty_scale_weight",
        "event_residual_norm_power",
        "pool_pair_k",
        "pool_pair_weight",
    )
    variants["provisional_simplified"] = simplified
    return {name: clip_candidate(candidate) for name, candidate in variants.items()}


def scope_name(code: int) -> str:
    return {1: "train", 2: "validation", 3: "test"}.get(code, "warmup")


def save_evolution_plot() -> None:
    if not SEARCH_HISTORY_PATH.exists():
        return
    history = pd.read_csv(SEARCH_HISTORY_PATH)
    if history.empty or "generation" not in history:
        return
    grouped = history.groupby("generation", as_index=False).agg(
        best_validation_mse=("validation_mse", "min"),
        best_fitness=("fitness", "min"),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grouped["generation"], grouped["best_validation_mse"], marker="o", label="best validation MSE")
    ax.plot(grouped["generation"], grouped["best_fitness"], marker="o", label="best fitness")
    ax.set_title("Postęp wyszukiwania ewolucyjnego")
    ax.set_xlabel("generacja")
    ax.set_ylabel("wartość metryki")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "evolution_progress.png", dpi=180)
    plt.close(fig)


def save_test_bar_plots(test: pd.DataFrame) -> None:
    ordered = test.sort_values("mse_all")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(ordered["model"], ordered["mse_all"])
    ax.set_title("MSE expected points na końcowym okresie testowym")
    ax.set_xlabel("MSE — mniej znaczy lepiej")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test_mse_all_by_model.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(ordered["model"], ordered["improvement_vs_classic_percent"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Poprawa względem klasycznego Elo na teście")
    ax.set_xlabel("poprawa MSE [%]")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test_improvement_vs_classic.png", dpi=180)
    plt.close(fig)


def save_monthly_plots(monthly: pd.DataFrame) -> None:
    selected = monthly[monthly["model"].isin(PLOT_SELECTED_MODELS)].copy()
    selected["date"] = pd.to_datetime(selected["month"] + "-01")
    if selected.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    for model_name, group in selected.groupby("model"):
        ax.plot(group["date"], group["mse_all"], linewidth=1.3, label=model_name)
    ax.set_title("Miesięczne MSE expected points")
    ax.set_xlabel("miesiąc")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "monthly_mse_selected_models.png", dpi=180)
    plt.close(fig)

    pivot = selected.pivot(index="date", columns="model", values="mse_all")
    if "classic_elo" in pivot.columns:
        fig, ax = plt.subplots(figsize=(15, 7))
        baseline = pivot["classic_elo"]
        for model_name in pivot.columns:
            if model_name == "classic_elo":
                continue
            improvement = (baseline - pivot[model_name]) / baseline * 100.0
            ax.plot(improvement.index, improvement, linewidth=1.3, label=model_name)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title("Miesięczna poprawa MSE względem klasycznego Elo")
        ax.set_xlabel("miesiąc")
        ax.set_ylabel("poprawa [%]")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "monthly_improvement_vs_classic.png", dpi=180)
        plt.close(fig)

    full = monthly[monthly["model"] == "provisional_full"].copy()
    full["date"] = pd.to_datetime(full["month"] + "-01")
    if not full.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(full["date"], full["mse_shared_with_fide"], linewidth=1.4, label="provisional_full")
        ax.plot(full["date"], full["fide_official_mse_on_shared"], linewidth=1.4, label="fide_official")
        ax.set_title("MSE modelu i oficjalnych ratingów FIDE na wspólnym podzbiorze")
        ax.set_xlabel("miesiąc")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "monthly_shared_mse_vs_fide.png", dpi=180)
        plt.close(fig)


def save_pool_plots(pool_offsets: np.ndarray, pair_offsets: np.ndarray) -> None:
    pool_df = pd.DataFrame({"pool_id": np.arange(len(pool_offsets)), "offset": pool_offsets})
    pool_df.to_csv(FINAL_POOL_OFFSETS_PATH, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(13, 6))
    non_fallback = pool_df[pool_df["pool_id"] > 0]
    ax.bar(non_fallback["pool_id"].astype(str), non_fallback["offset"])
    ax.set_title("Końcowe offsety latentnych pul")
    ax.set_xlabel("pool_id")
    ax.set_ylabel("offset ratingowy")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_pool_offsets.png", dpi=180)
    plt.close(fig)

    rows: list[dict[str, float | int]] = []
    for pool_a_id in range(1, pair_offsets.shape[0]):
        for pool_b_id in range(pool_a_id + 1, pair_offsets.shape[1]):
            rows.append({"pool_a": pool_a_id, "pool_b": pool_b_id, "offset_a_vs_b": float(pair_offsets[pool_a_id, pool_b_id])})
    pd.DataFrame(rows).to_csv(FINAL_PAIR_OFFSETS_PATH, index=False, encoding="utf-8-sig")
    if pair_offsets.shape[0] > 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(pair_offsets[1:, 1:], aspect="auto")
        ax.set_title("Końcowa macierz interakcji par pul")
        ax.set_xlabel("pool_b")
        ax.set_ylabel("pool_a")
        fig.colorbar(image, ax=ax, label="offset pool_a vs pool_b")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "final_pool_pair_offsets_heatmap.png", dpi=180)
        plt.close(fig)


def save_rating_range_plot(ranges: pd.DataFrame) -> None:
    if ranges.empty:
        return
    ordered = ranges.sort_values("effective_p50")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14, max(7, 0.52 * len(ordered))))
    ax.hlines(y, ordered["effective_min"], ordered["effective_max"], linewidth=1.0, label="min–max")
    ax.hlines(y, ordered["effective_p01"], ordered["effective_p99"], linewidth=3.0, label="p01–p99")
    ax.hlines(y, ordered["effective_p10"], ordered["effective_p90"], linewidth=6.0, label="p10–p90")
    ax.hlines(y, ordered["effective_p25"], ordered["effective_p75"], linewidth=10.0, label="p25–p75")
    ax.scatter(ordered["effective_p50"], y, s=35, label="mediana")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_title("Końcowy zakres ratingów zawodników według systemu")
    ax.set_xlabel("rating efektywny zawodnika")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "final_rating_ranges_by_model.png", dpi=180)
    plt.close(fig)


def rating_range_rows(
    *,
    model_name: str,
    effective_rating: np.ndarray,
    base_rating: np.ndarray,
    seen: np.ndarray,
    player_keys: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mask = seen.astype(bool) & np.isfinite(effective_rating) & np.isfinite(base_rating)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return {
            "model": model_name,
            "players_seen": 0,
        }, []

    effective = effective_rating[indices]
    base = base_rating[indices]
    min_local = int(np.argmin(effective))
    max_local = int(np.argmax(effective))
    min_index = int(indices[min_local])
    max_index = int(indices[max_local])

    summary: dict[str, object] = {
        "model": model_name,
        "players_seen": int(len(indices)),
        "effective_min": float(effective[min_local]),
        "effective_p01": float(np.quantile(effective, 0.01)),
        "effective_p10": float(np.quantile(effective, 0.10)),
        "effective_p25": float(np.quantile(effective, 0.25)),
        "effective_p50": float(np.quantile(effective, 0.50)),
        "effective_p75": float(np.quantile(effective, 0.75)),
        "effective_p90": float(np.quantile(effective, 0.90)),
        "effective_p99": float(np.quantile(effective, 0.99)),
        "effective_max": float(effective[max_local]),
        "effective_min_player_key": player_keys[min_index],
        "effective_max_player_key": player_keys[max_index],
        "base_min": float(np.min(base)),
        "base_p50": float(np.quantile(base, 0.50)),
        "base_max": float(np.max(base)),
        "sanity_warning": bool(
            effective[min_local] < SANITY_MIN_RATING
            or effective[max_local] > SANITY_MAX_RATING
        ),
    }

    order = np.argsort(effective)
    selected_positions = list(order[:TOP_BOTTOM_PLAYERS_PER_MODEL]) + list(order[-TOP_BOTTOM_PLAYERS_PER_MODEL:][::-1])
    extreme_rows: list[dict[str, object]] = []
    for rank, position in enumerate(selected_positions):
        player_index = int(indices[int(position)])
        side = "bottom" if rank < TOP_BOTTOM_PLAYERS_PER_MODEL else "top"
        side_rank = rank + 1 if side == "bottom" else rank - TOP_BOTTOM_PLAYERS_PER_MODEL + 1
        extreme_rows.append(
            {
                "model": model_name,
                "side": side,
                "rank_within_side": int(side_rank),
                "player_key": player_keys[player_index],
                "effective_rating": float(effective_rating[player_index]),
                "base_rating": float(base_rating[player_index]),
            }
        )
    return summary, extreme_rows


def save_monthly_rating_range_plots(monthly_ranges: pd.DataFrame) -> None:
    if monthly_ranges.empty:
        return
    for model_name, group in monthly_ranges.groupby("model"):
        group = group.copy()
        group["date"] = pd.to_datetime(group["month"] + "-01")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(group["date"], group["effective_min"], linewidth=0.9, label="min")
        ax.plot(group["date"], group["effective_p01"], linewidth=1.0, label="p01")
        ax.plot(group["date"], group["effective_p10"], linewidth=1.1, label="p10")
        ax.plot(group["date"], group["effective_p25"], linewidth=1.1, label="p25")
        ax.plot(group["date"], group["effective_p50"], linewidth=1.8, label="p50 — mediana")
        ax.plot(group["date"], group["effective_p75"], linewidth=1.1, label="p75")
        ax.plot(group["date"], group["effective_p90"], linewidth=1.1, label="p90")
        ax.plot(group["date"], group["effective_p99"], linewidth=1.0, label="p99")
        ax.plot(group["date"], group["effective_max"], linewidth=0.9, label="max")
        ax.set_title(f"Rozkład ratingów efektywnych w czasie: {model_name}")
        ax.set_xlabel("miesiąc")
        ax.set_ylabel("rating efektywny")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model_name)
        fig.savefig(PLOTS_DIR / f"monthly_rating_percentiles_{safe_name}.png", dpi=180)
        plt.close(fig)


def final_evaluation(cache: ReplayCache, best: np.ndarray) -> None:
    variants = named_candidate_variants(best)
    scope_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    rating_range_summary_rows: list[dict[str, object]] = []
    rating_extreme_rows: list[dict[str, object]] = []
    monthly_rating_range_rows: list[dict[str, object]] = []
    best_pool_offsets: np.ndarray | None = None
    best_pair_offsets: np.ndarray | None = None

    print("[FINAL] Evaluating best model and ablations on full history...")
    for model_name, parameters in tqdm(variants.items(), desc="final models", unit="model"):
        output = evaluate_candidate(
            cache,
            parameters,
            collect_monthly=True,
            collect_rating_ranges=model_name in RATING_RANGE_PLOT_MODELS,
        )
        (
            squared_error,
            absolute_error,
            counts,
            shared_squared_error,
            shared_counts,
            fide_squared_error,
            fide_counts,
            _,
            _,
            monthly_squared_error,
            monthly_counts,
            monthly_shared_squared_error,
            monthly_shared_counts,
            monthly_fide_squared_error,
            monthly_fide_counts,
            pool_offsets,
            pair_offsets,
            final_effective_rating,
            final_base_rating,
            final_seen,
            monthly_rating_min,
            monthly_rating_p01,
            monthly_rating_p10,
            monthly_rating_p25,
            monthly_rating_p50,
            monthly_rating_p75,
            monthly_rating_p90,
            monthly_rating_p99,
            monthly_rating_max,
            monthly_rating_seen_players,
        ) = output
        if model_name == "provisional_full":
            best_pool_offsets = pool_offsets.copy()
            best_pair_offsets = pair_offsets.copy()

        range_summary, extreme_rows = rating_range_rows(
            model_name=model_name,
            effective_rating=final_effective_rating,
            base_rating=final_base_rating,
            seen=final_seen,
            player_keys=cache.player_keys,
        )
        rating_range_summary_rows.append(range_summary)
        rating_extreme_rows.extend(extreme_rows)
        if model_name in RATING_RANGE_PLOT_MODELS:
            for slot in range(len(monthly_rating_min)):
                if np.isfinite(monthly_rating_min[slot]) and np.isfinite(monthly_rating_max[slot]):
                    monthly_rating_range_rows.append(
                        {
                            "model": model_name,
                            "month": index_to_month(cache.first_month_index + slot),
                            "players_seen": int(monthly_rating_seen_players[slot]),
                            "effective_min": float(monthly_rating_min[slot]),
                            "effective_p01": float(monthly_rating_p01[slot]),
                            "effective_p10": float(monthly_rating_p10[slot]),
                            "effective_p25": float(monthly_rating_p25[slot]),
                            "effective_p50": float(monthly_rating_p50[slot]),
                            "effective_p75": float(monthly_rating_p75[slot]),
                            "effective_p90": float(monthly_rating_p90[slot]),
                            "effective_p99": float(monthly_rating_p99[slot]),
                            "effective_max": float(monthly_rating_max[slot]),
                        }
                    )

        for code in (1, 2, 3):
            scope_rows.append(
                {
                    "model": model_name,
                    "scope": scope_name(code),
                    "games_all": int(counts[code]),
                    "mse_all": safe_mse(squared_error[code], int(counts[code])),
                    "mae_all": safe_mse(absolute_error[code], int(counts[code])),
                    "games_shared_with_fide": int(shared_counts[code]),
                    "mse_shared_with_fide": safe_mse(shared_squared_error[code], int(shared_counts[code])),
                    "fide_official_mse_on_shared": safe_mse(fide_squared_error[code], int(fide_counts[code])),
                    "max_abs_pool_offset": float(np.max(np.abs(pool_offsets))) if len(pool_offsets) else 0.0,
                    "max_abs_pair_offset": float(np.max(np.abs(pair_offsets))) if pair_offsets.size else 0.0,
                }
            )

        for slot in range(len(monthly_counts)):
            monthly_rows.append(
                {
                    "model": model_name,
                    "month": index_to_month(cache.first_month_index + slot),
                    "games_all": int(monthly_counts[slot]),
                    "mse_all": safe_mse(monthly_squared_error[slot], int(monthly_counts[slot])),
                    "games_shared_with_fide": int(monthly_shared_counts[slot]),
                    "mse_shared_with_fide": safe_mse(monthly_shared_squared_error[slot], int(monthly_shared_counts[slot])),
                    "fide_official_mse_on_shared": safe_mse(monthly_fide_squared_error[slot], int(monthly_fide_counts[slot])),
                }
            )

    scopes = pd.DataFrame(scope_rows)
    monthly = pd.DataFrame(monthly_rows)
    scopes.to_csv(FINAL_SCOPE_PATH, index=False, encoding="utf-8-sig")
    monthly.to_csv(FINAL_MONTHLY_PATH, index=False, encoding="utf-8-sig")
    ranges = pd.DataFrame(rating_range_summary_rows)
    extremes = pd.DataFrame(rating_extreme_rows)
    monthly_ranges = pd.DataFrame(monthly_rating_range_rows)
    ranges.to_csv(FINAL_RATING_RANGES_PATH, index=False, encoding="utf-8-sig")
    extremes.to_csv(FINAL_EXTREME_PLAYERS_PATH, index=False, encoding="utf-8-sig")
    monthly_ranges.to_csv(MONTHLY_RATING_RANGES_PATH, index=False, encoding="utf-8-sig")

    test = scopes[scopes["scope"] == "test"].copy()
    classic_mse = float(test[test["model"] == "classic_elo"].iloc[0]["mse_all"])
    test["improvement_vs_classic_percent"] = (classic_mse - test["mse_all"]) / classic_mse * 100.0
    test["improvement_vs_fide_on_shared_percent"] = (
        (test["fide_official_mse_on_shared"] - test["mse_shared_with_fide"])
        / test["fide_official_mse_on_shared"]
        * 100.0
    )
    test = test.sort_values("mse_all")
    test.to_csv(FINAL_TEST_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    if best_pool_offsets is not None and best_pair_offsets is not None:
        save_pool_plots(best_pool_offsets, best_pair_offsets)
    save_test_bar_plots(test)
    save_monthly_plots(monthly)
    save_evolution_plot()
    save_rating_range_plot(ranges)
    save_monthly_rating_range_plots(monthly_ranges)

    report_lines = [
        "STANDALONE DYNAMIC POOL RATING EXPERIMENT REPORT",
        "=" * 92,
        "",
        "Najlepsze parametry wybrane wyłącznie na walidacji:",
        json.dumps(params_array_to_dict(best), indent=2, ensure_ascii=False),
        "",
        "Wyniki TEST / końcowego okresu ewaluacyjnego:",
        test.to_string(index=False),
        "",
        "Interpretacja:",
        "- mse_all porównuje własne systemy na wszystkich unikalnych partiach.",
        "- mse_shared_with_fide porównuje model na partiach z benchmarkiem FIDE.",
        "- ablation study pokazuje, czy pomagają pule, dynamiczne ekspozycje,",
        "  interakcje par pul, shrinkage, centrowanie oraz liczba ekspozycji.",
        "- osobne ablacje mierzą wpływ performance ratingu pierwszych N partii",
        "  i zwiększonej reaktywności K dla zawodników prowizorycznych.",
        "- wariant provisional_fixed_1500_entry odtwarza wejście bez mechanizmu",
        "  performance: każdy nowy gracz zaczyna od 1500 i nie ma dodatkowego K.",
        "- parametr TEST nie wpływał na wybór najlepszego kandydata.",
        "",
        "Diagnostyka skali ratingów końcowych:",
        "- percentyle końcowe są liczone dokładnie; miesięczne percentyle histogramowo co 5 punktów.",
        "- min/max pomagają wychwycić eksplozję skali; p01..p99 pokazują typowy rozkład.",
        ranges.to_string(index=False),
        "",
        "Pliki diagnostyczne ratingów:",
        str(FINAL_RATING_RANGES_PATH),
        str(FINAL_EXTREME_PLAYERS_PATH),
        str(MONTHLY_RATING_RANGES_PATH),
        "",
        "Wykresy:",
        str(PLOTS_DIR),
    ]
    FINAL_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print("========== FINAL TEST ==========")
    print(test.to_string(index=False))
    print("================================")
    print(f"[FINAL] scope metrics:      {FINAL_SCOPE_PATH}")
    print(f"[FINAL] monthly metrics:    {FINAL_MONTHLY_PATH}")
    print(f"[FINAL] test comparison:    {FINAL_TEST_COMPARISON_PATH}")
    print(f"[FINAL] report:             {FINAL_REPORT_PATH}")
    print(f"[FINAL] rating ranges:      {FINAL_RATING_RANGES_PATH}")
    print(f"[FINAL] extreme players:    {FINAL_EXTREME_PLAYERS_PATH}")
    print(f"[FINAL] monthly ranges:     {MONTHLY_RATING_RANGES_PATH}")
    print(f"[FINAL] plots:              {PLOTS_DIR}")


def tune_and_select(
    con: duckdb.DuckDBPyConnection,
    parquet: str,
    membership: pd.DataFrame,
    *,
    profile_name: str,
    rebuild_cache: bool,
) -> np.ndarray:
    profile = search_profile(profile_name)
    quick = profile_name == "quick"
    fast_cache = build_replay_cache(
        con,
        parquet,
        membership,
        output_path=fast_cache_path(profile_name, profile.fast_sample_percent),
        end_month=SEARCH_VALID_END_MONTH,
        sample_percent=profile.fast_sample_percent,
        force=rebuild_cache,
        quick=quick,
    )
    finalists = evolve_parameters(fast_cache, profile=profile)

    full_cache = build_replay_cache(
        con,
        parquet,
        membership,
        output_path=full_tune_cache_path(profile_name),
        end_month=SEARCH_VALID_END_MONTH,
        sample_percent=None,
        force=rebuild_cache,
        quick=quick,
    )
    ranked = refine_on_full_cache(full_cache, finalists, profile=profile)
    best_result = ranked[0]
    best = best_result.parameters
    save_json(
        BEST_PARAMS_PATH,
        {
            "parameter_order": list(PARAMETER_NAMES),
            "parameters": params_array_to_dict(best),
            "selection_scope": "validation_only",
            "fitness": best_result.fitness,
            "validation_mse": best_result.validation_mse,
            "validation_shared_mse": best_result.validation_shared_mse,
            "validation_year_balanced_mse": best_result.validation_year_balanced_mse,
            "validation_worst_year_mse": best_result.validation_worst_year_mse,
            "train_mse": best_result.train_mse,
            "note": "TEST nie był używany do strojenia parametrów.",
        },
    )
    print(f"[TUNE] Full leaderboard:    {FULL_LEADERBOARD_PATH}")
    print(f"[TUNE] Best parameters:     {BEST_PARAMS_PATH}")
    print(f"[TUNE] Best validation MSE: {best_result.validation_mse:.8f}")
    return best


def load_best_parameters() -> np.ndarray:
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Brak {BEST_PARAMS_PATH}. Najpierw uruchom --mode tune albo --mode all.")
    payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return params_dict_to_array(payload["parameters"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.reset and OUTPUT_DIR.exists():
        print(f"[RESET] Usuwam poprzedni folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    ensure_directories()
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Nie ma pliku: {INPUT_PARQUET}")

    started = time.perf_counter()
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute("PRAGMA preserve_insertion_order=false")

    parquet = sql_path(INPUT_PARQUET)
    validate_input(con, parquet)
    latest_month = find_latest_month(con, parquet)
    final_end = FINAL_TEST_END_MONTH or latest_month
    profile = search_profile(args.profile)

    print("========== STANDALONE DYNAMIC POOL RATING EXPERIMENT ==========")
    print(f"input:                     {INPUT_PARQUET}")
    print(f"output:                    {OUTPUT_DIR}")
    print(f"cache:                     {CACHE_DIR}")
    print(f"plots:                     {PLOTS_DIR}")
    print(f"results:                   {RESULTS_DIR}")
    print(f"latest month:              {latest_month}")
    print(f"pool discovery end:        {POOL_DISCOVERY_END_MONTH}")
    print(f"search train:              {SEARCH_TRAIN_START_MONTH} .. {SEARCH_TRAIN_END_MONTH}")
    print(f"search validation:         {SEARCH_VALID_START_MONTH} .. {SEARCH_VALID_END_MONTH}")
    print(f"final evaluation:          {FINAL_TEST_START_MONTH} .. {final_end}")
    print(f"profile:                   {args.profile}")
    print(f"fast sample:               {profile.fast_sample_percent}%")
    print(f"population / generations:  {profile.population} / {profile.generations}")
    print(f"full refinement rounds:    {profile.full_refine_rounds}")
    print(f"latent pools disabled:     {args.no_pools}")
    print()

    membership = build_latent_pools(
        con,
        parquet,
        force=args.rebuild_pools,
        quick=args.profile == "quick",
        disabled=args.no_pools,
    )
    if args.mode == "pools":
        return

    if args.mode in {"all", "tune"}:
        best = tune_and_select(
            con,
            parquet,
            membership,
            profile_name=args.profile,
            rebuild_cache=(args.rebuild_cache or args.rebuild_pools),
        )
    else:
        best = load_best_parameters()

    if args.mode in {"all", "evaluate"}:
        final_cache = build_replay_cache(
            con,
            parquet,
            membership,
            output_path=final_cache_path(args.profile, final_end),
            end_month=final_end,
            sample_percent=None,
            force=(args.rebuild_cache or args.rebuild_pools),
            quick=args.profile == "quick",
        )
        final_evaluation(final_cache, best)

    con.close()
    elapsed = time.perf_counter() - started
    print()
    print(f"[DONE] total time: {elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
