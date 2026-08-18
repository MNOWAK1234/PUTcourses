from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def import_experiment_module(path: str | Path = "run_experiments.py"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("run_experiments_imported_robustness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs(out: str | Path = "outputs/robustness_ablation") -> tuple[Path, Path]:
    out = Path(out)
    results = out / "results"
    plots = out / "plots"
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return results, plots


def read_csv_required(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_best_parameters(exp, path: str | Path = "experiments/results/best_model_parameters.json") -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    incoming = payload.get("parameters", payload)
    values = exp.with_defaults()
    for name in exp.PARAMETER_NAMES:
        if name in incoming:
            values[name] = float(incoming[name])
    return exp.clip_candidate(exp.params_dict_to_array(values))


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


def safe_mse(error_sum: float, count: int) -> float:
    return float(error_sum / count) if int(count) > 0 else float("nan")


def output_scope_records(exp, cache, model_name: str, output):
    squared_error, absolute_error, counts, shared_error, shared_counts, fide_error, fide_counts = output[:7]
    rows = []
    for scope_code in (1, 2, 3):
        count = int(counts[scope_code])
        shared_count = int(shared_counts[scope_code])
        fide_count = int(fide_counts[scope_code])
        rows.append({
            "model": model_name,
            "scope": exp.scope_name(scope_code),
            "games_all": count,
            "mse_all": safe_mse(float(squared_error[scope_code]), count),
            "mae_all": float(absolute_error[scope_code] / count) if count else float("nan"),
            "games_shared_with_fide": shared_count,
            "mse_shared_with_fide": safe_mse(float(shared_error[scope_code]), shared_count),
            "fide_official_mse_on_shared": safe_mse(float(fide_error[scope_code]), fide_count),
        })
    return rows


def evaluate_model(exp, cache, name: str, params: np.ndarray):
    output = exp.evaluate_candidate(
        cache,
        exp.clip_candidate(params),
        collect_monthly=False,
        collect_rating_ranges=False,
    )
    return pd.DataFrame(output_scope_records(exp, cache, name, output))


def add_classic_improvement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "classic_elo" not in set(df["model"]):
        return df
    base = df[df["model"] == "classic_elo"][["scope", "mse_all"]].rename(columns={"mse_all": "classic_mse"})
    df = df.merge(base, on="scope", how="left")
    df["improvement_vs_classic_percent"] = (df["classic_mse"] - df["mse_all"]) / df["classic_mse"] * 100.0
    return df


def month_index(value: str) -> int:
    year, month = value.split("-")[:2]
    return int(year) * 12 + int(month) - 1


def add_previous_extra_to_path() -> None:
    previous = Path(__file__).resolve().parent.parent / "prediction_sensitivity"
    if previous.exists():
        sys.path.insert(0, str(previous))
