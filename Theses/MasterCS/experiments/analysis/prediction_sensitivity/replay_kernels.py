from __future__ import annotations

import inspect
import re
from typing import Any

import numpy as np


def _strip_decorators(source: str) -> str:
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


def _compile_from_replay_source(exp, new_source: str, function_name: str):
    """Compile a modified copy of run_experiments.replay_candidate."""
    try:
        from numba import njit
    except ImportError as exc:
        raise SystemExit("This script needs numba, the same dependency as run_experiments.py.") from exc

    namespace = dict(exp.__dict__)
    exec(new_source, namespace)
    py_func = namespace[function_name]
    return njit(cache=False, nogil=True)(py_func)


def compile_rating_shock_replay(exp):
    """Create replay_candidate_with_rating_shock by modifying the original replay kernel.

    Shock semantics:
    - At the first event whose month >= shock_month_index, add shock_rating_delta
      to the internal slow rating of all already-seen players.
    - If shock_pool_id >= 0, only players whose static pool equals shock_pool_id
      are affected.
    - If shock_pool_id == -1, all already-seen players are affected.
    """
    original = inspect.getsource(getattr(exp.replay_candidate, "py_func", exp.replay_candidate))
    source = _strip_decorators(original)
    source = source.replace("def replay_candidate(", "def replay_candidate_with_rating_shock(", 1)
    source = source.replace(
        "    last_month_index: int,\n):",
        "    last_month_index: int,\n    shock_month_index: int,\n    shock_pool_id: int,\n    shock_rating_delta: float,\n):",
        1,
    )
    source = source.replace(
        "    last_rating_range_month = -1\n    i = 0",
        "    last_rating_range_month = -1\n    shock_applied = 0\n    i = 0",
        1,
    )
    source = source.replace(
        "        current_month = month_index_values[i]\n",
        """        current_month = month_index_values[i]\n        if shock_month_index >= 0 and shock_applied == 0 and current_month >= shock_month_index:\n            for shock_player in range(players_count):\n                if seen[shock_player] != 0:\n                    if shock_pool_id < 0 or static_pool[shock_player] == shock_pool_id:\n                        rating[shock_player] = _clip(rating[shock_player] + shock_rating_delta, -2000.0, 6000.0)\n            shock_applied = 1\n""",
        1,
    )
    return _compile_from_replay_source(exp, source, "replay_candidate_with_rating_shock")


def run_rating_shock_replay(exp, cache, parameters, *, shock_month_index: int, shock_pool_id: int, shock_rating_delta: float):
    kernel = compile_rating_shock_replay(exp)
    return kernel(
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
        exp.POOL_NEW_PLAYER_ASSIGNMENT_VOTES,
        True,
        False,
        cache.first_month_index,
        cache.last_month_index,
        int(shock_month_index),
        int(shock_pool_id),
        float(shock_rating_delta),
    )


def compile_calibration_replay(exp):
    """Create replay_candidate_with_calibration by modifying the original replay kernel.

    The calibration buckets are collected on the final test scope only.
    For each prediction bucket, the script stores:
    - sum of predicted expected scores,
    - sum of observed scores,
    - number of games.
    """
    original = inspect.getsource(getattr(exp.replay_candidate, "py_func", exp.replay_candidate))
    source = _strip_decorators(original)
    source = source.replace("def replay_candidate(", "def replay_candidate_with_calibration(", 1)
    source = source.replace(
        "    last_month_index: int,\n):",
        "    last_month_index: int,\n    calibration_bins_count: int,\n):",
        1,
    )
    source = source.replace(
        "    fold_count = np.zeros(4, dtype=np.int64)\n",
        "    fold_count = np.zeros(4, dtype=np.int64)\n\n    calibration_sum_pred = np.zeros(calibration_bins_count, dtype=np.float64)\n    calibration_sum_actual = np.zeros(calibration_bins_count, dtype=np.float64)\n    calibration_count = np.zeros(calibration_bins_count, dtype=np.int64)\n",
        1,
    )
    source = source.replace(
        "            residual = actual - predicted\n\n            scope = scopes[row]\n",
        """            residual = actual - predicted\n\n            scope = scopes[row]\n            if scope == 3 and calibration_bins_count > 0:\n                calibration_bucket = int(predicted * calibration_bins_count)\n                if calibration_bucket < 0:\n                    calibration_bucket = 0\n                elif calibration_bucket >= calibration_bins_count:\n                    calibration_bucket = calibration_bins_count - 1\n                calibration_sum_pred[calibration_bucket] += predicted\n                calibration_sum_actual[calibration_bucket] += actual\n                calibration_count[calibration_bucket] += 1\n""",
        1,
    )
    source = source.replace(
        "        monthly_rating_seen_players,\n    )",
        "        monthly_rating_seen_players,\n        calibration_sum_pred,\n        calibration_sum_actual,\n        calibration_count,\n    )",
        1,
    )
    return _compile_from_replay_source(exp, source, "replay_candidate_with_calibration")


def run_calibration_replay(exp, cache, parameters, *, bins: int = 20):
    kernel = compile_calibration_replay(exp)
    return kernel(
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
        exp.POOL_NEW_PLAYER_ASSIGNMENT_VOTES,
        False,
        False,
        cache.first_month_index,
        cache.last_month_index,
        int(bins),
    )
