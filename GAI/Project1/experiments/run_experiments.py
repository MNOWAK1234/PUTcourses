import os
import sys
import csv
import time
import argparse
import contextlib

from tqdm import tqdm

# ---------------------------------------------------------------------
# Make repo root importable when running: python experiments/run_experiments.py
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from helper import *
from src.tetris import Tetris
from agents import AGENT_REGISTRY


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_RAW_CSV_PATH = os.path.join(RESULTS_DIR, "test_raw_runs.csv")

PREFERRED_ORDER = ["random", "heuristic", "dqn", "reinforce", "a2c", "ppo", "genetic", "es"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run raw experiments for one or more Tetris agents.")
    parser.add_argument(
        "--agent_types",
        type=str,
        default="all",
        help="Comma-separated list of agent types or 'all'. Example: random,heuristic,dqn,ppo",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10,
        help="Number of games per agent.",
    )
    parser.add_argument(
        "--max_pieces",
        type=int,
        default=None,
        help="Maximum number of pieces in one game. If omitted, uses config.MAX_PIECES_PER_EVAL_GAME.",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Optional wall-clock limit for one game.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_RAW_CSV_PATH,
        help="Path to output CSV with raw per-game results.",
    )
    parser.add_argument(
        "--allow_untrained_missing",
        action="store_true",
        help="If a model is missing, still run the agent untrained.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def ensure_results_dir_exists():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_agent_types(agent_types_raw: str):
    if agent_types_raw.strip().lower() == "all":
        return [a for a in PREFERRED_ORDER if a in AGENT_REGISTRY]

    requested = []
    for part in agent_types_raw.split(","):
        agent = part.strip().lower()
        if agent in AGENT_REGISTRY and agent not in requested:
            requested.append(agent)
    return requested


def set_eval_mode_if_possible(agent):
    networks_to_set_eval = [
        "policy_network",
        "v_network",
        "actor",
        "critic",
        "central_policy_net",
        "best_individual_network",
    ]

    for net_name in networks_to_set_eval:
        if hasattr(agent, net_name):
            network_component = getattr(agent, net_name)
            if (
                network_component is not None
                and hasattr(network_component, "eval")
                and callable(network_component.eval)
            ):
                network_component.eval()


def load_agent_for_test(agent_type: str, seed: int, model_base_dir: str, allow_untrained_missing: bool, verbose: bool):
    agent_class = AGENT_REGISTRY.get(agent_type)
    if not agent_class:
        if verbose:
            print(f"Agent '{agent_type}' not found in AGENT_REGISTRY.")
        return None, "missing-registry", ""

    with suppress_output(not verbose):
        agent = agent_class(state_size=config.STATE_SIZE, seed=seed)

    model_free_agents = {"random", "heuristic"}

    if agent_type in model_free_agents:
        set_eval_mode_if_possible(agent)
        return agent, ("random" if agent_type == "random" else "rule-based"), ""

    with suppress_output(not verbose):
        model_path = find_latest_or_best_model_path(agent_type, model_base_dir)

    if not model_path:
        if allow_untrained_missing:
            set_eval_mode_if_possible(agent)
            return agent, "untrained-missing-model", ""
        else:
            if verbose:
                print(f"No model found for agent '{agent_type}'.")
            return None, "missing-model", ""

    try:
        with suppress_output(not verbose):
            agent.load(model_path)
        set_eval_mode_if_possible(agent)
        return agent, "loaded", os.path.basename(model_path)
    except Exception as e:
        if verbose:
            print(f"Could not load model for {agent_type}: {e}")
        if allow_untrained_missing:
            set_eval_mode_if_possible(agent)
            return agent, "untrained-load-failed", os.path.basename(model_path)
        return None, "load-error", os.path.basename(model_path)


def select_action_safe(agent, state_features, env):
    try:
        return agent.select_action(state_features, env, epsilon_override=0.0)
    except TypeError:
        return agent.select_action(state_features, env)


def play_game(env: Tetris, agent, game_seed: int, max_pieces: int, max_seconds: float | None):
    with suppress_output(True):
        setup_seeds(game_seed)

    current_board_features = env.reset()
    if config.DEVICE.type == "cuda":
        current_board_features = current_board_features.cuda()

    if hasattr(agent, "reset") and callable(agent.reset):
        agent.reset()

    game_over = False
    pieces_played = 0
    start_time = time.time()

    while not game_over and pieces_played < max_pieces:
        if max_seconds is not None and (time.time() - start_time) >= max_seconds:
            break

        action_result = select_action_safe(agent, current_board_features, env)
        action_tuple = action_result[0] if isinstance(action_result, tuple) else action_result

        _, game_over = env.step(action_tuple, render=False)
        pieces_played += 1

        if not game_over:
            current_board_features = env.get_state_properties(env.board)
            if config.DEVICE.type == "cuda":
                current_board_features = current_board_features.cuda()

    elapsed_seconds = time.time() - start_time

    return {
        "score": env.score,
        "pieces_played": pieces_played,
        "tetrominoes": env.tetrominoes,
        "lines_cleared": env.cleared_lines,
        "elapsed_seconds": elapsed_seconds,
        "game_over": int(game_over),
    }


def csv_headers():
    return [
        "agent",
        "game_index",
        "seed",
        "load_status",
        "model_file",
        "score",
        "pieces_played",
        "tetrominoes",
        "lines_cleared",
        "elapsed_seconds",
        "game_over",
    ]


def main():
    opt = get_args()
    ensure_results_dir_exists()
    config.ensure_model_dir_exists()

    agent_types = normalize_agent_types(opt.agent_types)
    if not agent_types:
        print("No valid agent types selected.")
        return

    max_pieces = opt.max_pieces if opt.max_pieces is not None else config.MAX_PIECES_PER_EVAL_GAME
    model_base_dir = config.MODEL_DIR

    master_seed = config.SEED + 1000
    with suppress_output(True):
        setup_seeds(master_seed)

    output_dir = os.path.dirname(opt.csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = []

    if opt.verbose:
        print(f"Agents to test: {', '.join(agent_types)}")
        print(f"Games per agent: {opt.num_games}")
        print(f"Max pieces per game: {max_pieces}")
        if opt.max_seconds is not None:
            print(f"Max seconds per game: {opt.max_seconds}")

    for agent_index, agent_type in enumerate(agent_types):
        agent_seed = master_seed + 10000 + agent_index
        agent, load_status, model_file = load_agent_for_test(
            agent_type=agent_type,
            seed=agent_seed,
            model_base_dir=model_base_dir,
            allow_untrained_missing=opt.allow_untrained_missing,
            verbose=opt.verbose,
        )

        if agent is None:
            if opt.verbose:
                print(f"Skipping agent {agent_type} due to load_status={load_status}")
            continue

        env = Tetris()

        pbar = tqdm(
            total=opt.num_games,
            desc=f"{agent_type.upper()} eval",
            unit="game",
            dynamic_ncols=True,
            leave=True,
        )

        best_score_for_agent = None

        for game_idx in range(opt.num_games):
            game_seed = master_seed + agent_index * 100000 + game_idx
            result = play_game(
                env=env,
                agent=agent,
                game_seed=game_seed,
                max_pieces=max_pieces,
                max_seconds=opt.max_seconds,
            )

            row = {
                "agent": agent_type,
                "game_index": game_idx + 1,
                "seed": game_seed,
                "load_status": load_status,
                "model_file": model_file,
                "score": result["score"],
                "pieces_played": result["pieces_played"],
                "tetrominoes": result["tetrominoes"],
                "lines_cleared": result["lines_cleared"],
                "elapsed_seconds": round(result["elapsed_seconds"], 4),
                "game_over": result["game_over"],
            }
            rows.append(row)

            if best_score_for_agent is None or row["score"] > best_score_for_agent:
                best_score_for_agent = row["score"]

            pbar.update(1)
            pbar.set_postfix(
                {
                    "score": int(row["score"]),
                    "best": int(best_score_for_agent if best_score_for_agent is not None else 0),
                    "lines": int(row["lines_cleared"]),
                }
            )

        pbar.close()

    if not rows:
        print("No results to save.")
        return

    with open(opt.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved raw experiment results to: {opt.csv_path}")


if __name__ == "__main__":
    main()