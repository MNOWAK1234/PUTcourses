import os
import csv
import time
import argparse
import contextlib

from tqdm import tqdm

import config
from helper import *
from src.tetris import Tetris
from agents import AGENT_REGISTRY


LOCAL_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(LOCAL_PROJECT_ROOT, "results")


def get_args():
    parser = argparse.ArgumentParser("Run fixed-time benchmarks for non-learning agents.")
    parser.add_argument("--agents", type=str, default="random,heuristic")
    parser.add_argument("--time_budget_minutes", type=float, default=30.0)
    parser.add_argument("--render_game", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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


def init_minute_csv(agent_type: str) -> str:
    ensure_results_dir_exists()
    csv_path = os.path.join(RESULTS_DIR, f"{agent_type}_minute_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "minute",
                "elapsed_seconds",
                "games_played",
                "best_score",
                "avg_score",
                "last_score",
                "last_lines",
            ],
        )
        writer.writeheader()
    return csv_path


def append_minute_row(csv_path: str, row: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def normalize_agents(agent_str: str):
    valid = {"random", "heuristic"}
    result = []
    for a in agent_str.split(","):
        a = a.strip().lower()
        if a in valid and a not in result:
            result.append(a)
    return result


def run_agent(agent_type: str, time_budget_minutes: float, render_game: bool, verbose: bool):
    csv_path = init_minute_csv(agent_type)
    time_budget_seconds = time_budget_minutes * 60.0
    start_time = time.time()
    next_minute_mark = 1
    last_pbar_sync = 0.0

    env = Tetris()
    with suppress_output(not verbose):
        agent = AGENT_REGISTRY[agent_type](state_size=config.STATE_SIZE, seed=config.SEED)

    games_played = 0
    total_score = 0.0
    best_score = 0.0
    last_score = 0.0
    last_lines = 0

    current_features = env.reset()
    if config.DEVICE.type == "cuda":
        current_features = current_features.cuda()
    if hasattr(agent, "reset"):
        agent.reset()

    pbar = tqdm(
        total=time_budget_seconds,
        desc=f"{agent_type.upper()} benchmark",
        unit="s",
        dynamic_ncols=True,
        leave=True,
    )

    while True:
        elapsed = time.time() - start_time

        if elapsed - last_pbar_sync >= 1.0:
            target_n = min(elapsed, time_budget_seconds)
            delta = target_n - pbar.n
            if delta > 0:
                pbar.update(delta)
            last_pbar_sync = elapsed

        if elapsed >= time_budget_seconds:
            break

        action_tuple, _ = agent.select_action(current_features, env, epsilon_override=0.0)
        reward, game_over = env.step(action_tuple, render=render_game)

        best_score = max(best_score, float(env.score))

        while elapsed >= next_minute_mark * 60:
            avg_score = total_score / games_played if games_played > 0 else 0.0
            append_minute_row(
                csv_path,
                {
                    "minute": next_minute_mark,
                    "elapsed_seconds": round(next_minute_mark * 60, 2),
                    "games_played": games_played,
                    "best_score": round(best_score, 4),
                    "avg_score": round(avg_score, 4),
                    "last_score": round(last_score, 4),
                    "last_lines": int(last_lines),
                },
            )
            next_minute_mark += 1

        pbar.set_postfix(
            {
                "games": games_played,
                "score": int(env.score),
                "best": int(best_score),
                "t": f"{elapsed/60:.1f}m",
            }
        )

        if game_over:
            games_played += 1
            total_score += env.score
            last_score = env.score
            last_lines = env.cleared_lines
            best_score = max(best_score, float(env.score))

            current_features = env.reset()
            if config.DEVICE.type == "cuda":
                current_features = current_features.cuda()
            if hasattr(agent, "reset"):
                agent.reset()
        else:
            current_features = env.get_state_properties(env.board)
            if config.DEVICE.type == "cuda":
                current_features = current_features.cuda()

    final_elapsed = min(time.time() - start_time, time_budget_seconds)
    delta = final_elapsed - pbar.n
    if delta > 0:
        pbar.update(delta)

    pbar.close()

    avg_score = total_score / games_played if games_played > 0 else 0.0
    append_minute_row(
        csv_path,
        {
            "minute": max(1, int(final_elapsed // 60)),
            "elapsed_seconds": round(final_elapsed, 2),
            "games_played": games_played,
            "best_score": round(best_score, 4),
            "avg_score": round(avg_score, 4),
            "last_score": round(last_score, 4),
            "last_lines": int(last_lines),
        },
    )

    if verbose:
        print(f"{agent_type}: CSV saved to {csv_path}")


def main():
    opt = get_args()
    ensure_results_dir_exists()

    with suppress_output(not opt.verbose):
        setup_seeds()

    agents = normalize_agents(opt.agents)
    if not agents:
        print("No valid baseline agents selected.")
        return

    print(f"Running baseline agents: {', '.join(agents)}")
    print(f"Time budget per agent: {opt.time_budget_minutes} minute(s)")

    for agent_type in agents:
        print(f"=== Running {agent_type.upper()} ===")
        run_agent(agent_type, opt.time_budget_minutes, opt.render_game, opt.verbose)
        print(f"[OK] {agent_type.upper()} finished")


if __name__ == "__main__":
    main()