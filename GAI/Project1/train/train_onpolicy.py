import os
import csv
import glob
import time
import warnings
import argparse
import contextlib

from tqdm import tqdm

import config
from agents import PPOAgent, A2CAgent
from helper import *
from src.tetris import Tetris


warnings.filterwarnings(
    "ignore",
    message="Using a target size",
    category=UserWarning,
)

LOCAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(LOCAL_PROJECT_ROOT, "results")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("""Train A2C or PPO Agents for Tetris""")
    parser.add_argument("--agent_type", type=str, default="ppo", choices=["a2c", "ppo"])
    parser.add_argument("--total_steps", type=int, default=None, help="Used only when time budget is NOT provided.")
    parser.add_argument("--num_games", type=int, default=None, help="Used only when time budget is NOT provided.")
    parser.add_argument("--time_budget_minutes", type=float, default=None)
    parser.add_argument("--render_game", action="store_true", help="Render the game.")
    parser.add_argument("--print_every_games", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs.")
    return parser.parse_args()


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


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
                "steps_done",
                "best_score",
                "avg_score_recent",
                "last_score",
                "last_lines",
                "loss",
            ],
        )
        writer.writeheader()
    return csv_path


def append_minute_row(csv_path: str, row: dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def remove_existing_agent_models(model_dir: str, agent_prefix: str):
    pattern = os.path.join(model_dir, f"{agent_prefix}_score_*.pth")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass


def train(opt: argparse.Namespace):
    agent_type = opt.agent_type.lower()
    current_model_base_dir = config.MODEL_DIR
    use_time_budget = opt.time_budget_minutes is not None
    time_budget_seconds = opt.time_budget_minutes * 60 if use_time_budget else None

    if agent_type == "ppo":
        with suppress_output(not opt.verbose):
            controller = PPOAgent(state_size=config.STATE_SIZE, seed=config.SEED)
        max_steps = opt.total_steps if opt.total_steps is not None else config.PPO_TOTAL_PIECES
        max_games = opt.num_games if opt.num_games is not None else config.PPO_TRAIN_GAMES

    elif agent_type == "a2c":
        with suppress_output(not opt.verbose):
            controller = A2CAgent(state_size=config.STATE_SIZE, seed=config.SEED)
        max_steps = opt.total_steps if opt.total_steps is not None else config.A2C_TOTAL_PIECES
        max_games = opt.num_games if opt.num_games is not None else config.A2C_TRAIN_GAMES

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    env = Tetris()
    state_features = env.reset()
    if config.DEVICE.type == "cuda":
        state_features = state_features.cuda()

    current_total_steps = 0
    games_played_count = 0
    current_game_score = 0

    total_score_for_avg = 0
    total_lines_for_avg = 0
    games_since_last_refresh = 0
    highest_score_this_session = -1
    last_loss_str = ""
    saved_model_path = None

    start_time = time.time()
    next_minute_mark = 1
    csv_path = init_minute_csv(agent_type)

    if use_time_budget:
        pbar = tqdm(
            total=time_budget_seconds,
            desc=f"{agent_type.upper()} training",
            unit="s",
            dynamic_ncols=True,
            leave=True,
        )
    else:
        if max_steps is not None:
            progress_total = max_steps
            progress_unit = "step"
        else:
            progress_total = max_games
            progress_unit = "game"

        pbar = tqdm(
            total=progress_total,
            desc=f"{agent_type.upper()} training",
            unit=progress_unit,
            dynamic_ncols=True,
            leave=True,
        )

    last_pbar_sync = 0.0

    while True:
        elapsed = time.time() - start_time

        if use_time_budget:
            if elapsed - last_pbar_sync >= 1.0:
                target_n = min(elapsed, time_budget_seconds)
                delta = target_n - pbar.n
                if delta > 0:
                    pbar.update(delta)
                last_pbar_sync = elapsed

            if elapsed >= time_budget_seconds:
                break
        else:
            if max_steps is not None and current_total_steps >= max_steps:
                break
            if max_games is not None and games_played_count >= max_games:
                break

        action_tuple, aux_info = controller.select_action(state_features, env)

        reward, game_over = env.step(action_tuple, render=opt.render_game)
        current_game_score += reward
        current_total_steps += 1

        if not use_time_budget and max_steps is not None:
            pbar.update(1)

        next_state_features = env.get_state_properties(env.board)
        if config.DEVICE.type == "cuda":
            next_state_features = next_state_features.cuda()

        if agent_type == "ppo":
            controller.learn(
                state_features=state_features,
                reward=reward,
                next_state_features=next_state_features,
                done=game_over,
                aux_info=aux_info,
            )
        else:
            controller.learn(
                reward=reward,
                next_state_features=next_state_features,
                done=game_over,
                aux_info=aux_info,
            )

        state_features = next_state_features

        if game_over:
            games_played_count += 1
            if not use_time_budget and max_steps is None:
                pbar.update(1)

            games_since_last_refresh += 1
            total_score_for_avg += current_game_score
            total_lines_for_avg += env.cleared_lines

            if agent_type == "ppo":
                controller.learn_on_episode_end()
                loss_parts = [f"{k}:{float(v):.3f}" for k, v in controller.last_loss.items()]
                last_loss_str = ", ".join(loss_parts) if loss_parts else ""
            else:
                last_loss_str = f"A:{float(controller.last_loss[0]):.3f},C:{float(controller.last_loss[1]):.3f}"

            if current_game_score > highest_score_this_session:
                highest_score_this_session = current_game_score
                agent_prefix = get_agent_file_prefix(agent_type)

                remove_existing_agent_models(current_model_base_dir, agent_prefix)

                saved_model_path = os.path.join(
                    current_model_base_dir,
                    f"{agent_prefix}_score_{int(current_game_score)}.pth",
                )
                with suppress_output(not opt.verbose):
                    controller.save(saved_model_path)

            recent_avg_score = (
                total_score_for_avg / games_since_last_refresh if games_since_last_refresh > 0 else 0
            )
            recent_avg_lines = (
                total_lines_for_avg / games_since_last_refresh if games_since_last_refresh > 0 else 0
            )

            while elapsed >= next_minute_mark * 60:
                append_minute_row(
                    csv_path,
                    {
                        "minute": next_minute_mark,
                        "elapsed_seconds": round(next_minute_mark * 60, 2),
                        "games_played": games_played_count,
                        "steps_done": current_total_steps,
                        "best_score": int(highest_score_this_session if highest_score_this_session > -1 else 0),
                        "avg_score_recent": round(recent_avg_score, 4),
                        "last_score": int(current_game_score),
                        "last_lines": int(env.cleared_lines),
                        "loss": last_loss_str,
                    },
                )
                next_minute_mark += 1

            if games_since_last_refresh >= opt.print_every_games:
                pbar.set_postfix(
                    {
                        "games": games_played_count,
                        "steps": current_total_steps,
                        "score": int(current_game_score),
                        "avg": f"{recent_avg_score:.1f}",
                        "lines": f"{recent_avg_lines:.1f}",
                        "best": int(highest_score_this_session if highest_score_this_session > -1 else 0),
                        "t": f"{elapsed/60:.1f}m",
                    }
                )
                total_score_for_avg = 0
                total_lines_for_avg = 0
                games_since_last_refresh = 0
            else:
                pbar.set_postfix(
                    {
                        "games": games_played_count,
                        "steps": current_total_steps,
                        "score": int(current_game_score),
                        "best": int(highest_score_this_session if highest_score_this_session > -1 else 0),
                        "t": f"{elapsed/60:.1f}m",
                    }
                )

            controller.reset()
            current_game_score = 0
            state_features = env.reset()
            if config.DEVICE.type == "cuda":
                state_features = state_features.cuda()

    if use_time_budget:
        final_elapsed = min(time.time() - start_time, time_budget_seconds)
        delta = final_elapsed - pbar.n
        if delta > 0:
            pbar.update(delta)

    pbar.close()

    final_elapsed = time.time() - start_time
    final_recent_avg = total_score_for_avg / games_since_last_refresh if games_since_last_refresh > 0 else 0

    append_minute_row(
        csv_path,
        {
            "minute": max(1, int(final_elapsed // 60)),
            "elapsed_seconds": round(final_elapsed, 2),
            "games_played": games_played_count,
            "steps_done": current_total_steps,
            "best_score": int(highest_score_this_session if highest_score_this_session > -1 else 0),
            "avg_score_recent": round(final_recent_avg, 4),
            "last_score": int(current_game_score),
            "last_lines": int(env.cleared_lines),
            "loss": last_loss_str,
        },
    )

    if opt.verbose:
        print(f"Saved model: {saved_model_path}")
        print(f"Minute stats CSV: {csv_path}")


if __name__ == "__main__":
    opt = get_args()
    config.ensure_model_dir_exists()
    ensure_results_dir_exists()
    with suppress_output(not opt.verbose):
        setup_seeds()
    train(opt)