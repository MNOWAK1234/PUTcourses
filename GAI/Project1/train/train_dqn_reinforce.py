import os
import csv
import glob
import time
import argparse
import contextlib

from tqdm import tqdm

import config
from helper import *
from src.tetris import Tetris
from agents import DQNAgent, REINFORCEAgent


LOCAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(LOCAL_PROJECT_ROOT, "results")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("""Train DQN or REINFORCE Agents for Tetris""")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="dqn",
        choices=["dqn", "reinforce"],
        help="Type of agent to train (dqn or reinforce).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Used only when time budget is NOT provided.",
    )
    parser.add_argument("--time_budget_minutes", type=float, default=None)
    parser.add_argument("--render_game", action="store_true", help="Render the game during training.")
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
                "epochs_done",
                "best_score",
                "avg_score",
                "last_score",
                "last_lines",
                "epsilon",
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
    use_time_budget = opt.time_budget_minutes is not None
    time_budget_seconds = opt.time_budget_minutes * 60 if use_time_budget else None

    if agent_type == "dqn":
        with suppress_output(not opt.verbose):
            controller = DQNAgent(state_size=config.STATE_SIZE)

        total_learning_epochs = (
            opt.num_epochs if opt.num_epochs is not None else config.DQN_NUM_EPOCHS
        )
        initial_epsilon = config.DQN_EPSILON_START
        final_epsilon = config.DQN_EPSILON_MIN
        num_decay_learning_steps = config.DQN_EPSILON_DECAY_EPOCHS

    elif agent_type == "reinforce":
        with suppress_output(not opt.verbose):
            controller = REINFORCEAgent(state_size=config.STATE_SIZE)

        total_learning_epochs = (
            opt.num_epochs if opt.num_epochs is not None else config.REINFORCE_TRAIN_GAMES
        )
        initial_epsilon = None
        final_epsilon = None
        num_decay_learning_steps = None
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    current_model_base_dir = config.MODEL_DIR

    env = Tetris()
    state = env.reset()
    if config.DEVICE.type == "cuda":
        state = state.cuda()

    current_epoch = 0
    games_played_this_run = 0
    current_game_score = 0
    total_score_all_games = 0.0
    highest_score_this_session = -1
    start_time = time.time()
    next_minute_mark = 1
    last_loss_value = None
    csv_path = init_minute_csv(agent_type)
    saved_model_path = None

    if use_time_budget:
        pbar = tqdm(
            total=time_budget_seconds,
            desc=f"{agent_type.upper()} training",
            unit="s",
            dynamic_ncols=True,
            leave=True,
        )
    else:
        pbar = tqdm(
            total=total_learning_epochs,
            desc=f"{agent_type.upper()} training",
            unit="epoch",
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
            if current_epoch >= total_learning_epochs:
                break

        if agent_type == "dqn":
            epsilon = final_epsilon + (
                max(num_decay_learning_steps - current_epoch, 0)
                * (initial_epsilon - final_epsilon)
                / num_decay_learning_steps
            )
            action_tuple, state_info = controller.select_action(state, env, epsilon_override=epsilon)
        else:
            epsilon = None
            action_tuple, state_info = controller.select_action(state, env)

        reward, game_over = env.step(action_tuple, render=opt.render_game)
        new_state = env.get_state_properties(env.board)
        if config.DEVICE.type == "cuda":
            new_state = new_state.cuda()
        current_game_score += reward

        if agent_type == "dqn":
            controller.expand_memory(reward=reward, done=game_over, state_info=state_info)
        else:
            controller.expand_memory(reward=reward, state_info=state_info)

        if game_over:
            games_played_this_run += 1
            total_score_all_games += current_game_score
            avg_score = total_score_all_games / games_played_this_run if games_played_this_run > 0 else 0.0

            learned_this_epoch = False
            loss_value = None

            if agent_type == "dqn":
                if (
                    len(controller.memory) >= config.DQN_BATCH_SIZE
                    and len(controller.memory) >= (config.DQN_BUFFER_SIZE / 10)
                ):
                    experiences = controller.memory.sample()
                    if experiences[0].size(0) >= controller.memory.batch_size:
                        controller.learn_from_ReplayBuffer(experiences, config.DQN_GAMMA)
                        controller.learning_steps_done += 1
                        learned_this_epoch = True
                        loss_value = float(controller.last_loss)
            else:
                controller.learn_episode()
                controller.episodes_done += 1
                learned_this_epoch = True
                loss_value = float(controller.last_loss)

            last_loss_value = loss_value

            if learned_this_epoch:
                current_epoch += 1
                if not use_time_budget:
                    pbar.update(1)

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

            while elapsed >= next_minute_mark * 60:
                append_minute_row(
                    csv_path,
                    {
                        "minute": next_minute_mark,
                        "elapsed_seconds": round(next_minute_mark * 60, 2),
                        "games_played": games_played_this_run,
                        "epochs_done": current_epoch,
                        "best_score": int(highest_score_this_session if highest_score_this_session > -1 else 0),
                        "avg_score": round(avg_score, 4),
                        "last_score": int(current_game_score),
                        "last_lines": int(env.cleared_lines),
                        "epsilon": "" if epsilon is None else round(float(epsilon), 6),
                        "loss": "" if loss_value is None else round(float(loss_value), 6),
                    },
                )
                next_minute_mark += 1

            postfix = {
                "game": games_played_this_run,
                "score": int(current_game_score),
                "avg": f"{avg_score:.1f}",
                "lines": env.cleared_lines,
                "best": int(highest_score_this_session if highest_score_this_session > -1 else 0),
                "t": f"{elapsed/60:.1f}m",
            }
            if agent_type == "dqn":
                postfix["eps"] = f"{epsilon:.4f}"
            if loss_value is not None:
                postfix["loss"] = f"{loss_value:.2f}"

            pbar.set_postfix(postfix)

            controller.reset()
            current_game_score = 0
            state = env.reset()
            if config.DEVICE.type == "cuda":
                state = state.cuda()
        else:
            state = new_state

    if use_time_budget:
        final_elapsed = min(time.time() - start_time, time_budget_seconds)
        delta = final_elapsed - pbar.n
        if delta > 0:
            pbar.update(delta)

    pbar.close()

    final_elapsed = time.time() - start_time
    final_avg = total_score_all_games / games_played_this_run if games_played_this_run > 0 else 0.0
    final_minute = max(1, int(final_elapsed // 60))

    append_minute_row(
        csv_path,
        {
            "minute": final_minute,
            "elapsed_seconds": round(final_elapsed, 2),
            "games_played": games_played_this_run,
            "epochs_done": current_epoch,
            "best_score": int(highest_score_this_session if highest_score_this_session > -1 else 0),
            "avg_score": round(final_avg, 4),
            "last_score": int(current_game_score),
            "last_lines": int(env.cleared_lines),
            "epsilon": "" if epsilon is None else round(float(epsilon), 6),
            "loss": "" if last_loss_value is None else round(float(last_loss_value), 6),
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