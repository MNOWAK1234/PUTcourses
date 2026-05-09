import os
import sys
import csv
import time
import cv2
import imageio
import argparse
import contextlib

import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, deque


# ---------------------------------------------------------------------
# Make repo root importable when running: python experiments/save_gifs.py
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from helper import *
from src.tetris import Tetris
from agents import AGENT_REGISTRY
from agents.genetic_agent import GeneticAgent


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
GIF_DIR = os.path.join(RESULTS_DIR, "gifs")
DEFAULT_INPUT_CSV = os.path.join(RESULTS_DIR, "test_raw_runs.csv")

PREFERRED_ORDER = ["random", "heuristic", "dqn", "reinforce", "a2c", "ppo", "genetic", "es"]


def get_args():
    parser = argparse.ArgumentParser("Save GIFs of best runs based on previously saved raw results.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Raw results CSV produced by run_experiments.py",
    )
    parser.add_argument(
        "--agent_types",
        type=str,
        default="all",
        help="Comma-separated list of agent types or 'all'.",
    )
    parser.add_argument(
        "--max_pieces",
        type=int,
        default=400,
        help="Maximum number of pieces during best-run replay for GIF generation.",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Optional wall-clock limit for one replay.",
    )
    parser.add_argument(
        "--gif_target_mb",
        type=int,
        default=50,
        help="Maximum target GIF size in MB.",
    )
    parser.add_argument(
        "--max_frames_to_keep",
        type=int,
        default=1200,
        help="Maximum frames kept for one GIF.",
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


def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)


def normalize_agent_types(agent_types_raw: str):
    if agent_types_raw.strip().lower() == "all":
        return [a for a in PREFERRED_ORDER if a in AGENT_REGISTRY]

    requested = []
    for part in agent_types_raw.split(","):
        agent = part.strip().lower()
        if agent in AGENT_REGISTRY and agent not in requested:
            requested.append(agent)
    return requested


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "agent": row["agent"].strip().lower(),
                        "game_index": int(row["game_index"]),
                        "seed": int(row["seed"]),
                        "load_status": row["load_status"],
                        "model_file": row["model_file"],
                        "score": float(row["score"]),
                    }
                )
            except Exception:
                continue
    return rows


def pick_best_rows(rows, selected_agents):
    grouped = defaultdict(list)
    for row in rows:
        if row["agent"] in selected_agents:
            grouped[row["agent"]].append(row)

    best_rows = {}
    for agent in selected_agents:
        if agent not in grouped:
            continue
        best_rows[agent] = max(grouped[agent], key=lambda r: r["score"])
    return best_rows


def set_eval_mode_if_possible(agent):
    networks_to_eval = [
        "policy_network",
        "v_network",
        "actor",
        "critic",
        "central_policy_net",
        "best_individual_network",
    ]
    for net_name in networks_to_eval:
        if hasattr(agent, net_name):
            network = getattr(agent, net_name)
            if network is not None and hasattr(network, "eval") and callable(network.eval):
                network.eval()


def load_agent_for_eval(agent_type_to_load, state_size, model_base_dir, allow_untrained_missing=False, verbose=False):
    agent_class = AGENT_REGISTRY.get(agent_type_to_load)
    if not agent_class:
        return None, "missing-registry", ""

    with suppress_output(not verbose):
        agent_instance = agent_class(state_size=state_size, seed=config.SEED + 200)

    model_free_agents = {"random", "heuristic"}
    model_load_path = ""

    if agent_type_to_load not in model_free_agents:
        with suppress_output(not verbose):
            model_load_path = find_latest_or_best_model_path(agent_type_to_load, model_base_dir)

    try:
        if agent_type_to_load in model_free_agents:
            load_status = "rule-based" if agent_type_to_load == "heuristic" else "random"
        elif model_load_path:
            with suppress_output(not verbose):
                agent_instance.load(model_load_path)
            load_status = "loaded"
        else:
            if allow_untrained_missing:
                load_status = "untrained-missing-model"
            else:
                return None, "missing-model", ""

        set_eval_mode_if_possible(agent_instance)

        if agent_type_to_load == "genetic":
            if hasattr(agent_instance, "get_best_policy_network"):
                best_ga_net = agent_instance.get_best_policy_network()
                if best_ga_net:
                    eval_agent = GeneticAgent(state_size, policy_network_instance=best_ga_net)
                    return eval_agent, load_status, os.path.basename(model_load_path) if model_load_path else ""
                elif allow_untrained_missing:
                    return GeneticAgent(state_size, seed=config.SEED + 201), "untrained-genetic-wrapper", ""
                else:
                    return None, "missing-best-ga-network", ""

        return agent_instance, load_status, os.path.basename(model_load_path) if model_load_path else ""

    except Exception:
        if allow_untrained_missing:
            return agent_class(state_size=state_size, seed=config.SEED + 201), "untrained-load-failed", ""
        return None, "load-error", ""


def select_action_safe(agent, state_features, env):
    try:
        return agent.select_action(state_features, env, epsilon_override=0.0)
    except TypeError:
        return agent.select_action(state_features, env)


def _get_rgb_frame_from_env(env: Tetris) -> np.ndarray:
    if not env.gameover:
        img_data = [env.piece_colors[p] for row in env.get_current_board_state() for p in row]
    else:
        img_data = [env.piece_colors[p] for row in env.board for p in row]

    img = np.array(img_data).reshape((env.height, env.width, 3)).astype(np.uint8)
    img = Image.fromarray(img, "RGB")
    img = img.resize((env.width * env.block_size, env.height * env.block_size), Image.NEAREST)
    img = np.array(img)

    img[[i * env.block_size for i in range(env.height)], :, :] = 0
    img[:, [i * env.block_size for i in range(env.width)], :] = 0
    img = np.concatenate((img, env.extra_board), axis=1)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def put_text(local_img, text, org):
        cv2.putText(
            local_img,
            text,
            org,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=env.text_color,
        )

    put_text(img_bgr, "Score:", (env.width * env.block_size + int(env.block_size / 2), env.block_size))
    put_text(img_bgr, str(env.score), (env.width * env.block_size + int(env.block_size / 2), 2 * env.block_size))
    put_text(img_bgr, "Pieces:", (env.width * env.block_size + int(env.block_size / 2), 4 * env.block_size))
    put_text(img_bgr, str(env.tetrominoes), (env.width * env.block_size + int(env.block_size / 2), 5 * env.block_size))
    put_text(img_bgr, "Lines:", (env.width * env.block_size + int(env.block_size / 2), 7 * env.block_size))
    put_text(img_bgr, str(env.cleared_lines), (env.width * env.block_size + int(env.block_size / 2), 8 * env.block_size))

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_single_game_recorded(
    env: Tetris,
    agent,
    game_seed: int,
    max_pieces: int,
    max_seconds: float | None,
    max_frames_to_keep: int,
):
    with suppress_output(True):
        setup_seeds(game_seed)

    current_board_features = env.reset()
    if config.DEVICE.type == "cuda":
        current_board_features = current_board_features.cuda()

    if hasattr(agent, "reset") and callable(agent.reset):
        agent.reset()

    game_over = False
    pieces_played = 0
    frames = deque(maxlen=max_frames_to_keep)
    start_time = time.time()

    while not game_over and pieces_played < max_pieces:
        if max_seconds is not None and (time.time() - start_time) >= max_seconds:
            break

        action_result = select_action_safe(agent, current_board_features, env)
        action_tuple = action_result[0] if isinstance(action_result, tuple) else action_result

        frames.append(_get_rgb_frame_from_env(env))
        _, game_over = env.step(action_tuple, render=False)
        pieces_played += 1

        if not game_over:
            current_board_features = env.get_state_properties(env.board)
            if config.DEVICE.type == "cuda":
                current_board_features = current_board_features.cuda()

    frames.append(_get_rgb_frame_from_env(env))
    return list(frames), env.score


def create_optimized_gif(frames, output_path, target_mb=50, verbose=False):
    if not frames:
        return False

    try:
        imageio.mimsave(output_path, frames, fps=10, loop=0)
        gif_size = os.path.getsize(output_path) / (1024 * 1024)
        if gif_size <= target_mb:
            vprint(verbose, f"GIF saved: {output_path} ({gif_size:.2f} MB)")
            return True
    except Exception as e:
        vprint(verbose, f"Error creating GIF: {e}")
        return False

    reduction_factors = [0.5, 0.3, 0.2, 0.1]
    for factor in reduction_factors:
        try:
            keep_frames = max(1, int(len(frames) * factor))
            reduced_frames = frames[-keep_frames:]
            imageio.mimsave(output_path, reduced_frames, fps=10, loop=0)
            gif_size = os.path.getsize(output_path) / (1024 * 1024)
            if gif_size <= target_mb:
                vprint(verbose, f"Reduced GIF saved: {output_path} ({gif_size:.2f} MB)")
                return True
        except Exception:
            continue

    return False


def main():
    opt = get_args()
    ensure_dirs()

    if not os.path.exists(opt.input_csv):
        print(f"Input CSV not found: {opt.input_csv}")
        return

    rows = load_rows(opt.input_csv)
    if not rows:
        print("No valid rows found in input CSV.")
        return

    selected_agents = normalize_agent_types(opt.agent_types)
    best_rows = pick_best_rows(rows, selected_agents)

    if not best_rows:
        print("No matching agents found in raw CSV.")
        return

    pbar = tqdm(
        total=len(best_rows),
        desc="Saving GIFs",
        unit="agent",
        dynamic_ncols=True,
        leave=True,
    )

    for agent_type in PREFERRED_ORDER:
        if agent_type not in best_rows:
            continue

        best_row = best_rows[agent_type]

        agent, load_status, _ = load_agent_for_eval(
            agent_type_to_load=agent_type,
            state_size=config.STATE_SIZE,
            model_base_dir=config.MODEL_DIR,
            allow_untrained_missing=opt.allow_untrained_missing,
            verbose=opt.verbose,
        )

        if agent is None:
            pbar.update(1)
            continue

        env = Tetris()
        frames, replay_score = run_single_game_recorded(
            env=env,
            agent=agent,
            game_seed=best_row["seed"],
            max_pieces=opt.max_pieces,
            max_seconds=opt.max_seconds,
            max_frames_to_keep=opt.max_frames_to_keep,
        )

        gif_filename = f"{agent_type}_best_score_{int(best_row['score'])}.gif"
        gif_path = os.path.join(GIF_DIR, gif_filename)

        create_optimized_gif(
            frames,
            gif_path,
            target_mb=opt.gif_target_mb,
            verbose=opt.verbose,
        )

        pbar.update(1)
        pbar.set_postfix(
            {
                "agent": agent_type,
                "seed": int(best_row["seed"]),
                "score": int(replay_score),
            }
        )

    pbar.close()
    print(f"GIFs saved in: {GIF_DIR}")


if __name__ == "__main__":
    main()