import os
import csv
import glob
import time
import argparse
import contextlib

from tqdm import tqdm

import config
from agents import GeneticAlgorithmController, ESAgent
from helper import *
from src.tetris import Tetris


LOCAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(LOCAL_PROJECT_ROOT, "results")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("""Train Evolutionary Agents (GA, ES) for Tetris""")
    parser.add_argument("--agent_type", type=str, required=True, choices=["genetic", "es"])
    parser.add_argument("--fps", type=int, default=300)
    parser.add_argument("--num_generations", type=int, default=None, help="Used only when time budget is NOT provided.")
    parser.add_argument("--time_budget_minutes", type=float, default=None)
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
                "generations_done",
                "best_score",
                "mean_fitness",
                "best_generation_fitness",
                "central_fitness",
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


def train_evolutionary_agents(env_template: Tetris, opt: argparse.Namespace, model_base_dir: str) -> None:
    agent_type = opt.agent_type.lower()
    use_time_budget = opt.time_budget_minutes is not None
    time_budget_seconds = opt.time_budget_minutes * 60 if use_time_budget else None

    if agent_type == "genetic":
        with suppress_output(not opt.verbose):
            controller = GeneticAlgorithmController(state_size=config.STATE_SIZE)
        num_generations = opt.num_generations if opt.num_generations is not None else config.GA_N_GENERATIONS

    elif agent_type == "es":
        with suppress_output(not opt.verbose):
            controller = ESAgent(state_size=config.STATE_SIZE)
        num_generations = opt.num_generations if opt.num_generations is not None else config.ES_N_GENERATIONS

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    csv_path = init_minute_csv(agent_type)
    next_minute_mark = 1

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
            total=num_generations,
            desc=f"{agent_type.upper()} training",
            unit="gen",
            dynamic_ncols=True,
            leave=True,
        )

    current_overall_best_fitness = -1
    generations_done = 0
    start_time = time.time()
    last_mean_fit = 0.0
    last_best_gen_fit = 0.0
    last_central_fit = ""
    saved_model_path = None
    best_saved_score = None

    while True:
        elapsed = time.time() - start_time

        if use_time_budget:
            target_n = min(elapsed, time_budget_seconds)
            delta = target_n - pbar.n
            if delta > 0:
                pbar.update(delta)

            if elapsed >= time_budget_seconds:
                break
        else:
            if generations_done >= num_generations:
                break

        if agent_type == "genetic":
            with suppress_output(not opt.verbose):
                mean_pop_fit, max_pop_fit_this_gen = controller.evolve_population(env_template)

            last_mean_fit = float(mean_pop_fit)
            last_best_gen_fit = float(max_pop_fit_this_gen)
            last_central_fit = ""
            current_overall_best_fitness = int(controller.best_fitness)

            pbar.set_postfix(
                {
                    "gen": generations_done + 1,
                    "mean": f"{mean_pop_fit:.1f}",
                    "best_gen": f"{max_pop_fit_this_gen:.1f}",
                    "best_all": current_overall_best_fitness,
                    "t": f"{elapsed/60:.1f}m",
                }
            )

        else:
            with suppress_output(not opt.verbose):
                mean_pop_fit, max_pop_fit_this_gen, central_fit = controller.learn(env_template)

            last_mean_fit = float(mean_pop_fit)
            last_best_gen_fit = float(max_pop_fit_this_gen)
            last_central_fit = float(central_fit)
            current_overall_best_fitness = int(controller.current_best_fitness)

            pbar.set_postfix(
                {
                    "gen": generations_done + 1,
                    "mean": f"{mean_pop_fit:.1f}",
                    "best_gen": f"{max_pop_fit_this_gen:.1f}",
                    "central": f"{central_fit:.1f}",
                    "best_all": current_overall_best_fitness,
                    "t": f"{elapsed/60:.1f}m",
                }
            )

        generations_done += 1

        if best_saved_score is None or current_overall_best_fitness > best_saved_score:
            agent_prefix = get_agent_file_prefix(agent_type)
            remove_existing_agent_models(model_base_dir, agent_prefix)

            saved_model_path = os.path.join(
                model_base_dir,
                f"{agent_prefix}_score_{int(current_overall_best_fitness)}.pth",
            )
            with suppress_output(not opt.verbose):
                if agent_type == "genetic":
                    controller.save_best_individual(saved_model_path)
                else:
                    controller.save(saved_model_path)

            best_saved_score = current_overall_best_fitness

        while elapsed >= next_minute_mark * 60:
            append_minute_row(
                csv_path,
                {
                    "minute": next_minute_mark,
                    "elapsed_seconds": round(next_minute_mark * 60, 2),
                    "generations_done": generations_done,
                    "best_score": int(current_overall_best_fitness if current_overall_best_fitness > -1 else 0),
                    "mean_fitness": round(last_mean_fit, 6),
                    "best_generation_fitness": round(last_best_gen_fit, 6),
                    "central_fitness": last_central_fit if last_central_fit == "" else round(last_central_fit, 6),
                },
            )
            next_minute_mark += 1

        if not use_time_budget:
            pbar.update(1)

    pbar.close()

    final_elapsed = time.time() - start_time
    append_minute_row(
        csv_path,
        {
            "minute": max(1, int(final_elapsed // 60)),
            "elapsed_seconds": round(final_elapsed, 2),
            "generations_done": generations_done,
            "best_score": int(current_overall_best_fitness if current_overall_best_fitness > -1 else 0),
            "mean_fitness": round(last_mean_fit, 6),
            "best_generation_fitness": round(last_best_gen_fit, 6),
            "central_fitness": last_central_fit if last_central_fit == "" else round(last_central_fit, 6),
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

    current_model_base_dir = config.MODEL_DIR
    env_template = Tetris()
    train_evolutionary_agents(env_template, opt, current_model_base_dir)