import os
import sys
import shlex
import argparse
import subprocess


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    parser = argparse.ArgumentParser(description="Train multiple Tetris agents sequentially.")

    parser.add_argument(
        "--agents",
        type=str,
        default="all",
        help="Comma-separated list of agents to train, or 'all'.",
    )

    # Defaults = None, so child scripts fall back to config.py
    parser.add_argument("--dqn_epochs", type=int, default=None)
    parser.add_argument("--reinforce_epochs", type=int, default=None)

    parser.add_argument("--a2c_total_steps", type=int, default=None)
    parser.add_argument("--a2c_num_games", type=int, default=None)

    parser.add_argument("--ppo_total_steps", type=int, default=None)
    parser.add_argument("--ppo_num_games", type=int, default=None)

    parser.add_argument("--genetic_generations", type=int, default=None)
    parser.add_argument("--es_generations", type=int, default=None)

    parser.add_argument("--time_budget_minutes", type=float, default=None)
    parser.add_argument("--render_game", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def run_command(command, verbose=False):
    if verbose:
        print("\n[RUN]")
        print(" ".join(shlex.quote(str(x)) for x in command))
        print()

    result = subprocess.run(command, cwd=PROJECT_ROOT)
    return result.returncode


def normalize_agents(agent_str):
    all_agents = ["dqn", "reinforce", "a2c", "ppo", "genetic", "es"]
    if agent_str.strip().lower() == "all":
        return all_agents

    chosen = []
    for a in agent_str.split(","):
        a = a.strip().lower()
        if a in all_agents and a not in chosen:
            chosen.append(a)
    return chosen


def main():
    opt = get_args()
    agents = normalize_agents(opt.agents)

    if not agents:
        print("No valid agents selected.")
        return

    print(f"Training agents: {', '.join(agents)}")
    if opt.time_budget_minutes is not None:
        print(f"Time budget per agent: {opt.time_budget_minutes} minute(s)")

    failures = []

    for agent in agents:
        if agent in ["dqn", "reinforce"]:
            cmd = [
                sys.executable,
                "-m",
                "train.train_dqn_reinforce",
                "--agent_type",
                agent,
            ]

            if agent == "dqn" and opt.dqn_epochs is not None:
                cmd.extend(["--num_epochs", str(opt.dqn_epochs)])
            elif agent == "reinforce" and opt.reinforce_epochs is not None:
                cmd.extend(["--num_epochs", str(opt.reinforce_epochs)])

            if opt.render_game:
                cmd.append("--render_game")

        elif agent in ["a2c", "ppo"]:
            cmd = [
                sys.executable,
                "-m",
                "train.train_onpolicy",
                "--agent_type",
                agent,
            ]

            if agent == "a2c":
                if opt.a2c_total_steps is not None:
                    cmd.extend(["--total_steps", str(opt.a2c_total_steps)])
                if opt.a2c_num_games is not None:
                    cmd.extend(["--num_games", str(opt.a2c_num_games)])
            else:
                if opt.ppo_total_steps is not None:
                    cmd.extend(["--total_steps", str(opt.ppo_total_steps)])
                if opt.ppo_num_games is not None:
                    cmd.extend(["--num_games", str(opt.ppo_num_games)])

            if opt.render_game:
                cmd.append("--render_game")

        elif agent in ["genetic", "es"]:
            cmd = [
                sys.executable,
                "-m",
                "train.train_evolutionary",
                "--agent_type",
                agent,
            ]

            if agent == "genetic" and opt.genetic_generations is not None:
                cmd.extend(["--num_generations", str(opt.genetic_generations)])
            elif agent == "es" and opt.es_generations is not None:
                cmd.extend(["--num_generations", str(opt.es_generations)])

        else:
            print(f"Skipping unknown agent: {agent}")
            continue

        if opt.time_budget_minutes is not None:
            cmd.extend(["--time_budget_minutes", str(opt.time_budget_minutes)])

        if opt.verbose:
            cmd.append("--verbose")

        print(f"\n=== Training {agent.upper()} ===")
        code = run_command(cmd, verbose=opt.verbose)

        if code != 0:
            print(f"[ERROR] {agent.upper()} failed with exit code {code}")
            failures.append((agent, code))
            if opt.stop_on_error:
                break
        else:
            print(f"[OK] {agent.upper()} finished")

    print("\n=== Training summary ===")
    if failures:
        for agent, code in failures:
            print(f"- {agent.upper()}: FAILED ({code})")
    else:
        print("All selected agents finished successfully.")


if __name__ == "__main__":
    main()