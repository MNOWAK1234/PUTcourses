import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


DISPLAY_NAMES = {
    "random": "Random",
    "heuristic": "Heuristic",
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "a2c": "A2C",
    "ppo": "PPO",
    "genetic": "Genetic Algorithm",
    "es": "Evolution Strategies",
}

PREFERRED_ORDER = ["random", "heuristic", "dqn", "reinforce", "a2c", "ppo", "genetic", "es"]


def get_args():
    parser = argparse.ArgumentParser("Plot best-score-over-time curves from minute CSV files.")
    parser.add_argument("--max_minutes", type=float, default=30.0)
    return parser.parse_args()


def ensure_plots_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def find_minute_csv_files() -> List[Tuple[str, str]]:
    found = []
    if not os.path.isdir(RESULTS_DIR):
        return found

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith("_minute_stats.csv"):
            continue
        agent_key = filename.replace("_minute_stats.csv", "").lower()
        found.append((agent_key, os.path.join(RESULTS_DIR, filename)))

    order_map = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    found.sort(key=lambda x: order_map.get(x[0], 999))
    return found


def human_score(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(int(round(value)))


def y_axis_formatter(y, _):
    if abs(y) >= 1_000_000:
        return f"{y / 1_000_000:.1f}M"
    if abs(y) >= 1_000:
        return f"{y / 1_000:.0f}k"
    return f"{int(y)}"


def read_agent_curve(csv_path: str, max_minutes: float) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed_seconds_raw = row.get("elapsed_seconds", "").strip()
            best_score_raw = row.get("best_score", "").strip()

            if not elapsed_seconds_raw or not best_score_raw:
                continue

            try:
                elapsed_minutes = float(elapsed_seconds_raw) / 60.0
                best_score = float(best_score_raw)
            except ValueError:
                continue

            if elapsed_minutes > max_minutes:
                elapsed_minutes = max_minutes

            xs.append(elapsed_minutes)
            ys.append(best_score)

    if not xs:
        return [], []

    merged: Dict[float, float] = {}
    for x, y in zip(xs, ys):
        merged[x] = y

    xs = sorted(merged.keys())
    ys = [merged[x] for x in xs]

    # Dociągnij ostatnią wartość do dokładnie max_minutes,
    # żeby każda krzywa kończyła się w tym samym miejscu.
    if xs[-1] < max_minutes:
        xs.append(max_minutes)
        ys.append(ys[-1])

    return xs, ys


def load_all_curves(max_minutes: float) -> Dict[str, Tuple[List[float], List[float]]]:
    curves: Dict[str, Tuple[List[float], List[float]]] = {}
    for agent_key, csv_path in find_minute_csv_files():
        xs, ys = read_agent_curve(csv_path, max_minutes)
        if xs and ys:
            curves[agent_key] = (xs, ys)
    return curves


def agent_display_name(agent_key: str) -> str:
    return DISPLAY_NAMES.get(agent_key, agent_key.upper())


def save_combined_plot(curves: Dict[str, Tuple[List[float], List[float]]], max_minutes: float) -> str:
    fig, ax = plt.subplots(figsize=(15, 8))
    final_rows = []

    for agent_key in PREFERRED_ORDER:
        if agent_key not in curves:
            continue

        xs, ys = curves[agent_key]
        line = ax.step(xs, ys, where="post", linewidth=2.4, label=agent_display_name(agent_key))[0]

        if xs and ys:
            final_rows.append(
                {
                    "agent": agent_key,
                    "display": agent_display_name(agent_key),
                    "final_score": ys[-1],
                    "color": line.get_color(),
                }
            )

    ax.set_title("Best Score Over Time", fontsize=18)
    ax.set_xlabel("Time [minutes]", fontsize=13)
    ax.set_ylabel("Best score so far", fontsize=13)
    ax.set_xlim(0, max_minutes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    final_rows.sort(key=lambda r: r["final_score"], reverse=True)

    summary_lines = ["Final scores:"]
    for row in final_rows:
        summary_lines.append(f"{row['display']}: {human_score(row['final_score'])}")

    ax.text(
        1.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    fig.subplots_adjust(right=0.78)
    fig.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "best_score_over_time_combined.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_small_multiples_plot(curves: Dict[str, Tuple[List[float], List[float]]], max_minutes: float) -> str:
    available_agents = [a for a in PREFERRED_ORDER if a in curves]
    n = len(available_agents)
    if n == 0:
        raise ValueError("No curves to plot.")

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, agent_key in zip(axes, available_agents):
        xs, ys = curves[agent_key]
        ax.step(xs, ys, where="post", linewidth=2.2)
        ax.set_title(agent_display_name(agent_key), fontsize=14)
        ax.set_xlabel("Time [minutes]")
        ax.set_ylabel("Best score")
        ax.set_xlim(0, max_minutes)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        if xs and ys:
            ax.text(
                0.98,
                0.96,
                f"Final: {human_score(ys[-1])}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
            )

    for i in range(len(available_agents), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Training Progress by Agent", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(PLOTS_DIR, "best_score_over_time_panels.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    opt = get_args()
    ensure_plots_dir()
    curves = load_all_curves(opt.max_minutes)

    if not curves:
        print("No minute stats CSV files found in results/.")
        return

    combined_path = save_combined_plot(curves, opt.max_minutes)
    panels_path = save_small_multiples_plot(curves, opt.max_minutes)

    print("Plots saved:")
    print(combined_path)
    print(panels_path)


if __name__ == "__main__":
    main()