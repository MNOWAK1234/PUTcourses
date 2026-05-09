import os
import csv
import math
import statistics
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


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
    parser = argparse.ArgumentParser("Summarize raw test results and generate plots.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join(RESULTS_DIR, "test_raw_runs.csv"),
        help="Path to raw per-game results CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(RESULTS_DIR, "test_summary.csv"),
        help="Path to output summary CSV.",
    )
    return parser.parse_args()


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_median(values):
    return statistics.median(values) if values else 0.0


def safe_std(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0


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
                        "pieces_played": float(row["pieces_played"]),
                        "tetrominoes": float(row["tetrominoes"]),
                        "lines_cleared": float(row["lines_cleared"]),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                        "game_over": int(row["game_over"]),
                    }
                )
            except Exception:
                continue
    return rows


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(row)

    summary = []

    for agent in PREFERRED_ORDER:
        if agent not in grouped:
            continue

        agent_rows = grouped[agent]
        scores = [r["score"] for r in agent_rows]
        pieces = [r["pieces_played"] for r in agent_rows]
        tetrominoes = [r["tetrominoes"] for r in agent_rows]
        lines = [r["lines_cleared"] for r in agent_rows]
        elapsed = [r["elapsed_seconds"] for r in agent_rows]

        summary.append(
            {
                "agent": agent,
                "display_name": DISPLAY_NAMES.get(agent, agent.upper()),
                "num_games": len(agent_rows),
                "mean_score": safe_mean(scores),
                "median_score": safe_median(scores),
                "std_score": safe_std(scores),
                "best_score": max(scores) if scores else 0.0,
                "worst_score": min(scores) if scores else 0.0,
                "mean_pieces": safe_mean(pieces),
                "mean_tetrominoes": safe_mean(tetrominoes),
                "mean_lines": safe_mean(lines),
                "mean_elapsed_seconds": safe_mean(elapsed),
            }
        )

    return summary


def save_summary_csv(summary, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    headers = [
        "agent",
        "display_name",
        "num_games",
        "mean_score",
        "median_score",
        "std_score",
        "best_score",
        "worst_score",
        "mean_pieces",
        "mean_tetrominoes",
        "mean_lines",
        "mean_elapsed_seconds",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "agent": row["agent"],
                    "display_name": row["display_name"],
                    "num_games": row["num_games"],
                    "mean_score": f"{row['mean_score']:.4f}",
                    "median_score": f"{row['median_score']:.4f}",
                    "std_score": f"{row['std_score']:.4f}",
                    "best_score": f"{row['best_score']:.4f}",
                    "worst_score": f"{row['worst_score']:.4f}",
                    "mean_pieces": f"{row['mean_pieces']:.4f}",
                    "mean_tetrominoes": f"{row['mean_tetrominoes']:.4f}",
                    "mean_lines": f"{row['mean_lines']:.4f}",
                    "mean_elapsed_seconds": f"{row['mean_elapsed_seconds']:.4f}",
                }
            )


def print_summary_table(summary):
    headers = [
        "Agent",
        "Games",
        "Mean",
        "Median",
        "Std",
        "Best",
        "Worst",
        "Mean Lines",
        "Mean Pieces",
    ]

    table_rows = []
    for row in summary:
        table_rows.append(
            [
                row["display_name"],
                str(row["num_games"]),
                f"{row['mean_score']:.2f}",
                f"{row['median_score']:.2f}",
                f"{row['std_score']:.2f}",
                f"{row['best_score']:.2f}",
                f"{row['worst_score']:.2f}",
                f"{row['mean_lines']:.2f}",
                f"{row['mean_pieces']:.2f}",
            ]
        )

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))

    for r in table_rows:
        print(" | ".join(r[i].ljust(widths[i]) for i in range(len(r))))


def save_mean_score_plot(summary):
    labels = [row["display_name"] for row in summary]
    means = [row["mean_score"] for row in summary]
    stds = [row["std_score"] for row in summary]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.title("Mean Score by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Mean score")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(means):
        plt.annotate(
            f"{value:.0f}",
            (i, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "test_mean_score_bar.png")
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return output_path


def save_boxplot(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(row["score"])

    labels = []
    data = []
    for agent in PREFERRED_ORDER:
        if agent not in grouped:
            continue
        labels.append(DISPLAY_NAMES.get(agent, agent.upper()))
        data.append(grouped[agent])

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, tick_labels=labels)
    plt.title("Score Distribution by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "test_score_boxplot.png")
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return output_path


def save_best_worst_plot(summary):
    labels = [row["display_name"] for row in summary]
    bests = [row["best_score"] for row in summary]
    worsts = [row["worst_score"] for row in summary]

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], bests, width=width, label="Best")
    plt.bar([i + width / 2 for i in x], worsts, width=width, label="Worst")

    plt.title("Best and Worst Score by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Score")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "test_best_worst_bar.png")
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    opt = get_args()
    ensure_plots_dir()

    if not os.path.exists(opt.input_csv):
        print(f"Input CSV not found: {opt.input_csv}")
        return

    rows = load_rows(opt.input_csv)
    if not rows:
        print("No valid rows found in input CSV.")
        return

    summary = summarize_rows(rows)
    if not summary:
        print("No summary data produced.")
        return

    save_summary_csv(summary, opt.output_csv)
    print_summary_table(summary)

    mean_plot = save_mean_score_plot(summary)
    box_plot = save_boxplot(rows)
    best_worst_plot = save_best_worst_plot(summary)

    print(f"\nSummary CSV saved to: {opt.output_csv}")
    print("Plots saved:")
    print(mean_plot)
    print(box_plot)
    print(best_worst_plot)


if __name__ == "__main__":
    main()