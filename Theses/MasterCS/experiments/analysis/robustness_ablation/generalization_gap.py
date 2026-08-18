#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, read_csv_required


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare train, validation and test MSE for evaluated model variants.")
    parser.add_argument("--scope", default="experiments/results/final_scope_metrics.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    args = parser.parse_args()

    results_dir, plots_dir = ensure_dirs(args.out)
    df = read_csv_required(args.scope)

    pivot = df.pivot_table(index="model", columns="scope", values="mse_all", aggfunc="first").reset_index()
    for column in ["train", "validation", "test"]:
        if column not in pivot.columns:
            pivot[column] = float("nan")

    pivot["validation_minus_train"] = pivot["validation"] - pivot["train"]
    pivot["test_minus_validation"] = pivot["test"] - pivot["validation"]
    pivot["test_minus_train"] = pivot["test"] - pivot["train"]
    pivot["test_improvement_vs_classic_percent"] = None
    if "classic_elo" in set(pivot["model"]):
        classic_test = float(pivot.loc[pivot["model"] == "classic_elo", "test"].iloc[0])
        pivot["test_improvement_vs_classic_percent"] = (classic_test - pivot["test"]) / classic_test * 100.0

    out = pivot.sort_values("test", ascending=True)
    out_csv = results_dir / "generalization_gap.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    selected = out.head(12).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(selected["validation"], selected["test"])
    for _, row in selected.iterrows():
        ax.annotate(row["model"], (row["validation"], row["test"]), fontsize=7)
    mn = min(selected["validation"].min(), selected["test"].min())
    mx = max(selected["validation"].max(), selected["test"].max())
    ax.plot([mn, mx], [mn, mx], linewidth=1.0)
    ax.set_title("Validation MSE vs test MSE")
    ax.set_xlabel("validation MSE")
    ax.set_ylabel("test MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "validation_vs_test_mse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    selected2 = out.head(15).sort_values("test_minus_validation", ascending=True)
    ax.barh(selected2["model"], selected2["test_minus_validation"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Test-validation generalization gap")
    ax.set_xlabel("test MSE - validation MSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_validation_gap.png", dpi=180)
    plt.close(fig)

    print(f"[OK] saved {out_csv}")
    print(f"[OK] saved plots")


if __name__ == "__main__":
    main()
