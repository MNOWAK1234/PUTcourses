#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "classic_elo": "Classic Elo",
    "provisional_full": "Proposed full",
    "provisional_without_form": "No form",
    "provisional_without_pool": "No pool",
    "provisional_without_dynamic_exposure": "No dynamic exposure",
    "provisional_without_performance_entry": "No performance entry",
    "provisional_without_uncertainty_prediction": "No uncertainty prediction",
    "provisional_without_white_advantage": "No white advantage",
    "provisional_without_pair_interaction": "No pair interaction",
    "provisional_without_event_normalization": "No event normalization",
    "provisional_without_entry_acceleration": "No entry acceleration",
    "provisional_exactly_9_games": "Exactly 9 provisional games",
    "provisional_max_exposures_2": "Max exposures = 2",
    "provisional_max_exposures_3": "Max exposures = 3",
    "advanced_seed": "Advanced seed",
    "previous_dynamic_seed": "Previous dynamic",
    "provisional_9_seed": "Provisional seed",
}

COLORS = {
    "classic_elo": "#444444",
    "provisional_full": "#0057B8",
    "provisional_without_form": "#E69F00",
    "provisional_without_pool": "#D55E00",
    "provisional_without_dynamic_exposure": "#7B3294",
    "provisional_without_performance_entry": "#009E73",
    "provisional_without_uncertainty_prediction": "#CC79A7",
    "advanced_seed": "#56B4E9",
    "previous_dynamic_seed": "#999999",
}

IMPORTANT_MODELS = [
    "classic_elo",
    "provisional_full",
    "provisional_without_form",
    "provisional_without_pool",
    "provisional_without_dynamic_exposure",
    "advanced_seed",
]


def label(model: str) -> str:
    return LABELS.get(model, model.replace("_", " "))


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] missing {path}")
        return None
    return pd.read_csv(path)


def setup_ax(ax):
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_period_lines(ax):
    # Pool discovery ends at 2014-12. Search/train starts in 2015,
    # validation starts in 2019, and the final test period starts in 2022.
    for date, text in [("2019-01-01", "validation"), ("2022-01-01", "test")]:
        x = pd.to_datetime(date)
        ax.axvline(x, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ax.text(x, ymax, f" {text}", rotation=90, va="top", ha="left", fontsize=8, color="#555555")


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_monthly_mse(project: Path, plots: Path, start_month: str):
    df = read_csv(project / "experiments" / "results" / "final_monthly_metrics.csv")
    if df is None:
        return
    df = df[df["model"].isin(IMPORTANT_MODELS)].copy()
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df = df[df["month"] >= start_month].copy()

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for model in IMPORTANT_MODELS:
        g = df[df["model"] == model].sort_values("date")
        if g.empty:
            continue
        lw = 2.6 if model == "provisional_full" else 1.45
        alpha = 1.0 if model == "provisional_full" else 0.78
        ax.plot(g["date"], g["mse_all"], label=label(model), color=COLORS.get(model), linewidth=lw, alpha=alpha)
    ax.set_title(f"Monthly MSE of selected variants ({start_month} onward)")
    ax.set_xlabel("Month")
    ax.set_ylabel("MSE")
    setup_ax(ax)
    add_period_lines(ax)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    save(fig, plots / "clean_monthly_mse_selected_models.png")


def plot_monthly_improvement(project: Path, plots: Path, start_month: str):
    df = read_csv(project / "experiments" / "results" / "final_monthly_metrics.csv")
    if df is None:
        return
    baseline = df[df["model"] == "classic_elo"][["month", "mse_all"]].rename(columns={"mse_all": "classic_mse"})
    df = df.merge(baseline, on="month", how="left")
    df["improvement"] = (df["classic_mse"] - df["mse_all"]) / df["classic_mse"] * 100.0
    selected = [
        "provisional_full",
        "provisional_without_form",
        "provisional_without_pool",
        "provisional_without_dynamic_exposure",
        "advanced_seed",
        "previous_dynamic_seed",
    ]
    df = df[df["model"].isin(selected)].copy()
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df = df[df["month"] >= start_month].copy()

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    for model in selected:
        g = df[df["model"] == model].sort_values("date")
        if g.empty:
            continue
        lw = 2.8 if model == "provisional_full" else 1.45
        alpha = 1.0 if model == "provisional_full" else 0.78
        ax.plot(g["date"], g["improvement"], label=label(model), color=COLORS.get(model), linewidth=lw, alpha=alpha)
    ax.set_title(f"Monthly MSE improvement over Classic Elo ({start_month} onward)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Improvement over Classic Elo [%]")
    setup_ax(ax)
    add_period_lines(ax)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    ax.text(0.01, 0.03, "Dashed line: parity with Classic Elo", transform=ax.transAxes, fontsize=8, color="#555555")
    save(fig, plots / "clean_monthly_improvement_over_classic.png")


def plot_validation_vs_test(project: Path, plots: Path):
    df = read_csv(project / "experiments" / "results" / "final_scope_metrics.csv")
    if df is None:
        return
    pivot = df.pivot_table(index="model", columns="scope", values="mse_all", aggfunc="first").reset_index()
    selected = [
        "provisional_full",
        "provisional_without_form",
        "provisional_without_pool",
        "provisional_without_dynamic_exposure",
        "provisional_without_performance_entry",
        "provisional_without_uncertainty_prediction",
        "provisional_without_event_normalization",
        "advanced_seed",
        "classic_elo",
    ]
    p = pivot[pivot["model"].isin(selected)].copy()
    p = p.dropna(subset=["validation", "test"])

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.scatter(p["validation"], p["test"], s=58, color="#777777", alpha=0.75, zorder=2)
    full = p[p["model"] == "provisional_full"]
    if not full.empty:
        ax.scatter(full["validation"], full["test"], s=125, color=COLORS["provisional_full"], edgecolor="black", linewidth=0.8, zorder=4)

    x0, x1 = p["validation"].min(), p["validation"].max()
    y0, y1 = p["test"].min(), p["test"].max()
    lo = min(x0, y0) - 0.0004
    hi = max(x1, y1) + 0.0004
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1.0, zorder=1)

    offsets = {
        "provisional_full": (8, -5),
        "provisional_without_form": (8, 8),
        "provisional_without_pool": (-70, -8),
        "provisional_without_dynamic_exposure": (-110, 5),
        "provisional_without_performance_entry": (-125, -15),
        "provisional_without_uncertainty_prediction": (8, 8),
        "provisional_without_event_normalization": (-95, 5),
        "advanced_seed": (8, -10),
        "classic_elo": (-85, 8),
    }
    for _, row in p.iterrows():
        dx, dy = offsets.get(row["model"], (8, 8))
        ax.annotate(label(row["model"]), (row["validation"], row["test"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, alpha=0.7))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Validation MSE versus test MSE")
    ax.set_xlabel("Validation MSE")
    ax.set_ylabel("Test MSE")
    setup_ax(ax)
    ax.text(0.02, 0.03, "Lower-left is better. Dashed diagonal: equal validation and test MSE.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    save(fig, plots / "clean_validation_vs_test_mse.png")


def plot_test_improvement_bar(project: Path, plots: Path):
    df = read_csv(project / "experiments" / "results" / "final_test_comparison.csv")
    if df is None:
        return
    test = df[df["scope"] == "test"].copy()
    selected = [
        "provisional_full",
        "provisional_without_form",
        "provisional_without_dynamic_exposure",
        "provisional_without_performance_entry",
        "provisional_without_uncertainty_prediction",
        "advanced_seed",
        "previous_dynamic_seed",
        "provisional_without_pool",
        "classic_elo",
    ]
    test = test[test["model"].isin(selected)].copy()
    test["label"] = test["model"].map(label)
    test = test.sort_values("improvement_vs_classic_percent")
    colors = [COLORS.get(m, "#999999") for m in test["model"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(test["label"], test["improvement_vs_classic_percent"], color=colors, alpha=0.88)
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title("Final test improvement over Classic Elo")
    ax.set_xlabel("MSE improvement [%]")
    setup_ax(ax)
    save(fig, plots / "clean_test_improvement_key_models.png")


def plot_component_contribution(project: Path, plots: Path):
    df = read_csv(project / "outputs" / "robustness_ablation" / "results" / "ablation_component_contribution.csv")
    if df is None:
        return
    plot = df.sort_values("lost_improvement_points").tail(10).copy()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.barh(plot["component_removed_or_changed"], plot["lost_improvement_points"], color="#4C78A8", alpha=0.9)
    ax.set_title("Loss of improvement after removing model components")
    ax.set_xlabel("Lost improvement over Classic Elo [percentage points]")
    setup_ax(ax)
    save(fig, plots / "clean_ablation_component_contribution.png")


def plot_pool_weight(project: Path, plots: Path):
    df = read_csv(project / "outputs" / "prediction_sensitivity" / "results" / "pool_weight_sensitivity.csv")
    if df is None:
        return
    test = df[(df["scope"] == "test") & (df["model"] != "classic_elo")].copy()
    if "pool_weight_multiplier" not in test.columns:
        return
    test = test.dropna(subset=["pool_weight_multiplier"]).sort_values("pool_weight_multiplier")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(test["pool_weight_multiplier"], test["improvement_vs_baseline_percent"],
            marker="o", markersize=6, linewidth=2.2, color=COLORS["provisional_full"])
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(1.0, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_title("Sensitivity to the pool-weight multiplier")
    ax.set_xlabel("Multiplier applied to learned pool_weight")
    ax.set_ylabel("Improvement over Classic Elo [%]")
    setup_ax(ax)
    ax.text(0.02, 0.04, "Values above 1 may be clipped by the original parameter bounds.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    save(fig, plots / "clean_pool_weight_sensitivity.png")


def plot_calibration(project: Path, plots: Path):
    df = read_csv(project / "outputs" / "prediction_sensitivity" / "results" / "calibration_bins.csv")
    if df is None:
        return
    d = df[df["games"] > 0].dropna(subset=["mean_predicted_score", "observed_score"]).copy()
    if d.empty:
        return
    size = np.clip(np.sqrt(d["games"]) / 7.0, 25, 260)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax.scatter(d["mean_predicted_score"], d["observed_score"], s=size, color=COLORS["provisional_full"],
               alpha=0.75, edgecolor="black", linewidth=0.4, label="Prediction buckets")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration of predicted expected scores")
    ax.set_xlabel("Mean predicted expected score")
    ax.set_ylabel("Observed average score")
    setup_ax(ax)
    ax.legend(fontsize=8, frameon=True)
    save(fig, plots / "clean_calibration_expected_scores.png")


def write_index(project: Path, plots: Path, start_month: str):
    index = plots.parent / "CURATED_PLOTS_INDEX.md"
    index.write_text(f"""# Curated thesis plots

These figures are cleaned versions intended for the main thesis text.

Important convention:
- Monthly plots start at `{start_month}` because the latent pool discovery period ends in 2014.
- The vertical dashed lines mark the beginning of validation (`2019-01`) and test (`2022-01`).
- In improvement plots, the zero line is not a model. It denotes parity with Classic Elo.
- Positive improvement means lower MSE than Classic Elo.

Recommended figures:
- `clean_test_improvement_key_models.png`
- `clean_ablation_component_contribution.png`
- `clean_validation_vs_test_mse.png`
- `clean_monthly_mse_selected_models.png`
- `clean_monthly_improvement_over_classic.png`
- `clean_pool_weight_sensitivity.png`
- `clean_calibration_expected_scores.png`
""", encoding="utf-8")
    print(f"[OK] {index}")


def main():
    parser = argparse.ArgumentParser(description="Create cleaner thesis-ready plots from the experiment outputs.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--out", default="outputs/thesis_plots", help="Output folder under project root.")
    parser.add_argument("--start-month", default="2015-01", help="Start month for monthly plots.")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    out = project / args.out
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    plot_test_improvement_bar(project, plots)
    plot_component_contribution(project, plots)
    plot_validation_vs_test(project, plots)
    plot_monthly_mse(project, plots, args.start_month)
    plot_monthly_improvement(project, plots, args.start_month)
    plot_pool_weight(project, plots)
    plot_calibration(project, plots)
    write_index(project, plots, args.start_month)

    print("[DONE] curated plots generated.")


if __name__ == "__main__":
    main()
