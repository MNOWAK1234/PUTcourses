#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import ensure_dirs, read_csv_required


def read_pool_offsets(path: str) -> pd.DataFrame:
    df = read_csv_required(path)
    # Try to infer column names across possible versions.
    lower = {c.lower(): c for c in df.columns}
    if "pool_id" not in lower:
        # The first integer-like column is treated as pool_id.
        df = df.rename(columns={df.columns[0]: "pool_id"})
    if "pool_offset" not in lower and "offset" not in lower:
        numeric = [c for c in df.columns if c != "pool_id" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric:
            df = df.rename(columns={numeric[0]: "pool_offset"})
    elif "offset" in lower and "pool_offset" not in lower:
        df = df.rename(columns={lower["offset"]: "pool_offset"})
    return df


def read_pair_offsets(path: str) -> pd.DataFrame:
    df = read_csv_required(path)
    cols = list(df.columns)
    rename = {}
    for c in cols:
        lc = c.lower()
        if lc in {"pool_a", "pool_a_id", "source_pool", "row_pool"}:
            rename[c] = "pool_a"
        if lc in {"pool_b", "pool_b_id", "target_pool", "col_pool"}:
            rename[c] = "pool_b"
        if lc in {"pair_offset", "offset"}:
            rename[c] = "pair_offset"
    df = df.rename(columns=rename)
    if "pool_a" not in df.columns:
        df = df.rename(columns={df.columns[0]: "pool_a"})
    if "pool_b" not in df.columns:
        df = df.rename(columns={df.columns[1]: "pool_b"})
    if "pair_offset" not in df.columns:
        numeric = [c for c in df.columns if c not in {"pool_a", "pool_b"} and pd.api.types.is_numeric_dtype(df[c])]
        if numeric:
            df = df.rename(columns={numeric[0]: "pair_offset"})
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret final pool offsets and pool-pair offsets.")
    parser.add_argument("--pool-offsets", default="experiments/results/final_pool_offsets.csv")
    parser.add_argument("--pair-offsets", default="experiments/results/final_pool_pair_offsets.csv")
    parser.add_argument("--summary", default="experiments/results/latent_pool_summary.csv")
    parser.add_argument("--out", default="outputs/robustness_ablation")
    args = parser.parse_args()

    results_dir, plots_dir = ensure_dirs(args.out)
    pools = read_pool_offsets(args.pool_offsets)
    summary = read_csv_required(args.summary) if Path(args.summary).exists() else pd.DataFrame()

    if not summary.empty and "pool_id" in summary.columns:
        pools = pools.merge(summary, on="pool_id", how="left")

    if "pool_offset" not in pools.columns:
        raise RuntimeError(f"Cannot infer pool_offset column from {args.pool_offsets}: {pools.columns.tolist()}")

    pools["abs_pool_offset"] = pools["pool_offset"].abs()
    pools = pools.sort_values("abs_pool_offset", ascending=False)
    pools.to_csv(results_dir / "pool_offset_ranking.csv", index=False, encoding="utf-8-sig")

    top = pools.head(20).sort_values("pool_offset")
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = ["pool " + str(int(x)) for x in top["pool_id"]]
    ax.barh(labels, top["pool_offset"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Largest learned latent pool offsets")
    ax.set_xlabel("pool offset [rating points]")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "largest_pool_offsets.png", dpi=180)
    plt.close(fig)

    pair_df = read_pair_offsets(args.pair_offsets)
    if "pair_offset" in pair_df.columns:
        pair_df["abs_pair_offset"] = pair_df["pair_offset"].abs()
        pair_rank = pair_df.sort_values("abs_pair_offset", ascending=False)
        pair_rank.to_csv(results_dir / "pool_pair_offset_ranking.csv", index=False, encoding="utf-8-sig")

        top_pair = pair_rank.head(25).sort_values("pair_offset")
        fig, ax = plt.subplots(figsize=(11, 8))
        labels = [f"{int(a)} vs {int(b)}" for a, b in zip(top_pair["pool_a"], top_pair["pool_b"])]
        ax.barh(labels, top_pair["pair_offset"])
        ax.axvline(0.0, linewidth=1.0)
        ax.set_title("Largest learned pool-pair interactions")
        ax.set_xlabel("pair offset [rating points]")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "largest_pool_pair_offsets.png", dpi=180)
        plt.close(fig)

    print("[OK] saved pool interpretation outputs")


if __name__ == "__main__":
    main()
