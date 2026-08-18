#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import attach_metadata, ensure_dirs, import_experiment_module, load_cache, pool_membership_frame, read_or_build_player_metadata


def entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure alignment between learned pools and federations.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--games", default="games.parquet")
    parser.add_argument("--out", default="outputs/latent_pool_structure")
    parser.add_argument("--reference-year", type=int, default=2025)
    parser.add_argument("--min-fed-players", type=int, default=50)
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    membership = pool_membership_frame(cache)
    metadata = read_or_build_player_metadata(args.games, results_dir / "player_metadata_latest.csv")
    players = attach_metadata(membership, metadata, reference_year=args.reference_year)
    players = players[(players["fed"].notna()) & (players["fed"] != "UNK")].copy()

    fed_sizes = players.groupby("fed")["player_key"].count()
    keep_feds = fed_sizes[fed_sizes >= args.min_fed_players].index
    subset = players[players["fed"].isin(keep_feds)].copy()

    contingency = pd.crosstab(subset["fed"], subset["pool_id"])
    contingency.to_csv(results_dir / "federation_pool_contingency.csv", encoding="utf-8-sig")

    total = contingency.to_numpy().sum()
    p_joint = contingency.to_numpy(dtype=float) / total
    p_fed = p_joint.sum(axis=1)
    p_pool = p_joint.sum(axis=0)
    mi = 0.0
    for i in range(p_joint.shape[0]):
        for j in range(p_joint.shape[1]):
            if p_joint[i, j] > 0 and p_fed[i] > 0 and p_pool[j] > 0:
                mi += p_joint[i, j] * math.log2(p_joint[i, j] / (p_fed[i] * p_pool[j]))
    h_fed = entropy(p_fed)
    h_pool = entropy(p_pool)
    nmi_sqrt = mi / math.sqrt(h_fed * h_pool) if h_fed > 0 and h_pool > 0 else float("nan")
    nmi_min = mi / min(h_fed, h_pool) if min(h_fed, h_pool) > 0 else float("nan")

    rows = []
    for fed, row in contingency.iterrows():
        total_fed = int(row.sum())
        top_pool = int(row.idxmax())
        top_pool_count = int(row.max())
        rows.append({
            "fed": fed,
            "players": total_fed,
            "top_pool": top_pool,
            "top_pool_players": top_pool_count,
            "top_pool_share": top_pool_count / total_fed if total_fed else float("nan"),
            "pool_count": int((row > 0).sum()),
        })
    fed_summary = pd.DataFrame(rows).sort_values("players", ascending=False)
    fed_summary.to_csv(results_dir / "federation_pool_alignment_by_federation.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "players_used": int(total),
        "federations_used": int(contingency.shape[0]),
        "pools_used": int(contingency.shape[1]),
        "mutual_information_bits": mi,
        "federation_entropy_bits": h_fed,
        "pool_entropy_bits": h_pool,
        "normalized_mutual_information_sqrt": nmi_sqrt,
        "normalized_mutual_information_min": nmi_min,
        "min_fed_players": args.min_fed_players,
    }]).to_csv(results_dir / "federation_pool_alignment_summary.csv", index=False, encoding="utf-8-sig")

    plot = fed_summary.head(25).sort_values("top_pool_share")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(plot))))
    ax.barh(plot["fed"], plot["top_pool_share"])
    ax.set_title("How concentrated federations are in a single latent pool")
    ax.set_xlabel("share of federation players in the largest pool")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "federation_top_pool_share.png", dpi=180)
    plt.close(fig)

    print("[OK] saved federation-pool alignment outputs")


if __name__ == "__main__":
    main()
