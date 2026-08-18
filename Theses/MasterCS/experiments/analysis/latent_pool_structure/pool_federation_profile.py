#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import (
    attach_metadata,
    ensure_dirs,
    hhi_from_counts,
    import_experiment_module,
    load_cache,
    pool_membership_frame,
    read_or_build_player_metadata,
    top_values_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain latent pools using federation composition.")
    parser.add_argument("--module", default="run_experiments.py")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--games", default="games.parquet")
    parser.add_argument("--out", default="outputs/latent_pool_structure")
    parser.add_argument("--reference-year", type=int, default=2025)
    parser.add_argument("--top-pools", type=int, default=20)
    parser.add_argument("--force-metadata", action="store_true")
    args = parser.parse_args()

    exp = import_experiment_module(args.module)
    results_dir, plots_dir = ensure_dirs(args.out)
    cache = load_cache(exp, args.cache)
    membership = pool_membership_frame(cache)
    metadata = read_or_build_player_metadata(args.games, results_dir / "player_metadata_latest.csv", force=args.force_metadata)
    players = attach_metadata(membership, metadata, reference_year=args.reference_year)

    fed_counts = (
        players.groupby(["pool_id", "fed"], as_index=False)
        .agg(players=("player_key", "count"))
        .sort_values(["pool_id", "players"], ascending=[True, False])
    )
    fed_counts["pool_total"] = fed_counts.groupby("pool_id")["players"].transform("sum")
    fed_counts["share"] = fed_counts["players"] / fed_counts["pool_total"]
    fed_counts.to_csv(results_dir / "pool_federation_counts.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for pool_id, group in fed_counts.groupby("pool_id"):
        counts = group.set_index("fed")["players"]
        top = group.sort_values("players", ascending=False).head(1).iloc[0]
        summary_rows.append({
            "pool_id": int(pool_id),
            "players": int(group["players"].sum()),
            "federations": int(group["fed"].nunique()),
            "top_federation": top["fed"],
            "top_federation_players": int(top["players"]),
            "top_federation_share": float(top["share"]),
            "federation_hhi": hhi_from_counts(counts),
            "top_5_federations": top_values_text(group, "fed", "players", 5),
        })

    summary = pd.DataFrame(summary_rows).sort_values("players", ascending=False)
    summary.to_csv(results_dir / "pool_federation_summary.csv", index=False, encoding="utf-8-sig")

    top_pool_ids = summary.head(args.top_pools)["pool_id"].tolist()
    top_rows = fed_counts[fed_counts["pool_id"].isin(top_pool_ids)].copy()
    top_rows = top_rows.sort_values(["pool_id", "players"], ascending=[True, False])
    top_rows = top_rows.groupby("pool_id").head(5)

    pivot = top_rows.pivot_table(index="pool_id", columns="fed", values="share", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_title("Top federation shares inside the largest latent pools")
    ax.set_xlabel("federation")
    ax.set_ylabel("pool id")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    fig.colorbar(im, ax=ax, label="share")
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_federation_share_heatmap.png", dpi=180)
    plt.close(fig)

    plot = summary.head(args.top_pools).sort_values("top_federation_share")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot))))
    ax.barh(["pool " + str(x) for x in plot["pool_id"]], plot["top_federation_share"])
    ax.set_title("Dominance of the largest federation in each pool")
    ax.set_xlabel("share of largest federation")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pool_top_federation_share.png", dpi=180)
    plt.close(fig)

    print("[OK] saved federation latent-pool structure outputs")


if __name__ == "__main__":
    main()
