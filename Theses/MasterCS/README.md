# Chess rating pool model

This repository contains the code used for a master's thesis on a dynamic pool-aware rating system for chess players. The project reconstructs a chronological game dataset from public FIDE calculation records, runs an Elo-style rating replay with additional latent-pool components, and produces the analyses used in the thesis.

The repository intentionally contains only the scripts needed for the final pipeline and thesis experiments. Older exploratory experiments, screenshots, temporary archives and unrelated drafts are not included. Large raw and intermediate data files are also not included.

## Repository layout

```text
data_pipeline/
  scraping/              # FIDE data collection scripts
  preparation/           # conversion, rating recovery and unique-game construction
  vps_tools/             # helper scripts for distributed scraping
  config_examples/       # safe example CSV files, no credentials

experiments/
  run_experiments.py     # main rating model and final evaluation
  run_all_thesis_experiments.py
  make_thesis_plots.py
  analysis/
    prediction_sensitivity/
    robustness_ablation/
    latent_pool_structure/

reference_results/       # compact reference outputs used for checking tables/figures
thesis/                  # final thesis PDF or local thesis exports
```

## Data pipeline

The raw public FIDE calculation records are player-centric. A physical game can appear once from each player's perspective, so the preparation pipeline reconstructs a unique-game table before the rating replay.

Typical order:

```powershell
python data_pipeline/scraping/scrape_fide_local.py
python data_pipeline/preparation/convert_raw_games_to_parquet.py
python data_pipeline/preparation/resolve_opponent_ratings.py
python data_pipeline/preparation/build_unique_games.py
```

The final main experiment expects `games_unique.parquet` in the repository root. Metadata analyses also use `games.parquet`. These files are not committed because they are large.

## Main rating experiment

Run from the repository root:

```powershell
python experiments/run_experiments.py --mode all --profile max
```

This creates the local output folder `experiments/` with cache, plots and result tables. The generated output folders are ignored by Git.

## Thesis analyses

After the main experiment has generated `experiments/results/` and `experiments/cache/`, run:

```powershell
python experiments/run_all_thesis_experiments.py --project-root .
```

Useful options:

```powershell
python experiments/run_all_thesis_experiments.py --project-root . --skip-heavy-shocks
python experiments/run_all_thesis_experiments.py --project-root . --skip-latent-pool-structure
python experiments/run_all_thesis_experiments.py --project-root . --run-main evaluate
```

The script writes analysis outputs to `outputs/` and packs them into `thesis_experiment_outputs.zip`.

## Analysis groups

`prediction_sensitivity/` contains the expected-score curves, pool-weight sensitivity, provisional-entry sensitivity, calibration bins, monthly stability and global shock recovery.

`robustness_ablation/` contains component ablation, train-validation-test behaviour, pool-offset diagnostics, rating-scale diagnostics, pool-specific shocks, update-speed/scale sensitivity, white-advantage sensitivity and simplification reports.

`latent_pool_structure/` contains the descriptive analyses used in the appendix: federation composition, age and junior structure, elite-player distribution, federation-pool alignment, cross-pool interactions and metadata diagnostics.

## Security

Do not commit real `servers.csv`, `shard_plan.csv`, SSH keys, VPS passwords or downloaded raw data. Safe templates are in `data_pipeline/config_examples/`.
