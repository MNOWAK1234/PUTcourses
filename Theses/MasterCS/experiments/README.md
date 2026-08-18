# Experiments

`run_experiments.py` is the main rating replay and model-selection script. It produces the main results used in the thesis.

`run_all_thesis_experiments.py` runs the additional analyses after the main experiment has produced `experiments/results/` and `experiments/cache/`.

The analysis scripts are grouped by purpose:

- `analysis/prediction_sensitivity/`
- `analysis/robustness_ablation/`
- `analysis/latent_pool_structure/`

Generated outputs are written to `outputs/`.
