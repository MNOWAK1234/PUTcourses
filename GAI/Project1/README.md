# AI in Games — Tetris Agents Project

## Project overview

The aim of the project is to compare several approaches to controlling an agent in the Tetris environment:

- hand-crafted heuristic control
- reinforcement learning
- optimization / evolutionary methods

The repository includes:

- a **human-playable** Tetris version
- multiple trainable agents
- fixed-time benchmarks for non-learning baselines
- experiment scripts for evaluating agents over multiple games
- GIF generation for best runs
- analysis scripts for plots and summary tables

The main agent families used in the project are:

- **Random**
- **Heuristic**
- **DQN**
- **REINFORCE**
- **A2C**
- **PPO**
- **Genetic Algorithm**
- **Evolution Strategies**

---

## Repository structure

```text
.
├── agents/                         # Agent implementations
│   ├── __init__.py
│   ├── a2c_agent.py
│   ├── base_agent.py
│   ├── dqn_agent.py
│   ├── es_agent.py
│   ├── genetic_agent.py
│   ├── heuristic_agent.py
│   ├── human_agent.py
│   ├── ppo_agent.py
│   ├── random_agent.py
│   └── reinforce_agent.py
│
├── analysis/                       # Analysis and plotting scripts
│   ├── plot_training_curves.py
│   └── summarise_play_results.py
│
├── experiments/                    # Experiment scripts
│   ├── run_experiments.py
│   ├── run_baseline_time_benchmarks.py
│   └── save_gifs.py
│
├── models/                         # Saved trained models (.pth)
├── results/                        # Raw results, summaries, plots, GIFs
├── src/
│   └── tetris.py                   # Core Tetris environment
│
├── train/                          # Training scripts
│   ├── run_all_training.py
│   ├── train_dqn_reinforce.py
│   ├── train_evolutionary.py
│   └── train_onpolicy.py
│
├── config.py                       # Global configuration
├── helper.py                       # Helper functions
├── main.py                         # Original interactive launcher
├── play_human.py                   # Human-playable Tetris
└── README.md
```

---

## How to use the project

## 1. Play Tetris manually

To run the human-playable version:

```bash
python play_human.py
```

Controls:

- `A` / Left Arrow — move left
- `D` / Right Arrow — move right
- `W` / Up Arrow — rotate
- `S` / Down Arrow — soft drop
- `Space` — hard drop
- `Q` / `Esc` — quit

This is useful for checking the game mechanics and verifying that the environment works correctly.

---

## 2. Train learning and optimization agents

To train all trainable agents sequentially:

```bash
python train/run_all_training.py --agents all
```

To train all of them with a fixed time budget per agent:

```bash
python train/run_all_training.py --agents all --time_budget_minutes 30
```

This runs:

- DQN
- REINFORCE
- A2C
- PPO
- Genetic Algorithm
- Evolution Strategies

### Train selected agents only

Example:

```bash
python train/run_all_training.py --agents dqn,ppo,es --time_budget_minutes 30
```

### Optional examples

```bash
python train/run_all_training.py --agents dqn --dqn_epochs 1000
python train/run_all_training.py --agents genetic --genetic_generations 30
python train/run_all_training.py --agents a2c --a2c_total_steps 100000
python train/run_all_training.py --agents all --time_budget_minutes 30 --verbose
```

### Training outputs

During training, the project saves:

- best model files to `models/`
- minute-based training statistics to `results/`

Examples:

```text
models/dqn_score_304220.pth
models/ppo_score_231460.pth
results/dqn_minute_stats.csv
results/ppo_minute_stats.csv
results/genetic_minute_stats.csv
```

---

## 3. Run fixed-time baselines for Random and Heuristic

These two agents do not learn, but they can still be benchmarked for a fixed amount of time.

Run both:

```bash
python experiments/run_baseline_time_benchmarks.py --agents random,heuristic --time_budget_minutes 30
```

Run only the heuristic agent:

```bash
python experiments/run_baseline_time_benchmarks.py --agents heuristic --time_budget_minutes 30
```

These scripts save minute-based progress files to `results/`, compatible with the training curve plotting script.

---

## 4. Run experiments and save raw play results

To evaluate agents over multiple games and save raw per-game results:

```bash
python experiments/run_experiments.py --agent_types all --num_games 10
```

If `--max_pieces` is not provided, the script uses:

```text
config.MAX_PIECES_PER_EVAL_GAME
```

Example with an explicit limit:

```bash
python experiments/run_experiments.py --agent_types all --num_games 10 --max_pieces 400
```

You can also evaluate only selected agents:

```bash
python experiments/run_experiments.py --agent_types random,heuristic,ppo,es --num_games 10 --max_pieces 400
```

### Output

This script writes raw results to:

```text
results/test_raw_runs.csv
```

Each row corresponds to one played game and includes values such as:

- agent
- game index
- seed
- score
- pieces played
- lines cleared
- elapsed time

This raw CSV is the main input for later analysis.

---

## 5. Save GIFs for best runs

To create GIFs of the best runs found in `results/test_raw_runs.csv`:

```bash
python experiments/save_gifs.py --agent_types all
```

By default, GIF replay uses:

- `max_pieces = 400`

You can change it:

```bash
python experiments/save_gifs.py --agent_types all --max_pieces 400
```

GIFs are saved to:

```text
results/gifs/
```

Example filenames:

```text
results/gifs/dqn_best_score_8340.gif
results/gifs/heuristic_best_score_9100.gif
results/gifs/ppo_best_score_21140.gif
```

---

## 6. Plot training curves

To generate plots from the minute-based training logs:

```bash
python analysis/plot_training_curves.py --max_minutes 30
```

This script reads files such as:

```text
results/dqn_minute_stats.csv
results/reinforce_minute_stats.csv
results/a2c_minute_stats.csv
results/ppo_minute_stats.csv
results/genetic_minute_stats.csv
results/es_minute_stats.csv
results/random_minute_stats.csv
results/heuristic_minute_stats.csv
```

and saves plots to:

```text
results/plots/
```

Typical outputs:

```text
results/plots/best_score_over_time_combined.png
results/plots/best_score_over_time_panels.png
```

---

## 7. Summarize play results and generate comparison plots

After running experiments, summarize the raw results:

```bash
python analysis/summarise_play_results.py
```

This script reads:

```text
results/test_raw_runs.csv
```

and produces:

- a summary CSV
- comparison plots
- a printed summary table in the console

Typical outputs:

```text
results/test_summary.csv
results/plots/test_mean_score_bar.png
results/plots/test_score_boxplot.png
results/plots/test_best_worst_bar.png
```

The summary includes statistics such as:

- mean score
- median score
- standard deviation
- best score
- worst score
- mean lines cleared
- mean number of pieces

---

## Recommended workflow

A typical full workflow looks like this:

### Step 1 — verify the game manually

```bash
python play_human.py
```

### Step 2 — train the learning / optimization agents

```bash
python train/run_all_training.py --agents all --time_budget_minutes 30
```

### Step 3 — benchmark the non-learning baselines

```bash
python experiments/run_baseline_time_benchmarks.py --agents random,heuristic --time_budget_minutes 30
```

### Step 4 — plot training progress

```bash
python analysis/plot_training_curves.py --max_minutes 30
```

### Step 5 — run final evaluation games

```bash
python experiments/run_experiments.py --agent_types all --num_games 10 --max_pieces 400
```

### Step 6 — save GIFs for best runs

```bash
python experiments/save_gifs.py --agent_types all --max_pieces 400
```

### Step 7 — summarize and visualize evaluation results

```bash
python analysis/summarise_play_results.py
```

---

## Agent overview

### Random

Chooses a valid move at random.
Used as a simple baseline.

### Heuristic

Uses hand-crafted rules and a board evaluation strategy.
Represents the manual / rule-based approach.

### DQN

Value-based reinforcement learning agent.
Learns to estimate how good a resulting board state is.

### REINFORCE

Policy-gradient method.
Learns from episodic returns.

### A2C

Actor-Critic reinforcement learning method.
Uses separate policy and value estimation.

### PPO

A more stable on-policy Actor-Critic method.
Usually one of the strongest RL baselines in this project.

### Genetic Algorithm

Optimization-based method using population evolution, mutation and crossover.

### Evolution Strategies

Optimization-based method evolving policy parameters through perturbation and fitness-based updates.

---

## Main outputs of the repository

### Models

Saved to:

```text
models/
```

Examples:

```text
models/dqn_score_304220.pth
models/genetic_score_511500.pth
models/es_score_8500.pth
```

### Raw experiment results

Saved to:

```text
results/test_raw_runs.csv
```

### Experiment summary

Saved to:

```text
results/test_summary.csv
```

### Training logs

Saved to:

```text
results/*_minute_stats.csv
```

### Plots

Saved to:

```text
results/plots/
```

### GIFs

Saved to:

```text
results/gifs/
```

---

## Notes

- `random` and `heuristic` do not require model files.
- `run_experiments.py` is intended for collecting quantitative results.
- `save_gifs.py` is intended only for visualization of best runs.
- `plot_training_curves.py` uses the minute-based training logs.
- `summarise_play_results.py` uses the raw per-game experiment CSV.

---

## Example final command set

```bash
python play_human.py
python train/run_all_training.py --agents all --time_budget_minutes 30
python experiments/run_baseline_time_benchmarks.py --agents random,heuristic --time_budget_minutes 30
python analysis/plot_training_curves.py --max_minutes 30
python experiments/run_experiments.py --agent_types all --num_games 10 --max_pieces 400
python experiments/save_gifs.py --agent_types all --max_pieces 400
python analysis/summarise_play_results.py
```

---
