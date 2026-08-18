# Data pipeline

This folder contains only the scripts needed to collect and reconstruct the game data used by the rating experiments.

## Folders

- `scraping/` contains the local and VPS FIDE scraping scripts.
- `preparation/` converts collected rows to Parquet, resolves opponent ratings from reciprocal rows and builds the final unique-game table.
- `vps_tools/` contains helper scripts used to check, download from and stop remote scraping sessions.
- `config_examples/` contains safe CSV templates without real hosts or credentials.
