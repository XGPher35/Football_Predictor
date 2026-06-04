# Neural Network Notebooks — Football MTL Smoke Test

Minimal data loader and a PyTorch multi-task BiLSTM+Attention model that follows the system spec in `spec-sheet.md`.

## Quick start

Smoke test (runs a forward pass on a sample of CSV data):

```bash
python train_smoke.py
```

Train and save a model (stores checkpoints in `artifacts/`):

```bash
python -m nn_football.train --epochs 5 --save_dir artifacts
```

Prompt UI (interactive match selection with simple autocomplete):

```bash
python ui_prompt.py --ckpt_path artifacts/model_best.pt --scaler_path artifacts/scaler.pkl
```

## Dataset download

Download EPL CSVs into `data/epl/`:

```bash
mkdir -p data/epl
wget "https://datahub.io/football/english-premier-league/_r/-/season-2021.csv" -O data/epl/season-2021.csv
wget "https://datahub.io/football/english-premier-league/_r/-/season-2122.csv" -O data/epl/season-2122.csv
wget "https://datahub.io/football/english-premier-league/_r/-/season-2223.csv" -O data/epl/season-2223.csv
wget "https://datahub.io/football/english-premier-league/_r/-/season-2324.csv" -O data/epl/season-2324.csv
wget "https://datahub.io/football/english-premier-league/_r/-/season-2425.csv" -O data/epl/season-2425.csv
wget "https://datahub.io/football/english-premier-league/_r/-/season-2526.csv" -O data/epl/season-2526.csv
```

## Requirements

Dependencies are listed in `pyproject.toml` (PyTorch, pandas, numpy).
