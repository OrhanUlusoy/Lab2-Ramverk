## Körning

- (Valfritt) Om `data/` saknas, hämta datasetet med `dvc pull`.
- Starta hela flödet via `main.py` (krav i labben):
	- PowerShell: `./.venv/Scripts/Activate.ps1` och sedan `python -m uv run python main.py`
	- Git Bash: `source .venv/Scripts/activate` och sedan `python -m uv run python main.py`

## Struktur

- `dataset.py`: data-loading (CIFAR-10) via ett PyTorch `Dataset`
- `model.py`: enkel CNN-modell
- `train.py`: träningsloop
- `main.py`: läser `params.json`, kör träning och skriver ut `Final Accuracy`

## Experiments

| Experiment | epochs | batch_size | learning_rate | Final Accuracy |
|-----------|--------|------------|---------------|----------------|
| A (baseline) | 3 | 32 | 0.001 | 63.77% |
| B (higher lr) | 3 | 32 | 0.01  | 44.37% |
| C (bigger batch) | 3 | 64 | 0.001 | 61.71% |

