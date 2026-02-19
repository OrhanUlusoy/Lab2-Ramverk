## Körning

- Kör kommandon från repo-roten (mappen som innehåller `main.py`).
- Om `data/` saknas, hämta datasetet med `uv run dvc pull`.
- Starta hela flödet via `main.py` (krav i labben):
	- PowerShell: `./.venv/Scripts/Activate.ps1` och sedan `uv run python main.py`
	- Git Bash: `source .venv/Scripts/activate` och sedan `uv run python main.py`

Not: Om ni *inte* har DVC-data lokalt kan ni tillfälligt sätta `"download": true` i `params.json` för att låta torchvision ladda CIFAR-10.

## Struktur

- `dataset.py`: data-loading (CIFAR-10) via ett PyTorch `Dataset`
- `model.py`: enkel CNN-modell
- `train.py`: träningsloop
- `main.py`: läser `params.json`, kör alla experiment och skriver ut test-accuracy

## Experiments

Not: Resultaten kan variera något mellan körningar.

Accuracy nedan avser **test accuracy (sista epoken)**.

| Experiment | epochs | batch_size | learning_rate | Test Accuracy |
|-----------|--------|------------|---------------|----------------|
| A (baseline) | 3 | 32 | 0.001 | 62.53% |
| B (higher lr) | 3 | 32 | 0.01  | 44.54% |
| C (bigger batch) | 3 | 64 | 0.001 | 61.83% |

