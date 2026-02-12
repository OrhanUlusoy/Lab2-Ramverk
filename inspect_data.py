"""Snabb inspektion av lokala datafiler.

Den här filen är ett hjälp-script för att:
- kontrollera att `.npy`-filerna går att läsa
- se shapes/dtypes
- kika på `trainLabels.csv`
"""

import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path("data")

# Lista alla `.npy` i data-mappen och skriv ut enkel metadata.
for f in data_dir.glob("*.npy"):
    arr = np.load(f)
    print(f"{f.name}: shape={arr.shape}, dtype={arr.dtype}")

# Läs labels och skriv ut exempelrader för att verifiera format.
labels = pd.read_csv(data_dir / "trainLabels.csv")
print("trainLabels.csv head:")
print(labels.head())
print("label dtype:", labels["label"].dtype)
print("unique labels sample:", labels["label"].unique()[:10])
