import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path("data")

for f in data_dir.glob("*.npy"):
    arr = np.load(f)
    print(f"{f.name}: shape={arr.shape}, dtype={arr.dtype}")

labels = pd.read_csv(data_dir / "trainLabels.csv")
print("trainLabels.csv head:")
print(labels.head())
print("label dtype:", labels["label"].dtype)
print("unique labels sample:", labels["label"].unique()[:10])
