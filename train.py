"""Träningslogik.

Håller träning/evaluering separerad från:
- `dataset.py` (data)
- `model.py` (modell)
- `main.py` (orchestrering och param-läsning)
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CIFAR10Wrapper
from model import SimpleCNN

# Sätt seed för reproducibilitet
def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Försök göra körningar mer deterministiska (särskilt på GPU).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad() # För utvärdering behöver vi inte beräkna gradients, så vi kan spara minne och tid.
def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total if total else 0.0

# Huvudfunktion för att träna modellen. Tar in hyperparametrar och träningsinställningar, och returnerar test-accuracy.
def train_model(epochs=5, batch_size=32, lr=0.001, *, seed=42, data_root="data/cifar10", download=False):
    # Välj GPU om det finns, annars CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _set_seed(int(seed)) # Sätt seed för reproducibilitet.

    # Dataset + DataLoader.
    # Vi tränar på train-set och utvärderar på test-set (mer korrekt än train-accuracy).
    train_ds = CIFAR10Wrapper(root=data_root, train=True, download=download)
    test_ds = CIFAR10Wrapper(root=data_root, train=False, download=download)

    # num_workers=0 är robustast på Windows.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initiera modell och flytta till vald device.
    model = SimpleCNN().to(device)

    # För klassificering: cross-entropy, och Adam som optimerare.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    test_accuracy = 0.0 # För att hålla koll på senaste test-accuracy, som vi returnerar i slutet.
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        # Träningspass över train-set.
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / batches if batches else 0.0
        test_accuracy = _evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    # Returnera sista epochens test-accuracy (loggas i main.py/README).
    return float(test_accuracy)
