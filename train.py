"""Träningslogik.

Håller träning/evaluering separerad från:
- `dataset.py` (data)
- `model.py` (modell)
- `main.py` (orchestrering och param-läsning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CIFAR10Wrapper
from model import SimpleCNN


def train_model(epochs=5, batch_size=32, lr=0.001):
    # Välj GPU om det finns, annars CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + DataLoader (shuffle för att minska ordningsbias).
    dataset = CIFAR10Wrapper(root="data/cifar10", train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initiera modell och flytta till vald device.
    model = SimpleCNN().to(device)

    # För klassificering: cross-entropy, och Adam som optimerare.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Träningspass över hela datasetet.
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Framåtpass + loss.
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Bakåtpass + optimeringssteg.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Beräkna antal rätt (top-1).
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Epoch-accuracy i procent.
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Returnera sista epochens accuracy (loggas som Final Accuracy i main.py).
    return accuracy
