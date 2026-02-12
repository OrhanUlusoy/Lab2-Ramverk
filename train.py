import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CIFAR10Wrapper
from model import SimpleCNN


def train_model(epochs=5, batch_size=32, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CIFAR10Wrapper(root="data/cifar10", train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return accuracy
