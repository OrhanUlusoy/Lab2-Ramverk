"""Modell-definition.

En enkel CNN för CIFAR-10 som är tillräcklig för att demonstrera:
- conv/pool-steg
- fullt kopplade lager
- träningsloop i separat modul
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Konvolutionslager
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fullt kopplade lager
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv-lager 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv-lager 2
        x = self.pool(F.relu(self.conv2(x)))

        # Platta ut till (batch, features)
        x = torch.flatten(x, 1)

        # Fullt kopplade lager
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
