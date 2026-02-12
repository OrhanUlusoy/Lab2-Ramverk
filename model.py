import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Conv layer 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv layer 2
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
