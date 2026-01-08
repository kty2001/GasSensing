import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(
    model: str,
    input_length: int = 7300,
    num_classes: int = 3,
):
    if model == "cnn1d":
        return CNN1DClassifier(input_length, num_classes)
    elif model == "resnet1d":
        return ResNet1DClassifier(input_length, num_classes)
    elif model == "mlp":
        return MLPClassifier(input_length, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model}")


class CNN1DClassifier(nn.Module):
    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x).squeeze(-1)
        return self.fc(x)
    

class ResNet1DClassifier(nn.Module):
    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        self.stem = nn.Conv1d(1, 16, kernel_size=7, padding=3)

        self.block1 = ResidualBlock1D(16, 32)
        self.block2 = ResidualBlock1D(32, 64)
        self.block3 = ResidualBlock1D(64, 128)

        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.relu(x + residual)


class MLPClassifier(nn.Module):
    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
