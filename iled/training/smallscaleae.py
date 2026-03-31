# mytrainer3.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import joblib
import os

class TimeAutoEncoder(nn.Module):
    def __init__(self, input_dim=314, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)


class FNOEncoder(nn.Module):
    def __init__(self, in_channels=1, width=32, modes=16, latent_dim=8):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(1, width)

        self.conv = nn.Conv1d(width, width, kernel_size=1)
        self.fc1 = nn.Linear(width, latent_dim)

    def forward(self, x):
        # x: (B, 314)
        x = x.unsqueeze(-1)           # (B, 314, 1)
        x = self.fc0(x)               # (B, 314, width)
        x = x.permute(0, 2, 1)        # (B, width, 314)

        x = self.conv(x)              # (B, width, 314)

        x = x.permute(0, 2, 1)        # (B, 314, width)
        z = self.fc1(x)               # (B, 314, latent_dim)

        return z
    
class FNODecoder(nn.Module):
    def __init__(self, latent_dim=8, width=32):
        super().__init__()
        self.fc0 = nn.Linear(latent_dim, width)
        self.conv = nn.Conv1d(width, width, kernel_size=1)
        self.fc1 = nn.Linear(width, 1)

    def forward(self, z):
        # z: (B, 314, latent_dim)
        x = self.fc0(z)
        x = x.permute(0, 2, 1)

        x = self.conv(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x).squeeze(-1)

        return x
    
