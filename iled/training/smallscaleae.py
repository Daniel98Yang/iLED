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

class TimeFNOAutoEncoder(nn.Module):
    def __init__(self, input_dim=314, latent_dim=8, width=32):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.width = width

        # ── Encoder ─────────────────────────────
        self.enc_fc0 = nn.Linear(1, width)
        self.enc_conv = nn.Conv1d(width, width, kernel_size=1)
        self.enc_fc1 = nn.Linear(width, latent_dim)

        # ── Decoder ─────────────────────────────
        self.dec_fc0 = nn.Linear(latent_dim, width)
        self.dec_conv = nn.Conv1d(width, width, kernel_size=1)
        self.dec_fc1 = nn.Linear(width, 1)

    # ───────────────────────────────────────────
    # Encode: (B, 314) → (B, 314, d)
    # ───────────────────────────────────────────
    def encode(self, x):
        # x: (B, 314)
        x = x.unsqueeze(-1)                 # (B, 314, 1)
        x = self.enc_fc0(x)                 # (B, 314, width)
        x = x.permute(0, 2, 1)              # (B, width, 314)

        x = self.enc_conv(x)                # (B, width, 314)

        x = x.permute(0, 2, 1)              # (B, 314, width)
        z = self.enc_fc1(x)                 # (B, 314, latent_dim)

        return z

    # ───────────────────────────────────────────
    # Decode: (B, 314, d) → (B, 314)
    # ───────────────────────────────────────────
    def decode(self, z):
        x = self.dec_fc0(z)                 # (B, 314, width)
        x = x.permute(0, 2, 1)              # (B, width, 314)

        x = self.dec_conv(x)                # (B, width, 314)

        x = x.permute(0, 2, 1)              # (B, 314, width)
        x = self.dec_fc1(x).squeeze(-1)     # (B, 314)

        return x

    # ───────────────────────────────────────────
    # Forward (optional, not required by your code)
    # ───────────────────────────────────────────
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
