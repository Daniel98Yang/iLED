import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import joblib
import os

class TemporalAttention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.score = nn.Linear(latent_dim, 1)

    def forward(self, z_seq):
        """
        z_seq: (B, T, D)
        returns: (B, D)
        """
        # compute attention scores
        scores = self.score(z_seq)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)

        # weighted sum
        z_attn = (weights * z_seq).sum(dim=1)   # (B, D)
        return z_attn