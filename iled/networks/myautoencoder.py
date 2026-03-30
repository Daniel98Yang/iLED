# my_autoencoder.py
import torch
import torch.nn as nn
from iled.nn.autoencoders import AutoEncoder  # adjust import path to match your install

class MyEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),  # 3 latent variables
        )

    def forward(self, x):
        return self.net(x)


class MyDecoder(nn.Module):
    def __init__(self, output_dim, latent_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class MyAutoEncoder(AutoEncoder):
    """
    Wraps your encoder/decoder into the AutoEncoder interface.
    AutoEncoder already provides batch_transform() and batch_inverse_transform()
    which reshape (B, T, ...) → merge batch+time → encode → unmerge. Free.
    """
    def __init__(self, input_dim, latent_dim=3):
        encoder = MyEncoder(input_dim, latent_dim)
        decoder = MyDecoder(input_dim, latent_dim)
        super().__init__(encoder, decoder)