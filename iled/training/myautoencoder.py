# myautoencoder.py
# Uses RegularConvAutoencoder — confirmed by saved weight shapes:
#   encoder.encoder.0.weight: [128, 314, 7]  → Conv1d(314, 128, kernel=7)
#   encoder.encoder.6.weight: [3, 128, 3]    → Conv1d(128, 3, kernel=3)
#   decoder.0.weight:         [3, 128, 3]    → ConvTranspose1d(3, 128, kernel=3)
# PermutingConvAutoencoder would have encoder.encoder.0.weight: [48, 942, 7] — not what we have.

import torch
import torch.nn as nn
from autoencoder import RegularConvAutoencoder


class MyAutoEncoder(nn.Module):
    """
    Window-scale autoencoder wrapping the pretrained RegularConvAutoencoder.

    encode(x):  (B, 314, T) → CNN → (B, 3, T) → mean-pool → (B, 3)
    decode(z):  (B, 3)      → expand to (B, 3, T) → ConvTranspose → (B, 314, T)

    State dict key structure that must match saved .pth:
        encoder.encoder.{0,1,3,4,6,7}.*   (Conv + BN layers)
        decoder.{0,2,4}.*                  (ConvTranspose layers)
    """

    def __init__(self, num_features=314, latent_features=3,
                 seq_len=200, num_conv_filters=128, padding="same"):
        super().__init__()
        self.seq_len = seq_len

        # Instantiate with same args used during ProtoTSNet training
        self.model = RegularConvAutoencoder(
            num_features=num_features,
            latent_features=latent_features,
            padding=padding,
            do_max_pool=False,       # weights confirm no MaxPool
            do_batch_norm=True,      # BN layers present in saved weights
            num_conv_filters=num_conv_filters,
        )

    def encode(self, x):
        """
        x: (B, 314, T)
        Returns: (B, 3)  — mean-pooled over time axis
        """
        # RegularConvAutoencoder.forward returns (decoded, encoded)
        # We only need encoded here; using model.encoder directly avoids
        # the dropout and decoder pass
        z_seq = self.model.encoder(x)    # (B, 3, T)
        return z_seq.mean(dim=-1)        # (B, 3)

    def decode(self, z):
        """
        z: (B, 3)
        Returns: (B, 314, T)
        """
        # Expand flat latent back to temporal dimension
        z_seq = z.unsqueeze(-1).expand(-1, -1, self.seq_len)  # (B, 3, T)
        # model.decoder is nn.Sequential → returns a plain tensor, NOT a tuple
        return self.model.decoder(z_seq)                       # (B, 314, T)

    def forward(self, x):
        return self.decode(self.encode(x))

class CycleFNOAutoEncoder(nn.Module):
    def __init__(self, num_features=314, seq_len=200, latent_dim=8, width=32):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
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
    # Encode: (B, 314, 200) → (B, 314, 200, d)
    # ───────────────────────────────────────────
    def encode(self, x):
        B, C, T = x.shape

        x = x.view(B * C, T)              # treat each sensor separately
        x = x.unsqueeze(-1)              # (B*C, T, 1)

        x = self.enc_fc0(x)              # (B*C, T, width)
        x = x.permute(0, 2, 1)           # (B*C, width, T)

        x = self.enc_conv(x)             # (B*C, width, T)

        x = x.permute(0, 2, 1)           # (B*C, T, width)
        z = self.enc_fc1(x)              # (B*C, T, d)

        z = z.view(B, C, T, self.latent_dim)

        return z

    # ───────────────────────────────────────────
    # Decode: (B, 314, 200, d) → (B, 314, 200)
    # ───────────────────────────────────────────
    def decode(self, z):
        B, C, T, d = z.shape

        z = z.view(B * C, T, d)

        x = self.dec_fc0(z)              # (B*C, T, width)
        x = x.permute(0, 2, 1)

        x = self.dec_conv(x)

        x = x.permute(0, 2, 1)
        x = self.dec_fc1(x).squeeze(-1)  # (B*C, T)

        x = x.view(B, C, T)

        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z