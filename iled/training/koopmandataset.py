# koopmandataset.py
import torch
import numpy as np
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────
# WINDOW-SCALE DATASET
# Pairs adjacent full trajectories: (X_n, X_{n+1}, U_n)
# Used with the pretrained CNN autoencoder (window Koopman)
# ─────────────────────────────────────────────────────────────────
class CycleDataset(Dataset):
    """
    Window-level dataset for per-window Koopman dynamics.

    sensor_data : (N, 314, 200)  channels-first
    controls    : (N, 200, 8) or None
                  Controls are averaged over the 200-timestep window
                  to produce one control vector per window pair.
    """
    def __init__(self, sensor_data, controls=None):
        N, C, T = sensor_data.shape

        # Pair adjacent windows: (n) → (n+1)
        self.x_t    = torch.tensor(sensor_data[:-1], dtype=torch.float32)  # (N-1, 314, 200)
        self.x_next = torch.tensor(sensor_data[1:],  dtype=torch.float32)  # (N-1, 314, 200)

        if controls is not None:
            # controls: (N, 200, 8) → mean over time → (N, 8) → pair
            u = controls.mean(axis=1).astype(np.float32)   # (N, 8)
            self.u_t = torch.tensor(u[:-1])                 # (N-1, 8)
        else:
            self.u_t = None

    def __len__(self):
        return len(self.x_t)

    def __getitem__(self, idx):
        item = {'x_t': self.x_t[idx], 'x_next': self.x_next[idx]}
        if self.u_t is not None:
            item['u_t'] = self.u_t[idx]
        return item


# ─────────────────────────────────────────────────────────────────
# TIMESTEP-SCALE DATASET
# Pairs adjacent individual timesteps: (x_t, x_{t+1}, u_t)
# Used with TimeAutoEncoder (per-timestep Koopman)
# ─────────────────────────────────────────────────────────────────
class TimestepDataset(Dataset):
    """
    Timestep-level dataset for per-timestep Koopman dynamics.

    sensor_data : (N, 314, 200)  channels-first
    controls    : (N, 200, 8) or None

    Each item is a pair of consecutive single-timestep sensor snapshots.
    Produces N * 199 pairs total (one per consecutive timestep in each trajectory).
    """
    def __init__(self, sensor_data, controls=None):
        N, C, T = sensor_data.shape

        # Timestep pairs within each trajectory
        # sensor_data[:, :, :-1]: (N, 314, 199) → flatten → (N*199, 314)
        x_t    = sensor_data[:, :, :-1].transpose(0, 2, 1).reshape(-1, C)  # (N*199, 314)
        x_next = sensor_data[:, :, 1:].transpose(0, 2, 1).reshape(-1, C)   # (N*199, 314)

        self.x_t    = torch.tensor(x_t,    dtype=torch.float32)
        self.x_next = torch.tensor(x_next, dtype=torch.float32)

        if controls is not None:
            # controls: (N, 200, 8)
            # Take u_t from timesteps 0..T-2 to align with x_t
            u = controls[:, :-1, :].reshape(-1, controls.shape[-1])  # (N*199, 8)
            self.u_t = torch.tensor(u.astype(np.float32))
        else:
            self.u_t = None

    def __len__(self):
        return len(self.x_t)

    def __getitem__(self, idx):
        item = {'x_t': self.x_t[idx], 'x_next': self.x_next[idx]}
        if self.u_t is not None:
            item['u_t'] = self.u_t[idx]
        return item


# ─────────────────────────────────────────────────────────────────
# LEGACY — kept for backwards compatibility
# ─────────────────────────────────────────────────────────────────
class KoopmanDataset(TimestepDataset):
    """Alias for TimestepDataset. Use TimestepDataset in new code."""
    def __init__(self, sensor_data, controls=None, **kwargs):
        # Old calls passed (N, T, 314) or (N, 314, T) — normalise to (N, 314, T)
        if sensor_data.ndim == 3 and sensor_data.shape[1] != 314:
            sensor_data = sensor_data.transpose(0, 2, 1)
        super().__init__(sensor_data, controls)