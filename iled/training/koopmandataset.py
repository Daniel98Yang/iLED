# koopman_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset

class KoopmanDataset(Dataset):
    """
    Constructs ALL valid sequential (x_t, x_{t+1}, u_t) pairs from
    a dataset of trajectories.

    Args:
        trajectories : np.ndarray of shape (N, T, obs_dim)
        controls     : np.ndarray of shape (N, T-1, control_dim), or None
    """
    def __init__(self, trajectories, controls=None):
        N, T, obs_dim = trajectories.shape
        self.obs_dim = obs_dim

        # Flatten all trajectories into individual pairs
        # x_t  comes from timesteps 0..T-2, x_next from 1..T-1
        x_t    = trajectories[:, :-1, :]   # (N, T-1, obs_dim)
        x_next = trajectories[:, 1:,  :]   # (N, T-1, obs_dim)

        # Reshape to (N*(T-1), obs_dim)
        self.x_t    = torch.tensor(x_t.reshape(-1, obs_dim),    dtype=torch.float32)
        self.x_next = torch.tensor(x_next.reshape(-1, obs_dim), dtype=torch.float32)

        if controls is not None:
            ctrl_dim = controls.shape[-1]
            u = controls.reshape(-1, ctrl_dim)  # (N*(T-1), control_dim)
            self.u_t = torch.tensor(u, dtype=torch.float32)
        else:
            self.u_t = None

    def __len__(self):
        return len(self.x_t)

    def __getitem__(self, idx):
        item = {
            'x_t':    self.x_t[idx],
            'x_next': self.x_next[idx],
        }
        if self.u_t is not None:
            item['u_t'] = self.u_t[idx]
        return item