# koopmandataset.py

import torch
import numpy as np
from torch.utils.data import Dataset


class KoopmanDataset(Dataset):
    """
    Constructs sequential (x_n, x_{n+1}, u_n) pairs at the WINDOW level.

    Args:
        trajectories : np.ndarray of shape (N, 314, 200)
        controls     : np.ndarray of shape (N, 200, control_dim) or (N, T-1, control_dim)
    """
    def __init__(self, trajectories, controls=None):
        N, C, T = trajectories.shape

        # --- Pair whole windows ---
        self.x_t    = torch.tensor(trajectories[:-1], dtype=torch.float32)   # (N-1, 314, 200)
        self.x_next = torch.tensor(trajectories[1:],  dtype=torch.float32)

        # --- Controls ---
        if controls is not None:
            # ensure controls align with trajectories
            controls = controls[:N]

            # If controls are (N, 200, d), aggregate over time
            if controls.ndim == 3:
                # average over time → one control per window
                u = controls.mean(axis=1)   # (N, d)
            else:
                u = controls

            # match pairing (N-1)
            u = u[:-1]

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
    
class CycleDataset(Dataset):
    """
    Window-level dataset: (X_n, X_{n+1}, u_n)

    X: (N, 314, 200)
    u: (N, 200, control_dim)
    """
    def __init__(self, data, controls=None):
        N, C, T = data.shape

        self.x = torch.tensor(data, dtype=torch.float32)

        # pair windows
        self.x_t    = self.x[:-1]   # (N-1, 314, 200)
        self.x_next = self.x[1:]

        if controls is not None:
            # aggregate control over window
            u = controls.mean(axis=1)   # (N, control_dim)
            self.u_t = torch.tensor(u[:-1], dtype=torch.float32)
        else:
            self.u_t = None

    def __len__(self):
        return len(self.x_t)

    def __getitem__(self, idx):
        item = {
            'x_t': self.x_t[idx],
            'x_next': self.x_next[idx]
        }
        if self.u_t is not None:
            item['u_t'] = self.u_t[idx]
        return item