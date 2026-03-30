# my_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        data: numpy array of shape (N, T, feature_dim)
              N = number of trajectories
              T = sequence length
              feature_dim = your input size (e.g. 10, 128, etc.)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # shape: (T, feature_dim)