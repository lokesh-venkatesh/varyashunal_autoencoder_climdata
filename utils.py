import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.data = self.df.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.tensor(x)  # shape: (1536,)


def get_dataloaders(csv_path, batch_size=32, split_ratio=0.8):
    dataset = TimeSeriesDataset(csv_path)
    total = len(dataset)
    train_len = int(total * split_ratio)
    val_len = total - train_len

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
