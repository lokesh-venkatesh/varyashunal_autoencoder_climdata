import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class TimeSeriesDataset(Dataset):
    """THIS CLASS IS DEFINED FOR THE TIME SERIES DATASET
    
    IT READS THE CSV FILE AND CREATES A DATASET FOR TIME SERIES, 
    AND THEN RETURNS THE DATA AND TIME INDEX FOR EACH SAMPLE IN THE VECTOR."""
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.data = self.df.values.astype(np.float32)
        self.time_index = pd.to_datetime(self.df.index)  # convert to datetime

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])  # the shape of the input tensor is (1536,)
        day_of_year = self.time_index[idx].dayofyear / 365.0 # we extract the day of the year and divide it by 365 to get a float between 0 and 1
        time_tensor = torch.tensor(day_of_year, dtype=torch.float32) # this converts the day of the year to a tensor of float values
        return x, time_tensor # return the input tensor and the time tensor
    # In essence, the time tensor is basically the DateTime values corresponding to each temperature value in the tensor
    # And we simply take each of those DateTime values, convert them into a correspnding day_of_year float value between 0 and 1, 
    # and then return the corresponding time_tensor of float values between 0 and 1

def get_dataloaders(csv_path, batch_size=32, split_ratio=0.8):
    """THIS FUNCTION CREATES THE TRAINING AND VALIDATION DATA LOADERS
    THE DATALOADER FUNCTION ALSO ACCOUNTS FOR THE TIME TENSOR THAT NEEDS TO BE PLUGGED INTO THE SEASONAL PRIOR IN THE MODEL"""
    dataset = TimeSeriesDataset(csv_path)  # This is working fine only, it will basically return one unit containing a bunch of pairs of (x, time_tensor)
    train_len, val_len = int(len(dataset)*split_ratio), len(dataset)-int(len(dataset)*split_ratio)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader, val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader