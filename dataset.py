"""
This code basically takes the dataframe stored in final_timeseries.csv and expands the linear time series into blocks (or chunks) of 64*24 timestamps each.
This will be then fed into the model for training and processing.

The saved chunks are stored in reshaped_dataset.csv
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from utils import *

# Load full time series (hourly, 64 days per sequence)
time_series_df = pd.read_csv('data/final_timeseries.csv', index_col=None)

def flatten_time_series_into_chunks_of_data(dft):
    # 1536 columns = 64 days of hourly data
    k = INPUT_SIZE
    n = dft.shape[0] // k  # number of full k-length sequences
    dft_reshaped = pd.DataFrame(
        dft['temperature'].values[:n * k].reshape(n, k),
        index=dft['time'].iloc[::k][:n].values,
        #columns=range(1, k + 1)
        columns=range(k)
    )
    return dft_reshaped

if __name__ == "__main__":
    np.random.seed(42)
    dft_reshaped = flatten_time_series_into_chunks_of_data(time_series_df).sample(frac=1)
    dft_reshaped.to_csv(f'data/reshaped_dataset.csv')
    print(dft_reshaped.head())