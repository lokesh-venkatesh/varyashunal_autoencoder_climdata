"""
THIS SCRIPT IMPORTS THE TIME SERIES STORED IN data/final_timeseries.csv AND THEN FLATTENS THIS TIME SERIES INTO CHUNKS OF 64*24 TIME POINTS EACH.
THIS WILL BE THEN FED INTO THE MODEL FOR TRAINING, VALIDATION AND DATA GENERATION. THESE CHUNKS ARE STORED IN data/reshaped_dataset.csv
"""

DAYS, HOURS = 64, 24
INPUT_SIZE = DAYS*HOURS # 64 days of hourly temps

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *

time_series_df = pd.read_csv('data/final_timeseries.csv', index_col=None) 
time_series_df['time'] = pd.to_datetime(time_series_df["time"]) 

def flatten_time_series_into_chunks_of_data(dft):
    """THIS FUNCTION TAKES THE ENTIRE TIME SERIES, DIVIDES IT INTO 'n' SEQUENCES OF 'k'-LONG VALUES IN EACH"""
    k = INPUT_SIZE # this is the length of one vector that will be fed into the model for training, 
    # which is 64 days of hourly temperature dat, that is, 1536 values in one time-series vector
    n = dft.shape[0] // k  # number of full k-length sequences that can be gotten from the time-series (floor division)
    
    dft_reshaped = pd.DataFrame( # Defines the dataframe characteristics based on the time-series dataframe
        dft['temperature'].values[:n*k].reshape(n, k), # This takes the first n*k temperature values,
        # and reshapes them into a dataframe of k columns and n rows
        index=dft['time'].iloc[::k][:n].values, # This simply takes the time-stamp for range(0, n*k, n)
        columns=range(k) # This will define a list of columns corresponding to each of the 1536 entries in a time-series vector
    )
    return dft_reshaped

if __name__ == "__main__":
    np.random.seed(42) # Defines a random seed for consistency of dataset formed
    dft_reshaped = flatten_time_series_into_chunks_of_data(time_series_df).sample(frac=1) # This is the reshaped dataset for processing
    dft_reshaped.to_csv(f'data/reshaped_dataset.csv') # Saved to data/reshaped_dataset.csv
    print(dft_reshaped.head())