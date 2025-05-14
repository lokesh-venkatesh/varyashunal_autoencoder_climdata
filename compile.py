"""
CODE WILL COMPILE THE .nc FILES IN THE SUBDIRECTORIES OF THE 'data' DIRECTORY AND THEN SAVE THEM AS .csv FILES,
AS WELL AS COMPILE ALL OF THEM AND SAVE THEM INTO A SINGLE .csv FILE CALLED raw_data.csv

THIS CODE WILL ALSO PROCESS THE DATA AND PREPARE IT FOR THE PIPELINE BY:
1. ADJUSTING THE DATA FOR CLIMATE CHANGE (SLOPE OF THE LINE IS cc_slope SAVED IN processing_params_deets.csv)
2. NORMALISING THE DATASET (MEAN AND VARIANCE ARE SAVED AS norm_offset AND norm_scale IN processing_params_deets.csv)
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

base_dir = "data/raw_data_files"
output_dir = "data"
combined_csv_path = os.path.join(output_dir, "raw_data.csv")
final_csv_path = os.path.join(output_dir, "final_dataset.csv")

os.makedirs(output_dir, exist_ok=True)
os.makedirs('results', exist_ok=True)

all_dataframes = [] # This list will contain each dataframe corresponding to an .nc file for a triplet of years

def save_nc_as_csv(nc_file_path, csv_file_path):
    """WILL SAVE A NEW .csv FILE AFTER PARSING THROUGH THE .nc FILE, 
    THIS IS DONE REGARDLESS OF WHETHER IT IS PRESENT OR NOT, IF IT IS THEN IT IS SIMPLY OVERWRITTEN."""
    try:
        ds = xr.open_dataset(nc_file_path) # Uses xarray to open the .nc file
        df = ds.to_dataframe().reset_index() # Opens the xarray file as a dataframe and resets the default indices

        df.to_csv(csv_file_path, index=False) # Save this opened dataframe as a .csv file in the same location
        print(f"Saved CSV: {csv_file_path}")

        all_dataframes.append(df) # This dataframe for a triplet of years is added to the list 
        # this is for later compiling all data from all years together
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")

def adjust_timeseries_for_climchange(dataframe):
    """THIS FUNCTION WILL TAKE IN ONE TIME SERIES DATAFRAME AND ADJUST THE DATA FOR ANY LONG-TERM INCREASING TRENDS
    (LIKE GRADUALLY INCREASING ANNUAL MEAN TEMPERATURES ON ACCOUNT OF GLOBAL WARMING)"""
    dataframe['dummy_col'] = dataframe['time'] # Copy of the 'time' column created in the dataframe for use
    dataframe['time'] = pd.to_datetime(dataframe['time']) # Ensure that the 'time' column is actually in datetime format
    dataframe['temperature_time_hours'] = (dataframe['time'] - dataframe['time'].iloc[0]).dt.total_seconds() / 3600 
    # The column 'temperature_time_hours' will store time as 'number of hours' from the latest time stamp
    
    # We perform a Linear regression using scipy/stats/linregress and get a slope in units of temperature/hours
    slope, intercept, r_value, p_value, std_err = linregress(dataframe['temperature_time_hours'], dataframe['temperature'])
    print(f"The rate of temperature increase per hour is {slope} degrees/hour.")
    
    most_recent_time = dataframe['time'].max() # This basically returns the latest time 
    # (it is redundant actually because the code for this is already in line 47, but I don't want to change anything right now)
    hours_from_recent = (dataframe['time'] - most_recent_time).dt.total_seconds() / 3600 # This returns a column of the hours from the latest time
    
    dataframe['temperature'] = dataframe['temperature'] - slope*hours_from_recent 
    # This line does the actual temperature adjustment, subtracting slope*house_from_most_recent_time from each time stamp
    dataframe['time'] = dataframe['dummy_col'] # This restores the original 'time' column to the dataframe
    dataframe = dataframe.drop(columns=['dummy_col', 'temperature_time_hours']) # This deletes the columns created in processing while running this function
    return dataframe, slope # We return the slope so that we can save it to the preprocessing_params_deets.csv file for later reference

def normalise_timeseries(dataframe):
    """THIS FUNCTION WILL NORMALISE THE ENTIRE LIST OF DATA INTO A GAUSSIAN WITH MEAN=0 AND VARIANCE=1,
    THE ORIGINAL MEAN AND VARIANCE OF THE DATA WILL SET AS norm_offset AND norm_scale IN THE FILE preprocessing_params_deets.csv"""
    offset = dataframe['temperature'].mean() # mean of the distribution
    scale = dataframe['temperature'].std() # standard deviation of the distribution
    dataframe['temperature'] = (dataframe['temperature']-offset)/scale # normalisation of the data based on these parameters
    return dataframe, offset, scale # We return the offset and scale too for saving to preprocessing_params_deets.csv file for later reference

def main():
    """THE BELOW BLOCK OF CODE WILL ITERATE THROUGH THE raw_data_files DIRECTORY AND SAVE ALL OF THE .nc FILES AS .csv FILES IN THE SAME LOCATIONS."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nc"):
                nc_file_path = os.path.join(root, file)
                csv_file_path = os.path.splitext(nc_file_path)[0] + ".csv"
                save_nc_as_csv(nc_file_path=nc_file_path, csv_file_path=csv_file_path)
    
    """THE BELOW CODE BLOCK COMPILES ALL OF THE DATAFRAMES STORED IN MEMORY INTO ONE MASTER .csv FILE AND SAVES IT.
    IT DELETES THE COLUMNS 'number' AND 'expver' AND ALSO RENAMES THE COLUMNS 'valid_time' TO 'time' AND 't2m' TO 'temperature' RESPECTIVELY
    THE VALUES IN THE .csv FILE ARE THEN ALSO SORTED BY THE 'time' COLUMN"""

    if all_dataframes:
        combined_raw_df = pd.concat(all_dataframes)

        if 'valid_time' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.rename(columns={'valid_time': 'time'})
            combined_raw_df = combined_raw_df.sort_values(by='time') # renames the 'valid_time' column to 'time' and sorts it by increasing time
        if 't2m' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.rename(columns={'t2m': 'temperature'}) # renames the 't2m' column to 'temperature
        if 'number' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.drop('number', axis=1) # drops the 'number' column
        if 'expver' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.drop('expver', axis=1) # drops the 'expver' column

        """SINCE THE DATA IN THE RAW .nc FILES IS WRITTEN IN GREENWICH MERIDIAN TIME, IT CONVERTS THE DATASETS TO GMT+7:00 HOURS FOR CONSISTENCY,
        AND ALSO CONVERTS THE TEMPERATURE FROM KELVIN INTO CELSIUS"""
        combined_raw_df['time'] = pd.to_datetime(combined_raw_df['time']) - pd.Timedelta(hours=7) # Conversion from GMT to Arizona Local Time
        combined_raw_df['temperature'] = combined_raw_df['temperature'] - 273.15 # Conversion from Kelvin to Celsius

        """THIS LINE AVERAGES THE TEMPERATURE VALUES SPATIALLY FOR THE SAME TIME STAMP"""
        raw_df = combined_raw_df.groupby("time")["temperature"].mean().reset_index() # Groups datapoints according to time, and averages values within the same group
        raw_df.to_csv(combined_csv_path, index=False) # Saves this file to data/raw_data.csv
        print(f"Combined CSV saved at: {combined_csv_path}")

        raw_df_desc = raw_df.drop(columns=['time']).describe() # Saves a description of this dataset to data/raw_data_stats.csv
        raw_df_desc.to_csv("data/raw_data_stats.csv")
    else:
        print("No .nc files were processed.")

    raw_df = pd.read_csv(combined_csv_path, index_col=None)
    raw_df['time'] = pd.to_datetime(raw_df["time"]) 
    # NOTE that every time you import the .csv file with datetime values in it, 
    # you will need to explicitly convert the time column from string values into DateTime values!
    
    cc_adjstd_df, cc_slope = adjust_timeseries_for_climchange(raw_df) # adjusts the time series for climate change
    cc_adjstd_nrmlsd_df, norm_offset, norm_scale = normalise_timeseries(cc_adjstd_df) # normalises the time series data

    processing_params = {'cc_slope':cc_slope,'norm_offset': norm_offset,'norm_scale':norm_scale} # this is the dataframe with the processing parameters used in this script
    processing_params_df = pd.DataFrame.from_dict(processing_params, orient='index', columns=['processing_parameter'])
    processing_params_df.to_csv("data/processing_params_deets.csv") # saved for later reference and usage
    
    cc_adjstd_nrmlsd_df.to_csv('data/final_timeseries.csv', index=False) # This is the final time series after climate change adjusting and normalisation 
    cc_adjstd_nrmlsd_df_desc = cc_adjstd_nrmlsd_df.drop(columns=['time']).describe() # This describes statistics for the dataframe
    cc_adjstd_nrmlsd_df_desc.to_csv('data/final_timeseries_stats.csv')

if __name__ == "__main__":
    main()