"""
This is code for compiling all of the raw .nc files into a large .csv file, as well as saving stats for the same.

Then, this file takes the raw dataset from the compiled .csv file, and then:
1. performs a linear regression on it, to account for climate change, 
and then collapses it from a straight line with a slope into a flat line.
2. normalises that collapsed dataset into a normal distribution with mean=0 and standard deviation=1, 
basically z = (x - mu)/sigma
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

all_dataframes = [] #This list will contain all the dataframes for each triple of annums

def save_nc_as_csv(nc_file_path, csv_file_path):
    """
    Checks if the .csv files for each triple of years exists or not, 
    """
    #if not os.path.exists(csv_file_path):
    try:
        ds = xr.open_dataset(nc_file_path)
        df = ds.to_dataframe().reset_index()

        df.to_csv(csv_file_path, index=False) # Save the dataframe as a CSV file in the same location
        print(f"Saved CSV: {csv_file_path}")

        all_dataframes.append(df) # Append the dataframe to the list for combining later
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")
    #else:
    #    print(f"The .csv file {csv_file_path} already exists")

def adjust_timeseries_for_climchange(dataframe):
    dataframe['dummy_col'] = dataframe['time']
    dataframe['time'] = pd.to_datetime(dataframe['time']) # Ensure time is datetime
    dataframe['temperature_time_hours'] = (dataframe['time'] - dataframe['time'].iloc[0]).dt.total_seconds() / 3600 # Convert time to hours since the first timestamp
    # Linear regression: time in hours vs. temperature
    slope, intercept, r_value, p_value, std_err = linregress(dataframe['temperature_time_hours'], dataframe['temperature'])
    print(f"The rate of temperature increase per hour is {slope} degrees/hour.")
    
    most_recent_time = dataframe['time'].max() # Compute time difference (in hours) from most recent time
    hours_from_recent = (dataframe['time'] - most_recent_time).dt.total_seconds() / 3600
    
    dataframe['temperature'] = dataframe['temperature'] - slope * hours_from_recent # Adjust temperature
    dataframe['time'] = dataframe['dummy_col'] # Restore original time column
    dataframe = dataframe.drop(columns=['dummy_col', 'temperature_time_hours'])
    return dataframe, slope

def normalise_timeseries(dataframe):
    offset = dataframe['temperature'].mean()
    scale = dataframe['temperature'].std()
    dataframe['temperature'] = (dataframe['temperature']-offset)/scale
    return dataframe, offset, scale

def main():
    """This block of code iterates through each subfolder of 'data' and saves each .nc file as a .csv file 
    along with the former."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nc"):
                nc_file_path = os.path.join(root, file)
                csv_file_path = os.path.splitext(nc_file_path)[0] + ".csv"
                save_nc_as_csv(nc_file_path=nc_file_path, csv_file_path=csv_file_path)
    
    """This block of code combines all of the .csv files into one huge .csv file
    It also drops the columns 'number' and 'expver' if they are in the dataframe at all.
    Then, it sorts the values in the .csv file by the 'valid_time' column"""

    if all_dataframes:
        combined_raw_df = pd.concat(all_dataframes)

        if 'valid_time' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.rename(columns={'valid_time': 'time'})
            combined_raw_df = combined_raw_df.sort_values(by='time')
        if 't2m' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.rename(columns={'t2m': 'temperature'})
        if 'number' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.drop('number', axis=1)
        if 'expver' in combined_raw_df.columns:
            combined_raw_df = combined_raw_df.drop('expver', axis=1)

        """Then, since the raw .nc files are written with respect to GMT, it converts the datasets to GMT + 7:00 hrs, 
        and also converts the temperature values from Kelvin to Celsius"""
        combined_raw_df['time'] = pd.to_datetime(combined_raw_df['time']) - pd.Timedelta(hours=7)
        combined_raw_df['temperature'] = combined_raw_df['temperature'] - 273.15

        """This line of code spatially averages the temperature values,
        giving exactly one value per time point for the whole dataset"""
        raw_df = combined_raw_df.groupby("time")["temperature"].mean().reset_index()
        raw_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved at: {combined_csv_path}")

        raw_df_desc = raw_df.drop(columns=['time']).describe()
        raw_df_desc.to_csv("data/raw_data_stats.csv")
    else:
        print("No .nc files were processed.")

    raw_df = pd.read_csv(combined_csv_path, index_col=None)
    raw_df['time'] = pd.to_datetime(raw_df["time"])
    
    cc_adjstd_df, cc_slope = adjust_timeseries_for_climchange(raw_df)
    cc_adjstd_nrmlsd_df, norm_offset, norm_scale = normalise_timeseries(cc_adjstd_df)

    processing_params = {'cc_slope':cc_slope,'norm_offset': norm_offset,'norm_scale':norm_scale}
    processing_params_df = pd.DataFrame.from_dict(processing_params, orient='index', columns=['processing_parameter'])
    processing_params_df.to_csv("data/processing_params_deets.csv")
    
    cc_adjstd_nrmlsd_df.to_csv('data/final_timeseries.csv', index=False)
    cc_adjstd_nrmlsd_df_desc = cc_adjstd_nrmlsd_df.drop(columns=['time']).describe()
    cc_adjstd_nrmlsd_df_desc.to_csv('data/final_timeseries_stats.csv')
    
    '''
    cc_adjstd_nrmlsd_df['time'] = pd.to_datetime(cc_adjstd_nrmlsd_df['time']) 
    # Convert 'valid_time' to datetime if it's not already
    
    cc_adjstd_nrmlsd_df = cc_adjstd_nrmlsd_df[(cc_adjstd_nrmlsd_df['time'].dt.year >= 1970) & (cc_adjstd_nrmlsd_df['time'].dt.year <= 2020)] 
    cc_adjstd_df = cc_adjstd_df[(cc_adjstd_df['time'].dt.year >= 1970) & (cc_adjstd_df['time'].dt.year <= 2020)] 
    # Filter out data from after 2020 and before 1970
    cc_adjstd_nrmlsd_df['year'] = cc_adjstd_nrmlsd_df['time'].dt.year # Add a 'year' column for grouping
    
    print(cc_adjstd_nrmlsd_df.head())
    # Calculate the yearly average temperatures
    #raw_temps = pd.read_csv(combined_csv_path, index_col=None)
    #raw_temps['time'] = pd.to_datetime(raw_temps["time"])
    cc_adjstd_df['year'] = cc_adjstd_df['time'].dt.year
    yearly_avg_raw_temp = cc_adjstd_df.groupby('year')['temperature'].mean()
    yearly_avg_adj_temp = cc_adjstd_nrmlsd_df.groupby('year')['temperature'].mean()
    
    # Plot the yearly average temperatures
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_avg_raw_temp.index, yearly_avg_raw_temp.values, label='Raw Temperature', color='blue')
    plt.plot(yearly_avg_adj_temp.index, yearly_avg_adj_temp.values, label='Adjusted Temperature', color='red')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Yearly Average Temperature Comparison')
    plt.xticks(yearly_avg_raw_temp.index, rotation=90)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig('results/plot6_impact_of_climate_change.png', dpi=300)
    '''

if __name__ == "__main__":
    main()