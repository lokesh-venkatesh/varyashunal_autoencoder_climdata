import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Define the base directory
base_dir = "data/raw_data"
output_dir = "data"
combined_csv_path = os.path.join(output_dir, "raw_data.csv")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store all dataframes
all_dataframes = []

def save_nc_as_csv(nc_file_path, csv_file_path):
    # Open the .nc file and convert it to a dataframe
    try:
        ds = xr.open_dataset(nc_file_path)
        df = ds.to_dataframe().reset_index()

        # Save the dataframe as a CSV file in the same location
        df.to_csv(csv_file_path, index=False)
        print(f"Saved CSV: {csv_file_path}")

        # Append the dataframe to the list for combining later
        all_dataframes.append(df)
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")

def adjust_timeseries_for_climchange(dataframe=pd.read_csv('data/raw_data.csv')):
    dataframe['dummy_col'] = dataframe['valid_time']
    dataframe['valid_time'] = (pd.to_datetime(dataframe['valid_time'])-pd.to_datetime(dataframe['valid_time'].iloc[0])).dt.total_seconds()/3600
    slope, intercept, r_value, p_value, std_err = linregress(dataframe['valid_time'], dataframe['t2m']) # Perform linear regression

    print(f"The rate of temperature increase per hour is {slope} degrees/hour.")

    most_recent_time = dataframe['valid_time'].max()
    dataframe['t2m_adj'] = dataframe['t2m'] - slope * (dataframe['valid_time'] - most_recent_time)
    dataframe['valid_time'] = dataframe['dummy_col']
    dataframe = dataframe.drop(columns=['dummy_col'])
    return dataframe

def normalise_timeseries(dataframe):
    offset = dataframe['t2m_adj'].mean()
    scale = dataframe['t2m_adj'].std()
    dataframe['t2m_norm_adj'] = (dataframe['t2m_adj'] - offset) / scale
    return dataframe
    

def main():
    # Iterate over all subfolders in the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nc"):
                nc_file_path = os.path.join(root, file)
                csv_file_path = os.path.splitext(nc_file_path)[0] + ".csv"
                save_nc_as_csv(nc_file_path=nc_file_path, csv_file_path=csv_file_path)
    
    # Combine all dataframes into one large dataframe
    if all_dataframes:
        combined_df = pd.concat(all_dataframes)

        # Sort the combined dataframe by time (assuming a 'time' column exists)
        if 'valid_time' in combined_df.columns:
            combined_df = combined_df.sort_values(by='valid_time')
        if 'number' in combined_df.columns:
            combined_df = combined_df.drop('number', axis=1)
        if 'expver' in combined_df.columns:
            combined_df = combined_df.drop('expver', axis=1)

        # To account for time zone difference with respect to GMT
        combined_df['valid_time'] = pd.to_datetime(combined_df['valid_time']) - pd.Timedelta(hours=7)

        # Convert Kelvin to Celsius:
        combined_df['t2m'] = combined_df['t2m']-273.15

        # Average temperature spatially
        combined_df = combined_df.groupby("valid_time")["t2m"].mean().reset_index()

        # Save the combined dataframe as a CSV file
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved at: {combined_csv_path}")

        df_desc = combined_df.drop(columns=['valid_time'])
        df_desc = df_desc.describe() # Get descriptive statistics of the temperature data
        df_desc.to_csv("data/raw_data_stats.csv")
    else:
        print("No .nc files were processed.")
    
    df = pd.read_csv('data/raw_data.csv') # Assuming the dataframe has columns 'valid_time' (in hours) and 'Observed'
    df_copy = adjust_timeseries_for_climchange(df)
    #df_copy.to_csv('data/adj_data.csv', index=False)
    #df_copy = df_copy.drop(columns=['valid_time'])
    df_desc = df_copy.describe()
    #df_desc.to_csv('data/adj_data_stats.csv')
    df_norm = normalise_timeseries(df_copy)
    #df_norm.to_csv('data/norm_adj_data.csv', index=False)
    dft_desc = df_norm.describe()
    #dft_desc.to_csv('data/norm_adj_data_stats.csv')
    df_norm = df_norm.drop(columns=['t2m', 't2m_adj'])
    df_norm.to_csv('data/final_timeseries.csv', index=False)
    df_norm_desc = df_norm.describe()
    df_norm_desc.to_csv('data/final_timeseries_stats.csv')

    
    # Preview the DataFrame
    print(df.head())

    # Convert 'valid_time' to datetime if it's not already
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Filter out data from the year 2021
    df = df[(df['valid_time'].dt.year >= 1970) & (df['valid_time'].dt.year <= 2020)]
    #df = df[df['valid_time'].dt.year < 2021]

    # Add a 'year' column for grouping
    df['year'] = df['valid_time'].dt.year

    # Calculate yearly average temperatures
    yearly_avg_raw_temp = df.groupby('year')['t2m'].mean()
    yearly_avg_adj_temp = df.groupby('year')['t2m_adj'].mean()

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

if __name__ == "__main__":
    main()