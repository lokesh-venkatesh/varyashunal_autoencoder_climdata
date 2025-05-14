import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from model import VariationalAutoencoder


def generate_time_series(start_time_str='1969-12-31 17:00:00', # Start time of the time series
                         end_time_str='2020-12-31 17:00:00', # End time of the time series
                         model_path="results/model_weights.pth",
                         output_csv="results/gnrtd_timeseries.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    input_dim = 1536
    latent_dim = 10

    # Load model
    model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_model(model_path)
    model.eval() # Set the model to evaluation mode

    # Parse date range
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S") 
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
    delta_hours = int((end_time - start_time).total_seconds() // 3600) # Calculate the number of hours between start and end time

    # Number of blocks to generate
    block_size = input_dim 
    n_blocks = (delta_hours + block_size - 1) // block_size

    all_generated = []
    for i in range(n_blocks):
        with torch.no_grad():
            block_start_time = start_time + timedelta(hours=i * block_size)
            block_times = [block_start_time + timedelta(hours=j) for j in range(block_size)]

            # Compute day-of-year for seasonal prior
            day_of_year = [t.timetuple().tm_yday for t in block_times]
            time_tensor = torch.tensor(day_of_year, dtype=torch.float32, device=device)

            # Get learned seasonal prior
            seasonal_prior = model.seasonal_prior(time_tensor)  # (1536, latent_dim)
            seasonal_prior = seasonal_prior.transpose(0, 1).unsqueeze(0)  # (1, latent_dim, 1536)

            # Add noise to the seasonal prior
            z = torch.randn_like(seasonal_prior) + seasonal_prior

            # Decode
            x_recon = model.decoder(z).cpu().numpy().flatten()
            all_generated.extend(x_recon)

    # Clip to exact time range
    all_generated = all_generated[:delta_hours]
    generated_times = [start_time + timedelta(hours=i) for i in range(delta_hours)]

    # Save to CSV
    df = pd.DataFrame({"time": generated_times, "temperature": all_generated})
    df.to_csv(output_csv, index=False)
    print(f"✅ Generated data saved to {output_csv}")

    df_stats = df.drop(columns=['time']).describe()
    df_stats_filepath = "results/gnrtd_timeseries_stats.csv"
    df_stats.to_csv(df_stats_filepath)
    print(f"✅ Generated data saved to {df_stats_filepath}")


if __name__ == "__main__":
    generate_time_series()
