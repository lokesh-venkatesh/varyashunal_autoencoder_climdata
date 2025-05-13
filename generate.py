import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

from model import VariationalAutoencoder


def generate_time_series(start_time_str='1969-12-31 17:00:00', 
                         end_time_str='2020-12-31 16:00:00', 
                         model_path="results/model_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    input_dim = 1536
    latent_dim = 10
    model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_model(model_path)
    model.eval()

    # Parse dates
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
    delta_hours = int((end_time - start_time).total_seconds() // 3600)

    # Determine how many full 1536-hour blocks we need to generate
    block_size = 1536
    n_blocks = (delta_hours + block_size - 1) // block_size

    # Generate synthetic sequences from prior
    all_generated = []
    timestamps = []
    for i in range(n_blocks):
        with torch.no_grad():
            z = torch.randn(1, latent_dim).to(device)
            dummy_time = start_time + timedelta(hours=i * block_size)
            season_input = model.seasonal_prior(torch.tensor([[dummy_time.timetuple().tm_yday]], dtype=torch.float32).to(device))
            z = z + season_input
            x_recon = model.decoder(z).cpu().numpy().flatten()
            all_generated.extend(x_recon)

    # Slice to match exact duration requested
    all_generated = all_generated[:delta_hours]

    # Build time index and DataFrame
    generated_times = [start_time + timedelta(hours=i) for i in range(delta_hours)]
    df = pd.DataFrame({"time": generated_times, "temperature": all_generated})
    df.to_csv("results/gnrtd_timeseries.csv", index=False)
    print(f"Generated data saved to results/gnrtd_timeseries.csv")


if __name__ == "__main__":
    generate_time_series()
