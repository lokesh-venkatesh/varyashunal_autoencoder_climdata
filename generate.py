import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from model import VariationalAutoencoder


def generate_time_series(start_time_str='1969-12-31 17:00:00', 
                         end_time_str='2020-12-31 17:00:00', 
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

    all_generated = []
    for i in range(n_blocks):
        with torch.no_grad():
            block_start_time = start_time + timedelta(hours=i * block_size)
            
            # Create time tensor for all 1536 hours in the block
            block_times = [block_start_time + timedelta(hours=j) for j in range(block_size)]
            day_of_year = [t.timetuple().tm_yday for t in block_times]
            time_tensor = torch.tensor(day_of_year, dtype=torch.float32, device=device).unsqueeze(1)

            # Generate seasonal prior for the whole sequence
            seasonal_prior = model.seasonal_prior(time_tensor)  # (1536, latent_dim)
            seasonal_prior = seasonal_prior.transpose(0, 1).unsqueeze(0)  # (1, latent_dim, 1536)

            # Sample latent vector with noise
            z = torch.randn_like(seasonal_prior) + seasonal_prior  # (1, latent_dim, 1536)

            # Decode
            x_recon = model.decoder(z).cpu().numpy().flatten()
            all_generated.extend(x_recon)

    # Slice to match exact duration
    all_generated = all_generated[:delta_hours]

    # Build time index and DataFrame
    generated_times = [start_time + timedelta(hours=i) for i in range(delta_hours)]
    df = pd.DataFrame({"time": generated_times, "temperature": all_generated})
    df.to_csv("results/gnrtd_timeseries.csv", index=False)
    print("✅ Generated data saved to results/gnrtd_timeseries.csv")


if __name__ == "__main__":
    generate_time_series()
