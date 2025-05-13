# generate.py

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from model import VAE, Encoder, Decoder, SeasonalPrior, fourier  # make sure these are properly defined
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Generate temperature time series.")
parser.add_argument('--start', type=str, default='1974-01-01', help='Start date (YYYY-MM-DD)')
parser.add_argument('--end', type=str, default='2023-12-31 23:00:00', help='End date (YYYY-MM-DD HH:MM:SS)')
parser.add_argument('--output', type=str, default='results/generated.csv', help='Output CSV path')
args = parser.parse_args()

# Generate datetime range
start_date = args.start
end_date = args.end
dt = pd.date_range(start=start_date, end=end_date, freq='h')

# Seasonal inputs: (1, T, 6)
day_of_year = (dt.dayofyear.values[:, np.newaxis] - 1) / 365
gen_seasonal_inputs = fourier(torch.tensor(day_of_year.T[0][np.newaxis, :], dtype=torch.float32))[np.newaxis]

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#vae = torch.load("results/vae_final.pth", map_location=device)  # adjust if saved differently
vae = VAE()
vae.load_state_dict(torch.load("results/vae_final.pth"))
vae.eval()

# Generate latent z
with torch.no_grad():
    _, _, z_gen = SeasonalPrior(torch.tensor(gen_seasonal_inputs, dtype=torch.float32).to(device))
    gen_mean = vae.decoder(z_gen).cpu().numpy()

    # Add noise if desired (currently deterministic)
    # noise = np.random.normal(size=gen_mean.shape) * np.exp(0.5 * vae.noise_log_var[0].cpu().numpy())
    # gen = gen_mean + noise
    gen = gen_mean

# Scale back to original (Celsius) using fixed stats
# (based on David Kyle's scaling): x = x_norm * 18.29 + 75.08
gen_scaled = gen[0, :len(dt)] * 18.29 + 75.08
gen_series = pd.Series(gen_scaled.astype(np.float32), index=dt)

# Save to CSV
os.makedirs(os.path.dirname(args.output), exist_ok=True)
gen_series.to_csv(args.output)

print(f"Saved generated time series to {args.output}")
