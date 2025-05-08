import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import load_model
from model_components import construct_decoder, construct_encoder, construct_seasonal_prior  # Assuming separate functions for encoder/decoder
from model import VAE  # Make sure this imports your VAE class

def generate_time_series(start_time, stop_time, model, latent_dim=64, time_step=1, batch_size=512):
    # Create a time range from start to stop time
    time_range = pd.date_range(start=start_time, end=stop_time, freq=f'{time_step}H')
    
    # Generate the data in batches to avoid any memory issues
    generated_t2m = []
    
    # Process the data in smaller batches to avoid large memory allocation
    for i in range(0, len(time_range), batch_size):
        batch_time_range = time_range[i:i + batch_size]
        batch_size_actual = len(batch_time_range)
        
        # Sample latent vector z from a normal distribution (mean=0, std=1) matching the latent dimension
        z = torch.randn(batch_size_actual, latent_dim).to(model.device)  # Adjust the size for batch
        seasonal_input = torch.zeros(batch_size_actual, latent_dim).to(model.device)  # Adjust as needed
        
        # Ensure the input is of shape [batch_size, channels, sequence_length]
        z = z.unsqueeze(1).repeat(1, 32, 1)  # Repeat the latent vector to have 32 channels
        seasonal_input = seasonal_input.unsqueeze(1).repeat(1, 32, 1)  # Repeat seasonal input to have 32 channels
        
        # Pass the latent code through the decoder to generate the output
        try:
            generated_data = model.decoder(z, seasonal_input)
            
            # Assuming the output is the generated time series in some form (e.g., t2m values)
            generated_t2m_batch = generated_data.cpu().detach().numpy().flatten()
            generated_t2m.extend(generated_t2m_batch)
        except Exception as e:
            print(f"Error during generation: {e}")
            break  # Optionally continue with next batch if there's an error

    # Ensure the lengths match between time range and generated data
    if len(generated_t2m) > len(time_range):
        generated_t2m = generated_t2m[:len(time_range)]  # Truncate if needed
    elif len(generated_t2m) < len(time_range):
        time_range = time_range[:len(generated_t2m)]  # Adjust time range to match data length

    # Create a DataFrame with the generated data
    df = pd.DataFrame({
        'time': time_range,
        'generated_t2m': generated_t2m
    })
    
    # Save the generated time series to a CSV file
    df.to_csv('data/generated_timeseries.csv', index=False)

    print(f"Generated time series saved to 'data/generated_timeseries.csv'")
    return df

# Load encoder, decoder, and prior from components or architecture
encoder = construct_encoder()  # Adjust this as per your encoder architecture
decoder = construct_decoder()  # Adjust this based on your decoder
prior = construct_seasonal_prior()  # Define if you have a prior
input_size = (64, 24)  # Example input size, adjust as needed

# Now initialize the VAE model
model = VAE(encoder=encoder, decoder=decoder, prior=prior, input_size=input_size)
load_model(model, 'models/vae_model_final.pth')

# Specify start and stop time for generating data
start_time = '1969-12-31 17:00:00'
stop_time = '2020-12-31 16:00:00'

# Generate the time series
generated_df = generate_time_series(start_time, stop_time, model)
