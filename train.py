import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from components import construct_encoder, construct_decoder, construct_seasonal_prior
from model import construct_VAE
from utils import train_and_evaluate, save_model, analyze_latent_space

# Configurations
from config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, INPUT_SIZE, LATENT_DIM, DEGREE, DEVICE, MODEL_SAVE_PATH
)

# Load and preprocess data
data = pd.read_csv('data/phoenix_64days.csv', index_col=0, parse_dates=True)

# Fourier basis for seasonal encoding
fourier = lambda x: np.stack(
    [np.sin(2 * np.pi * i * x) for i in range(1, DEGREE + 1)] +
    [np.cos(2 * np.pi * i * x) for i in range(1, DEGREE + 1)],
    axis=-1
)

# Generate seasonal input (daily basis for each sample)
starting_day = np.array(data.index.dayofyear)[:, np.newaxis] - 1
data_days = (starting_day + np.arange(0, INPUT_SIZE // 24, LATENT_DIM // 24)) % 365
seasonal_data = fourier(data_days / 365)

# Split data into train/test
n_train = int(len(data) * 0.8)
train = data[:n_train]
test = data[n_train:]
train_seasonal = seasonal_data[:n_train]
test_seasonal = seasonal_data[n_train:]

# Convert to tensors
train_tensor = torch.tensor(train.values, dtype=torch.float32)
test_tensor = torch.tensor(test.values, dtype=torch.float32)
train_seasonal_tensor = torch.tensor(train_seasonal, dtype=torch.float32)
test_seasonal_tensor = torch.tensor(test_seasonal, dtype=torch.float32)

# Create TensorDatasets and Dataloaders
train_dataset = TensorDataset(train_tensor, train_seasonal_tensor)
test_dataset = TensorDataset(test_tensor, test_seasonal_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Construct model components
encoder = construct_encoder()
decoder = construct_decoder()
seasonal_prior = construct_seasonal_prior()

vae_model = construct_VAE()
vae_model.to(DEVICE)

# Optimizer
optimizer = Adam(vae_model.parameters(), lr=LEARNING_RATE)

# Train and evaluate the model
train_and_evaluate(
    model=vae_model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    epochs=EPOCHS,
    input_size=INPUT_SIZE,
    output_dir="output"
)

# Save model
save_model(vae_model, filename='vae_model_final.pth', output_dir='models')

# Analyze latent space after training
vae_model.eval()
z_means = []
z_log_vars = []

with torch.no_grad():
    for batch in test_loader:
        values, seasonal = batch
        values = values.to(DEVICE)
        seasonal = seasonal.to(DEVICE)
        z_mean, z_log_var = vae_model.encode(values, seasonal)
        z_mean = z_mean.view(z_mean.size(0), -1)
        z_log_var = z_log_var.view(z_log_var.size(0), -1)

        z_means.append(z_mean)
        z_log_vars.append(z_log_var)



z_mean_tensor = torch.cat(z_means, dim=0)
z_log_var_tensor = torch.cat(z_log_vars, dim=0)

analyze_latent_space(z_mean_tensor, z_log_var_tensor, output_dir="output")

print("Training and analysis complete!")
