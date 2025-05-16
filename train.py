import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import VariationalAutoencoder
from utils import get_dataloaders, set_seed

set_seed(42)

def elbo_loss(recon_x, x, mu, logvar):
    """CALCULATE THE EVIDENCE LOWER BOUND (ELBO) LOSS FUNCTION USING THE RECONSTRUCTION LOSS AND KL DIVERGENCE"""
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon_loss + kl_div, recon_loss.item(), kl_div.item()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 1536
    latent_dim = 10
    batch_size = 32
    epochs = 100

    # Prepare data
    train_loader, val_loader = get_dataloaders("data/reshaped_dataset.csv", batch_size=batch_size)

    # Initialize model
    model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, time_tensor in loop:
            x = x.to(device) # This is the input vector of the 1536-timepoints length time series
            time_tensor = time_tensor.to(device) # This is the time tensor of the same length as x

            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x, time_tensor)  # The model returns the reconstructed output, mean, log variance, and the latent representation

            elbo, recon_loss, kl_loss = elbo_loss(recon, x, mu, logvar) # returns the ELBO loss, reconstruction loss, and KL divergence

            elbo.backward() # Backpropagation
            optimizer.step() # Update the model parameters

            train_loss += elbo.item() # Accumulate the total loss
            recon_loss_total += recon_loss # Accumulate the reconstruction loss
            kl_loss_total += kl_loss # Accumulate the KL divergence

            loop.set_postfix({"Loss": elbo.item(), "Recon": recon_loss, "KL": kl_loss}) 

        print(f"Epoch {epoch+1}: Total Loss = {train_loss:.8f}, Recon = {recon_loss_total:.8f}, KL = {kl_loss_total:.8f}")

    os.makedirs("results", exist_ok=True)
    model.save_model("results/model_weights.pth")
    print("Model saved to results/model_weights.pth")

if __name__ == "__main__":
    train()
