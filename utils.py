import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import math
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from config import *

def kl_divergence_sum(mu1=0.0, log_var1=0.0, mu2=0.0, log_var2=0.0):
    var1 = torch.exp(log_var1)
    var2 = torch.exp(log_var2)
    axis0 = 0.5 * torch.mean(log_var2 - log_var1 + (var1 + (mu1 - mu2) ** 2) / var2 - 1, dim=0)
    return torch.sum(axis0)

def log_lik_normal_sum(x, mu=0.0, log_var=0.0):
    # Shape sanity check
    if x.shape != mu.shape:
        raise ValueError(f"Shape mismatch in log_lik_normal_sum: x {x.shape}, mu {mu.shape}")

    axis0 = -0.5 * (math.log(2 * np.pi) + torch.mean(log_var + ((x - mu) ** 2) * torch.exp(-log_var), dim=0))
    return torch.sum(axis0)

# Logging training and validation losses
def plot_loss_curves(train_losses, test_losses, output_dir="output"):
    """
    Plot the training and testing loss curves over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    
    # Create directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

# Evaluate model metrics
def evaluate_vae(reconstructed, original, z_mean, z_log_var, input_size):
    """
    Compute evaluation metrics for VAE - MSE for reconstruction and KL divergence.
    """
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum') / input_size
    kl_loss = 0.5 * torch.sum(z_log_var - torch.square(z_mean) - torch.exp(z_log_var) + 1) / input_size
    return recon_loss.item(), kl_loss.item()

# Compute final accuracy or evaluate VAE metrics
def final_metrics(model, test_loader, input_size):
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            values, seasonal = batch
            values = values.to(model.device)
            seasonal = seasonal.to(model.device)
            
            reconstructed, recon_loss, kl_loss = model(values, seasonal)
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss

    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_kl_loss = total_kl_loss / len(test_loader)
    
    print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"Average KL Loss: {avg_kl_loss:.4f}")
    
    return avg_recon_loss, avg_kl_loss

# Saving the trained model
def save_model(model, filename='vae_model.pth', output_dir='models'):
    """
    Save the trained VAE model to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to location {model_path}")
    
# Boxplot analysis of latent space
def analyze_latent_space(z_mean, z_log_var, output_dir="output"):
    """
    Plot boxplots for latent space means and variances.
    """
    z_mean_np = z_mean.cpu().numpy()
    z_log_var_np = z_log_var.cpu().numpy()

    plt.figure(figsize=(12, 6))
    
    # Plot boxplot for latent means
    plt.subplot(1, 2, 1)
    sns.boxplot(data=z_mean_np)
    plt.title('Latent Space Means')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')

    # Plot boxplot for latent log variances
    plt.subplot(1, 2, 2)
    sns.boxplot(data=z_log_var_np)
    plt.title('Latent Space Log Variances')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Log Variance')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'latent_space_analysis.png'))
    plt.close()

# Training loop for one epoch
def train_epoch(model, train_loader, optimizer, input_size):
    model.train()
    device = next(model.parameters()).device  # safely get model's device
    total_loss = 0

    for batch in train_loader:
        values, seasonal = batch
        values = values.to(device)
        seasonal = seasonal.to(device)

        # Print the shapes of the input and reconstructed data
        # print("Input shape:", values.shape)
        
        # Forward pass
        _, recon_loss, kl_loss = model(values, seasonal)
        
        # Print the shape of the reconstructed output
        # print("Reconstructed shape:", recon_loss.shape)  # or use the correct variable for reconstructed output
        
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)



# Testing loop for one epoch
def evaluate_epoch(model, test_loader, input_size):
    model.eval()
    device = next(model.parameters()).device  # safely get model's device
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            values, seasonal = batch
            values = values.to(device)
            seasonal = seasonal.to(device)
            
            _, recon_loss, kl_loss = model(values, seasonal)
            loss = recon_loss + kl_loss
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Function to run the full training and evaluation loop
def train_and_evaluate(model, train_loader, test_loader, optimizer, epochs, input_size, output_dir="output"):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, input_size)
        train_losses.append(train_loss)
        
        # Evaluate
        test_loss = evaluate_epoch(model, test_loader, input_size)
        test_losses.append(test_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    # Plot Loss Curves
    plot_loss_curves(train_losses, test_losses, output_dir)
    
    # Final metrics
    final_metrics(model, test_loader, input_size)
    
    # Save Model
    save_model(model, filename='vae_model_final.pth', output_dir=output_dir)

import torch

# Function to load model weights
def load_model(model, model_path):
    """
    Load the trained model weights from a specified file.

    Parameters:
    - model: The model instance (VAE)
    - model_path: Path to the saved model weights file
    """
    model.load_state_dict(torch.load(model_path))
    print(f"Model weights loaded from {model_path}")