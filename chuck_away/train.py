
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import VAE
from config import INPUT_SIZE, LATENT_DIM, BATCH_SIZE, EPOCHS, SEED

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Create output directory
os.makedirs("results", exist_ok=True)

# --- Load and preprocess data ---
df = pd.read_csv("data/reshaped_dataset.csv", index_col=0)
x = torch.tensor(df.values, dtype=torch.float32)
x = (x - x.mean()) / x.std()  # Standardize

# --- Train/Val Split ---
dataset = TensorDataset(x)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- Initialize Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = Adam(vae.parameters(), lr=1e-3)

# --- Training Loop ---
train_log = []
for epoch in range(1, EPOCHS + 1):
    vae.train()
    total_loss = total_recon = total_kl = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        x_batch = batch[0].to(device)
        x_hat, z_mean, z_log_var, _ = vae(x_batch)

        recon_loss = torch.nn.functional.mse_loss(x_hat, x_batch, reduction='mean')
        kl_loss = vae.compute_kl(z_mean, z_log_var)
        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        total_recon += recon_loss.item() * x_batch.size(0)
        total_kl += kl_loss.item() * x_batch.size(0)

    train_log.append([epoch, total_loss / train_len, total_recon / train_len, total_kl / train_len])
    print(f"Epoch {epoch}: Loss={train_log[-1][1]:.4f}, Recon={train_log[-1][2]:.4f}, KL={train_log[-1][3]:.4f}")

# --- Save Model and Logs ---
torch.save(vae.state_dict(), "results/vae_final.pth")
np.savetxt("results/train_log.csv", train_log, delimiter=",", header="epoch,loss,recon,kl", comments="")

# --- Latent Representation and Visualization ---
vae.eval()
with torch.no_grad():
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    latents = []
    for batch in full_loader:
        x_batch = batch[0].to(device)
        z = vae.encode_latent(x_batch)
        latents.append(z.cpu().numpy())
    z_all = np.concatenate(latents, axis=0)
    np.save("results/latent_vectors.npy", z_all)

# Visualize (first 2 dims if possible)
if LATENT_DIM >= 2:
    plt.figure(figsize=(6, 5))
    plt.scatter(z_all[:, 0, 0], z_all[:, 1, 0], alpha=0.5)
    plt.title("Latent Space Projection (dim 0 vs 1)")
    plt.xlabel("z[:,0,0]")
    plt.ylabel("z[:,1,0]")
    plt.savefig("results/latent_plot.png")
    plt.close()
