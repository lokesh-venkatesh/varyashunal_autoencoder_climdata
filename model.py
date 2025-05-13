import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, input_dim=1536, latent_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=10, output_dim=1536):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class SeasonalPrior(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.freqs = torch.arange(1, 4).float().view(1, -1)  # 1st to 3rd harmonic

    def forward(self, times):
        # times: (batch,) in days of year scaled to [0, 1]
        times = times.view(-1, 1)  # shape: (batch, 1)
        phases = 2 * math.pi * self.freqs * times  # (batch, freqs)
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)  # (batch, 6)
        seasonal = F.linear(features, torch.randn(self.latent_dim, features.shape[1]))
        return seasonal


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=1536, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)
        self.seasonal_prior = SeasonalPrior(latent_dim=latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def save_model(self, path="results/model_weights.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="results/model_weights.pth"):
        self.load_state_dict(torch.load(path))
        self.eval()
