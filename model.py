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
    def __init__(self, latent_dim=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=5, stride=4, padding=1)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(20, 64, kernel_size=3, stride=2, padding=1)

        self.mu_layer = nn.Conv1d(64, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv1d(64, latent_dim, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 1536)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))  # (batch, 64, ~3)
        mu = self.mu_layer(x)      # (batch, latent_dim, seq_len_reduced)
        logvar = self.logvar_layer(x)  # (batch, latent_dim, seq_len_reduced)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=10, output_length=1536):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose1d(20, 1, kernel_size=5, stride=4, padding=1, output_padding=1)

    def forward(self, z):
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        z = self.deconv5(z)
        return z.squeeze(1)  # (batch, 1536)


class SeasonalPrior(nn.Module):
    def __init__(self, latent_dim=10, num_freqs=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_freqs = num_freqs

        # Frequencies for Fourier features: 1, 2, ..., num_freqs
        self.freqs = nn.Parameter(torch.arange(1, num_freqs + 1).float().view(1, -1), requires_grad=False)

        # Learnable projection from Fourier features to latent_dim
        self.linear = nn.Linear(2 * num_freqs, latent_dim)

    def forward(self, time_tensor):
        # time_tensor: (batch,)
        time_tensor = time_tensor.view(-1, 1)  # (batch, 1)
        phases = 2 * math.pi * self.freqs * time_tensor  # (batch, num_freqs)
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)  # (batch, 2 * num_freqs)
        seasonal = self.linear(features)  # (batch, latent_dim)
        return seasonal


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=1536, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_length=input_dim)
        self.sampling = Sampling()
        self.seasonal_prior = SeasonalPrior(latent_dim=latent_dim)

    def forward(self, x, time_tensor):
        mu, logvar = self.encoder(x)  # (batch, latent_dim, seq_len)
        z = self.sampling(mu, logvar)  # (batch, latent_dim, seq_len)

        # Compute seasonal embedding for each sample and broadcast over sequence
        seasonal = self.seasonal_prior(time_tensor)  # (batch, latent_dim)
        seasonal = seasonal.unsqueeze(-1).expand_as(z)  # (batch, latent_dim, seq_len)
        z = z + seasonal  # Add seasonality to latent representation

        recon = self.decoder(z)  # (batch, 1536)

        # Mean pooling for ELBO loss
        mu_avg = mu.mean(dim=2)
        logvar_avg = logvar.mean(dim=2)

        return recon, mu_avg, logvar_avg, z

    def save_model(self, path="results/model_weights.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="results/model_weights.pth"):
        self.load_state_dict(torch.load(path))
        self.eval()
