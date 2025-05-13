import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_SIZE, LATENT_DIM, INTERIM_FILTERS, LATENT_FILTERS, FOURIER_PERIOD, FOURIER_DEGREE

# --- Fourier Feature Function (Seasonality Injection) ---
def fourier(x: torch.Tensor) -> torch.Tensor:
    """
    Computes Fourier features to capture seasonal patterns (e.g., yearly cycles).

    Args:
        x (Tensor): 1D tensor of shape (batch,) representing time values (e.g., day-of-year).

    Returns:
        Tensor: Shape (batch, 2 * FOURIER_DEGREE) containing sine and cosine terms.
    """
    x = x.view(-1, 1).float()
    freq = torch.arange(1, FOURIER_DEGREE + 1, device=x.device).view(1, -1)
    sin = torch.sin(2 * torch.pi * x * freq / FOURIER_PERIOD)
    cos = torch.cos(2 * torch.pi * x * freq / FOURIER_PERIOD)
    return torch.cat([sin, cos], dim=-1)


# --- Sampling Layer for Reparameterization ---
class Sampling(nn.Module):
    """
    Reparameterization trick to sample from N(z_mean, exp(z_log_var))
    while allowing gradients to flow through the network.
    """
    def forward(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps


# --- Encoder Network ---
class Encoder(nn.Module):
    """
    Encoder network using 1D convolutions to map input sequence
    into a latent mean and log variance.
    """
    def __init__(self, input_size=INPUT_SIZE, interim_filters=INTERIM_FILTERS, latent_filters=LATENT_FILTERS):
        super().__init__()
        self.latent_filters = latent_filters
        self.sampling = Sampling()

        self.encode = nn.Sequential(
            nn.Conv1d(1, interim_filters, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(interim_filters, 2 * latent_filters, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, INPUT_SIZE)

        Returns:
            z_mean (Tensor): Mean of latent distribution
            z_log_var (Tensor): Log variance of latent distribution
            z (Tensor): Sampled latent vector after reparameterization
        """
        x = x.view(x.size(0), 1, -1)
        x = self.encode(x)
        x = x.permute(0, 2, 1)
        z_mean = x[:, :, :self.latent_filters]
        z_log_var = x[:, :, self.latent_filters:]
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


# --- Decoder Network ---
class Decoder(nn.Module):
    """
    Decoder network to reconstruct input from latent vector using ConvTranspose1D layers.
    """
    def __init__(self, latent_filters=LATENT_FILTERS, interim_filters=INTERIM_FILTERS, output_size=INPUT_SIZE):
        super().__init__()
        self.output_size = output_size

        self.decode = nn.Sequential(
            nn.ConvTranspose1d(latent_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, 1, kernel_size=5, stride=3, padding=2, output_padding=2)
        )

    def forward(self, z):
        z = z.permute(0, 2, 1)
        x_hat = self.decode(z)
        x_hat = x_hat.view(x_hat.size(0), -1)
        return x_hat[:, :self.output_size]


# --- Seasonal Prior Generator ---
class SeasonalPrior(nn.Module):
    """
    Module to generate a prior latent representation based on seasonal features.
    Useful for regularizing latent space using known temporal structure.
    """
    def __init__(self, latent_dim=LATENT_DIM, latent_filters=LATENT_FILTERS):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_filters = latent_filters
        self.linear = nn.Linear(2 * FOURIER_DEGREE, latent_dim * latent_filters)

    def forward(self, time_tensor):
        """
        Args:
            time_tensor (Tensor): Shape (batch,) representing time indices (e.g., day of year)

        Returns:
            Tensor: Prior latent tensor of shape (batch, latent_dim, latent_filters)
        """
        f = fourier(time_tensor)  # (batch, 2 * DEGREE)
        out = self.linear(f)      # (batch, latent_dim * latent_filters)
        return out.view(-1, self.latent_dim, self.latent_filters)


# --- Variational Autoencoder (VAE) ---
class VAE(nn.Module):
    """
    Full VAE architecture combining Encoder, Decoder, and optional seasonal prior.
    Includes KL divergence and utility methods for training.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input sequence of shape (batch, INPUT_SIZE)

        Returns:
            x_hat (Tensor): Reconstructed input
            z_mean (Tensor): Latent mean
            z_log_var (Tensor): Latent log variance
            z (Tensor): Sampled latent vector
        """
        z_mean, z_log_var, z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z_mean, z_log_var, z

    def compute_kl(self, z_mean, z_log_var):
        """
        Computes KL divergence between encoded posterior and standard normal prior.

        Returns:
            kl_loss (Tensor): Scalar KL divergence
        """
        kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=[1, 2])
        return kl.mean()

    def encode_latent(self, x):
        """
        Utility method to return latent vector z from input.

        Returns:
            z (Tensor): Sampled latent tensor
        """
        _, _, z = self.encoder(x)
        return z
