import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*eps


class Encoder(nn.Module):
    def __init__(self, input_shape=INPUT_SIZE, interim_filters=INTERIM_FILTERS, latent_filter=LATENT_FILTER):
        super().__init__()
        self.latent_filter = latent_filter
        self.sampling = Sampling()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=interim_filters, kernel_size=5, stride=3, padding=2),  # same padding
            nn.ReLU(),
            nn.Conv1d(in_channels=interim_filters, out_channels=interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=interim_filters, out_channels=interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=interim_filters, out_channels=interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=interim_filters, out_channels=interim_filters, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=interim_filters, out_channels=2*latent_filter, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # (batch, 1, input_shape)
        x = self.conv(x)              # (batch, 2*latent_filter, latent_dim)
        x = x.permute(0, 2, 1)        # (batch, latent_dim, 2*latent_filter)
        z_mean = x[:, :, :self.latent_filter]
        z_log_var = x[:, :, self.latent_filter:]
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, latent_filter=LATENT_FILTER, interim_filters=INTERIM_FILTERS, output_size=INPUT_SIZE):
        super().__init__()
        self.latent_filter = latent_filter
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_filter, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(interim_filters, 1, kernel_size=5, stride=3, padding=2, output_padding=0)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, latent_filter, latent_dim)
        x = self.decoder(x)     # (batch, 1, seq_len)
        x = x.view(x.size(0), -1)
        return x[:, :self.output_size]  # Trim extra padding


class SeasonalPrior(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, degree=DEGREE, latent_filter=LATENT_FILTER):
        super().__init__()
        self.linear = nn.Linear(2 * degree, 2 * latent_filter, bias=False)
        self.latent_filter = latent_filter
        self.sampling = Sampling()

    def forward(self, x):
        # x: (batch_size, latent_dim, 2 * degree)
        x = self.linear(x)  # (batch, latent_dim, 2 * latent_filter)
        z_mean = x[:, :, :self.latent_filter]
        z_log_var = x[:, :, self.latent_filter:]
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z