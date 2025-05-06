import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps

def construct_encoder(input_shape=INPUT_SIZE, interim_filters=64, latent_filter=32):
    """
    Constructs and returns the Encoder network with default parameters.
    """
    class Encoder(nn.Module):
        def __init__(self, input_shape, interim_filters, latent_filter):
            super(Encoder, self).__init__()
            self.input_shape = input_shape
            self.latent_filter = latent_filter
            self.sampling = Sampling()

            self.conv = nn.Sequential(
                nn.Conv1d(1, interim_filters, kernel_size=5, stride=3, padding=2),  # same padding
                nn.ReLU(),
                nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(interim_filters, interim_filters, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(interim_filters, 2 * latent_filter, kernel_size=3, stride=2, padding=1),
            )

        def forward(self, x, seasonal):
            # Here, both `x` and `seasonal` are used
            x = x.view(x.size(0), 1, -1)  # Reshape to (batch, 1, input_shape)
            x = self.conv(x)
            z_mean = x[:, :self.latent_filter, :]
            z_log_var = x[:, self.latent_filter:, :]
            z = self.sampling(z_mean, z_log_var)
            return z_mean, z_log_var, z
    
    return Encoder(input_shape, interim_filters, latent_filter)


def construct_decoder(latent_dim=LATENT_DIM, latent_filter=32, interim_filters=64):
    """
    Constructs and returns the Decoder network with default parameters.
    """
    class Decoder(nn.Module):
        def __init__(self, latent_dim, latent_filter, interim_filters):
            super(Decoder, self).__init__()
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
                nn.ConvTranspose1d(interim_filters, 1, kernel_size=6, stride=3, padding=2, output_padding=1),  # <-- Changed from kernel_size=5 to 6
            )

        def forward(self, x, seasonal):
            x = self.decoder(x)
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, seq_len)
            x = x[:, :1536]  # Trim in case of overshoot
            return x

    return Decoder(latent_dim, latent_filter, interim_filters)

def construct_seasonal_prior(latent_dim=LATENT_DIM, DEGREE=DEGREE, latent_filter=32):
    """
    Constructs and returns the Seasonal Prior network with default parameters.
    """
    class SeasonalPrior(nn.Module):
        def __init__(self, latent_dim, DEGREE, latent_filter):
            super(SeasonalPrior, self).__init__()
            self.linear = nn.Linear(2 * DEGREE, 2 * latent_filter, bias=False)
            self.latent_filter = latent_filter
            self.sampling = Sampling()

        def forward(self, x):
            # x shape: (batch_size, latent_dim, 2*DEGREE)
            x = self.linear(x)  # -> shape: (batch_size, latent_dim, 2*latent_filter)
            z_mean = x[:, :, :self.latent_filter]
            z_log_var = x[:, :, self.latent_filter:]
            z = self.sampling(z_mean, z_log_var)
            return z_mean, z_log_var, z
    
    return SeasonalPrior(latent_dim, DEGREE, latent_filter)