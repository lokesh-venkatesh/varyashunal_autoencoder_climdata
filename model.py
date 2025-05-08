import torch
import torch.nn as nn
from utils import *
from model_components import construct_encoder, construct_decoder, construct_seasonal_prior
from config import *

class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, input_size, noise_log_var=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.input_size = input_size

        # Learnable noise log variance (if not provided)
        if noise_log_var is None:
            self.noise_log_var = nn.Parameter(torch.zeros(1))
        else:
            self.noise_log_var = noise_log_var

    def forward(self, x, seasonal):
        # Encoder: generates the mean and log variance of the latent space
        z_mean, z_log_var, z = self.encoder(x, seasonal)

        # Sample z from the latent space (using reparameterization trick)
        z = self.reparameterize(z_mean, z_log_var)

        # Decoder: reconstruct the input based on z and seasonal info
        reconstructed = self.decoder(z, seasonal)

        # Print shapes for debugging
        print(f"Input shape: {x.shape}")  # Shape of the input (x)
        print(f"Reconstructed shape: {reconstructed.shape}")  # Shape of the reconstructed output

        # Compute the reconstruction loss and KL divergence
        recon_loss = log_lik_normal_sum(x, reconstructed, self.noise_log_var) / self.input_size
        kl_loss = self.kl_divergence(z_mean, z_log_var)

        return reconstructed, recon_loss, kl_loss

    def reparameterize(self, z_mean, z_log_var):
        """
        Reparameterization trick to sample z from the normal distribution defined by 
        z_mean and z_log_var.
        """
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps

    def kl_divergence(self, mu, log_var):
        """
        Computes KL divergence between N(mu, sigma) and standard normal N(0,1).
        """
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)

    def training_step(self, batch, optimizer):
        self.train()
        values, seasonal = batch
        optimizer.zero_grad()
        _, recon_loss, kl_loss = self(values, seasonal)
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        optimizer.step()
        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def evaluation_step(self, batch):
        self.eval()
        with torch.no_grad():
            values, seasonal = batch
            _, recon_loss, kl_loss = self(values, seasonal)
            total_loss = recon_loss + kl_loss
            return {
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            }
        
    def encode(self, x, seasonal):
        self.eval()
        with torch.no_grad():
            z_mean, z_log_var, _ = self.encoder(x, seasonal)
        return z_mean, z_log_var

    @property
    def device(self):
        return next(self.parameters()).device


def construct_VAE(input_size=INPUT_SIZE, latent_size=LATENT_DIM, DEGREE=DEGREE, 
                  interim_filters=64, latent_filter=32, noise_log_var=None):
    """
    Constructs and returns a VAE model with default parameters
    """
    encoder = construct_encoder(input_shape=input_size, interim_filters=interim_filters, latent_filter=latent_filter)
    decoder = construct_decoder(latent_dim=latent_size, latent_filter=latent_filter, interim_filters=interim_filters)
    seasonal_prior = construct_seasonal_prior(latent_dim=latent_size, DEGREE=DEGREE, latent_filter=latent_filter)

    vae = VAE(encoder=encoder, decoder=decoder, prior=seasonal_prior, 
              input_size=input_size, noise_log_var=noise_log_var)
    
    return vae
