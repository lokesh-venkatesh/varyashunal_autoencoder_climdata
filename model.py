import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

class Encoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=5, stride=4, padding=1)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(20, 64, kernel_size=3, stride=2, padding=1)

        # Separate heads for mu and logvar
        self.mu_layer = nn.Conv1d(64, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv1d(64, latent_dim, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 1536)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))  # (batch, 64, seq_len_reduced)
        mu = self.mu_layer(x)      # (batch, latent_dim, seq_len_reduced)
        logvar = self.logvar_layer(x)  # (batch, latent_dim, seq_len_reduced)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=10, output_length=1536):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(latent_dim, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
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
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.freqs = torch.arange(1, 4).float().view(1, -1)

    def forward(self, time_tensor):
        time_tensor = time_tensor.view(-1, 1)
        phases = 2 * math.pi*self.freqs * time_tensor
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)
        seasonal = F.linear(features, torch.randn(self.latent_dim, features.shape[1]))
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

    def forward(self, x):
        mu, logvar = self.encoder(x)  # (batch, latent_dim, seq_len)
        z = self.sampling(mu, logvar)  # (batch, latent_dim, seq_len)
        recon = self.decoder(z)  # (batch, 1536)
        mu_avg = mu.mean(dim=2)
        logvar_avg = logvar.mean(dim=2)
        return recon, mu_avg, logvar_avg, z

    def save_model(self, path="results/model_weights.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="results/model_weights.pth"):
        self.load_state_dict(torch.load(path))
        self.eval()



'''
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
        # Encoder convolutional layers
        self.conv1 = nn.Conv1d(1, 20, kernel_size=5, stride=4, padding=1)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(20, 64, kernel_size=3, stride=2, padding=1)

        # Separate heads for mu and logvar
        self.mu_layer = nn.Conv1d(64, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv1d(64, latent_dim, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 1536)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))  # (batch, 64, seq_len_reduced)
        mu = self.mu_layer(x)      # (batch, latent_dim, seq_len_reduced)
        logvar = self.logvar_layer(x)  # (batch, latent_dim, seq_len_reduced)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=10, output_length=1536):
        super().__init__()
        # Decoder convolutional layers
        self.deconv1 = nn.ConvTranspose1d(latent_dim, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)
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
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.freqs = torch.arange(1, 4).float().view(1, -1)

    def forward(self, time_tensor):
        # Project the time tensor into a seasonal representation
        time_tensor = time_tensor.view(-1, 1)
        phases = 2 * math.pi * self.freqs * time_tensor
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)
        
        # Use a fixed learnable projection of the seasonal features
        seasonal = F.linear(features, torch.randn(self.latent_dim, features.shape[1]))
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
        # Apply the seasonal prior
        seasonal_latent = self.seasonal_prior(time_tensor)  # (batch, latent_dim)
        
        mu, logvar = self.encoder(x)  # (batch, latent_dim, seq_len)
        z = self.sampling(mu, logvar)  # (batch, latent_dim, seq_len)
        
        # Combine the seasonal latent with the learned latent space (mu)
        z = z + seasonal_latent.unsqueeze(-1)  # Add seasonal prior influence
        
        recon = self.decoder(z)  # (batch, 1536)
        
        mu_avg = mu.mean(dim=2)
        logvar_avg = logvar.mean(dim=2)
        
        return recon, mu_avg, logvar_avg, z

    def save_model(self, path="results/model_weights.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="results/model_weights.pth"):
        self.load_state_dict(torch.load(path))
        self.eval()

'''