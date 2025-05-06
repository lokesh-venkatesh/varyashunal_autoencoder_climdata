import torch

DAYS = 64
INPUT_SIZE = DAYS*24        # 64 days of hourly temps
LATENT_DIM = 32             # size of latent vector z
SEASONAL_INTERVAL = 24      # use one seasonal embedding per day
DEGREE = 3                  # for Fourier features for the prior distribution
HIDDEN_DIM = 128            # used in encoder/decoder hidden layers
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3

# Optimizer and other configurations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model saving path
MODEL_SAVE_PATH = 'models/vae_model_final.pth'