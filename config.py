import torch

DAYS = 64
HOURS = 24
INPUT_SIZE = DAYS*HOURS        # 64 days of hourly temps
#LATENT_DIM = 32             # size of latent vector z
LATENT_SIZE = 4*24          # the hours associated with each latent variable
LATENT_DIM = INPUT_SIZE // LATENT_SIZE  # 16
LATENT_FILTER = 10
INTERIM_FILTERS = 2*LATENT_FILTER     # 20
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