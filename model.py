import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Sampling(nn.Module):
    """THIS CLASS IS FOR ADDING SAMPLING TO THE MIDDLE OF THE VAE NETWORK,
    THE ACTUAL REPARAMETRISATION IS DONE THROUGH THE forward METHOD DEFINED BELOW"""
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """REQUIRES THE MEAN AND LOG-VARIANCE TO BE PROVIDED AS INPUT PARAMETERS,
        AND THEN DOES THE SAMPLING PROCESS AND RETURNS A TENSOR OF THE SAME DIMENSIONS AS THE PARAMS"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Encoder(nn.Module):
    """THIS DEFINES THE ENCODER ARCHITECTURE FOR THE VAE MODEL,
    FOR REFERENCE, THE BELOW LINES ARE THE STRUCTURE THAT DAVID KYLE USES:

    # Parameters
    input_shape = None #INPUT_SIZE
    latent_dim = None #INPUT_SIZE//LATENT_SIZE
    latent_filter = 10
    interim_filters = 2*latent_filter

    def build_encoder():
        inputs = layers.Input(shape=(input_shape,))
        x = layers.Reshape((-1, 1))(inputs)
        x = layers.Conv1D(interim_filters, 5, strides=3, padding='same', activation='relu')(x)
        x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(2*latent_filter, 3, strides=2, padding='same')(x)
        z_mean = x[: ,:, :latent_filter]
        z_log_var = x[:, :, latent_filter:]
        z = Sampling()([z_mean, z_log_var])
        encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        return encoder
    """
    def __init__(self, latent_dim=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=5, stride=3, padding=2) # input is of length 1536, output is of length 512
        # NOTE that the kernel size is 5, and the stride is 3, so the output length is (input_length - kernel_size) / stride + 1
        # padding=2 has the same effect as padding='same' on Keras
        # NOTE that this takes one channel per timepoint on the vector as input, and then returns 20 channels per timepoint as output
        self.conv2 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1) # output is of length 256
        self.conv3 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1) # output is of length 128
        self.conv4 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1) # output is of length 64
        self.conv5 = nn.Conv1d(20, 20, kernel_size=3, stride=2, padding=1) # output is of length 32
        # Now the output is of length 32, and we want to reduce it to the latent dimension
        # We can do this by using a 1D convolution with kernel size 1, which will not change the length of the output
        self.mu_layer = nn.Conv1d(20, latent_dim, kernel_size=1) # output is of length 32
        self.logvar_layer = nn.Conv1d(20, latent_dim, kernel_size=1) # output is of length 32
        # NOTE: We can use the same layer for both mu and logvar
        # because they are both of the same size, and we can just split the output into two parts
        # NOTE that the output of this layer is of size (batch, latent_dim, seq_len_reduced)
        # We can then use the mean and logvar to sample from the latent space and then use the decoder to reconstruct the input

    def forward(self, x):
        """TAKES IN THE 1536-DIMENSIONAL INPUT TIME SERIES AND
        RETURNS THE MEAN AND THE LOG-VARIANCE OF THE INPUT VECTOR IN THE CORRESOPONDING LATENT SPACE"""
        x = x.unsqueeze(1) # (batch, 1, 1536)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class Decoder(nn.Module):
    """THIS DEFINES THE DECODER ARCHITECTURE FOR THE VAE MODEL,
    FOR REFERENCE, THE BELOW LINES ARE THE STRUCTURE THAT DAVID KYLE USES:

    # Parameters
    input_shape = None #INPUT_SIZE
    latent_dim = None #INPUT_SIZE//LATENT_SIZE
    latent_filter = 10
    interim_filters = 2*latent_filter

    def build_decoder():
        latent_inputs = layers.Input(shape=(latent_dim, latent_filter))
        x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(latent_inputs)
        x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(interim_filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1DTranspose(1, 5, strides=3, padding='same')(x)
        outputs = layers.Reshape((-1,))(x)
        decoder = models.Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        return decoder

    decoder = build_decoder()
    """
    def __init__(self, latent_dim=10, output_length=1536):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(latent_dim, 20, kernel_size=3, stride=2, padding=1, output_padding=1) # input is of length 32, output is of length 64
        # NOTE that the kernel size is 3, and the stride is 2, so the output length is (input_length - kernel_size) / stride + 1
        # padding=1 has the same effect as padding='same' on Keras
        self.deconv2 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1) # output is of length 128
        self.deconv3 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1) # output is of length 256
        self.deconv4 = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1) # output is of length 512
        self.deconv5 = nn.ConvTranspose1d(20, 1, kernel_size=5, stride=3, padding=1, output_padding=0) # output is of length 1536

    def forward(self, z):
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        z = self.deconv5(z)
        return z.squeeze(1)  # (batch, 1536)


class SeasonalPrior(nn.Module):
    """THIS CLASS DEFINES THE SEASONALITY PRIOR FOR THE VAE MODEL, IT TAKES IN A TIME-TENSOR AND RETURNS A SEASONALITY EMBEDDING
    
    THE TIME-TENSOR IS A 1D TENSOR OF LENGTH BATCH_SIZE, AND THE SEASONALITY EMBEDDING IS A 2D TENSOR OF SIZE (BATCH_SIZE, LATENT_DIM)

    THE SEASONALITY EMBEDDING IS COMPUTED BY TAKING THE SINE AND COSINE OF THE TIME-TENSOR MULTIPLIED BY A FREQUENCY PARAMETER, 
    AND THEN LINEARLY PROJECTING IT TO THE LATENT DIMENSION. THE FREQUENCY PARAMETER IS A LEARNABLE PARAMETER THAT IS INITIALIZED TO 1, 2, ..., NUM_FREQS.
    THE SINE AND COSINE FUNCTIONS ARE USED TO CAPTURE THE SEASONALITY IN THE DATA. THE LINEAR PROJECTION IS USED TO MAP 
    THE SINE AND COSINE FUNCTIONS TO THE LATENT DIMENSION. THE SEASONALITY EMBEDDING IS THEN ADDED TO THE LATENT REPRESENTATION OF THE INPUT DATA, 
    AND COMPUTED FOR EACH SAMPLE IN THE BATCH
    """
    def __init__(self, latent_dim=10, num_freqs=3):
        super().__init__()
        self.latent_dim = latent_dim # This is the dimension of the latent space vector
        self.num_freqs = num_freqs # This is the highest degree of the Fourier series we wish to use for modelling the seasonality
        self.freqs = nn.Parameter(torch.arange(1, num_freqs + 1).float().view(1, -1), requires_grad=False) 
        # This is a learnable parameter that is initialized to 1, 2, ..., num_freqs
        self.linear = nn.Linear(2 * num_freqs, latent_dim) # Learnable projection from Fourier features to latent_dim

    def forward(self, time_tensor): # time_tensor: (batch,)
        """TAKES IN A TIME-TENSOR OF SIZE (BATCH_SIZE,) AND RETURNS A SEASONALITY EMBEDDING OF SIZE (BATCH_SIZE, LATENT_DIM)"""
        time_tensor = time_tensor.view(-1, 1)  # (batch, 1)
        # Note that the time tensor is of the same length as the temperature values, which is 1536 values long, and the 'phase' is a float value between 0 and 1
        phases = 2*math.pi*self.freqs*time_tensor  # (batch, num_freqs) -> simply returns 2pi(x), 2pi(2x) and 2pi(3x)
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)  # (batch, 2 * num_freqs) -> cos(2pi(x), 2pi(2x), 2pi(3x)), sin(2pi(x), 2pi(2x), 2pi(3x))
        # This will return the corresponding phase vectors, which is simply a nested array with six values of fourier terms for each timepoint
        seasonal = self.linear(features)  # (batch, latent_dim)
        # This term basically passes this 'features' vector through the linear layer in the SeasonalPrior unit of the model, and returns whatever output is obtained
        return seasonal

class VariationalAutoencoder(nn.Module):
    """THIS CLASS WILL DEFINE THE ENTIRE VARIATIONAL AUTOENCODER ARCHITECTURE IN ONE GO."""
    def __init__(self, input_dim=1536, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sampling = Sampling()
        self.seasonal_prior = SeasonalPrior()

    def forward(self, x, time_tensor):
        mu, logvar = self.encoder(x) # NOTE that the mu and logvar are of size (batch, latent_dim, seq_len), and we want to sample from this distribution
        z = self.sampling(mu, logvar) # NOTE that the seasonal prior is of size (batch, latent_dim), and we want to broadcast this over the sequence length

        seasonal = self.seasonal_prior(time_tensor)  # (batch, latent_dim)
        # NOTE that the seasonal prior is of size (batch, latent_dim), and we want to add this to the latent representation
        # We need to expand the seasonal prior to the same size as z
        seasonal = seasonal.unsqueeze(-1).expand_as(z)  # (batch, latent_dim, seq_len)
        # This will expand the seasonal prior to the same size as z, and then we can add it to z
        z = z + seasonal  # Add seasonality to latent representation
        # NOTE that the z is of size (batch, latent_dim, seq_len), and we want to pass this through the decoder
        
        recon = self.decoder(z)  # (batch, 1536)
        mu_avg, logvar_avg = mu.mean(dim=2), logvar.mean(dim=2)


        # NOTE THAT HERE, recon IS THE RECONSTRUCTED OUTPUT AND z IS THE LATENT REPRESENTATION OF THE INPUT DATA.
        # mu_avg AND logvar_avg ARE THE AVERAGE MEAN AND LOGVARIANCE OF THE INPUT DATA, AND THEY ARE USED FOR THE ELBO LOSS FUNCTION.    
        # BASICALLY THIS mu_avg AND logvar_avg ARE THE AVERAGED VECTORS OF THE mu AND logvar VECTORS FOR EACH TIMEPOINT IN THE ENTIRE SEQUENCE
        return recon, mu_avg, logvar_avg, z

    def save_model(self, path="results/model_weights.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="results/model_weights.pth"):
        self.load_state_dict(torch.load(path))
        self.eval()
