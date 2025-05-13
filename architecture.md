# ===== ENCODER =====

Input: temperature_sequence           # shape = (1536,)

Step 1: Reshape for Conv1D
    temperature_sequence → reshaped_input
    shape: (1536,) → (1536, 1)

Step 2: Conv1D Layer 1
    kernel_size=5, strides=3, padding='same', filters=interim_filters, activation='relu'
    (1536, 1) → (512, interim_filters)

Step 3: Conv1D Layer 2
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (512, interim_filters) → (256, interim_filters)

Step 4: Conv1D Layer 3
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (256, interim_filters) → (128, interim_filters)

Step 5: Conv1D Layer 4
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (128, interim_filters) → (64, interim_filters)

Step 6: Conv1D Layer 5
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (64, interim_filters) → (32, interim_filters)

Step 7: Conv1D Layer 6
    kernel_size=3, strides=2, padding='same', filters=2 * latent_filters, activation=None
    (32, interim_filters) → (16, 2 * latent_filters)

Step 8: Split output into mean and log variance
    z_mean     = x[:, :, :latent_filters]      # shape: (16, latent_filters)
    z_log_var  = x[:, :, latent_filters:]      # shape: (16, latent_filters)

# ===== LATENT SPACE & SAMPLING =====

Step 9: Sampling with reparameterization trick
    epsilon ~ Normal(0, 1)
    z = z_mean + exp(0.5 * z_log_var) * epsilon   # shape: (16, latent_filters)

# ===== DECODER =====

Step 10: Conv1DTranspose Layer 1
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (16, latent_filters) → (32, interim_filters)

Step 11: Conv1DTranspose Layer 2
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (32, interim_filters) → (64, interim_filters)

Step 12: Conv1DTranspose Layer 3
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (64, interim_filters) → (128, interim_filters)

Step 13: Conv1DTranspose Layer 4
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (128, interim_filters) → (256, interim_filters)

Step 14: Conv1DTranspose Layer 5
    kernel_size=3, strides=2, padding='same', filters=interim_filters, activation='relu'
    (256, interim_filters) → (512, interim_filters)

Step 15: Conv1DTranspose Layer 6
    kernel_size=5, strides=3, padding='same', filters=1, activation=None
    (512, interim_filters) → (1536, 1)

Step 16: Reshape back to 1D
    (1536, 1) → (1536,)

Output: reconstructed_sequence         # shape = (1536,)
