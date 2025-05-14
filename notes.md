# Notes from meeting with Shaon:

1. Build the pipeline for univariate time series data
2. Analyse and plot the state space - maybe try reducing the latent space from 10 variables to 2 using a PCA or something like that
3. Alternatively, depending on model performance, see if you can just use a 2-D latent space instead of a 10-D one, or maybe a 4/5-D one depending on how good the model gets trained
4. Understand the mathematics, don't be blind
5. See how one could incorporate LSTMs into this model
6. Think about how to adapt this structure for multivariate patient time-series data

---

# Explanation for the structure of the forward-method in the VariationalAutoencoder Class:

Ah, great question — and it's exactly the kind of intuition that helps *really* understand David Kyle's time series VAE design.

Let's break it down **in the context of David Kyle’s implementation** of the Seasonal VAE (from his [repository](https://github.com/davidthemathman/vae_for_time_series/)). You're asking specifically:

> **How does simply adding the seasonal prior vector to the latent vector work?**

Let’s build up the answer from the pieces in his architecture.

---

### 🧠 1. What's the purpose of the seasonal prior?

David Kyle’s idea is to **guide the latent space** using knowledge about the *seasonal phase* (i.e., where in the year the sample falls). Instead of a **standard Normal prior** for the VAE latent space, he builds a **structured prior** that varies smoothly across the year — by using **Fourier features** of the timestamp (e.g., day of year / 365.0).

So the prior for each latent dimension is now a function of time — call this `s(t)`.

> Think of this like:
> → “If the input time series came from day 120 (spring), then its latent encoding should be close to the latent prior vector for spring.”

---

### 🧬 2. How is this prior computed?

In `SeasonalPrior`, he uses a neural net that maps the scalar `time_tensor` (day-of-year normalized between 0 and 1) to a `latent_dim`-dimensional vector using a set of **Fourier basis functions** (like `sin(2πft), cos(2πft)`).

This gives you a time-aware prior vector:

```python
seasonal = self.seasonal_prior(time_tensor)  # shape: (batch, latent_dim)
```

---

### 🔁 3. Why do we **add** the seasonal prior vector to the latent code?

This is subtle, but clever. In his design:

* The **encoder** outputs `mu` and `logvar` → giving a stochastic latent sample `z` per time step.
* Then:

  ```python
  z = z + seasonal.unsqueeze(-1)
  ```

  where `z.shape = (batch, latent_dim, seq_len)` and `seasonal.shape = (batch, latent_dim)`, which is expanded to match `z`.

#### This addition does 2 things:

1. **Centers the latent code around the seasonal prior**
   Instead of encoding `z ~ N(0, I)`, you're nudging `z` to be:

   $$
   z \sim \mathcal{N}(\mu + s(t), \sigma^2)
   $$

   Effectively, you're telling the decoder:

   > “Hey, here’s a version of the latent code that carries the usual compressed features of the sequence **plus** an embedded hint about what season we're in.”

2. **Encodes structured variability** in the latent space
   This allows the model to learn a **disentangled representation** — i.e., one part of the latent space learns seasonal features (from `s(t)`), and the rest can focus on residual variation (e.g., anomalies, noise, or sequence-specific patterns).

---

### 🎯 Why not multiply, concatenate, or condition differently?

* **Addition** is a natural way to impose a **shift in the prior mean** in latent space.
* It preserves the shape and format of the latent variable while still modifying its location.
* It avoids introducing unnecessary parameters (as concatenation would) or complicating the decoder interface.

This trick is common in conditional VAEs:
→ If you want a latent vector to depend on some context (like class label or time), you shift it.

---

### ✅ Summary (Kyle-style):

* You **encode** the input time series `x` into a latent representation `z ~ N(mu, σ²)`.
* You **add** a **seasonal prior vector** to `z`, based on `time_tensor`, making the final latent:

  $$
  z' = z + s(t)
  $$
* The **decoder** reconstructs the time series from this `z'`, which now includes both the learned content and the seasonal context.
* This lets the model learn how sequences vary with **time of year**, and capture patterns like seasonal trends, daily temperature rhythms, etc.

---

Let me know if you'd like a visual schematic of this flow — I can draw one for you.
