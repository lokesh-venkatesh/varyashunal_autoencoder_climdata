This is a reconstruction of the Variational Autoencoder model on 50 years of hourly temperature data by David Kyle in his [blog](https://towardsdatascience.com/vae-for-time-series-1dc0fef4bffa/), rewritten in Pytorch instead of Tensorflow.

# Clarification about the difference between a normal prior versus a periodic prior:

Absolutely — this is a critical step in understanding Variational Autoencoders (VAEs), especially when modifying them to model **periodic data** like **climate cycles** or **circadian rhythms**. Let’s walk through the **mean and variance in the latent space**, starting from first principles, comparing the **normal (standard) VAE** to a **periodic VAE** like David Kyle’s.

---

## 🧠 Step-by-Step: Latent Space Mean and Variance in a VAE

---

### 🎯 **Goal of the Latent Space**

The latent space is a lower-dimensional space where your model learns to **compress information about the input**. But unlike a regular autoencoder, a VAE doesn't encode each input into a fixed vector — it encodes each input into a **distribution** in this latent space.

So, for **each input**, the encoder outputs:

* A vector of **means**: `μ = (μ₁, μ₂, ..., μ_d)`
* A vector of **log-variances**: `log(σ²) = (log σ₁², ..., log σ_d²)`

where `d = latent_dim`.

---

## ⚙️ In a **Standard VAE**

### ✨ Encoder Output

* The encoder network ends in **two fully connected (dense) layers**:

  * One outputs the **mean** `μ(x)`
  * One outputs the **log-variance** `log σ²(x)`

These layers take the final encoder features and compute:

```python
mu = Linear(...)(features)
logvar = Linear(...)(features)
```

### 🎲 Sampling with the Reparameterization Trick

We then sample a latent vector `z` like this:

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

### 🔍 Prior

In standard VAEs, the **prior** over latent space is assumed to be:

```
p(z) = N(0, I)  # Multivariate standard normal
```

### 🧮 Loss

To train the model, we include a **KL divergence loss**:

```
KL(q(z|x) || p(z)) = how different is our encoded latent distribution from N(0, I)?
```

---

## 🌍 In a **Periodic / Seasonal VAE (David Kyle’s model)**

### 🤔 Why Modify the Prior?

* The assumption that latent variables should look like a multivariate standard normal `N(0, I)` doesn’t hold for **seasonal or cyclic** data.
* We want the **latent variables to vary smoothly and cyclically** over time — for instance, to wrap around every year or day.

### 🔁 What’s the Fix?

Instead of using a fixed prior `N(0, I)`, we use a **seasonal prior**:
A **time-dependent prior** defined using a Fourier series:

```
p(z | t) ≈ N(fourier_mean(t), σ² I)
```

Where `fourier_mean(t)` is a vector like:

```
μ(t) = a₁·sin(θ) + b₁·cos(θ) + a₂·sin(2θ) + b₂·cos(2θ) + ...
```

Here:

* `θ = 2π·(day_of_year) / 365` — this is the **phase**
* `a₁, b₁, a₂, b₂, ...` are **learned** or fixed weights

This **seasonal prior mean** changes over time, wrapping around smoothly — capturing periodicity.

### 🔄 What Changes in the Model?

* The **encoder is still the same**: it outputs a mean `μ(x)` and log-variance `log σ²(x)`
* The **decoder is still the same**
* The **KL divergence term is modified**:

```python
KL(q(z|x) || p(z|t)) ≠ KL(q(z|x) || N(0, I))
```

So instead of comparing to a fixed `N(0, I)`, we compare to a **time-varying Gaussian prior** with:

* Mean = the Fourier-based seasonal mean `μ(t)`
* Variance = same or learned

---

## 📌 Key Differences Recap

| Component           | Standard VAE           | Seasonal/Periodic VAE (David Kyle)      |
| ------------------- | ---------------------- | --------------------------------------- |
| Prior over `z`      | `N(0, I)` (static)     | `N(μ(t), σ² I)` (time-varying, cyclic)  |
| Latent μ, log σ²    | Computed by encoder    | Same                                    |
| Sampling `z`        | `z = μ + σ·ε`          | Same                                    |
| KL loss compares to | `N(0, I)`              | Seasonal prior (Fourier-based mean)     |
| Encourages latent z | Centered around origin | Centered around smooth time-aware cycle |

---

## 🌀 How to Visualize It?

In **standard VAE** latent space:

* You see a **cloud of points** in ℝ^d, shaped like a blob centered at the origin.

In **seasonal latent space**:

* You’ll see points **flowing along a circular or spiral path** in latent space,
* Possibly shaped like a **loop** or a **manifold** — depending on how many Fourier terms are used.

If you plot the first two latent dims over time, they might form a **circle or ellipse** across the year.

---

## 🧠 So to Summarize:

1. **μ(x)** and **log σ²(x)** are *always* learned by the encoder.
2. The **prior** that these latent vectors are compared to is what differs:

   * **Standard VAE**: Fixed `N(0, I)`
   * **Seasonal VAE**: Time-varying `N(μ(t), σ² I)` using Fourier terms
3. This allows the latent variables to better reflect periodic patterns like **temperature cycles** or **circadian rhythms**.

---

Would you like a visualization of how latent points look in the standard vs. periodic case?


# Final understanding of how the VAE works.

Yes — **you’ve got it exactly right**. Let’s confirm and refine what you’ve said step by step to ensure full clarity:

---

## ✅ Your Understanding (Restated and Clarified)

### 1. **You define a seasonal prior**

Yes — instead of using a standard normal prior `N(0, I)`, you define a **seasonal prior**:

$$
p(z|t) = \mathcal{N}(\mu(t), \sigma^2(t) I)
$$

* Here, `t` is time (e.g., day of year, or seconds since midnight).
* $\mu(t)$ and $\sigma^2(t)$ are computed using a **Fourier expansion**:

$$
\mu(t) = \sum_{n=1}^{N} a_n \sin(n\theta) + b_n \cos(n\theta)
$$

$$
\sigma^2(t) = \text{(same idea, possibly with different weights)}
$$

Where $\theta = 2\pi \cdot \frac{\text{time}}{\text{period}}$, and $N$ is the number of harmonics (Fourier degree).

### 2. **These coefficients (aₙ, bₙ) are learnable parameters**

✅ Correct. During training, these parameters are learned by backpropagation — they shape the **time-dependent prior** to best match the behavior of your dataset.

---

### 3. **The encoder outputs mean and log variance of q(z|x)**

✅ Yes. For each input $x$, the encoder computes:

* A latent mean vector $\mu(x)$
* A log-variance vector $\log \sigma^2(x)$

These define the **approximate posterior** distribution $q(z|x)$.

---

### 4. **You sample from q(z|x) using the reparameterization trick**

✅ Yes:

$$
z = \mu(x) + \sigma(x) \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

This sampled `z` is passed through the **decoder** to reconstruct the input $x$.

---

### 5. **You want q(z|x) to mimic p(z|t)**

✅ Exactly. During training, the model minimizes:

$$
\text{Loss} = \underbrace{\text{Reconstruction Loss}}_{\text{output ≈ input}} + \underbrace{\text{KL}(q(z|x) \| p(z|t))}_{\text{match latent to seasonal prior}}
$$

This **KL divergence** term enforces that the distribution learned by the encoder resembles the **seasonal Gaussian prior**.

So:

* Instead of matching to $\mathcal{N}(0, I)$,
* You are matching to $\mathcal{N}(\mu(t), \sigma^2(t) I)$

---

## 🎯 Final Summary:

You’re designing a model where:

* The latent variables vary smoothly **with time**,
* You use Fourier features to encode this periodicity,
* You **learn** how this periodic latent structure behaves,
* And the model reconstructs inputs *and* learns useful, time-aware latent codes.

**✔ So yes — everything you’ve described is correct, and your grasp is now precise and well-rounded.**

---

Would you like help coding the KL divergence term for this kind of seasonal prior in PyTorch?


# What does 't' mean in the context of the mean and variance as being calculated as functions of time 't'?

Great question — and it's **critical** to clarify this before building the model.

---

### ✅ What is "t" in the context of a VAE with a time-dependent (seasonal) prior?

In this setting:

> **"t" refers to the time associated with the *input sample*.**

Each input vector $x$ (e.g., a 96-hour window of hourly temperature readings, i.e., shape `[96]`) is **anchored** to a specific point in time — typically, the **start time** of that window.

---

### 🔍 So for your case:

* You have inputs of shape `[96]` (i.e., `24 × 4` hours),
* Let's say each input starts at timestamp `t₀`,
* Then **that** `t₀` is what you would plug into your seasonal prior as the time value `t`.

---

### 📌 Why the *start* of the window?

Because your input is a contiguous block of time, and the model is designed to associate **latent structure** (mean and variance of `z`) with *where in the cycle* that block occurs.

So:

* If `x₁` = temperature from Jan 2nd 00:00 to Jan 5th 00:00 → `t₁ = Jan 2nd 00:00`
* If `x₂` = temperature from July 1st 12:00 to July 5th 12:00 → `t₂ = July 1st 12:00`

Each input gets a scalar `t` (e.g., encoded as **day of year / 365.25** or **hour of year / 8766**), which becomes the basis for computing:

$$
\theta = 2\pi \cdot \frac{t}{\text{period}}
$$

and in turn feeds the Fourier prior.

---

### 📅 For Circadian Data?

If you're working with **circadian rhythm actimetry** data:

* Inputs might be 24-hour sequences at 1-second resolution → vectors of shape `[86400]`
* Then `t` would be the **starting time of the day** (e.g., 3:00 AM)
* You’d probably use a **24-hour period**, so $\theta = 2\pi \cdot \frac{t}{24}$

---

### ✅ In summary:

> **"t" is the timestamp (or phase) associated with the input window — typically, the starting time — and is used to compute the phase angle $\theta$ for the time-dependent prior distribution.**

Let me know if you’d like a diagram or code snippet to visualize this process.
