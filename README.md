# Diffusion Models from Scratch — DDPM + Butterfly Generator 🦋

This repository documents my learning journey building **Denoising Diffusion Probabilistic Models (DDPMs)** from first principles and validating the theory through experiments. It includes:

* A minimal **DDPM** implementation in PyTorch
* A **Butterfly model** that generates butterfly images
* Notes on the **theory**, **key equations**, and the **U‑Net** backbone used by diffusion models

> **Why this project?**
> I wanted to go beyond reading the paper and actually test my understanding by implementing the forward and reverse diffusion processes, training the noise-prediction model, and sampling high‑quality images.

---

## Table of Contents

* [What is a Diffusion Model?](#what-is-a-diffusion-model)
* [Key DDPM Equations](#key-ddpm-equations)
* [U‑Net Backbone (for ε-prediction)](#u-net-backbone-for-ε-prediction)
* [Butterfly Generator](#butterfly-generator)
* [Sampling](#sampling)
* [Results](#results)
* [References](#references)

---

## Results

<img width="531" height="527" alt="Screenshot 2025-07-30 at 11 31 24 PM" src="https://github.com/user-attachments/assets/b1414577-7360-4874-9265-70ad80cd58b0" />

<img width="517" height="521" alt="Screenshot 2025-07-30 at 11 50 50 PM" src="https://github.com/user-attachments/assets/e9879226-70d3-46b9-abc5-1a2f7b21f73a" />

<img width="674" height="548" alt="Screenshot 2025-07-31 at 12 11 26 AM" src="https://github.com/user-attachments/assets/9bffa93b-4afb-4580-bca3-e2e9529d0ecf" />

## What is a Diffusion Model?

A **diffusion model** learns to invert a gradual noising process. During training, we **add Gaussian noise** to clean data over many timesteps; during sampling, we **denoise step‑by‑step**, using a neural network that predicts the noise present at each step. The original DDPM paper shows this can be framed as a **variational inference** problem with a tractable training objective and a simple MSE loss for noise prediction.

---

## Key DDPM Equations

### 1) Forward (noising) process $q$

We progressively add Gaussian noise with a variance schedule $\{\beta_t\}_{t=1}^T$.

x_t = sqrt(alphā_t) * x_0 + sqrt(1 - alphā_t) * ε  
where ε ~ N(0, I)

A closed form exists that jumps from $x_0$ to any $x_t$:

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\; \sqrt{\bar{\alpha}_t}\,x_0,\; (1-\bar{\alpha}_t)\mathbf{I}\right)
$$

$$
\Rightarrow\quad x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\mathbf{I})
$$

### 2) Reverse (denoising) process $p_\theta$

We learn a model $\varepsilon_\theta(x_t, t)$ that predicts the noise at each step and defines a Gaussian transition:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 \mathbf{I}\big)
$$

Using noise‑prediction parameterization (from the paper), the mean is:

μ_θ(x_t, t) = 1 / sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alphā_t)) * ε_θ(x_t, t))

(With $\sigma_t^2$ typically set to a fixed or learned variance per timestep—DDPM uses fixed variance; improved variants learn it.)

### 3) Training objective (simple loss)

The variational bound reduces to a simple MSE on the noise:

L_simple = E_{x_0, ε, t} [ || ε - ε_θ(x_t, t) ||^2 ]

### 4) Sampling loop (high‑level)

Start at pure noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ and iterate:

$$
x_{t-1} \leftarrow \mu_\theta(x_t, t) + \sigma_t\,z,\quad z \sim \mathcal{N}(0,\mathbf{I})\ \text{if } t>1;\ \ z=0\ \text{if } t=1
$$

> **Schedules:** I experimented with linear $\beta_t$ schedules; cosine schedules can also improve sample quality.

---

## U‑Net Backbone (for ε-prediction)

I used a **U‑Net** to predict $\varepsilon_\theta(x_t, t)$:

* **Encoder–Decoder with Skips:** Downsample (Conv/ResBlocks) to capture global context; upsample to recover spatial detail; **skip connections** fuse matching resolutions.
* **Time Embeddings:** Sinusoidal $t$-embeddings passed through MLPs and injected into residual blocks via FiLM‑style or additive conditioning.
* **Attention (optional):** Self‑attention at lower resolutions helps model long‑range dependencies.
* **Blocks:** GroupNorm + SiLU activation in ResBlocks; strided Conv for downsampling; nearest/bilinear + Conv (or transposed Conv) for upsampling.

This structure is well‑suited to image‑to‑image mappings like denoising at multiple scales.

---

## Butterfly Generator

A small, focused experiment to verify the pipeline end‑to‑end.

* **Task:** Unconditional image generation of butterflies 🦋
* **Backbone:** Same U‑Net used for DDPM
* **Objective:** $\mathcal{L}_{\text{simple}}$ noise MSE
* **Resolution:** Start with $64\times64$ or $128\times128$ (easier training)
* **Timesteps:** $T \in [0, 1000]$ (I typically used 1000 for first runs)
* **Schedule:** Linear $\beta_t$ for baseline; try cosine for improvements
* **Augmentations:** Light augmentations (random crop/flip) can help generalization

**Why butterflies?** The class has distinctive textures and symmetries; it’s a good sanity check before scaling to more complex datasets.

## References

* J. Ho, A. Jain, P. Abbeel. **Denoising Diffusion Probabilistic Models (DDPM)**, NeurIPS 2020.
* P. Dhariwal, A. Nichol. **Diffusion Models Beat GANs on Image Synthesis**, NeurIPS 2021.
* O. Ronneberger, P. Fischer, T. Brox. **U‑Net: Convolutional Networks for Biomedical Image Segmentation**, 2015.

---

*Author:* Kunal Sahu

---

