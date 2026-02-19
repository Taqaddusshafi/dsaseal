# Appendix A: Mathematical Formulations

## A.1 LoRA Decomposition — Complete Derivation

### A.1.1 Problem Setup

Given a pre-trained weight matrix **W₀ ∈ ℝ^{d_out × d_in}** in an attention layer, the standard linear transformation is:

```
h = W₀ · x    where x ∈ ℝ^{d_in}, h ∈ ℝ^{d_out}
```

During fine-tuning, the weight is updated:

```
W = W₀ + ΔW
```

**Key observation** (Aghajanyan et al., 2021): The update ΔW has a low intrinsic rank. That is, the effective dimensionality of the update is much smaller than min(d_out, d_in).

### A.1.2 Low-Rank Factorization

LoRA parameterizes ΔW as a product of two low-rank matrices:

```
ΔW = B · A

where:
  B ∈ ℝ^{d_out × r}     (down-projection)
  A ∈ ℝ^{r × d_in}      (up-projection)
  r << min(d_out, d_in)  (rank constraint)
```

### A.1.3 Modified Forward Pass

```
h = W₀ · x + (α/r) · B · A · x
  = W₀ · x + (α/r) · ΔW · x
```

where α is the scaling hyperparameter. The scaling factor α/r ensures:

- When α = r, the scaling is 1 (same as unscaled LoRA)
- When α = 2r (recommended), the scaling is 2 (slightly amplified updates)

### A.1.4 Initialization

```
B is initialized to zeros: B ← 0^{d_out × r}
A is initialized with Kaiming uniform: A ~ U(-√(1/d_in), √(1/d_in))
```

**Why this initialization?**

- B = 0 ensures ΔW = BA = 0 at the start
- This means the model begins with exactly the pre-trained behavior
- A is random to break symmetry during gradient updates
- Kaiming initialization scales with input dimension for stable gradients

### A.1.5 Gradient Computation

The loss gradient with respect to LoRA matrices:

```
∂L/∂B = (α/r) · (∂L/∂h) · (A · x)ᵀ
       = (α/r) · δ · zᵀ

where:
  δ = ∂L/∂h ∈ ℝ^{d_out}  (upstream gradient)
  z = A · x ∈ ℝ^{r}       (intermediate)

∂L/∂A = (α/r) · Bᵀ · (∂L/∂h) · xᵀ
       = (α/r) · Bᵀ · δ · xᵀ
```

### A.1.6 Parameter Count Analysis

```
Full fine-tuning parameters per layer:
  P_full = d_out × d_in

LoRA parameters per layer:
  P_LoRA = d_out × r + r × d_in = r × (d_out + d_in)

Compression ratio:
  ρ = P_LoRA / P_full = r × (d_out + d_in) / (d_out × d_in)

For square layers (d_out = d_in = d):
  ρ = 2r / d

Example (d=1536, r=8):
  ρ = 2 × 8 / 1536 = 0.0104 = 1.04%

  P_full = 1536 × 1536 = 2,359,296
  P_LoRA = 8 × (1536 + 1536) = 24,576

  Reduction: 96× fewer parameters
```

### A.1.7 Merging LoRA for Inference

At inference time, LoRA can be merged into the base weights with zero latency overhead:

```
W_merged = W₀ + (α/r) · B · A

# This is a one-time computation
# After merging, inference uses W_merged directly
# No additional matrix multiplications needed
```

## A.2 Quantization Mathematics (QLoRA)

### A.2.1 Normal Float (NF4) Quantization

NF4 uses 16 quantization levels optimized for normally distributed weights:

```
Given weight w ~ N(0, σ²):

Step 1: Normalize
  w_norm = w / max(|w|)    →  w_norm ∈ [-1, 1]

Step 2: Quantize to nearest NF4 value
  q_i = argmin_j |w_norm - nf4_j|

  where nf4 values are the 16 optimal quantiles of N(0,1):
  {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
   -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,
   0.4407, 0.5626, 0.7230, 1.0}

Step 3: Store as 4-bit index
  memory per param = 4 bits
```

### A.2.2 Double Quantization

The absmax scaling factors are further quantized:

```
Standard quantization: w ÷ absmax → 4-bit
  Scale stored as FP32 per block of 64 weights
  Overhead: 32/64 = 0.5 bits per param

Double quantization: scale ÷ absmax(scales) → FP8
  Overhead: 8/64 + 32/(64×256) ≈ 0.127 bits per param

Total: 4 + 0.127 ≈ 4.127 bits per param
```

### A.2.3 Memory Computation

```
Model: Qwen2.5-1.5B
Parameters: 1,500,000,000

FP16 (no quantization):
  Memory = 1.5B × 2 bytes = 3.0 GB

NF4 quantization:
  Memory = 1.5B × 0.5 bytes + overhead ≈ 0.8 GB

NF4 + LoRA adapters:
  LoRA params: 2.75M × 2 bytes (FP16) = 0.0055 GB
  Total ≈ 0.8 + 0.006 ≈ 0.806 GB

NF4 + LoRA + Optimizer:
  AdamW states: 2 × 2.75M × 4 bytes = 0.022 GB
  Total ≈ 0.83 GB
```

## A.3 Elastic Weight Consolidation — Complete Theory

### A.3.1 Bayesian Perspective

EWC derives from a Bayesian view of neural network training:

```
Given tasks A and B, the posterior after learning both:

log p(θ|D_A, D_B) = log p(D_B|θ) + log p(θ|D_A) - log p(D_B)
                   ≈ log p(D_B|θ) + log p(θ|D_A) + const.
```

The key: p(θ|D_A) is approximated as a Gaussian:

```
log p(θ|D_A) ≈ log p(θ*_A|D_A) - (1/2) Σᵢ Fᵢ(θᵢ - θ*_{A,i})²
```

where Fᵢ is the diagonal of the Fisher Information Matrix.

### A.3.2 Fisher Information Matrix

The Fisher Information Matrix measures the curvature of the log-likelihood:

```
Full Fisher:
  F = E_{x~p(x|θ)} [ ∇θ log p(x|θ) · (∇θ log p(x|θ))ᵀ ]

Diagonal approximation:
  F_ii = E_{x~p(x|θ)} [ (∂log p(x|θ)/∂θᵢ)² ]

Empirical approximation:
  F̂_ii = (1/N) Σₙ (∂L(xₙ, θ)/∂θᵢ)²
```

### A.3.3 EWC Loss Function

```
L_EWC(θ) = L_B(θ) + (λ/2) Σᵢ Fᵢ · (θᵢ - θ*_{A,i})²
```

**Gradient**:

```
∂L_EWC/∂θᵢ = ∂L_B/∂θᵢ + λ · Fᵢ · (θᵢ - θ*_{A,i})
```

### A.3.4 Online EWC (Schwarz et al., 2018)

For sequential tasks T₁, T₂, ..., Tₖ, the Fisher is accumulated:

```
After task k:
  F_accumulated = γ · F_{accumulated} + (1-γ) · F_k
  θ* = θ_k (current optimal)

where γ ∈ [0, 1] is the decay factor (typically 0.9)
```

This avoids storing separate Fisher matrices for each task.

### A.3.5 EWC for LoRA Parameters

In our setting, EWC is applied ONLY to LoRA parameters:

```
θ_LoRA = {B₁, A₁, B₂, A₂, ..., B_L, A_L}

F is computed for θ_LoRA only:
  Memory: 2 × |θ_LoRA| values

For Qwen2.5-1.5B:
  |θ_LoRA| = 2,752,512
  F storage: 2,752,512 × 4 bytes = 11 MB
  θ* storage: 2,752,512 × 4 bytes = 11 MB
  Total EWC overhead: ~22 MB (negligible)
```

## A.4 Loss Function Design

### A.4.1 Task Loss (Cross-Entropy)

```
L_task(θ) = -(1/B) Σᵢ₌₁ᴮ wᵢ · Σⱼ₌₁ᵀⁱ log P(yᵢⱼ | xᵢ, y_{i,<j}; θ)

where:
  B: batch size
  wᵢ: quality weight from evaluator for sample i
  T_i: length of answer i
  xᵢ: question (input context)
  yᵢⱼ: j-th token of answer i
  y_{i,<j}: all tokens before position j
```

### A.4.2 Total Loss

```
L_total(θ) = L_task(θ) + L_EWC(θ)

= -(1/B) Σᵢ wᵢ · Σⱼ log P(yᵢⱼ|...) + (λ/2) Σₖ Fₖ(θₖ - θ*ₖ)²
```

### A.4.3 Quality-Weighted Training

The evaluator score wᵢ ∈ [0, 1] modulates the loss contribution:

```
High score (wᵢ → 1): Full learning signal
  - Model strongly reinforces this answer pattern

Low score (wᵢ → 0): Reduced learning signal
  - Model weakly adjusts based on this example
  - Combined with corrective prompt information

Effective batch loss:
  L_eff = (Σᵢ wᵢ · Lᵢ) / (Σᵢ wᵢ)
```

## A.5 Optimization — AdamW with EWC

### A.5.1 Update Rule

```
Compute gradient:
  gₜ = ∇θ L_total(θ_{t-1})
     = ∇θ L_task + λ · F ⊙ (θ_{t-1} - θ*)

AdamW updates:
  mₜ = β₁ · m_{t-1} + (1-β₁) · gₜ           (momentum)
  vₜ = β₂ · v_{t-1} + (1-β₂) · gₜ²          (adaptive LR)
  m̂ₜ = mₜ / (1 - β₁ᵗ)                        (bias correction)
  v̂ₜ = vₜ / (1 - β₂ᵗ)                        (bias correction)

  θₜ = θ_{t-1} - η · [m̂ₜ/(√v̂ₜ + ε) + λ_wd · θ_{t-1}]

Hyperparameters:
  η = 2 × 10⁻⁴     (learning rate)
  β₁ = 0.9          (first moment decay)
  β₂ = 0.999        (second moment decay)
  ε = 10⁻⁸          (numerical stability)
  λ_wd = 0.01       (weight decay)
```

### A.5.2 Gradient Clipping

```
If ||gₜ|| > max_norm:
  gₜ ← gₜ × (max_norm / ||gₜ||)

max_norm = 1.0
```

### A.5.3 Learning Rate Schedule (Cosine Annealing)

```
η(t) = η_min + (η_max - η_min) × (1 + cos(π × t/T_max)) / 2

where:
  η_max = 2 × 10⁻⁴   (initial LR)
  η_min = 1 × 10⁻⁶   (final LR)
  t: current step
  T_max: total steps
```

## A.6 Catastrophic Forgetting — Quantitative Framework

### A.6.1 Forgetting Metric

```
Forgetting(topic_k, time_t) = max_{t'<t} Score(topic_k, t') - Score(topic_k, t)

Global Forgetting = max_k Forgetting(topic_k, t)
Average Forgetting = (1/K) Σ_k Forgetting(topic_k, t)
```

### A.6.2 Backward Transfer

```
BWT = (1/K) Σ_{k=1}^{K} [Score(k, final) - Score(k, after_k)]

Positive BWT: later training helped earlier topics
Negative BWT: later training hurt earlier topics (forgetting)
```

### A.6.3 Forward Transfer

```
FWT = (1/K) Σ_{k=2}^{K} [Score(k, before_k) - Score(k, random)]

Measures how much learning earlier topics helps with new topics.
```

## A.7 Information-Theoretic Analysis

### A.7.1 LoRA Capacity

```
Information capacity of LoRA adapter:
  Bits ≈ r × (d_out + d_in) × precision_bits

For r=8, d=1536, FP16:
  Bits = 8 × 3072 × 16 = 393,216 bits per layer
  Total (28 layers): 11,010,048 bits ≈ 1.3 MB

This represents the maximum "new knowledge" that
LoRA can encode per attention module.
```

### A.7.2 Trade-off: Plasticity vs. Stability

```
Plasticity: ability to learn new information
  ∝ learning_rate × LoRA_rank × (1 - EWC_lambda)

Stability: ability to retain old information
  ∝ (1 - learning_rate) × EWC_lambda × Fisher_strength

Optimal trade-off:
  λ* = argmin_λ [α · Forgetting(λ) + β · (1 - Accuracy(λ))]

Our choice: λ = 0.4 (empirically determined)
```
