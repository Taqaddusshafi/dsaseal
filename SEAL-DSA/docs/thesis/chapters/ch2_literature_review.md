# Chapter 2: Literature Review

## 2.1 Large Language Models for Education

### 2.1.1 Evolution of Educational AI

The application of AI in education has evolved from simple rule-based tutoring systems (1970s) to modern LLM-powered assistants. Key milestones include:

- **ELIZA (1966)**: Pattern-matching conversational agent
- **Carnegie Learning's Mathia (2000s)**: Adaptive math tutoring
- **Khan Academy's Khanmigo (2023)**: GPT-4 powered educational assistant
- **MIT SEAL (2025)**: Self-adapting framework for continuous improvement

### 2.1.2 Limitations of Static LLMs in Education

Current educational LLMs suffer from the "deploy and forget" paradigm:

1. **Knowledge Staleness**: Training data has a cutoff date
2. **No Personalization**: Cannot adapt to individual learning patterns
3. **Error Persistence**: Same mistakes repeated across sessions
4. **Domain Gaps**: General-purpose training may not cover specialized topics deeply

## 2.2 MIT CSAIL SEAL Framework (2025)

The SEAL (Self-Adapting Language model) framework, proposed by MIT CSAIL in 2025, introduces the concept of **autonomous self-improvement** for language models. The key contributions are:

### 2.2.1 Core Architecture

SEAL implements a four-stage learning loop:

1. **Self-Question Generation**: Model generates questions about its domain
2. **Self-Answer Attempt**: Model attempts to answer its own questions
3. **Self-Evaluation**: Model evaluates the quality of its answers
4. **Self-Update**: Model updates its parameters based on the evaluation

### 2.2.2 Key Theoretical Contributions

- Demonstrated that LLMs can generate meaningful training signals autonomously
- Showed that micro-updates can improve domain-specific performance without degrading general capabilities
- Introduced evaluation metrics for self-improvement loops

### 2.2.3 Our Simplifications

Our SEAL-DSA differs from the original in several important ways:

| Aspect         | MIT SEAL                   | SEAL-DSA (Ours)             |
| -------------- | -------------------------- | --------------------------- |
| Model Size     | 7B-70B                     | 1B-4B                       |
| Evaluator      | Model-based (LLM-as-judge) | Rule-based rubric           |
| Domain         | Multi-domain               | DSA only                    |
| Infrastructure | Multi-GPU                  | Colab Free (single T4)      |
| Fine-tuning    | Full / LoRA                | LoRA only (4-bit quantized) |
| Cost           | >$100                      | <$20                        |

## 2.3 Low-Rank Adaptation (LoRA)

### 2.3.1 Hu et al. (2022) — Original LoRA Paper

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer.

**Core Insight**: The weight updates during fine-tuning have a low "intrinsic rank," meaning they can be captured by low-rank matrices without significant loss of expressiveness.

**Mathematical Formulation**:

For a pre-trained weight matrix **W₀ ∈ ℝ^{d×k}**:

```
W = W₀ + ΔW = W₀ + BA
```

where:

- **B ∈ ℝ^{d×r}** (down-projection, initialized to zeros)
- **A ∈ ℝ^{r×k}** (up-projection, initialized with random Gaussian)
- **r << min(d, k)** is the rank

The modified forward pass:

```
h = W₀x + (α/r) · BAx
```

where α/r is the scaling factor.

**Parameter Savings**:

```
Full fine-tuning: d × k parameters per layer
LoRA: r × (d + k) parameters per layer

Example (d=k=4096, r=8):
  Full: 16,777,216 parameters
  LoRA: 65,536 parameters (0.39% of full)
```

### 2.3.2 QLoRA (Dettmers et al., 2023)

QLoRA extends LoRA by quantizing the base model to 4-bit precision, enabling fine-tuning of larger models on smaller GPUs.

**Key Innovations**:

1. **4-bit NormalFloat (NF4)**: A quantization scheme optimized for normally distributed weights
2. **Double Quantization**: Quantizing the quantization constants for further memory savings
3. **Paged Optimizers**: GPU memory management for handling memory spikes

**Memory Savings**:

```
Model: 7B parameters
FP16: ~14GB
NF4 (QLoRA): ~3.5GB (75% reduction)
```

For our 1.5B model:

```
FP16: ~3GB
NF4: ~0.8GB
With LoRA adapters: ~1GB total during training
```

### 2.3.3 Why LoRA Instead of Full Fine-tuning

| Criterion          | Full Fine-tuning | LoRA                |
| ------------------ | ---------------- | ------------------- |
| Parameters updated | 100%             | 0.1-0.5%            |
| GPU memory         | Very high        | Low                 |
| Training time      | Hours            | Minutes             |
| Risk of forgetting | High             | Low                 |
| Modular            | No               | Yes (swap adapters) |
| Colab compatible   | No (for >1B)     | Yes                 |

## 2.4 Self-Rewarding Language Models (Yuan et al., 2024)

This paper demonstrates that LLMs can evaluate their own outputs and use these evaluations as training signals:

- **LLM-as-a-Judge**: Model rates its own responses on multiple dimensions
- **Iterative DPO**: Direct Preference Optimization using self-generated preferences
- **Improvement Trajectory**: Models show consistent improvement across iterations

**Relevance to SEAL-DSA**: Our evaluation module draws from this concept, though we use rule-based evaluation instead of model-based evaluation due to the small model size.

## 2.5 Continual Learning and Catastrophic Forgetting

### 2.5.1 The Problem

When a neural network is trained sequentially on tasks A, then B, it tends to "forget" task A. This is known as catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999).

### 2.5.2 Elastic Weight Consolidation (EWC) — Kirkpatrick et al. (2017)

EWC prevents catastrophic forgetting by identifying which parameters were important for previous tasks and penalizing changes to them.

**Mathematical Foundation**:

The loss function with EWC regularization:

```
L_total = L_task(θ) + (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²
```

where:

- **L_task(θ)**: Loss on the current task
- **Fᵢ**: Diagonal of the Fisher Information Matrix (parameter importance)
- **θ\*ᵢ**: Optimal parameters from previous task
- **λ**: Regularization strength

**Fisher Information Matrix**:

```
Fᵢ = E[( ∂log p(x|θ) / ∂θᵢ )²]
```

The Fisher diagonal measures the expected squared gradient. Parameters with large Fisher values had large gradients during training on the previous task, indicating high importance.

**Practical Approximation**:

```
F̂ᵢ ≈ (1/N) Σₙ (∂Lₙ / ∂θᵢ)²
```

### 2.5.3 Other Continual Learning Methods

| Method                   | Type           | Key Idea                             |
| ------------------------ | -------------- | ------------------------------------ |
| EWC (2017)               | Regularization | Penalize changes to important params |
| SI (2017)                | Regularization | Track importance during training     |
| PackNet (2018)           | Architecture   | Prune and freeze for each task       |
| Progressive Nets (2016)  | Architecture   | Add new columns per task             |
| Experience Replay (2019) | Replay         | Store and replay old data            |
| GEM (2017)               | Gradient       | Constrain gradients                  |

**Our Choice**: We use EWC because:

1. It is memory-efficient (stores only 2× LoRA parameters)
2. It does not require storing training data (important for Colab)
3. It is well-studied with strong theoretical foundations
4. It integrates naturally with the LoRA update mechanism

## 2.6 Continuous Alignment (2024)

Recent work on continuous alignment extends RLHF to ongoing scenarios:

- Models can be aligned using streaming feedback
- Importance weighting prevents forgetting of alignment
- Online DPO enables continuous preference learning

**Relevance**: SEAL-DSA's continuous learning loop is analogous to continuous alignment, where the "human feedback" is replaced by automated evaluation signals.

## 2.7 Curriculum Learning (Bengio et al., 2009)

Curriculum learning proposes training models on examples ordered from easy to hard, mimicking human learning:

- **Key Finding**: Training on easier examples first leads to better convergence
- **Theoretical Basis**: Smoother optimization landscape in early training
- **Application to SEAL-DSA**: Our 16-week curriculum introduces DSA topics progressively, from foundational (arrays) to advanced (dynamic programming)

## 2.8 Summary of Literature

```
┌──────────────────────────────────────────────────────────────┐
│              Theoretical Foundation Map                       │
│                                                               │
│  MIT SEAL (2025)──────────▶ Self-improving loop design       │
│         │                                                     │
│  LoRA (2022) ─────────────▶ Parameter-efficient updates      │
│         │                                                     │
│  QLoRA (2023) ────────────▶ 4-bit quantization for Colab     │
│         │                                                     │
│  Self-Rewarding LM (2024)─▶ Self-evaluation methodology     │
│         │                                                     │
│  EWC (2017) ──────────────▶ Forgetting prevention            │
│         │                                                     │
│  Curriculum Learning (2009)▶ Progressive topic introduction  │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │            SEAL-DSA (This Thesis)                     │    │
│  │  Combines all above for educational AI on Colab      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```
