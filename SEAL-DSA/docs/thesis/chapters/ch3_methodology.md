# Chapter 3: Methodology

## 3.1 System Architecture

The SEAL-DSA system consists of six interconnected modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SEAL-DSA Architecture                        │
│                                                                  │
│  ╔══════════════════════════════════════════════════════════╗    │
│  ║                  SEAL Core Loop                          ║    │
│  ║                                                          ║    │
│  ║  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  ║    │
│  ║  │   Question    │───▶│   Answer     │───▶│ Evaluator │  ║    │
│  ║  │  Generator    │    │  Generator   │    │  Module   │  ║    │
│  ║  │              │    │              │    │           │  ║    │
│  ║  │  Generates    │    │ Attempts to  │    │ Scores on │  ║    │
│  ║  │  DSA questions│    │ answer using │    │ 5 rubric  │  ║    │
│  ║  │  for current  │    │ current model│    │ dimensions│  ║    │
│  ║  │  topic        │    │ weights      │    │           │  ║    │
│  ║  └──────────────┘    └──────────────┘    └─────┬─────┘  ║    │
│  ║         ▲                                       │        ║    │
│  ║         │            ┌──────────────┐           │        ║    │
│  ║         └────────────│  Parameter   │◀──────────┘        ║    │
│  ║                      │   Updater    │                    ║    │
│  ║                      │  (LoRA)      │                    ║    │
│  ║                      │              │                    ║    │
│  ║                      │ Updates only  │                    ║    │
│  ║                      │ low-rank     │                    ║    │
│  ║                      │ matrices     │                    ║    │
│  ║                      └──────────────┘                    ║    │
│  ╚══════════════════════════════════════════════════════════╝    │
│                                                                  │
│  ╔════════════════════════════════════════════════════════╗      │
│  ║              Support Systems                           ║      │
│  ║  ┌──────────────┐  ┌──────────┐  ┌────────────────┐  ║      │
│  ║  │  Curriculum   │  │ EWC      │  │  Checkpoint    │  ║      │
│  ║  │  Scheduler    │  │ Module   │  │  Manager       │  ║      │
│  ║  │              │  │          │  │                │  ║      │
│  ║  │ Controls     │  │ Prevents │  │ Saves/loads    │  ║      │
│  ║  │ topic order  │  │ forgetting│  │ model state    │  ║      │
│  ║  └──────────────┘  └──────────┘  └────────────────┘  ║      │
│  ╚════════════════════════════════════════════════════════╝      │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 Module Descriptions

### 3.2.1 Question Generator Module

**Purpose**: Self-generate DSA questions that serve as training data for the learning loop.

**Design Rationale**: By generating its own questions, the model creates a self-supervised training signal. The diversity and quality of generated questions directly impact learning effectiveness.

**Process**:

1. Receive current topic from Curriculum Scheduler
2. Construct prompt using topic-specific templates
3. Generate questions using the LLM with temperature=0.8 (for diversity)
4. Filter questions through quality checks
5. Categorize by difficulty and type

**Quality Checks**:

- Minimum length threshold (20+ characters)
- Contains question indicators (?, "implement", "explain", etc.)
- No excessive word repetition (unique word ratio > 0.3)
- DSA-relevant content detected

### 3.2.2 Answer Generator Module

**Purpose**: Generate answers to the self-generated questions using the current model weights.

**Key Design Decision**: The SAME model generates both questions and answers. This is critical because:

- The answers reflect the model's CURRENT knowledge level
- Evaluation of these answers reveals gaps in knowledge
- The learning signal comes from the difference between current ability and ideal performance

**Process**:

1. Receive question from Question Generator
2. Construct answer prompt with chain-of-thought reasoning
3. Generate answer with temperature=0.3 (for focused, deterministic responses)
4. Estimate confidence based on generation characteristics
5. Package as (question, answer) pair for evaluation

### 3.2.3 Evaluator Module (Rule-Based)

**Purpose**: Score generated answers to produce training signals.

**Why Rule-Based Instead of Model-Based**:
Small models (1-4B) are unreliable self-evaluators. A rule-based evaluator provides:

- Deterministic, reproducible scores
- No additional GPU memory overhead
- Interpretable feedback
- Validated against known-correct patterns

**Evaluation Dimensions** (5 dimensions, weighted scoring):

```
Score = 0.35 × Correctness + 0.25 × Completeness +
        0.20 × Complexity  + 0.10 × Code + 0.10 × Explanation
```

| Dimension           | Weight | What It Measures                               |
| ------------------- | ------ | ---------------------------------------------- |
| Correctness         | 0.35   | Presence of correct DSA concepts/keywords      |
| Completeness        | 0.25   | Coverage of question requirements              |
| Complexity Analysis | 0.20   | Big-O notation and complexity reasoning        |
| Code Quality        | 0.10   | Code presence and basic syntax (for coding Qs) |
| Explanation Quality | 0.10   | Structure, examples, reasoning flow            |

### 3.2.4 Parameter Updater Module (LoRA)

**Purpose**: Perform micro-updates to the model's LoRA adapter weights based on evaluation results.

**Training Signal Construction**:

For evaluated (question, answer) pairs:

1. **High-score answers** (score > threshold):
   - Used as positive training examples
   - The model reinforces these response patterns
   - Weight in loss = score value

2. **Low-score answers** (score < threshold):
   - Used as corrective examples
   - Paired with evaluation feedback
   - Lower weight in loss to prevent overwriting

**Loss Function**:

```
L_total = L_task + λ_ewc · L_ewc

L_task = -(1/B) Σᵢ wᵢ · Σⱼ log P(yᵢⱼ | x_i, y_{i,<j}; θ)

L_ewc = (λ/2) Σₖ Fₖ · (θₖ - θ*ₖ)²
```

where:

- B: batch size
- wᵢ: quality weight from evaluator for sample i
- P(yᵢⱼ|...): probability of token j in answer i
- Fₖ: Fisher Information diagonal for parameter k
- θ\*ₖ: optimal parameters from previous topic

## 3.3 Mathematical Framework

### 3.3.1 LoRA Decomposition

For each target attention module with pre-trained weight **W₀ ∈ ℝ^{d_out × d_in}**:

**Forward Pass**:

```
h = (W₀ + ΔW)x = W₀x + (α/r) · BAx
```

**Parameter Matrices**:

```
B ∈ ℝ^{d_out × r}    (initialized to zeros)
A ∈ ℝ^{r × d_in}     (initialized ~ N(0, σ²))
```

**Initialization Rationale**:

- B = 0 ensures ΔW = 0 at start → model begins with pre-trained behavior
- A is random to break symmetry in the gradient update

**Scaling Factor**: α/r where typically α = 2r

**Gradient Computation**:

```
∂L/∂B = (α/r) · (∂L/∂h) · (Ax)ᵀ
∂L/∂A = (α/r) · Bᵀ · (∂L/∂h) · xᵀ
```

### 3.3.2 Quantization (QLoRA)

Base model weights are stored in 4-bit NF4 format:

**Normal Float Quantization**:

```
QNF4(w) = argminᵢ |w - qᵢ|
```

where q₁, ..., q₁₆ are 16 quantization levels optimized for N(0, σ²) distributions.

**De-quantization during forward pass**:

```
w_fp16 = scale × QNF4⁻¹(w_nf4) + zero_point
```

**Double Quantization**: The scale factors themselves are quantized in FP8:

```
Memory per param ≈ 4 bits + 8/block_size bits
```

### 3.3.3 EWC Regularization

**Fisher Information Matrix** (diagonal approximation):

```
Fᵢ = E_{x~D}[( ∂/∂θᵢ log p(x|θ) )²]

Approximated as:
F̂ᵢ = (1/N) Σₙ₌₁ᴺ (∂Lₙ/∂θᵢ)²
```

**EWC Loss**:

```
L_EWC(θ) = (λ/2) Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²
```

**Gradient of EWC Loss**:

```
∂L_EWC/∂θᵢ = λ · Fᵢ · (θᵢ - θ*ᵢ)
```

**Online EWC Update** (Schwarz et al., 2018):

```
F_new = γ · F_old + (1-γ) · F_current
```

where γ = 0.9 is the decay factor.

### 3.3.4 Total Optimization Objective

```
θ* = argmin_θ [ L_task(θ) + (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)² ]
```

Subject to constraints:

- Only LoRA parameters θ_LoRA are updated
- Base parameters θ₀ are frozen
- Memory budget ≤ 15GB (T4 GPU)

**Update Rule** (AdamW with EWC):

```
mₜ = β₁ · m_{t-1} + (1-β₁) · gₜ        (first moment)
vₜ = β₂ · v_{t-1} + (1-β₂) · gₜ²       (second moment)
m̂ₜ = mₜ / (1-β₁ᵗ)                       (bias correction)
v̂ₜ = vₜ / (1-β₂ᵗ)                       (bias correction)
θₜ = θ_{t-1} - η · (m̂ₜ/(√v̂ₜ + ε) + λ_wd · θ_{t-1})

where gₜ = ∂L_task/∂θ + λ_ewc · Fᵢ · (θᵢ - θ*ᵢ)
```

## 3.4 Curriculum Learning Design

### 3.4.1 Topic Ordering

The 16-week curriculum follows a prerequisite-based ordering:

```
Week 1-2:   Arrays & Strings ──────────┐
                                        │
Week 3-4:   Linked Lists ──────────────┤
                                        │
Week 5-6:   Stacks & Queues ───────────┤
                                        ├──▶ Foundation
Week 7-8:   Trees ─────────────────────┤
                                        │
Week 9-10:  Graphs ────────────────────┤
                                        │
Week 11-12: Sorting & Searching ───────┤
                                        ├──▶ Advanced
Week 13-14: Dynamic Programming ───────┤
                                        │
Week 15-16: Advanced & Review ─────────┘
```

### 3.4.2 Scheduling Strategies

1. **Progressive**: Introduce one new topic per epoch, always reviewing previous
2. **Adaptive**: Focus on weakest topics based on evaluation scores
3. **Random**: Baseline comparison with random topic selection

### 3.4.3 Anti-Forgetting Review

When the Forgetting Detector identifies score degradation > 5% on a previously learned topic, that topic is automatically added to the current epoch's training set.

## 3.5 Data Flow

Complete data flow through one SEAL iteration:

```
Topic="Binary Trees"
      │
      ▼
[Question Generator]
      │
      ▼
Questions = [
  "What is the time complexity of BST search?",
  "Implement inorder traversal recursively.",
  "Find the lowest common ancestor in a BST.",
  ...
]
      │
      ▼
[Answer Generator]
      │
      ▼
Answers = [
  ("What is...", "The time complexity is O(h) where h..."),
  ("Implement...", "def inorder(root): ..."),
  ...
]
      │
      ▼
[Evaluator]
      │
      ▼
Evaluations = [
  (Q1, A1, score=0.72, feedback="Good concept coverage..."),
  (Q2, A2, score=0.55, feedback="Add complexity analysis..."),
  ...
]
      │
      ▼
[Parameter Updater]
      │
      ▼
LoRA weights updated:
  B_new = B_old - η · ∇_B (L_task + L_ewc)
  A_new = A_old - η · ∇_A (L_task + L_ewc)
      │
      ▼
Model improved for "Binary Trees" topic
```

## 3.6 Evaluation Methodology

### 3.6.1 Metrics

1. **DSA Accuracy**: Percentage of answers scoring above threshold on held-out test set
2. **Quality Score**: Average multi-dimensional rubric score (0-1)
3. **Forgetting Rate**: max(best_score - current_score) for previous topics
4. **Training Efficiency**: GPU hours and cost per percentage point improvement
5. **Inference Speed**: Tokens per second during answer generation

### 3.6.2 Experimental Design

```
Experiment: SEAL-DSA Evaluation
────────────────────────────────
Independent Variable: Training method
  - Static (no training)
  - Traditional fine-tuning
  - RAG baseline
  - SEAL-DSA (proposed)

Dependent Variables:
  - DSA accuracy
  - Forgetting rate
  - Training cost

Control Variables:
  - Same base model (Qwen2.5-1.5B)
  - Same test set (140 questions)
  - Same evaluation rubric

Protocol:
  1. Evaluate baseline (static model)
  2. Run SEAL for 5 epochs
  3. Evaluate after each epoch
  4. Compare against baselines
```
