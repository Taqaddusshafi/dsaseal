# Appendix B: Potential Viva Questions and Answers

## Category 1: Fundamental Understanding

### Q1: Explain the SEAL framework in one sentence.

**A**: SEAL (Self-Adapting Language model) is an autonomous learning loop where a language model continuously improves by generating its own questions, attempting answers, evaluating quality, and performing micro-parameter updates using LoRA.

### Q2: Why did you choose LoRA over full fine-tuning?

**A**: Three key reasons:

1. **Memory**: Full fine-tuning of even a 1.5B model requires ~6GB VRAM for optimizer states alone. LoRA only updates ~0.18% of parameters, needing ~30MB.
2. **Forgetting**: LoRA keeps base weights frozen, providing a first line of defense against catastrophic forgetting. Full fine-tuning would modify all parameters.
3. **Colab compatibility**: Google Colab Free Tier provides a T4 GPU with 15GB VRAM. With 4-bit quantization + LoRA, total training memory is ~5GB, well within limits.

### Q3: What is the mathematical formulation of LoRA?

**A**: For a pre-trained weight matrix W₀ ∈ ℝ^{d×k}:

- W = W₀ + (α/r) · B·A where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}
- B is initialized to zeros, A to random Gaussian
- Only B and A are trained; W₀ is frozen
- Parameter savings: from d×k to r×(d+k), with r << min(d,k)

### Q4: How does EWC prevent catastrophic forgetting?

**A**: EWC adds a quadratic penalty to the loss: L_EWC = (λ/2) Σᵢ Fᵢ(θᵢ - θ\*ᵢ)². The Fisher Information Matrix diagonal Fᵢ measures how important each parameter was for the previous task. High Fisher values → parameter is important → changes are penalized heavily. This preserves old knowledge while allowing new learning.

### Q5: What is the Fisher Information Matrix and how do you compute it?

**A**: The Fisher Information Matrix measures the curvature of the log-likelihood landscape. The diagonal entry Fᵢ = E[(∂log p(x|θ)/∂θᵢ)²]. In practice, we approximate it empirically as F̂ᵢ = (1/N) Σₙ (∂Lₙ/∂θᵢ)² using N samples. We use self-supervised samples (the model's own prompts) to compute gradients, making it tractable on Colab.

## Category 2: Architecture and Design

### Q6: Why rule-based evaluation instead of LLM-as-a-judge?

**A**: Small models (1-4B) are unreliable self-evaluators. Our experiments with 1.5B models showed inconsistent and often incorrect self-ratings. Rule-based evaluation is:

- Deterministic and reproducible
- Requires no additional GPU memory
- Can be validated against human judgments
- Provides interpretable feedback through specific rubric dimensions

This is acknowledged as a project limitation and could be addressed with larger models.

### Q7: Explain your five evaluation dimensions.

**A**:

1. **Correctness (35%)**: Presence of relevant DSA concepts and algorithm keywords
2. **Completeness (25%)**: Coverage of all parts of the question
3. **Complexity Analysis (20%)**: Big-O notation and time/space analysis
4. **Code Quality (10%)**: Syntactic correctness and function structure
5. **Explanation Quality (10%)**: Structured reasoning with examples

The weights reflect the relative importance in DSA education — correctness is most critical, while code quality has lower weight since not all questions require code.

### Q8: How does your curriculum learning approach work?

**A**: Following Bengio et al. (2009), topics are ordered from foundational to advanced:

- Arrays → Linked Lists → Stacks → Trees → Graphs → Sorting → DP
- Each epoch introduces one new topic while reviewing all previous ones
- The adaptive scheduler can re-prioritize weak topics based on evaluation scores
- If forgetting > 5% on any topic, it's automatically added to the review schedule

### Q9: What is your training signal? How do you get supervision without labeled data?

**A**: The training signal comes from the evaluator's scores. High-scoring answers (above threshold) become positive training examples—the model learns to reproduce answers of this quality. Low-scoring answers are paired with corrective feedback. The weighted cross-entropy loss uses the evaluation score as the sample weight, creating a quality-modulated learning signal.

### Q10: How do you handle Colab session timeouts?

**A**: Three mechanisms:

1. **Checkpoint-and-resume**: Save LoRA adapter (~15MB) and optimizer state to Google Drive after each epoch
2. **Resume protocol**: On reconnection, reload base model from HuggingFace, load LoRA adapter from Drive, and continue from the saved epoch
3. **Efficient storage**: Only LoRA adapters are saved, not the full model (base model is reloaded from HuggingFace)

## Category 3: Technical Depth

### Q11: Explain the complete loss function.

**A**:

```
L_total = L_task + L_EWC

L_task = -(1/B) Σᵢ wᵢ · Σⱼ log P(yᵢⱼ | xᵢ, y_{i,<j}; θ)
L_EWC = (λ/2) Σₖ Fₖ · (θₖ - θ*ₖ)²
```

- L_task: Quality-weighted cross-entropy over generated answers
- wᵢ: Evaluation score for answer i (higher score = larger gradient)
- L_EWC: Fisher-weighted quadratic penalty around optimal parameters
- Only LoRA parameters θ_LoRA are updated via backpropagation

### Q12: What is QLoRA and why do you use it?

**A**: QLoRA (Dettmers et al., 2023) quantizes base model weights to 4-bit NormalFloat format while keeping LoRA adapters in FP16. This reduces model memory from ~3GB (FP16) to ~0.8GB (4-bit) for our 1.5B model. NF4 is optimized for normally distributed weights (which neural network weights approximately follow). Double quantization further compresses the scaling factors from FP32 to FP8.

### Q13: What optimizer do you use and why?

**A**: AdamW with the following hyperparameters:

- lr=2×10⁻⁴, β₁=0.9, β₂=0.999, ε=10⁻⁸
- Weight decay=0.01 (decoupled from gradient updates)
- Cosine annealing schedule from 2×10⁻⁴ to 10⁻⁶
- Gradient clipping at norm=1.0

AdamW is preferred because:

1. Adaptive learning rates handle varying gradient magnitudes
2. Decoupled weight decay is more principled than L2 regularization
3. Well-suited for transformer fine-tuning (standard in the field)

### Q14: How many trainable parameters does your model have?

**A**: For Qwen2.5-1.5B with LoRA (r=8, target=[q,k,v,o]\_proj):

- Parameters per LoRA module: 2 × 1536 × 8 = 24,576
- Modules per layer: 4
- Layers: 28
- Total LoRA parameters: 28 × 4 × 24,576 = 2,752,512 (~2.75M)
- Base model parameters: ~1,500,000,000 (frozen)
- Trainable percentage: 0.18%

### Q15: What is gradient accumulation and why do you use it?

**A**: Gradient accumulation computes gradients over multiple micro-batches before updating weights. With batch_size=4 and accumulation_steps=4, the effective batch size is 16. This is necessary because:

1. Colab's limited VRAM constrains the micro-batch size
2. Larger effective batches produce more stable gradients
3. The optimizer steps only once per 4 micro-batches, saving memory

## Category 4: Results and Analysis

### Q16: What improvement do you expect and why?

**A**: We target 15-25% accuracy improvement (from ~35% to ~50-55%). This is based on:

1. LoRA fine-tuning studies showing 10-30% improvement on domain-specific tasks
2. The self-improvement loop provides diverse training data across multiple rounds
3. EWC ensures improvements don't come at the cost of forgetting
4. The 1.5B model has significant room for improvement on specialized DSA content

### Q17: How do you measure catastrophic forgetting?

**A**: Forgetting rate for topic k at time t:

```
F(k,t) = max_{t'<t} Score(k,t') - Score(k,t)
```

We use quick-check evaluation with canned prompts and keyword matching. If F(k,t) > 5%, the topic is flagged for review. Our target is max forgetting < 5% across all topics.

### Q18: How do you ensure statistical validity of your results?

**A**:

1. Fixed held-out test set (140 questions) not used during training
2. Three independent runs with different random seeds
3. Report mean ± standard deviation
4. Paired t-test for significance (p < 0.05 threshold)
5. Ablation studies isolating the effect of each component (LoRA rank, EWC λ, curriculum strategy)

### Q19: What are the main limitations of your evaluation approach?

**A**:

1. **Rule-based evaluation**: Cannot assess deep semantic correctness (keywords ≠ understanding)
2. **No human evaluation**: True educational impact not measured
3. **Small test set**: 140 questions may not cover all edge cases
4. **Synthetic setup**: Real student interaction not tested
5. **Single trial variance**: Individual runs may vary due to stochastic generation

## Category 5: Broader Impact and Future Work

### Q20: How would you scale this to larger models?

**A**:

1. **Colab Pro** (A100, 40GB): Could run 7B models with 4-bit quantization
2. **Multiple GPUs**: Model parallelism for 13B+ models
3. **QLoRA improvements**: AQLM, GPTQ for even more compression
4. **Model-based evaluation**: Larger models can self-evaluate more reliably
5. **Framework remains the same**: SEAL loop architecture scales naturally

### Q21: How does your work differ from RAG (Retrieval-Augmented Generation)?

**A**:
| Aspect | RAG | SEAL-DSA |
|--------|-----|----------|
| Knowledge source | External documents | Model's own parameters |
| Improvement | Static retrieval quality | Continuously improving model |
| Latency | Higher (retrieval step) | Lower (no retrieval) |
| Storage | Requires document index | Only LoRA adapters |
| Adaptability | Limited by document quality | Self-improving |
| Forgetting | No forgetting (external docs) | Managed via EWC |

### Q22: What are the ethical considerations of self-improving AI?

**A**:

1. **Accuracy drift**: Self-generated training data may contain errors that compound over time ("hallucination amplification")
2. **Evaluation quality**: Rule-based evaluation may miss subtle errors
3. **Bias amplification**: The model may reinforce its own biases
4. **Transparency**: Students should know they're interacting with a self-improved model
5. **Mitigation**: Regular human evaluation checkpoints, diverse evaluation rubrics, and forgetting detection act as safeguards

### Q23: Could this framework be applied to other domains?

**A**: Yes, with modifications:

1. **Change topic definitions**: Replace DSA_TOPICS with domain-specific topics
2. **Update evaluator**: New keyword dictionaries and rubric dimensions
3. **Adjust curriculum**: Domain-specific prerequisite ordering
4. **Potential domains**: Mathematics, Physics, Operating Systems, Databases, Programming Languages

### Q24: What would you do differently if you had unlimited resources?

**A**:

1. **Model**: Use 70B parameter model (LLaMA-2-70B or better)
2. **Evaluator**: Use a separate large model as evaluator (LLM-as-a-judge)
3. **Training**: Full SEAL loop with model-based self-evaluation
4. **Evaluation**: Large-scale human evaluation study with actual students
5. **Multi-domain**: Test across 10+ academic domains
6. **Deployment**: Production-ready system with real-time learning from student interactions

### Q25: Explain the connection between your work and the original MIT SEAL paper.

**A**: The MIT SEAL paper (2025) introduces the concept of self-adapting language models that improve through autonomous interaction loops. Our work:

- **Inspires from**: The four-stage loop (generate, attempt, evaluate, update)
- **Simplifies**: Uses smaller models (1.5B vs 7B-70B)
- **Adapts**: Rule-based evaluation instead of model-based
- **Focuses**: Single domain (DSA) instead of multi-domain
- **Democratizes**: Runs on free infrastructure (Colab)
- **Extends**: Adds curriculum learning and EWC, which the original paper does not emphasize

This trade-off between complexity and accessibility is the core contribution—we show that the SEAL concept works even with significant resource constraints.
