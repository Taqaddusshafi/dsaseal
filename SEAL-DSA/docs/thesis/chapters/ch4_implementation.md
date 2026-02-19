# Chapter 4: Implementation

## 4.1 Technology Stack

| Component      | Technology               | Version   | Purpose                    |
| -------------- | ------------------------ | --------- | -------------------------- |
| Language       | Python                   | 3.10+     | Core implementation        |
| ML Framework   | PyTorch                  | 2.1+      | Training and inference     |
| Model Library  | HuggingFace Transformers | 4.36+     | Model loading              |
| Fine-tuning    | PEFT (LoRA)              | 0.7+      | Parameter-efficient tuning |
| Quantization   | bitsandbytes             | 0.41+     | 4-bit quantization         |
| Infrastructure | Google Colab             | Free Tier | T4 GPU                     |
| Storage        | Google Drive             | Free      | Checkpoint persistence     |

## 4.2 Model Selection

### 4.2.1 Evaluation of Candidate Models

| Model                 | Params | VRAM (4-bit) | Quality   | Colab Compatible |
| --------------------- | ------ | ------------ | --------- | ---------------- |
| TinyLlama-1.1B        | 1.1B   | ~2GB         | Basic     | ✅ Excellent     |
| Qwen2.5-1.5B-Instruct | 1.5B   | ~3GB         | Good      | ✅ Recommended   |
| Microsoft Phi-2       | 2.7B   | ~4GB         | Very Good | ✅ Good          |
| Qwen2.5-3B-Instruct   | 3B     | ~4.5GB       | Very Good | ⚠️ Tight         |

### 4.2.2 Recommended: Qwen2.5-1.5B-Instruct

Selected for the following reasons:

1. **Instruction-tuned**: Already follows instructions well
2. **Size**: Fits comfortably in T4 VRAM with 4-bit quantization
3. **Performance**: Best quality-to-size ratio for instruction following
4. **Community**: Well-supported with extensive documentation
5. **License**: Apache 2.0, suitable for research

## 4.3 LoRA Configuration Details

```python
# Optimal LoRA configuration for SEAL-DSA
lora_config = LoraConfig(
    r=8,                    # Rank: 8 is a good balance
    lora_alpha=16,          # Scaling: alpha/r = 2
    lora_dropout=0.05,      # Light dropout
    bias="none",            # Don't train biases
    task_type="CAUSAL_LM",  # Autoregressive task
    target_modules=[
        "q_proj",           # Query projection
        "k_proj",           # Key projection
        "v_proj",           # Value projection
        "o_proj",           # Output projection
    ],
)
```

**Parameter Budget**:

```
For Qwen2.5-1.5B with hidden_size=1536, num_layers=28:

LoRA parameters per module = 2 × 1536 × 8 = 24,576
Modules per layer = 4 (q, k, v, o)
Parameters per layer = 4 × 24,576 = 98,304
Total LoRA parameters = 28 × 98,304 = 2,752,512

Total model parameters: ~1,500,000,000
LoRA percentage: 2,752,512 / 1,500,000,000 = 0.18%
```

## 4.4 Training Configuration

```python
# Training hyperparameters
training_config = {
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "num_epochs": 5,
    "questions_per_topic": 20,
    "warmup_steps": 10,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "mixed_precision": "fp16",
}
```

## 4.5 Google Colab Setup

### 4.5.1 Environment Setup

```python
# Cell 1: Install dependencies
!pip install -q torch transformers peft accelerate bitsandbytes
!pip install -q datasets evaluate rouge-score wandb

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Clone repository
!git clone https://github.com/yourusername/SEAL-DSA.git
%cd SEAL-DSA

# Cell 4: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 4.5.2 Handling Colab Limitations

| Limitation          | Solution                                  |
| ------------------- | ----------------------------------------- |
| 12h session limit   | Save checkpoints to Drive every epoch     |
| ~12.7GB RAM         | Use 4-bit quantization + small batch size |
| Session disconnects | Auto-save + resume from checkpoint        |
| Limited disk space  | Store only LoRA adapters (~15MB each)     |

### 4.5.3 Memory Budget

```
Component                     Memory
───────────────────────────────────────
Base model (4-bit quantized)  ~0.8 GB
LoRA adapters                 ~0.01 GB
Optimizer states (AdamW)      ~0.04 GB
Gradient buffers              ~0.04 GB
KV Cache (inference)          ~1-2 GB
Tokenized batch               ~0.1 GB
EWC Fisher + θ*               ~0.02 GB
PyTorch overhead              ~1-2 GB
───────────────────────────────────────
Total estimated               ~3-5 GB
T4 Available                  15 GB
Headroom                     ~10 GB ✅
```

## 4.6 Code Structure

### 4.6.1 Entry Point (`main.py`)

```python
# Pseudo-code for the main training script
def main():
    config = load_config("configs/default.yaml")
    model, tokenizer = load_model_with_lora(config)

    question_gen = QuestionGenerator(model, tokenizer)
    answer_gen = AnswerGenerator(model, tokenizer)
    evaluator = DSAEvaluator()
    updater = ParameterUpdater(model, config)
    curriculum = CurriculumScheduler(config)
    ewc = EWC(model, config)

    for epoch in range(num_epochs):
        topics = curriculum.get_topics(epoch)

        for topic in topics:
            # SEAL Loop
            questions = question_gen.generate(topic, n=20)
            answers = answer_gen.answer(questions)
            evaluations = evaluator.evaluate(answers)
            updater.update(evaluations, ewc_loss=ewc.loss)

        ewc.update_fisher(model)
        checkpoint.save(model, epoch)
```

### 4.6.2 Training Loop Pseudo-code

```
Algorithm: SEAL Training Loop
══════════════════════════════

Input: Base model M, Curriculum C, Config K
Output: Fine-tuned model M*

1.  M ← LoadModel(K.model_name) with LoRA(r=8, α=16)
2.  EWC ← Initialize()
3.
4.  FOR epoch = 1 to K.num_epochs:
5.    topics ← C.get_topics(epoch)
6.
7.    FOR each topic T in topics:
8.      // Step 1: Generate Questions
9.      Q ← GenerateQuestions(M, T, n=20, temp=0.8)
10.     Q ← FilterByQuality(Q)
11.
12.     // Step 2: Generate Answers
13.     A ← []
14.     FOR each q in Q:
15.       a ← GenerateAnswer(M, q, temp=0.3)
16.       A.append((q, a))
17.
18.     // Step 3: Evaluate
19.     E ← []
20.     FOR each (q, a) in A:
21.       score ← Evaluate(a, rubric)
22.       E.append((q, a, score))
23.
24.     // Step 4: Update Parameters
25.     L_task ← ComputeLoss(M, E)
26.     L_ewc ← EWC.compute_loss(M)
27.     L_total ← L_task + λ · L_ewc
28.
29.     // Backpropagate through LoRA only
30.     θ_LoRA ← θ_LoRA - η · ∇_LoRA(L_total)
31.
32.   // Update Fisher for forgetting prevention
33.   EWC.update_fisher(M)
34.
35.   // Check for forgetting
36.   forgetting ← CheckForgetting(M, previous_topics)
37.   IF forgetting > 5%: add_to_review(affected_topics)
38.
39.   // Checkpoint
40.   SaveCheckpoint(M, epoch)
41.
42. RETURN M
```

### 4.6.3 Evaluation Loop Pseudo-code

```
Algorithm: Evaluation Protocol
══════════════════════════════

Input: Model M, Test set S (140 questions)
Output: Metrics dictionary

1.  results ← {}
2.
3.  FOR each topic T in S.topics:
4.    scores_T ← []
5.
6.    FOR each question q in S[T]:
7.      answer ← M.generate(q, temp=0.3, greedy=True)
8.      score ← Evaluate(answer, rubric)
9.      scores_T.append(score)
10.
11.   results[T] = {
12.     "accuracy": mean(scores_T > threshold),
13.     "avg_score": mean(scores_T),
14.     "best": max(scores_T),
15.     "worst": min(scores_T),
16.   }
17.
18. // Compute global metrics
19. results["overall"] = {
20.   "accuracy": mean(all_scores > threshold),
21.   "forgetting": max(best_historical - current, per topic),
22.   "improvement": final_accuracy - baseline_accuracy,
23. }
24.
25. RETURN results
```

## 4.7 Checkpoint Strategy

### 4.7.1 What is Saved

- **LoRA adapter weights** (~7-15MB per checkpoint)
- **Optimizer state** (~15MB)
- **Training metadata** (JSON, < 1KB)
- **Metrics history** (JSON, < 10KB)

### 4.7.2 What is NOT Saved

- Base model weights (reloaded from HuggingFace)
- Tokenizer (reloaded from HuggingFace)
- Training data (regenerated each epoch)

### 4.7.3 Recovery Procedure

```
1. Mount Google Drive (checkpoints are synced automatically)
2. Load base model from HuggingFace
3. Load LoRA adapter from latest checkpoint
4. Restore optimizer state
5. Resume training from saved epoch number
```

## 4.8 Detection of Catastrophic Forgetting

### 4.8.1 Detection Mechanism

After each epoch, evaluate the model on quick-check questions for all previously seen topics:

```
forgetting_rate(topic) = max_historical_score(topic) - current_score(topic)

If forgetting_rate > 5%: FLAG for review
If forgetting_rate > 10%: CRITICAL warning, add to next epoch
```

### 4.8.2 Prevention Stack

1. **LoRA** (first line): Only small adapters updated, base model frozen
2. **EWC** (second line): Important parameters protected via Fisher
3. **Curriculum Review** (third line): Declining topics re-added to training
4. **Score Monitoring** (detection): Continuous tracking of per-topic performance

## 4.9 Measuring Improvement

### 4.9.1 Improvement Formula

```
Improvement(%) = [(final_accuracy - baseline_accuracy) / baseline_accuracy] × 100

Target: 15-25% improvement
```

### 4.9.2 Statistical Validity

- Run 3 independent trials with different random seeds
- Report mean ± standard deviation
- Use paired t-test to verify significance (p < 0.05)
