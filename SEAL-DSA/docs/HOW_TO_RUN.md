# 🚀 How to Train & Run SEAL-DSA — Complete Guide

## Overview

You will run everything on **Google Colab Free Tier** (free T4 GPU).  
No local GPU needed. No money needed (under $0 for free tier).

**Total Time**: ~3-5 hours across 1-2 Colab sessions  
**What You Get**: A fine-tuned model + metrics + thesis-ready results

---

## Step 1: Upload Project to GitHub

### 1.1 — Initialize Git & Push

Open a terminal in the `SEAL-DSA` folder and run:

```bash
cd "c:\Users\lenovo\Desktop\New folder\SEAL-DSA"

git init
git add .
git commit -m "Initial commit: SEAL-DSA framework"
```

### 1.2 — Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `SEAL-DSA`
3. Keep it **Public** (or Private — your choice)
4. **Don't** add README (we already have one)
5. Click "Create repository"

### 1.3 — Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/SEAL-DSA.git
git branch -M main
git push -u origin main
```

---

## Step 2: Open Google Colab

1. Go to **[colab.research.google.com](https://colab.research.google.com)**
2. Click **"New Notebook"**
3. Go to **Runtime → Change runtime type → T4 GPU** ← IMPORTANT!
4. Click **Save**

---

## Step 3: Run These Cells in Colab (Copy-Paste Each)

### Cell 1: Install Dependencies & Clone Repo

```python
# Install required packages
!pip install -q torch transformers peft accelerate bitsandbytes
!pip install -q datasets evaluate rouge-score pyyaml

# Mount Google Drive (for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Clone YOUR repository
!git clone https://github.com/YOUR_USERNAME/SEAL-DSA.git
%cd SEAL-DSA

# Verify GPU
import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

> ⚠️ Replace `YOUR_USERNAME` with your actual GitHub username!

---

### Cell 2: Load Configuration

```python
import sys
sys.path.insert(0, '.')

from seal_dsa.config import load_config

config = load_config("configs/colab_optimized.yaml")

print(f"✅ Model: {config.model.name}")
print(f"✅ LoRA rank: {config.lora.r}")
print(f"✅ Epochs: {config.seal.num_epochs}")
print(f"✅ Questions per topic: {config.seal.questions_per_topic}")
```

---

### Cell 3: Load Model with LoRA (Takes 2-5 minutes)

```python
from seal_dsa.models.model_loader import load_model_and_tokenizer

print("⏳ Loading model... (2-5 minutes)")
model, tokenizer = load_model_and_tokenizer(config)

# Show parameter summary
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✅ Model loaded!")
print(f"   Total params: {total:,}")
print(f"   Trainable (LoRA): {trainable:,} ({trainable/total*100:.2f}%)")
print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

### Cell 4: Quick Test — Make Sure Model Works

```python
# Test with a simple DSA question
test_q = "What is the time complexity of binary search?"
inputs = tokenizer(test_q, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, temperature=0.3,
                         do_sample=True, pad_token_id=tokenizer.pad_token_id)

answer = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"❓ Question: {test_q}")
print(f"💬 Answer: {answer[:300]}")
```

You should see a reasonable (but probably imperfect) answer. That's the **baseline** — SEAL will improve this!

---

### Cell 5: Initialize All SEAL Modules

```python
from seal_dsa.modules.question_generator import QuestionGenerator
from seal_dsa.modules.answer_generator import AnswerGenerator
from seal_dsa.modules.evaluator import DSAEvaluator
from seal_dsa.modules.parameter_updater import ParameterUpdater
from seal_dsa.curriculum.scheduler import CurriculumScheduler
from seal_dsa.training.checkpoint import CheckpointManager
from seal_dsa.training.ewc import EWC
from seal_dsa.evaluation.metrics import MetricsTracker
from seal_dsa.evaluation.forgetting_detector import ForgettingDetector

device = torch.device("cuda")

question_gen = QuestionGenerator(model, tokenizer, config)
answer_gen = AnswerGenerator(model, tokenizer, config)
evaluator = DSAEvaluator(config)
updater = ParameterUpdater(model, config, device)
curriculum = CurriculumScheduler(config.curriculum)
checkpoint_mgr = CheckpointManager(config.checkpoint)
metrics = MetricsTracker()
forgetting_det = ForgettingDetector(config)

print("✅ All SEAL modules initialized!")
```

---

### Cell 6: 🚀 RUN THE SEAL TRAINING LOOP

```python
from seal_dsa.training.seal_loop import SEALTrainingLoop

seal = SEALTrainingLoop(
    model=model,
    tokenizer=tokenizer,
    question_generator=question_gen,
    answer_generator=answer_gen,
    evaluator=evaluator,
    parameter_updater=updater,
    curriculum=curriculum,
    checkpoint_manager=checkpoint_mgr,
    metrics_tracker=metrics,
    forgetting_detector=forgetting_det,
    config=config,
    device=device,
)

# START TRAINING!
seal.run()
```

**This will take ~2-3 hours.** You'll see live output like:

```
══════════════════════════════════════════════════
EPOCH 1/3
══════════════════════════════════════════════════
--- Topic 1/7: arrays_strings ---
  Step 1: Generating questions...
  Generated 10 questions
  Step 2: Generating answers...
  Generated 10 answers
  Step 3: Evaluating answers...
  Avg score: 0.421, Correct: 4/10
  Step 4: Updating parameters...
  Loss: 2.3451, Grad norm: 0.5234
```

---

### Cell 7: View Results

```python
# Print summary table
metrics.print_summary()

# Save metrics to file
import os
os.makedirs("results", exist_ok=True)
metrics.save("results/training_metrics.json")

# Also save to Google Drive
import shutil
drive_path = "/content/drive/MyDrive/SEAL-DSA/results"
os.makedirs(drive_path, exist_ok=True)
shutil.copy("results/training_metrics.json", drive_path)
print(f"\n✅ Metrics saved to Google Drive: {drive_path}")
```

---

### Cell 8: Save Final Trained Model

```python
# Save LoRA adapter (only ~15MB!)
model.save_pretrained("results/final_model")
tokenizer.save_pretrained("results/final_model")

# Copy to Google Drive for persistence
import shutil
drive_model_path = "/content/drive/MyDrive/SEAL-DSA/final_model"
if os.path.exists(drive_model_path):
    shutil.rmtree(drive_model_path)
shutil.copytree("results/final_model", drive_model_path)

print(f"✅ Model saved to Google Drive!")
print(f"   Path: {drive_model_path}")
print(f"   Size: ~15MB (LoRA adapter only)")
```

---

### Cell 9: Test the IMPROVED Model

```python
# Ask the SAME question again — compare with Cell 4!
test_q = "What is the time complexity of binary search?"
inputs = tokenizer(test_q, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.3,
                         do_sample=True, pad_token_id=tokenizer.pad_token_id)

answer = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("🔥 AFTER SEAL TRAINING:")
print(f"❓ Question: {test_q}")
print(f"💬 Answer: {answer[:500]}")
```

You should see a **noticeably better** answer compared to Cell 4!

---

### Cell 10 (Optional): Generate Visualization Plots

```python
# Run the evaluation analysis script
exec(open("notebooks/Evaluation_Analysis.py").read())

# Copy figures to Drive
import shutil
drive_fig = "/content/drive/MyDrive/SEAL-DSA/figures"
if os.path.exists("results/figures"):
    if os.path.exists(drive_fig):
        shutil.rmtree(drive_fig)
    shutil.copytree("results/figures", drive_fig)
    print(f"✅ Figures saved to: {drive_fig}")
```

---

## Step 4: Resume Training (If Colab Disconnects)

Colab sessions timeout after ~12 hours. If it disconnects:

1. Open a **new notebook**
2. Run Cell 1 (install + clone + mount Drive)
3. Run Cell 2 (load config)
4. Run Cell 3 (load model)
5. Then run this **resume cell**:

```python
# Load checkpoint from Google Drive
checkpoint_path = "/content/drive/MyDrive/SEAL-DSA/checkpoints"
import os
latest = sorted([d for d in os.listdir(checkpoint_path)
                 if d.startswith("checkpoint_")])[-1]
full_path = os.path.join(checkpoint_path, latest)

# Load LoRA adapter from checkpoint
from peft import PeftModel
model = PeftModel.from_pretrained(model, full_path)
print(f"✅ Resumed from: {latest}")

# Continue training from where you left off
# (re-initialize modules as in Cell 5, then run Cell 6)
```

---

## Step 5: What You Get (For Your Thesis)

After training completes, you'll have:

| Output                  | Location                      | Use In Thesis             |
| ----------------------- | ----------------------------- | ------------------------- |
| Training metrics JSON   | `Drive/SEAL-DSA/results/`     | Chapter 5: Results        |
| Learning curve plots    | `Drive/SEAL-DSA/figures/`     | Chapter 5: Figures        |
| Final LoRA model        | `Drive/SEAL-DSA/final_model/` | Chapter 4: Implementation |
| Per-topic scores        | In metrics JSON               | Chapter 5: Tables         |
| Before/After comparison | Cell 4 vs Cell 9 output       | Chapter 5: Qualitative    |

---

## Quick Reference: Time & Cost Estimate

| Item                  | Estimate               |
| --------------------- | ---------------------- |
| Setup (Cells 1-5)     | ~10 minutes            |
| Training (Cell 6)     | ~2-3 hours per epoch   |
| Total (3 epochs)      | ~6-9 hours             |
| Colab sessions needed | 1-2 sessions           |
| GPU cost              | **$0** (Free Tier)     |
| Storage used          | ~100MB on Google Drive |

---

## Troubleshooting

| Problem                | Solution                                                       |
| ---------------------- | -------------------------------------------------------------- |
| "CUDA out of memory"   | Reduce `questions_per_topic` to 5 in config                    |
| "Session disconnected" | Follow Step 4 (Resume) above                                   |
| "Model not found"      | Check internet connection; HuggingFace downloads the model     |
| Slow generation        | Normal — small GPU. Be patient                                 |
| "Module not found"     | Make sure you ran `%cd SEAL-DSA` and `sys.path.insert(0, '.')` |

---

## 🎯 Summary: Just 10 Cells!

```
Cell 1:  Install + Clone + Mount Drive     (2 min)
Cell 2:  Load Config                        (instant)
Cell 3:  Load Model                         (3 min)
Cell 4:  Test BEFORE training               (10 sec)
Cell 5:  Initialize modules                 (instant)
Cell 6:  🚀 RUN TRAINING                   (2-3 hours)
Cell 7:  View results                       (instant)
Cell 8:  Save model to Drive                (1 min)
Cell 9:  Test AFTER training                (10 sec)
Cell 10: Generate plots                     (1 min)
```

**That's it — 10 cells and you have a trained, improved DSA tutor!** 🎉
