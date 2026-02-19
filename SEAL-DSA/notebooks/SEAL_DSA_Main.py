"""
SEAL-DSA Main Training Notebook
==================================
Run this script in Google Colab to execute the full SEAL training loop.

To use in Colab:
  1. Upload this file or clone the repo
  2. Run cells sequentially
  3. Checkpoints auto-save to Google Drive

Estimated Time: ~2-3 hours per epoch on T4 GPU
Estimated VRAM: ~5GB peak
"""

# ============================================================
# Cell 1: Environment Setup
# ============================================================

# !pip install -q torch transformers peft accelerate bitsandbytes
# !pip install -q datasets evaluate rouge-score pyyaml

# Mount Google Drive for checkpoint persistence
# from google.colab import drive
# drive.mount('/content/drive')

# Clone repository (uncomment if running from scratch)
# !git clone https://github.com/yourusername/SEAL-DSA.git
# %cd SEAL-DSA

import os
import sys
import torch

# Add project to path
if os.path.exists("seal_dsa"):
    sys.path.insert(0, ".")

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# Cell 2: Load Configuration
# ============================================================

from seal_dsa.config import load_config

# Use colab-optimized config for free tier
config_path = "configs/colab_optimized.yaml"
if not os.path.exists(config_path):
    config_path = "configs/default.yaml"

config = load_config(config_path)
print(f"Config loaded: {config_path}")
print(f"  Model: {config.model.name}")
print(f"  LoRA rank: {config.lora.r}")
print(f"  Epochs: {config.seal.num_epochs}")
print(f"  Questions/topic: {config.seal.questions_per_topic}")

# ============================================================
# Cell 3: Load Model with LoRA
# ============================================================

from seal_dsa.models.model_loader import load_model_and_tokenizer

print("Loading model (this may take 2-5 minutes)...")
model, tokenizer = load_model_and_tokenizer(config)

# Print parameter summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel loaded successfully!")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable (LoRA): {trainable_params:,}")
print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")

if torch.cuda.is_available():
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================
# Cell 4: Quick Test - Verify Model Works
# ============================================================

print("\n--- Quick Model Test ---")
test_prompt = "Explain what a binary search tree is in one sentence."
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Prompt: {test_prompt}")
print(f"Response: {response[:200]}")

# ============================================================
# Cell 5: Initialize SEAL Modules
# ============================================================

from seal_dsa.modules.question_generator import QuestionGenerator
from seal_dsa.modules.answer_generator import AnswerGenerator
from seal_dsa.modules.evaluator import DSAEvaluator
from seal_dsa.modules.parameter_updater import ParameterUpdater
from seal_dsa.curriculum.scheduler import CurriculumScheduler
from seal_dsa.training.checkpoint import CheckpointManager
from seal_dsa.training.ewc import EWC
from seal_dsa.evaluation.metrics import MetricsTracker
from seal_dsa.evaluation.forgetting_detector import ForgettingDetector
from seal_dsa.training.seal_loop import SEALTrainingLoop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question_gen = QuestionGenerator(model, tokenizer, config)
answer_gen = AnswerGenerator(model, tokenizer, config)
evaluator = DSAEvaluator(config)
updater = ParameterUpdater(model, config, device)
curriculum = CurriculumScheduler(config.curriculum)
checkpoint_mgr = CheckpointManager(config.checkpoint)
metrics = MetricsTracker()
forgetting = ForgettingDetector(config)

print("All SEAL modules initialized!")
print(f"  Curriculum strategy: {config.curriculum.strategy}")
print(f"  EWC enabled: {config.ewc.enabled}")

# ============================================================
# Cell 6: Check for Existing Checkpoint (Resume Support)
# ============================================================

start_epoch = 0
latest_ckpt = checkpoint_mgr.get_latest_checkpoint()

if latest_ckpt:
    print(f"Found checkpoint: {latest_ckpt}")
    user_input = input("Resume from checkpoint? [y/n]: ").strip().lower()
    if user_input == 'y':
        start_epoch = checkpoint_mgr.load(latest_ckpt, model, updater.optimizer)
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Starting fresh training")
else:
    print("No checkpoint found. Starting fresh training.")

# ============================================================
# Cell 7: Run SEAL Training Loop
# ============================================================

print("\n" + "=" * 60)
print("STARTING SEAL TRAINING LOOP")
print("=" * 60)

seal_loop = SEALTrainingLoop(
    model=model,
    tokenizer=tokenizer,
    question_generator=question_gen,
    answer_generator=answer_gen,
    evaluator=evaluator,
    parameter_updater=updater,
    curriculum=curriculum,
    checkpoint_manager=checkpoint_mgr,
    metrics_tracker=metrics,
    forgetting_detector=forgetting,
    config=config,
    device=device,
)

seal_loop.run(start_epoch=start_epoch)

# ============================================================
# Cell 8: View Results
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RESULTS")
print("=" * 60)

metrics.print_summary()

# Save metrics
metrics.save("results/training_metrics.json")
print("\nMetrics saved to results/training_metrics.json")

# ============================================================
# Cell 9: Post-Training Evaluation
# ============================================================

from seal_dsa.evaluation.baseline import BaselineComparison

print("\n--- Running Final Evaluation ---")
evaluator_final = BaselineComparison(model, tokenizer, config)
results = evaluator_final.run_full_evaluation()

print("\nFinal evaluation complete!")

# ============================================================
# Cell 10: Save Final Model to Google Drive
# ============================================================

from seal_dsa.utils.colab_utils import save_to_drive

# Save final LoRA adapter
final_save_path = "results/final_model"
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final model saved to {final_save_path}")

# Sync to Google Drive if available
try:
    save_to_drive(final_save_path)
    save_to_drive("results/training_metrics.json")
    print("Results synced to Google Drive!")
except Exception as e:
    print(f"Drive sync skipped: {e}")

print("\n🎉 SEAL-DSA training complete!")
