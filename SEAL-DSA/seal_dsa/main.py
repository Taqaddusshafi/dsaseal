"""
SEAL-DSA Main Entry Point
===========================
Orchestrates the complete SEAL learning loop:
  1. Load model with LoRA adapters
  2. Initialize curriculum scheduler  
  3. For each topic in curriculum:
     a. Generate questions (Question Generator)
     b. Generate answers (Answer Generator)
     c. Evaluate answers (Evaluator)
     d. Update parameters (Parameter Updater with LoRA)
  4. Detect catastrophic forgetting
  5. Save checkpoints and metrics
"""

import argparse
import os
import sys
import torch
import logging
from pathlib import Path

from seal_dsa.config import load_config, SEALDSAConfig
from seal_dsa.models.model_loader import load_model_and_tokenizer
from seal_dsa.modules.question_generator import QuestionGenerator
from seal_dsa.modules.answer_generator import AnswerGenerator
from seal_dsa.modules.evaluator import DSAEvaluator
from seal_dsa.modules.parameter_updater import ParameterUpdater
from seal_dsa.curriculum.scheduler import CurriculumScheduler
from seal_dsa.training.seal_loop import SEALTrainingLoop
from seal_dsa.training.checkpoint import CheckpointManager
from seal_dsa.evaluation.metrics import MetricsTracker
from seal_dsa.evaluation.forgetting_detector import ForgettingDetector
from seal_dsa.utils.logger import setup_logger
from seal_dsa.utils.colab_utils import is_colab, setup_colab_environment


def main():
    """Main entry point for SEAL-DSA training."""
    parser = argparse.ArgumentParser(
        description="SEAL-DSA: Self-Adapting LM for DSA Education"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--topics", type=str, nargs="+", default=None,
        help="Specific topics to train on (overrides curriculum)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only, no training"
    )
    args = parser.parse_args()
    
    # ── Load Configuration ──────────────────────────────────────
    config = load_config(args.config)
    logger = setup_logger(config.log_level)
    
    logger.info("=" * 60)
    logger.info("SEAL-DSA: Self-Adapting Language Model for DSA Education")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"LoRA rank: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    logger.info(f"Device: {config.device}")
    
    # ── Setup Environment ───────────────────────────────────────
    if is_colab():
        logger.info("Google Colab detected — applying optimizations")
        setup_colab_environment()
    
    device = _resolve_device(config.device)
    logger.info(f"Using device: {device}")
    
    # ── Load Model + LoRA ───────────────────────────────────────
    logger.info("Loading base model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info(f"Trainable parameters: {_count_trainable_params(model):,}")
    logger.info(f"Total parameters: {_count_total_params(model):,}")
    logger.info(f"Trainable %: {100 * _count_trainable_params(model) / _count_total_params(model):.4f}%")
    
    # ── Initialize Modules ──────────────────────────────────────
    question_gen = QuestionGenerator(model, tokenizer, config)
    answer_gen = AnswerGenerator(model, tokenizer, config)
    evaluator = DSAEvaluator(config)
    updater = ParameterUpdater(model, tokenizer, config)
    
    # ── Initialize Support Systems ──────────────────────────────
    curriculum = CurriculumScheduler(config.curriculum)
    checkpoint_mgr = CheckpointManager(config.checkpoint)
    metrics = MetricsTracker()
    forgetting_detector = ForgettingDetector(config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = checkpoint_mgr.load(args.resume, model, updater.optimizer)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # ── Evaluation Only Mode ────────────────────────────────────
    if args.eval_only:
        logger.info("Running evaluation only...")
        _run_evaluation(model, tokenizer, config, metrics, logger)
        return
    
    # ── SEAL Training Loop ──────────────────────────────────────
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
        forgetting_detector=forgetting_detector,
        config=config,
        device=device,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Starting SEAL Training Loop")
    logger.info("=" * 60)
    
    seal_loop.run(
        start_epoch=start_epoch,
        num_epochs=config.seal.num_epochs,
        topics=args.topics,
    )
    
    # ── Final Report ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SEAL Training Complete!")
    logger.info("=" * 60)
    metrics.print_summary()
    

def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def _count_trainable_params(model) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total_params(model) -> int:
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def _run_evaluation(model, tokenizer, config, metrics, logger):
    """Run standalone evaluation."""
    from seal_dsa.evaluation.baseline import BaselineComparison
    
    comparison = BaselineComparison(model, tokenizer, config)
    results = comparison.run_full_evaluation()
    
    logger.info("\n=== Evaluation Results ===")
    for method, scores in results.items():
        logger.info(f"{method}: {scores}")


if __name__ == "__main__":
    main()
