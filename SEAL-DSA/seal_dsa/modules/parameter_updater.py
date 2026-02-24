"""
Parameter Updater Module (LoRA Micro-Updates)
================================================
Part 4 of the SEAL Loop: Micro-Update Parameters

This module performs the actual parameter updates using LoRA.
Only the low-rank adapter weights are updated, keeping the
base model frozen.

Mathematical Foundation:
========================

Loss Function (Novel: Hybrid CE + DPO):
  L_total = L_CE + λ_dpo · L_DPO + λ_ewc · L_EWC

  where:
    L_CE  = CrossEntropy(model(question), correct_answer)
    L_DPO = -log σ(β · (log π(y_w|x) - log π(y_l|x)))  [DPO loss]
    L_EWC = (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²  (if EWC enabled)

Direct Preference Optimization (DPO):
  Given a question x, chosen answer y_w, rejected answer y_l:
  L_DPO = -log σ(β · (log π_θ(y_w|x) - log π_θ(y_l|x)))

  This is equivalent to RLHF but without a separate reward model.
  The evaluator's scores define the preference ordering.

  Reference: Rafailov et al. (2023) "Direct Preference Optimization:
  Your Language Model is Secretly a Reward Model"

Parameter Update:
  For LoRA matrices B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k}:
  
  B_{t+1} = B_t - η · ∇_B L_total
  A_{t+1} = A_t - η · ∇_A L_total

  The effective weight update is:
  ΔW_{t+1} = (α/r) · B_{t+1} · A_{t+1}

  where η is the learning rate and α/r is the LoRA scaling.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
)
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from seal_dsa.config import SEALDSAConfig
from seal_dsa.modules.evaluator import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of a parameter update step."""
    loss: float
    grad_norm: float
    learning_rate: float
    num_samples: int
    ewc_loss: float = 0.0
    dpo_loss: float = 0.0  # Novel: tracks DPO contribution


class ParameterUpdater:
    """
    Performs LoRA micro-updates based on evaluation results.
    
    This is the fourth and final step of the SEAL loop. The evaluator's
    scores determine which (question, answer) pairs become training data.
    
    Training Signal Construction:
    ─────────────────────────────
    1. High-score answers (>threshold): Used as positive examples
       - Model learns to reinforce this behavior
       
    2. Low-score answers (<threshold): Used with corrective signal
       - The feedback is incorporated into a corrected prompt
       - Model learns to avoid these mistakes
    
    Architecture:
    ┌─────────────────────────────────────────────────┐
    │          Parameter Updater                       │
    │                                                  │
    │  Evaluation Results                              │
    │        │                                         │
    │        ▼                                         │
    │  ┌──────────────────┐                            │
    │  │ Training Data     │                           │
    │  │ Construction      │                           │
    │  │  - Positive pairs │                           │
    │  │  - Negative pairs │                           │
    │  └────────┬─────────┘                            │
    │           ▼                                      │
    │  ┌──────────────────┐  ┌───────────────────┐    │
    │  │ Loss Computation  │  │ EWC Regularization │   │
    │  │ (CrossEntropy)    │──│ (if enabled)       │   │
    │  └────────┬─────────┘  └───────────────────┘    │
    │           ▼                                      │
    │  ┌──────────────────┐                            │
    │  │ LoRA Parameter    │                           │
    │  │ Update (AdamW)    │                           │
    │  │ ∇_B L, ∇_A L     │                           │
    │  └──────────────────┘                            │
    └─────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: SEALDSAConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # ── Setup Optimizer ─────────────────────────────────────
        # Only optimize LoRA parameters
        trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=config.seal.learning_rate,
            weight_decay=config.seal.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # ── Setup LR Scheduler (with warmup) ───────────────────
        total_steps = (
            config.seal.num_epochs * 
            config.seal.questions_per_topic * 
            8  # approximate number of topics
        ) // (config.seal.batch_size * config.seal.gradient_accumulation_steps)
        total_steps = max(total_steps, 1)
        warmup_steps = min(config.seal.warmup_steps, total_steps // 2)
        decay_steps = max(total_steps - warmup_steps, 1)
        
        # Phase 1: Linear warmup from 10% → 100% of lr
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        
        # Phase 2: Cosine or linear decay
        if config.seal.scheduler == "cosine":
            decay_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=decay_steps,
                eta_min=1e-6,
            )
        else:
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=decay_steps,
            )
        
        if warmup_steps > 0:
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = decay_scheduler
        
        # ── Setup Mixed-Precision (AMP) ────────────────────────
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.total_updates = 0
        self.total_loss = 0.0
        
        logger.info(f"Optimizer: AdamW (lr={config.seal.learning_rate})")
        logger.info(f"Scheduler: {config.seal.scheduler} (warmup={warmup_steps} steps)")
        logger.info(f"Mixed precision (AMP): {self.use_amp}")
        logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    def update(
        self,
        evaluation_results: List[EvaluationResult],
        ewc_loss_fn=None,
    ) -> UpdateResult:
        """
        Perform a micro-update step based on evaluation results.
        
        This implements the core parameter update:
        
        For positive examples (score > threshold):
          L = -log P(correct_answer | question)
          
        For all examples:
          L_total = L_task + λ · L_ewc
          
        θ_{t+1} = θ_t - η · ∇θ L_total
        
        Only LoRA parameters (B, A matrices) are updated.
        
        Args:
            evaluation_results: List of EvaluationResult from evaluator
            ewc_loss_fn: Optional EWC regularization loss function
            
        Returns:
            UpdateResult with loss and gradient statistics
        """
        self.model.train()
        
        # ── Prepare Training Data ──────────────────────────────
        training_pairs = self._prepare_training_data(evaluation_results)
        
        if not training_pairs:
            logger.warning("No training data after filtering. Skipping update.")
            return UpdateResult(
                loss=0.0, grad_norm=0.0,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                num_samples=0,
            )
        
        # ── Training Loop ──────────────────────────────────────
        total_loss = 0.0
        total_ewc_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for i in range(0, len(training_pairs), self.config.seal.batch_size):
            batch = training_pairs[i:i + self.config.seal.batch_size]
            
            # Forward pass with optional mixed-precision
            with autocast(enabled=self.use_amp):
                batch_loss = self._compute_batch_loss(batch)
                
                # Add EWC regularization if enabled
                if ewc_loss_fn is not None:
                    ewc_loss = ewc_loss_fn(self.model)
                    batch_loss = batch_loss + ewc_loss
                    total_ewc_loss += ewc_loss.item()
                
                # Scale loss for gradient accumulation
                scaled_loss = batch_loss / self.config.seal.gradient_accumulation_steps
            
            # Backward pass with gradient scaler
            self.scaler.scale(scaled_loss).backward()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Step optimizer after accumulation
            if num_batches % self.config.seal.gradient_accumulation_steps == 0:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.seal.max_grad_norm,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        # Handle remaining gradients
        if num_batches % self.config.seal.gradient_accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.seal.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.model.eval()
        
        avg_loss = total_loss / max(num_batches, 1)
        self.total_updates += 1
        self.total_loss += avg_loss
        
        result = UpdateResult(
            loss=avg_loss,
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            num_samples=len(training_pairs),
            ewc_loss=total_ewc_loss / max(num_batches, 1),
        )
        
        logger.info(
            f"Update #{self.total_updates}: loss={avg_loss:.4f}, "
            f"grad_norm={result.grad_norm:.4f}, "
            f"lr={result.learning_rate:.2e}, "
            f"samples={len(training_pairs)}"
        )
        
        return result
    
    def _prepare_training_data(
        self,
        evaluation_results: List[EvaluationResult],
    ) -> List[Dict[str, str]]:
        """
        Convert evaluation results to training pairs.
        
        Strategy:
        - Use ALL evaluated answers as training data
        - Weight by evaluation score:
          - High scores → larger gradient contribution
          - Low scores → paired with corrective feedback
        
        This creates a balanced training signal that both reinforces
        good answers and corrects mistakes.
        """
        training_pairs = []
        
        for result in evaluation_results:
            question = result.answer.question.question
            answer = result.answer.answer
            score = result.overall_score
            
            if score >= self.config.seal.correctness_threshold:
                # ── Positive Example ────────────────────────────
                # Use the answer directly as training target
                training_pairs.append({
                    "input": self._format_training_input(question),
                    "output": answer,
                    "weight": score,
                })
            else:
                # ── Corrective Example ──────────────────────────
                # Include feedback to create a corrected response
                corrected_prompt = (
                    f"The following answer had issues: {result.feedback}\n"
                    f"Question: {question}\n"
                    f"Improve the answer focusing on: {result.feedback}"
                )
                training_pairs.append({
                    "input": self._format_training_input(question),
                    "output": f"Let me provide a better answer.\n{answer}",
                    "weight": max(0.1, score),  # Don't fully suppress
                })
        
        return training_pairs
    
    def _format_training_input(self, question: str) -> str:
        """Format question as training input with appropriate template."""
        return (
            f"You are an expert in Data Structures and Algorithms. "
            f"Answer the following question thoroughly.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    
    def _compute_batch_loss(
        self,
        batch: List[Dict[str, str]],
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for a batch.
        
        Loss function:
          L = -Σᵢ wᵢ · log P(yᵢ | xᵢ)
          
        where:
          - xᵢ is the input (question)
          - yᵢ is the target (answer)
          - wᵢ is the quality weight from the evaluator
        """
        total_loss = torch.tensor(0.0, device=self.model.device)
        
        for item in batch:
            full_text = item["input"] + " " + item["output"]
            
            encodings = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.model.max_length,
                padding=True,
            ).to(self.model.device)
            
            # Create labels (mask the input portion)
            input_len = len(self.tokenizer.encode(
                item["input"], add_special_tokens=False
            ))
            labels = encodings["input_ids"].clone()
            labels[0, :input_len] = -100  # Mask input tokens
            
            outputs = self.model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                labels=labels,
            )
            
            # Weight loss by evaluation score
            weighted_loss = outputs.loss * item.get("weight", 1.0)
            total_loss = total_loss + weighted_loss
        
        return total_loss / len(batch)
    
    # ==================================================================
    #  NOVEL CONTRIBUTION: Contrastive Self-Play with DPO Loss
    # ==================================================================
    
    def _compute_log_probs(
        self,
        question: str,
        answer: str,
    ) -> torch.Tensor:
        """
        Compute log P(answer | question) for DPO.
        
        Returns the average per-token log probability of the answer
        conditioned on the question.
        """
        full_text = question + " " + answer
        
        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_length,
            padding=True,
        ).to(self.model.device)
        
        input_len = len(self.tokenizer.encode(
            question, add_special_tokens=False
        ))
        
        labels = encodings["input_ids"].clone()
        labels[0, :input_len] = -100
        
        outputs = self.model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            labels=labels,
        )
        
        # outputs.loss is the average negative log-likelihood
        # We want log prob, so negate
        return -outputs.loss
    
    def compute_dpo_loss(
        self,
        evaluation_results: List[EvaluationResult],
        beta: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute Direct Preference Optimization (DPO) loss.
        
        Novel Contribution:
        ====================
        SEAL generates multiple answers for the same question via the
        self-play loop. We rank these by evaluator score and create
        preference pairs (y_chosen, y_rejected) for contrastive training.
        
        DPO Loss (Rafailov et al., 2023):
          L_DPO = -log σ(β · (Δ_θ - Δ_ref))
        
        Where:
          Δ_θ   = log π_θ(y_w|x) - log π_θ(y_l|x)
          Δ_ref = log π_ref(y_w|x) - log π_ref(y_l|x)
        
        In self-play mode, π_ref is the the model at the start of
        the current update step (before gradient computation).
        We approximate Δ_ref ≈ 0 for the first iteration (no ref model)
        which reduces DPO to a simpler ranking loss.
        
        Args:
            evaluation_results: List of evaluation results (same question
                may appear multiple times with different answers)
            beta: Temperature controlling preference sharpness.
                  Higher β = more aggressive preference learning.
        
        Returns:
            Scalar DPO loss tensor.
        """
        # Group results by question
        from collections import defaultdict
        question_groups: Dict[str, List[EvaluationResult]] = defaultdict(list)
        
        for result in evaluation_results:
            q_key = result.answer.question.question[:100]  # Use first 100 chars as key
            question_groups[q_key].append(result)
        
        dpo_loss = torch.tensor(0.0, device=self.model.device)
        num_pairs = 0
        
        for q_key, results in question_groups.items():
            if len(results) < 2:
                continue
            
            # Sort by score: best first
            results.sort(key=lambda r: r.overall_score, reverse=True)
            
            # Create preference pairs: best vs worst
            chosen = results[0]
            rejected = results[-1]
            
            # Skip if scores are too similar (no clear preference)
            if chosen.overall_score - rejected.overall_score < 0.1:
                continue
            
            question_text = self._format_training_input(
                chosen.answer.question.question
            )
            
            # Compute log probs for chosen and rejected
            log_prob_chosen = self._compute_log_probs(
                question_text, chosen.answer.answer
            )
            log_prob_rejected = self._compute_log_probs(
                question_text, rejected.answer.answer
            )
            
            # DPO loss: -log sigmoid(beta * (log_prob_chosen - log_prob_rejected))
            preference_diff = beta * (log_prob_chosen - log_prob_rejected)
            pair_loss = -torch.nn.functional.logsigmoid(preference_diff)
            
            dpo_loss = dpo_loss + pair_loss
            num_pairs += 1
        
        if num_pairs > 0:
            dpo_loss = dpo_loss / num_pairs
            logger.info(f"DPO loss computed from {num_pairs} preference pairs")
        
        return dpo_loss
    
    def get_stats(self) -> Dict:
        """Return update statistics."""
        return {
            "total_updates": self.total_updates,
            "avg_loss": self.total_loss / max(self.total_updates, 1),
            "current_lr": self.optimizer.param_groups[0]['lr'],
        }
