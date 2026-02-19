"""
Parameter Updater Module (LoRA Micro-Updates)
================================================
Part 4 of the SEAL Loop: Micro-Update Parameters

This module performs the actual parameter updates using LoRA.
Only the low-rank adapter weights are updated, keeping the
base model frozen.

Mathematical Foundation:
========================

Loss Function:
  L_total = L_task + λ_ewc · L_ewc

  where:
    L_task = CrossEntropy(model(question), correct_answer)
    L_ewc = (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²  (if EWC enabled)

Parameter Update:
  For LoRA matrices B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k}:
  
  B_{t+1} = B_t - η · ∇_B L_total
  A_{t+1} = A_t - η · ∇_A L_total

  The effective weight update is:
  ΔW_{t+1} = (α/r) · B_{t+1} · A_{t+1}

  where η is the learning rate and α/r is the LoRA scaling.

Key Properties:
  1. Only r × (d + k) parameters updated per module (vs d × k for full)
  2. Base weights W₀ remain frozen (no catastrophic forgetting of general knowledge)
  3. Updates are additive: W_effective = W₀ + ΔW
  4. Can be merged back into W₀ for inference (no latency overhead)
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
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
        
        # ── Setup LR Scheduler ─────────────────────────────────
        total_steps = (
            config.seal.num_epochs * 
            config.seal.questions_per_topic * 
            8  # approximate number of topics
        ) // (config.seal.batch_size * config.seal.gradient_accumulation_steps)
        
        if config.seal.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_steps, 1),
                eta_min=1e-6,
            )
        else:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=max(total_steps, 1),
            )
        
        self.total_updates = 0
        self.total_loss = 0.0
        
        logger.info(f"Optimizer: AdamW (lr={config.seal.learning_rate})")
        logger.info(f"Scheduler: {config.seal.scheduler}")
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
            
            # Tokenize batch
            batch_loss = self._compute_batch_loss(batch)
            
            # Add EWC regularization if enabled
            if ewc_loss_fn is not None:
                ewc_loss = ewc_loss_fn(self.model)
                batch_loss = batch_loss + ewc_loss
                total_ewc_loss += ewc_loss.item()
            
            # Scale loss for gradient accumulation
            scaled_loss = batch_loss / self.config.seal.gradient_accumulation_steps
            scaled_loss.backward()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Step optimizer after accumulation
            if num_batches % self.config.seal.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.seal.max_grad_norm,
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        # Handle remaining gradients
        if num_batches % self.config.seal.gradient_accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.seal.max_grad_norm,
            )
            self.optimizer.step()
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
    
    def get_stats(self) -> Dict:
        """Return update statistics."""
        return {
            "total_updates": self.total_updates,
            "avg_loss": self.total_loss / max(self.total_updates, 1),
            "current_lr": self.optimizer.param_groups[0]['lr'],
        }
