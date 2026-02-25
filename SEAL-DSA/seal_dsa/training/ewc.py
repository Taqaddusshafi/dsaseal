"""
Elastic Weight Consolidation (EWC)
=====================================
Prevents catastrophic forgetting during continual learning.

Mathematical Foundation:
========================

The core idea: When learning task B, penalize changes to parameters
that were important for task A.

"Importance" is measured by the Fisher Information Matrix:

  F_i = E[( ∂log p(x|θ) / ∂θ_i )²]

The EWC loss adds a quadratic penalty:

  L_EWC = (λ/2) Σᵢ F_i · (θ_i - θ*_i)²

where:
  - F_i:    Diagonal of the Fisher Information Matrix
  - θ*_i:   Parameters after learning the previous task
  - θ_i:    Current parameters being optimized
  - λ:      Regularization strength (higher = more protection)

Intuition:
  - F_i large → parameter θ_i was important for previous task
    → penalize changes heavily
  - F_i small → parameter θ_i was not important
    → allow free modification

For LoRA:
  We only compute Fisher for LoRA parameters (B, A matrices),
  not the frozen base model. This makes EWC tractable on small GPUs.

Reference:
  Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting
  in neural networks" (PNAS)
"""

import logging
from typing import Dict, List, Optional
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F

from seal_dsa.config import EWCConfig

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation for catastrophic forgetting prevention.
    
    Implementation Details:
    ──────────────────────
    1. After training on each topic, compute Fisher Information Matrix
    2. Store optimal parameters θ* and Fisher diagonal F
    3. During subsequent training, add EWC penalty to the loss
    4. Fisher is computed using only LoRA parameters (memory efficient)
    
    Novel: Dynamic Lambda (Adaptive Regularisation Strength)
    =========================================================
    Instead of a fixed λ, SEAL dynamically adjusts λ based on
    real-time forgetting detector signals:
    
      λ^{t+1} = clip(λ^{t} + η_λ · (forgetting_rate - target_rate), λ_min, λ_max)
    
    When forgetting_rate > target_rate: λ increases (more protection)
    When forgetting_rate < target_rate: λ decreases (faster learning)
    
    This eliminates the need for expensive λ hyperparameter tuning.
    
    Memory Cost:
    ──────────────
    Stores 2× the LoRA parameters:
    - θ* (optimal parameters copy)
    - F (Fisher diagonal)
    
    For Qwen2.5-1.5B with rank-8 LoRA:
    ~2 × 1.7M ≈ 3.4M float32 values ≈ ~14MB
    
    This is negligible compared to model memory.
    """
    
    def __init__(self, model: nn.Module, config: EWCConfig):
        self.config = config
        self.lambda_ = config.lambda_
        self.fisher_sample_size = config.fisher_sample_size
        
        # Storage for Fisher diagonal and optimal parameters
        # These are populated after the first topic is learned
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        self.initialized = False
        self.update_count = 0
        
        # ── Dynamic Lambda State (Novel) ──────────────────────
        self.lambda_min = 0.05
        self.lambda_max = 2.0
        self.lambda_lr = 0.1          # Learning rate for lambda adjustment
        self.target_forgetting = 0.05  # Target forgetting rate
        self._lambda_history: List[float] = [self.lambda_]
        
        logger.info(f"EWC initialized: λ={self.lambda_} (dynamic, "
                     f"range=[{self.lambda_min}, {self.lambda_max}]), "
                     f"fisher_samples={config.fisher_sample_size}")
    
    def adapt_lambda(self, forgetting_report: Dict) -> float:
        """
        Dynamically adjust λ based on forgetting detector signals.
        
        Novel Contribution:
        ====================
        Traditional EWC uses a fixed λ that must be manually tuned.
        SEAL's dynamic λ uses a simple control loop:
        
          λ^{t+1} = clip(λ^{t} + η_λ · (f_rate - f_target), λ_min, λ_max)
        
        where:
          f_rate   = current forgetting rate (from detector)
          f_target = desired maximum forgetting rate
          η_λ      = lambda learning rate
        
        Intuition:
          - If forgetting_rate > target: λ increases → stronger regularisation
          - If forgetting_rate < target: λ decreases → more plasticity
        
        This creates a self-regulating system that balances stability
        and plasticity automatically.
        
        Args:
            forgetting_report: Dict from ForgettingDetector.check_all_topics()
                Expected keys: 'avg_forgetting', 'topics_at_risk'
        
        Returns:
            Updated lambda value.
        """
        current_forgetting = forgetting_report.get("avg_forgetting", 0.0)
        topics_at_risk = len(forgetting_report.get("topics_at_risk", []))
        
        # Compute forgetting signal (combine average forgetting + risk count)
        total_topics = max(forgetting_report.get("total_topics", 1), 1)
        risk_rate = topics_at_risk / total_topics
        forgetting_signal = 0.7 * current_forgetting + 0.3 * risk_rate
        
        # Control loop: adjust lambda
        error = forgetting_signal - self.target_forgetting
        old_lambda = self.lambda_
        self.lambda_ = max(
            self.lambda_min,
            min(self.lambda_max, self.lambda_ + self.lambda_lr * error)
        )
        
        self._lambda_history.append(self.lambda_)
        
        if abs(self.lambda_ - old_lambda) > 0.001:
            logger.info(
                f"Dynamic EWC: λ {old_lambda:.4f} → {self.lambda_:.4f} "
                f"(forgetting={forgetting_signal:.4f}, "
                f"target={self.target_forgetting:.4f}, "
                f"topics_at_risk={topics_at_risk})"
            )
        
        return self.lambda_
    
    def get_lambda_history(self) -> List[float]:
        """Return the history of lambda values for analysis."""
        return self._lambda_history
    
    def update_fisher(
        self,
        model: nn.Module,
        tokenizer=None,
        dataloader=None,
    ):
        """
        Compute/update the Fisher Information Matrix diagonal.
        
        The Fisher diagonal is approximated as:
        
          F_i ≈ (1/N) Σ_{n=1}^{N} (∂L_n / ∂θ_i)²
        
        where L_n is the loss on sample n.
        
        For efficiency, we use a small number of samples
        (fisher_sample_size) and only compute for LoRA params.
        
        Args:
            model: The current model
            tokenizer: Tokenizer for generating Fisher samples
            dataloader: Optional dataloader with samples
        """
        logger.info("Computing Fisher Information Matrix...")
        
        model.eval()
        fisher_dict = {}
        
        # Initialize Fisher to zeros for all trainable params
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
        
        # Compute Fisher using gradient squares
        # We use the model's own predictions as the "true" labels
        # This is the empirical Fisher approximation
        
        num_samples = 0
        
        if dataloader is not None:
            # Use provided dataloader
            for batch in dataloader:
                if num_samples >= self.fisher_sample_size:
                    break
                self._accumulate_fisher(model, batch, fisher_dict)
                num_samples += 1
        else:
            # Generate synthetic samples using the model itself
            # This is the self-supervised Fisher computation
            for _ in range(self.fisher_sample_size):
                self._accumulate_fisher_self_supervised(
                    model, tokenizer, fisher_dict
                )
                num_samples += 1
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= max(num_samples, 1)
        
        # ── Online Fisher Update ───────────────────────────────
        # If this isn't the first update, blend with previous Fisher
        # This implements "online EWC" (Schwarz et al., 2018)
        if self.initialized:
            gamma = 0.9  # Decay factor for old Fisher
            for name in fisher_dict:
                if name in self.fisher:
                    self.fisher[name] = (
                        gamma * self.fisher[name] + 
                        (1 - gamma) * fisher_dict[name]
                    )
                else:
                    self.fisher[name] = fisher_dict[name]
        else:
            self.fisher = fisher_dict
        
        # ── Store optimal parameters ──────────────────────────
        self.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        self.initialized = True
        self.update_count += 1
        
        # Log Fisher statistics
        fisher_norms = {
            name: f.norm().item()
            for name, f in self.fisher.items()
        }
        max_fisher = max(fisher_norms.values()) if fisher_norms else 0
        logger.info(f"Fisher updated (update #{self.update_count}), "
                     f"max norm: {max_fisher:.4f}")
    
    def _accumulate_fisher(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        fisher_dict: Dict[str, torch.Tensor],
    ):
        """Accumulate Fisher from a data batch."""
        model.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
    
    def _accumulate_fisher_self_supervised(
        self,
        model: nn.Module,
        tokenizer,
        fisher_dict: Dict[str, torch.Tensor],
    ):
        """
        Accumulate Fisher using self-supervised samples.
        
        Generate a short sequence, compute loss on it, and use
        the gradient squares as Fisher approximation.
        """
        model.zero_grad()
        
        # Generate a short prompt about DSA
        prompts = [
            "Explain the time complexity of binary search.",
            "What is a linked list and how does it work?",
            "Describe the BFS algorithm for graphs.",
            "How does dynamic programming solve the knapsack problem?",
            "What is the difference between a stack and a queue?",
            "Explain merge sort with its complexity analysis.",
            "What is a binary search tree?",
            "Describe Dijkstra's shortest path algorithm.",
        ]
        
        import random
        prompt = random.choice(prompts)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(next(model.parameters()).device)
        
        # Use the input as both input and label (self-supervised)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )
        
        outputs.loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
    
    def compute_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularization loss.
        
        L_EWC = (λ/2) Σᵢ F_i · (θ_i - θ*_i)²
        
        Args:
            model: Current model with updated parameters
            
        Returns:
            EWC loss tensor (to be added to the task loss)
        """
        if not self.initialized:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                # (λ/2) × F_i × (θ_i - θ*_i)²
                fisher = self.fisher[name].to(param.device)
                optimal = self.optimal_params[name].to(param.device)
                
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return (self.lambda_ / 2) * ewc_loss
    
    def get_importance_summary(self) -> Dict[str, float]:
        """Get summary of parameter importance (Fisher norms)."""
        if not self.fisher:
            return {}
        
        return {
            name: fisher.norm().item()
            for name, fisher in self.fisher.items()
        }
