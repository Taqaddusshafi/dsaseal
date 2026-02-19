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
from typing import Dict, Optional
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
        
        logger.info(f"EWC initialized: λ={self.lambda_}, "
                     f"fisher_samples={self.fisher_sample_size}")
    
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
