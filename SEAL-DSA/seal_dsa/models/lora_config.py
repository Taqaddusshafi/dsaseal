"""
LoRA Configuration Details
============================
Provides detailed LoRA configuration and parameter computation.

Mathematical Detail:
====================
For each target module with weight W₀ ∈ ℝ^{d_out × d_in}:

  ΔW = B · A where B ∈ ℝ^{d_out × r}, A ∈ ℝ^{r × d_in}

Initialization:
  - A ~ N(0, σ²) with Kaiming uniform initialization
  - B = 0 (ensures ΔW = 0 at start, preserving pre-trained behavior)

Scaling:
  - Output is scaled by α/r
  - Higher α → larger updates
  - Higher r → more expressive but more parameters

Number of LoRA parameters per module:
  params = d_out × r + r × d_in = r × (d_out + d_in)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class LoRAParameterBudget:
    """Computes the parameter budget for different LoRA configurations."""
    
    rank: int
    target_modules: List[str]
    model_hidden_size: int
    num_layers: int
    
    def compute_params_per_module(self) -> int:
        """Parameters added per LoRA module.
        
        For a square weight matrix (d × d):
            params = 2 × d × r
        """
        d = self.model_hidden_size
        r = self.rank
        return 2 * d * r
    
    def compute_total_lora_params(self) -> int:
        """Total LoRA parameters across all layers and modules."""
        per_module = self.compute_params_per_module()
        total = per_module * len(self.target_modules) * self.num_layers
        return total
    
    def compute_compression_ratio(self, total_model_params: int) -> float:
        """Ratio of LoRA params to full model params."""
        lora_params = self.compute_total_lora_params()
        return lora_params / total_model_params
    
    def report(self, total_model_params: int) -> str:
        """Generate a human-readable parameter budget report."""
        lora_total = self.compute_total_lora_params()
        ratio = self.compute_compression_ratio(total_model_params)
        
        report = f"""
╔══════════════════════════════════════════════════╗
║           LoRA Parameter Budget Report           ║
╠══════════════════════════════════════════════════╣
║ Configuration:                                   ║
║   Rank (r):            {self.rank:<25}║
║   Target Modules:      {', '.join(self.target_modules):<25}║
║   Hidden Size (d):     {self.model_hidden_size:<25}║
║   Num Layers:          {self.num_layers:<25}║
╠══════════════════════════════════════════════════╣
║ Parameter Count:                                 ║
║   Per Module:          {self.compute_params_per_module():>15,}     ║
║   Per Layer:           {self.compute_params_per_module() * len(self.target_modules):>15,}     ║
║   Total LoRA:          {lora_total:>15,}     ║
║   Total Model:         {total_model_params:>15,}     ║
║   LoRA / Model:        {ratio:>14.4%}      ║
╚══════════════════════════════════════════════════╝
"""
        return report


# Pre-computed budgets for recommended models
MODEL_BUDGETS = {
    "TinyLlama-1.1B": LoRAParameterBudget(
        rank=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        model_hidden_size=2048,
        num_layers=22,
    ),
    "Qwen2.5-1.5B": LoRAParameterBudget(
        rank=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        model_hidden_size=1536,
        num_layers=28,
    ),
    "Phi-2-2.7B": LoRAParameterBudget(
        rank=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        model_hidden_size=2560,
        num_layers=32,
    ),
}
