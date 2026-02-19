"""
Configuration Management for SEAL-DSA
======================================
Loads and validates YAML configuration files.
Supports hierarchical configuration with base configs.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    quantization_enabled: bool = True
    quantization_bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration.
    
    Mathematical Foundation:
    ========================
    For a pre-trained weight matrix W₀ ∈ ℝ^{d×k}, LoRA decomposes
    the update as:
        W = W₀ + ΔW = W₀ + BA
    where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, and r << min(d, k).
    
    The scaling factor is: α/r (lora_alpha / r)
    
    Parameters:
        r: Rank of down-projection (lower = fewer params, less capacity)
        lora_alpha: Scaling factor (controls update magnitude)
        lora_dropout: Dropout applied to LoRA layers
        bias: Whether to train bias parameters
        target_modules: Which attention layers to apply LoRA
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])


@dataclass
class SEALConfig:
    """SEAL loop configuration."""
    num_epochs: int = 5
    questions_per_topic: int = 20
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    correctness_threshold: float = 0.7
    quality_threshold: float = 0.5
    improvement_threshold: float = 0.15
    forgetting_threshold: float = 0.05


@dataclass
class EWCConfig:
    """Elastic Weight Consolidation configuration.
    
    Mathematical Foundation:
    ========================
    EWC adds a regularization term to prevent catastrophic forgetting:
        L_total = L_task + (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²
    
    where:
        - Fᵢ: Fisher Information Matrix diagonal element
        - θ*ᵢ: Optimal parameters from previous task
        - λ: Regularization strength (higher = more protection)
    """
    enabled: bool = True
    lambda_: float = 0.4
    fisher_sample_size: int = 200


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    strategy: str = "progressive"
    weeks: Dict[int, str] = field(default_factory=lambda: {
        1: "arrays_strings", 2: "arrays_strings",
        3: "linked_lists", 4: "linked_lists",
        5: "stacks_queues", 6: "stacks_queues",
        7: "trees", 8: "trees",
        9: "graphs", 10: "graphs",
        11: "sorting_searching", 12: "sorting_searching",
        13: "dynamic_programming", 14: "dynamic_programming",
        15: "advanced_topics", 16: "comprehensive_review",
    })


@dataclass
class CheckpointConfig:
    """Checkpoint management configuration."""
    save_every_n_epochs: int = 1
    save_dir: str = "checkpoints"
    max_checkpoints: int = 5
    save_to_drive: bool = True


@dataclass
class SEALDSAConfig:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    seal: SEALConfig = field(default_factory=SEALConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    device: str = "auto"
    mixed_precision: bool = True
    log_level: str = "INFO"


def load_config(config_path: str) -> SEALDSAConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        SEALDSAConfig: Populated configuration object.
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    config = SEALDSAConfig()
    
    # Parse model config
    if 'model' in raw:
        m = raw['model']
        config.model.name = m.get('name', config.model.name)
        config.model.max_length = m.get('max_length', config.model.max_length)
        config.model.temperature = m.get('temperature', config.model.temperature)
        config.model.top_p = m.get('top_p', config.model.top_p)
        config.model.top_k = m.get('top_k', config.model.top_k)
        config.model.do_sample = m.get('do_sample', config.model.do_sample)
        if 'quantization' in m:
            q = m['quantization']
            config.model.quantization_enabled = q.get('enabled', True)
            config.model.quantization_bits = q.get('bits', 4)
            config.model.quant_type = q.get('quant_type', 'nf4')
            config.model.double_quant = q.get('double_quant', True)
            config.model.compute_dtype = q.get('compute_dtype', 'bfloat16')
    
    # Parse LoRA config
    if 'lora' in raw:
        l = raw['lora']
        config.lora.r = l.get('r', config.lora.r)
        config.lora.lora_alpha = l.get('lora_alpha', config.lora.lora_alpha)
        config.lora.lora_dropout = l.get('lora_dropout', config.lora.lora_dropout)
        config.lora.bias = l.get('bias', config.lora.bias)
        config.lora.task_type = l.get('task_type', config.lora.task_type)
        config.lora.target_modules = l.get('target_modules', config.lora.target_modules)
    
    # Parse SEAL config
    if 'seal' in raw:
        s = raw['seal']
        config.seal.num_epochs = s.get('num_epochs', config.seal.num_epochs)
        config.seal.questions_per_topic = s.get('questions_per_topic', config.seal.questions_per_topic)
        config.seal.batch_size = s.get('batch_size', config.seal.batch_size)
        config.seal.gradient_accumulation_steps = s.get('gradient_accumulation_steps', config.seal.gradient_accumulation_steps)
        config.seal.learning_rate = s.get('learning_rate', config.seal.learning_rate)
        config.seal.warmup_steps = s.get('warmup_steps', config.seal.warmup_steps)
        config.seal.max_grad_norm = s.get('max_grad_norm', config.seal.max_grad_norm)
        config.seal.weight_decay = s.get('weight_decay', config.seal.weight_decay)
        if 'evaluation' in s:
            e = s['evaluation']
            config.seal.correctness_threshold = e.get('correctness_threshold', config.seal.correctness_threshold)
            config.seal.quality_threshold = e.get('quality_threshold', config.seal.quality_threshold)
            config.seal.improvement_threshold = e.get('improvement_threshold', config.seal.improvement_threshold)
            config.seal.forgetting_threshold = e.get('forgetting_threshold', config.seal.forgetting_threshold)
    
    # Parse EWC config
    if 'ewc' in raw:
        e = raw['ewc']
        config.ewc.enabled = e.get('enabled', config.ewc.enabled)
        config.ewc.lambda_ = e.get('lambda', config.ewc.lambda_)
        config.ewc.fisher_sample_size = e.get('fisher_sample_size', config.ewc.fisher_sample_size)
    
    # Parse curriculum
    if 'curriculum' in raw:
        c = raw['curriculum']
        config.curriculum.strategy = c.get('strategy', config.curriculum.strategy)
        if 'weeks' in c:
            config.curriculum.weeks = {int(k): v for k, v in c['weeks'].items()}
    
    # Parse checkpoint
    if 'checkpoint' in raw:
        cp = raw['checkpoint']
        config.checkpoint.save_every_n_epochs = cp.get('save_every_n_epochs', config.checkpoint.save_every_n_epochs)
        config.checkpoint.save_dir = cp.get('save_dir', config.checkpoint.save_dir)
        config.checkpoint.max_checkpoints = cp.get('max_checkpoints', config.checkpoint.max_checkpoints)
        config.checkpoint.save_to_drive = cp.get('save_to_drive', config.checkpoint.save_to_drive)
    
    # Parse hardware
    if 'hardware' in raw:
        h = raw['hardware']
        config.device = h.get('device', config.device)
        config.mixed_precision = h.get('mixed_precision', config.mixed_precision)
    
    if 'logging' in raw:
        config.log_level = raw['logging'].get('level', 'INFO')
    
    return config
