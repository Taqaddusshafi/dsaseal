"""
Model Loader with LoRA Integration
=====================================
Loads a pre-trained causal language model with:
  - 4-bit quantization (QLoRA) for memory efficiency
  - LoRA adapter injection for parameter-efficient fine-tuning
  - Automatic device mapping for Colab T4 GPU

Mathematical Foundation of LoRA:
================================
Given a pre-trained weight matrix W₀ ∈ ℝ^{d×k}:

    h = W₀x + ΔWx = W₀x + BAx

where:
    - B ∈ ℝ^{d×r} (initialized to zeros)
    - A ∈ ℝ^{r×k} (initialized with Kaiming uniform)  
    - r << min(d, k) is the rank (typically 4-16)

The forward pass becomes:
    h = W₀x + (α/r) · BAx

where α is the scaling factor (lora_alpha).

Trainable parameters comparison:
    Full fine-tuning: d × k parameters per layer
    LoRA: (d + k) × r parameters per layer
    
    For d=k=4096, r=8: 
        Full: 16,777,216 params
        LoRA: 65,536 params (0.39% of full)
"""

import torch
import logging
from typing import Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from seal_dsa.config import SEALDSAConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    config: SEALDSAConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pre-trained model with LoRA adapters.
    
    Architecture:
    ┌──────────────────────────────────────────────────┐
    │  Pre-trained Model (Frozen W₀)                   │
    │  ┌────────────────────────────────────────────┐  │
    │  │ Attention Layer                             │  │
    │  │  ┌──────────┐  ┌──────────┐               │  │
    │  │  │ W₀(q)    │  │ LoRA(q)  │  ← Trainable  │  │
    │  │  │ (frozen) │  │ B·A      │               │  │
    │  │  └──────────┘  └──────────┘               │  │
    │  │  ┌──────────┐  ┌──────────┐               │  │
    │  │  │ W₀(k)    │  │ LoRA(k)  │  ← Trainable  │  │
    │  │  │ (frozen) │  │ B·A      │               │  │
    │  │  └──────────┘  └──────────┘               │  │
    │  │  ┌──────────┐  ┌──────────┐               │  │
    │  │  │ W₀(v)    │  │ LoRA(v)  │  ← Trainable  │  │
    │  │  │ (frozen) │  │ B·A      │               │  │
    │  │  └──────────┘  └──────────┘               │  │
    │  │  ┌──────────┐  ┌──────────┐               │  │
    │  │  │ W₀(o)    │  │ LoRA(o)  │  ← Trainable  │  │
    │  │  │ (frozen) │  │ B·A      │               │  │
    │  │  └──────────┘  └──────────┘               │  │
    │  └────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────┘
    
    Args:
        config: SEALDSAConfig with model and LoRA settings.
        
    Returns:
        Tuple of (model with LoRA, tokenizer).
    """
    logger.info(f"Loading model: {config.model.name}")
    
    # ── Step 1: Configure Quantization ──────────────────────────
    bnb_config = None
    if config.model.quantization_enabled:
        compute_dtype = getattr(torch, config.model.compute_dtype, torch.float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(config.model.quantization_bits == 4),
            load_in_8bit=(config.model.quantization_bits == 8),
            bnb_4bit_quant_type=config.model.quant_type,
            bnb_4bit_use_double_quant=config.model.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(f"Quantization: {config.model.quantization_bits}-bit "
                     f"({config.model.quant_type})")
    
    # ── Step 2: Load Base Model ─────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",  # Compatible with all models
    )
    
    # ── Step 3: Load Tokenizer ──────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # ── Step 4: Prepare for k-bit Training ──────────────────────
    if config.model.quantization_enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        logger.info("Model prepared for k-bit training with gradient checkpointing")
    
    # ── Step 5: Apply LoRA Adapters ─────────────────────────────
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    
    # ── Report Statistics ───────────────────────────────────────
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    reduction = (1 - trainable_params / total_params) * 100
    
    logger.info(f"LoRA applied successfully:")
    logger.info(f"  Trainable: {trainable_params:,} params")
    logger.info(f"  Total: {total_params:,} params")
    logger.info(f"  Reduction: {reduction:.2f}% fewer trainable params")
    logger.info(f"  Target modules: {config.lora.target_modules}")
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def get_model_info(model_name: str) -> dict:
    """Get information about recommended models.
    
    Returns a dict with model recommendations for different use cases.
    """
    models = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "params": "1.1B",
            "vram_4bit": "~2GB",
            "vram_fp16": "~2.2GB",
            "speed": "fastest",
            "quality": "basic",
            "recommended_for": "Quick experiments, debugging",
        },
        "Qwen/Qwen2.5-1.5B-Instruct": {
            "params": "1.5B",
            "vram_4bit": "~3GB",
            "vram_fp16": "~3GB",
            "speed": "fast",
            "quality": "good",
            "recommended_for": "Default choice, balanced",
        },
        "microsoft/phi-2": {
            "params": "2.7B",
            "vram_4bit": "~4GB",
            "vram_fp16": "~5.4GB",
            "speed": "moderate",
            "quality": "very good",
            "recommended_for": "Best quality within Colab limits",
        },
        "Qwen/Qwen2.5-3B-Instruct": {
            "params": "3B",
            "vram_4bit": "~4.5GB",
            "vram_fp16": "~6GB",
            "speed": "moderate",
            "quality": "very good",
            "recommended_for": "High quality, Colab Pro",
        },
    }
    return models.get(model_name, {"info": "Unknown model"})
