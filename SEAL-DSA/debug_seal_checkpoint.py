#!/usr/bin/env python3
"""
Diagnose whether a SEAL/PEFT checkpoint is actually changing the base model.

Run in Colab after mounting Drive:

    python debug_seal_checkpoint.py \
      --checkpoint /content/drive/MyDrive/SEAL-DSA/checkpoints/checkpoint_epoch_5

This script checks four things:
  1. checkpoint files exist
  2. adapter weight key names match the PEFT model
  3. LoRA tensors are non-zero after loading
  4. logits and generated answers differ from the base model
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTIONS = [
    "Explain the two-pointer technique and give a brief example in Python.",
    "What is the time and space complexity of Kadane's algorithm? Explain why.",
    "Compare memoization and tabulation in dynamic programming. Which one is generally faster?",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--checkpoint",
        default="/content/drive/MyDrive/SEAL-DSA/checkpoints/checkpoint_epoch_5",
    )
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def print_checkpoint_summary(path: Path):
    print("\n=== Checkpoint summary ===")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    for item in sorted(path.iterdir()):
        print(f"{item.name:<35} {item.stat().st_size / 1_000_000:.3f} MB")

    adapter_config = path / "adapter_config.json"
    adapter_weights = path / "adapter_model.safetensors"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}")
    if not adapter_weights.exists():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {path}")

    with adapter_config.open() as f:
        cfg = json.load(f)
    print("\nAdapter config:")
    print(f"  base_model_name_or_path: {cfg.get('base_model_name_or_path')}")
    print(f"  r: {cfg.get('r')}")
    print(f"  lora_alpha: {cfg.get('lora_alpha')}")
    print(f"  target_modules: {cfg.get('target_modules')}")


def load_tokenizer(base_name):
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(base_name, device_map):
    return AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )


def chat_prompt(tokenizer, question):
    messages = [
        {
            "role": "system",
            "content": "You are a Data Structures and Algorithms expert. Keep your answers clear, concise, and accurate.",
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, question, max_new_tokens):
    prompt = chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_adapter_state(path: Path):
    state = load_file(path / "adapter_model.safetensors")
    print("\n=== Adapter state dict ===")
    print(f"Tensor count: {len(state)}")
    for key in list(state.keys())[:8]:
        print(f"  {key}  shape={tuple(state[key].shape)}  norm={state[key].float().norm().item():.6f}")
    nonzero_b = sum(1 for key, value in state.items() if "lora_B" in key and value.float().norm().item() > 1e-8)
    total_b = sum(1 for key in state if "lora_B" in key)
    print(f"Saved non-zero lora_B tensors: {nonzero_b}/{total_b}")
    return state


def saved_key_candidates(param_name):
    canonical = param_name.replace(".default.weight", ".weight")
    candidates = [
        canonical,
        canonical.replace("base_model.model.base_model.model.", "base_model.model."),
        canonical.replace("base_model.model.", "base_model.model.base_model.model.", 1),
    ]

    if "base_model.model.model.layers." in canonical:
        candidates.append(
            canonical.replace(
                "base_model.model.model.layers.",
                "base_model.model.base_model.model.model.layers.",
            )
        )

    return list(dict.fromkeys(candidates))


def force_load_lora_weights(peft_model, state):
    copied = 0
    missed = []

    for name, param in peft_model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue

        found_key = None
        for candidate in saved_key_candidates(name):
            if candidate in state:
                found_key = candidate
                break

        if found_key is None:
            suffix = name.split("model.layers.", 1)[-1].replace(".default.weight", ".weight")
            matches = [key for key in state if key.endswith(suffix)]
            if len(matches) == 1:
                found_key = matches[0]

        if found_key is None:
            missed.append(name)
            continue

        param.data.copy_(state[found_key].to(device=param.device, dtype=param.dtype))
        copied += 1

    print("\n=== Manual adapter injection ===")
    print(f"Copied LoRA tensors: {copied}")
    print(f"Missed LoRA tensors: {len(missed)}")
    if missed[:5]:
        print("First missed names:")
        for name in missed[:5]:
            print(f"  {name}")
    return copied


def lora_nonzero_report(peft_model):
    lora = [(name, param) for name, param in peft_model.named_parameters() if "lora_A" in name or "lora_B" in name]
    nonzero = [(name, param.float().norm().item()) for name, param in lora if param.float().norm().item() > 1e-8]
    nonzero_b = [(name, norm) for name, norm in nonzero if "lora_B" in name]
    print("\n=== Loaded LoRA tensor norms ===")
    print(f"Non-zero LoRA tensors: {len(nonzero)}/{len(lora)}")
    print(f"Non-zero lora_B tensors: {len(nonzero_b)}/{sum(1 for name, _ in lora if 'lora_B' in name)}")
    for name, norm in nonzero[:8]:
        print(f"  {name} norm={norm:.6f}")


def logits_for(model, tokenizer, question):
    prompt = chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float().detach().cpu()
    return logits


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)

    print_checkpoint_summary(checkpoint)
    state = load_adapter_state(checkpoint)

    tokenizer = load_tokenizer(args.base_model)

    print("\n=== Base model answer ===")
    base_model = load_base(args.base_model, args.device_map)
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_logits = logits_for(base_model, tokenizer, QUESTIONS[0])
    base_answers = [generate(base_model, tokenizer, q, args.max_new_tokens) for q in QUESTIONS]
    for i, answer in enumerate(base_answers, 1):
        print(f"\n[Base {i}]\n{answer[:700]}")

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== SEAL model answer ===")
    fresh_base = load_base(args.base_model, args.device_map)
    if tokenizer.pad_token_id is not None:
        fresh_base.config.pad_token_id = tokenizer.pad_token_id

    peft_model = PeftModel.from_pretrained(fresh_base, str(checkpoint), is_trainable=False)
    lora_nonzero_report(peft_model)

    copied = force_load_lora_weights(peft_model, state)
    lora_nonzero_report(peft_model)

    seal_logits = logits_for(peft_model, tokenizer, QUESTIONS[0])
    diff = (seal_logits - base_logits).abs()
    print("\n=== Logit difference before merge ===")
    print(f"max_abs_delta:  {diff.max().item():.8f}")
    print(f"mean_abs_delta: {diff.mean().item():.8f}")
    if copied == 0 or diff.max().item() < 1e-6:
        print("WARNING: The adapter is not changing the model meaningfully.")

    seal_model = peft_model.merge_and_unload()
    seal_answers = [generate(seal_model, tokenizer, q, args.max_new_tokens) for q in QUESTIONS]
    for i, answer in enumerate(seal_answers, 1):
        print(f"\n[SEAL {i}]\n{answer[:700]}")

    print("\n=== Final equality check ===")
    for i, (base_answer, seal_answer) in enumerate(zip(base_answers, seal_answers), 1):
        print(f"Question {i}: {'IDENTICAL' if base_answer == seal_answer else 'different'}")


if __name__ == "__main__":
    main()
