#!/usr/bin/env python3
"""
=============================================================================
SEAL-DSA  ·  Google Colab  ·  Baseline vs SEAL Comparison
=============================================================================
PURPOSE:
    Run this on Google Colab to compare:
      (A) Vanilla pre-trained model  (no fine-tuning  =  baseline)
      (B) Your SEAL-trained model    (loaded directly from Google Drive)

HOW TO USE — paste these 3 cells in a Colab notebook:

  ── Cell 1 ───────────────────────────────────────────────
  !pip install -q torch transformers peft accelerate bitsandbytes \\
               datasets evaluate rouge-score pyyaml tabulate tqdm

  ── Cell 2 ───────────────────────────────────────────────
  from google.colab import drive
  drive.mount('/content/drive')

  !rm -rf dsaseal
  !git clone https://github.com/Taqaddusshafi/dsaseal.git
  %cd dsaseal/SEAL-DSA

  ── Cell 3 ───────────────────────────────────────────────
  !python colab_drive_compare.py

=============================================================================
"""

import sys
import os
import time
import json
import torch
from pathlib import Path
from datetime import datetime

# ─── Add repo to Python path (matches training script setup) ─────────────────
sys.path.insert(0, '/content/dsaseal/SEAL-DSA')

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =============================================================================
#  CONFIG  ·  Edit these values before running
# =============================================================================
CONFIG = {
    # Base model — must match what was used in SEAL training (from colab_optimized.yaml)
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",

    # Google Drive checkpoint directory (matches CheckpointManager._setup_drive_path)
    "checkpoint_dir": "/content/drive/MyDrive/SEAL-DSA/checkpoints",

    # Leave None to auto-pick the latest epoch, or set explicitly e.g.:
    # "/content/drive/MyDrive/SEAL-DSA/checkpoints/checkpoint_epoch_5"
    "explicit_checkpoint_path": None,

    # Evaluation data — the repo's held-out eval set
    "eval_json_path": "/content/dsaseal/SEAL-DSA/data/evaluation_sets/dsa_eval_set.json",

    # Generation settings (match colab_optimized.yaml)
    "max_new_tokens": 200,
    "max_input_length": 512,
    "use_4bit": True,           # matches quantization.bits: 4 in colab_optimized.yaml

    # Questions per topic — None = all, set 3 for a quick smoke-test
    "limit_per_topic": 3,
}

# =============================================================================
#  FALLBACK EVAL SET  (used if eval_json_path is missing)
# =============================================================================
MINI_EVAL = {
    "arrays_strings": [
        {
            "question": "Implement Kadane's algorithm for maximum subarray sum.",
            "type": "coding",
            "function_name": "max_subarray_sum",
            "test_cases": [
                {"input": "([-2,1,-3,4,-1,2,1,-5,4],)", "expected": 6},
                {"input": "([1],)", "expected": 1},
                {"input": "([-1,-2,-3],)", "expected": -1},
            ],
        },
        {"question": "Explain the two-pointer technique with two examples.", "type": "conceptual"},
        {"question": "Write a function to check if two strings are anagrams.", "type": "coding"},
    ],
    "trees": [
        {"question": "Explain preorder, inorder, and postorder traversals.", "type": "conceptual"},
        {"question": "What is the time complexity of search in a balanced BST?", "type": "analytical"},
        {"question": "Implement level-order (BFS) traversal of a binary tree.", "type": "coding"},
    ],
    "dynamic_programming": [
        {"question": "Explain memoization vs tabulation with the Fibonacci example.", "type": "conceptual"},
        {
            "question": "Solve the coin change problem: minimum coins to make a target.",
            "type": "coding",
            "function_name": "coin_change",
            "test_cases": [
                {"input": "([1,5,10,25],30)", "expected": 2},
                {"input": "([2],3)", "expected": -1},
            ],
        },
        {"question": "What are the two conditions for a problem to be solvable by DP?", "type": "conceptual"},
    ],
}

# =============================================================================
#  HELPERS
# =============================================================================

def load_eval_data(cfg: dict) -> dict:
    path = cfg.get("eval_json_path", "")
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        topics = data.get("topics", {})
        print(f"✅ Loaded eval JSON → {sum(len(v) for v in topics.values())} questions across {len(topics)} topics")
    else:
        topics = MINI_EVAL
        print(f"ℹ️  eval JSON not found — using built-in mini eval set ({sum(len(v) for v in topics.values())} questions)")

    limit = cfg.get("limit_per_topic")
    if limit:
        topics = {k: v[:limit] for k, v in topics.items()}
        print(f"ℹ️  Limiting to {limit} question(s) per topic")

    return topics


def _build_bnb_config(cfg: dict):
    if not cfg["use_4bit"]:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(cfg: dict):
    """Load a fresh copy of the base model (no LoRA)."""
    name = cfg["base_model"]
    print(f"\n⏳ Loading base model: {name} ...")
    bnb = _build_bnb_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",   # matches model_loader.py
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
        padding_side="right",          # matches model_loader.py
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("✅ Base model ready.")
    return model, tokenizer


def find_latest_checkpoint(checkpoint_dir: str):
    """
    Mirror the same discovery logic used in the training notebook:
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_epoch_")]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
        latest = checkpoints[-1]
    """
    if not os.path.exists(checkpoint_dir):
        return None
    entries = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint_epoch_") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not entries:
        return None
    entries.sort(key=lambda x: int(x.split("_")[-1]))
    latest = entries[-1]
    print(f"ℹ️  Auto-selected latest checkpoint: {latest}")
    return os.path.join(checkpoint_dir, latest)


def load_seal_model(cfg: dict):
    """
    Load a FRESH base model then overlay the LoRA adapter from Drive.
    Fresh load avoids any state bleed from the baseline evaluation.
    Includes full diagnostics to surface missing-weight issues.
    """
    ckpt_path = cfg["explicit_checkpoint_path"] or find_latest_checkpoint(cfg["checkpoint_dir"])

    if not ckpt_path:
        print(f"\n⚠️  No checkpoint found in: {cfg['checkpoint_dir']}")
        print("    Make sure SEAL training has run at least 1 epoch and Drive is mounted.")
        return None, None, None

    if not os.path.exists(ckpt_path):
        print(f"\n❌ Checkpoint path does not exist: {ckpt_path}")
        return None, None, None

    p = Path(ckpt_path)
    label = f"SEAL ({p.name})"
    print(f"\n⏳ Loading SEAL model from: {p.name}")

    # ── Diagnostic: list checkpoint files ─────────────────────
    ckpt_files = list(p.iterdir())
    print(f"  📂 Files in checkpoint ({len(ckpt_files)} total):")
    for f in sorted(ckpt_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"      {f.name:<45} {size_mb:.2f} MB")

    has_adapter_config = (p / "adapter_config.json").exists()
    has_safetensors = (p / "adapter_model.safetensors").exists()
    has_bin = (p / "adapter_model.bin").exists()
    has_weights = has_safetensors or has_bin

    if not has_adapter_config:
        print("  ❌ adapter_config.json NOT found — not a valid PEFT checkpoint")
        return None, None, None

    if not has_weights:
        print("  ❌ CRITICAL: adapter_model.safetensors / adapter_model.bin NOT found!")
        print("     This means the Drive sync was incomplete during training.")
        print("     The LoRA weight file was never saved to Drive.")
        print("\n  🔧 How to fix:")
        print("     1. Go back to your TRAINING Colab session")
        print("     2. Run this to manually save the model to Drive:")
        print("        model.save_pretrained('/content/drive/MyDrive/SEAL-DSA/checkpoints/checkpoint_epoch_manual')")
        print("     3. Re-run this comparison script")
        return None, None, None

    print(f"  ✅ adapter_config.json : found")
    print(f"  ✅ adapter weights     : found ({'safetensors' if has_safetensors else 'bin'})")

    # ── Reload base model fresh ────────────────────────────────
    name = cfg["base_model"]
    bnb = _build_bnb_config(cfg)
    print(f"\n  ⏳ Reloading fresh base model for SEAL...")
    fresh_base = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        fresh_base.config.pad_token_id = fresh_base.config.eos_token_id

    # ── Load LoRA adapter ──────────────────────────────────────
    import warnings
    print("  ⏳ Applying LoRA adapter...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress verbose PEFT warnings
        model = PeftModel.from_pretrained(fresh_base, str(p), is_trainable=False)

    print("  ⏳ Merging LoRA weights...")
    model = model.merge_and_unload()
    print("  ✅ LoRA merged successfully.")
    print("✅ SEAL model ready.")
    return model, tokenizer, label


# =============================================================================
#  GENERATION & SCORING
# =============================================================================

def generate_answer(model, tokenizer, question: str, cfg: dict):
    """Generate an answer for a DSA question."""
    prompt = (
        "You are an expert in Data Structures and Algorithms. "
        "Answer the following DSA question clearly and concisely.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["max_input_length"],
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            # do_sample=True lets temperature work; False = greedy (ignores temperature)
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer, elapsed


def _try_run_tests(answer: str, q: dict):
    """Execute extracted code against test cases. Returns pass-rate or None."""
    fn = q.get("function_name")
    tests = q.get("test_cases", [])
    if not fn or not tests:
        return None

    code = ""
    if "```python" in answer:
        code = answer.split("```python")[1].split("```")[0]
    elif "```" in answer:
        code = answer.split("```")[1].split("```")[0]
    elif f"def {fn}" in answer:
        code = answer[answer.find(f"def {fn}"):]

    if not code.strip():
        return 0.0

    ns = {}
    passed = 0
    try:
        exec(code, ns)
        func = ns.get(fn)
        if func:
            for tc in tests:
                try:
                    args = eval(tc["input"])
                    args = args if isinstance(args, tuple) else (args,)
                    if func(*args) == tc["expected"]:
                        passed += 1
                except Exception:
                    pass
    except Exception:
        pass

    return passed / len(tests)


# Keyword sets matching DSA_TOPICS in curriculum/dsa_topics.py
_KW_MAP = {
    "arrays_strings":      ["array", "string", "index", "pointer", "window", "subarray", "slice"],
    "linked_lists":        ["node", "next", "head", "pointer", "list", "tail", "link"],
    "stacks_queues":       ["stack", "queue", "push", "pop", "fifo", "lifo", "top", "deque"],
    "trees":               ["tree", "root", "node", "traversal", "height", "bst", "leaf", "child"],
    "graphs":              ["graph", "vertex", "edge", "bfs", "dfs", "visited", "cycle", "path"],
    "sorting_searching":   ["sort", "binary", "search", "pivot", "merge", "complexity", "partition"],
    "dynamic_programming": ["dp", "memoization", "subproblem", "state", "transition", "cache", "optimal"],
}


def score_answer(answer: str, q: dict) -> float:
    """
    Heuristic scorer:
        30% length score    — penalises empty / very short answers
        40% keyword score   — topic-specific DSA keyword coverage
        30% code score      — rewards code presence for coding questions
        +   test pass rate  — blended in (50/50) when test cases exist
    """
    # 1. Length
    words = len(answer.split())
    if words == 0:
        length_s = 0.0
    elif words < 10:
        length_s = 0.3
    elif words < 30:
        length_s = 0.6
    else:
        length_s = 1.0

    # 2. Keywords
    topic = q.get("topic", "")
    kws = _KW_MAP.get(topic, [])
    al = answer.lower()
    if kws:
        hits = sum(1 for k in kws if k in al)
        kw_s = min(hits / max(len(kws) * 0.4, 1), 1.0)
    else:
        kw_s = 0.5

    # 3. Code presence (for coding questions only)
    q_type = q.get("type", "conceptual")
    if q_type == "coding":
        has_code = any(tok in answer for tok in ["def ", "return ", "for ", "while ", "```", "if "])
        code_s = 1.0 if has_code else 0.0
    else:
        code_s = 1.0  # N/A — full marks

    overall = length_s * 0.3 + kw_s * 0.4 + code_s * 0.3

    # 4. Blend with test pass rate if available
    test_rate = _try_run_tests(answer, q)
    if test_rate is not None:
        overall = overall * 0.5 + test_rate * 0.5

    return round(overall, 4)


# =============================================================================
#  EVALUATION RUNNER
# =============================================================================

def evaluate_model(model, tokenizer, eval_data: dict, cfg: dict, label: str) -> dict:
    print(f"\n{'─'*60}")
    print(f"🚀 Evaluating: {label}")
    print(f"{'─'*60}")

    model.eval()
    results = {
        "label": label,
        "topics": {},
        "overall_score": 0.0,
        "total_time_s": 0.0,
        "total_questions": 0,
        "timestamp": datetime.now().isoformat(),
        "sample_answers": {},  # Store one sample answer per topic for visual check
    }
    all_scores = []

    for topic, questions in eval_data.items():
        print(f"  📌 {topic} ({len(questions)} Qs) ...", end=" ", flush=True)
        scores = []
        t_total = 0.0
        first_answer = None

        for i, q in enumerate(questions):
            q["topic"] = topic
            ans, t = generate_answer(model, tokenizer, q["question"], cfg)
            s = score_answer(ans, q)
            scores.append(s)
            t_total += t
            if i == 0:
                first_answer = (q["question"], ans)  # Save first Q&A for display

        avg = sum(scores) / len(scores) if scores else 0.0
        results["topics"][topic] = {"avg_score": round(avg, 4), "scores": scores}
        results["total_time_s"] += t_total
        results["sample_answers"][topic] = first_answer
        all_scores.extend(scores)
        print(f"score = {avg:.3f}")

    results["overall_score"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    results["total_questions"] = len(all_scores)
    return results


# =============================================================================
#  REPORT
# =============================================================================

def print_sample_answers(base_res: dict, seal_res: dict, topic: str = None):
    """Print one sample Q&A from each model side-by-side to visually verify they differ."""
    topics = list(base_res.get("sample_answers", {}).keys())
    check_topic = topic or (topics[0] if topics else None)
    if not check_topic:
        return

    print(f"\n{'═'*65}")
    print(f"  🔎 SAMPLE ANSWER CHECK  (topic: {check_topic})")
    print(f"  (Verify the two models give DIFFERENT answers)")
    print(f"{'═'*65}")

    base_qa = base_res["sample_answers"].get(check_topic)
    seal_qa = seal_res["sample_answers"].get(check_topic) if seal_res else None

    if base_qa:
        print(f"\n  Q: {base_qa[0]}")
        print(f"\n  📘 BASE MODEL answer:")
        print(f"  {base_qa[1][:400]}..." if len(base_qa[1]) > 400 else f"  {base_qa[1]}")

    if seal_qa:
        print(f"\n  📗 SEAL MODEL answer:")
        print(f"  {seal_qa[1][:400]}..." if len(seal_qa[1]) > 400 else f"  {seal_qa[1]}")

    if base_qa and seal_qa:
        if base_qa[1].strip() == seal_qa[1].strip():
            print("\n  ⚠️  WARNING: Answers are IDENTICAL — LoRA weights may not have loaded!")
        else:
            print("\n  ✅ Answers differ — models are genuinely different.")
    print(f"{'═'*65}")


def print_report(base_res: dict, seal_res: dict):
    print("\n" + "=" * 65)
    print("  📊  SEAL-DSA  ·  COMPARISON RESULTS  (Base vs SEAL)")
    print("=" * 65)

    topics = list(base_res["topics"].keys())
    C = 16

    print(f"  {'Topic':<24} {'Base Model':>{C}} {'SEAL Model':>{C}} {'   Δ':>8}")
    print("  " + "─" * (24 + C + C + 9))

    for topic in topics:
        b = base_res["topics"][topic]["avg_score"]
        s = seal_res["topics"].get(topic, {}).get("avg_score", 0.0)
        delta = s - b
        sign = "+" if delta >= 0 else ""
        arrow = "▲" if delta >= 0 else "▼"
        print(f"  {topic:<24} {b:>{C}.3f} {s:>{C}.3f}  {arrow}{sign}{delta:.3f}")

    print("  " + "─" * (24 + C + C + 9))

    bo = base_res["overall_score"]
    so = seal_res["overall_score"]
    delta = so - bo
    pct = (delta / max(bo, 0.001)) * 100
    sign = "+" if delta >= 0 else ""
    arrow = "▲" if delta >= 0 else "▼"
    print(f"  {'⭐ OVERALL':<24} {bo:>{C}.3f} {so:>{C}.3f}  {arrow}{sign}{delta:.3f}")
    print("=" * 65)

    print(f"\n  🎯 SEAL improvement  : {sign}{delta:.3f}  ({pct:+.1f}%)")
    print(f"  ⏱️  Base  eval time  : {base_res['total_time_s']:.1f}s  "
          f"({base_res['total_time_s']/max(base_res['total_questions'],1):.2f}s/q)")
    print(f"  ⏱️  SEAL  eval time  : {seal_res['total_time_s']:.1f}s  "
          f"({seal_res['total_time_s']/max(seal_res['total_questions'],1):.2f}s/q)")
    print(f"  📦 Checkpoint used   : {seal_res['label']}")


def save_report(base_res: dict, seal_res: dict, out_dir: str = "/content"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"seal_comparison_{ts}.json"
    with open(path, "w") as f:
        json.dump({"base": base_res, "seal": seal_res, "generated_at": ts}, f, indent=2)
    print(f"\n💾 Full report saved → {path}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  🔬 SEAL-DSA  ·  Google Colab Comparison Script")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 1. Eval data
    print("\n[1/4] Loading evaluation data...")
    eval_data = load_eval_data(CONFIG)

    # 2. Load + evaluate BASE model (clean — no LoRA)
    print("\n[2/4] Setting up base model...")
    base_model, tokenizer = load_base_model(CONFIG)
    base_res = evaluate_model(base_model, tokenizer, eval_data, CONFIG, "Base Model (no fine-tuning)")

    # Free base model memory before loading SEAL model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Load + evaluate SEAL model (fresh base + LoRA merged)
    print("\n[3/4] Setting up SEAL model...")
    seal_model, seal_tokenizer, seal_label = load_seal_model(CONFIG)

    if seal_model is None:
        print("\n⚠️  Skipping SEAL evaluation — no checkpoint available.")
        print("   Base model results only:")
        print(f"   Overall score: {base_res['overall_score']:.3f}")
        return

    seal_res = evaluate_model(seal_model, seal_tokenizer, eval_data, CONFIG, seal_label)

    # 4. Print + save report
    print("\n[4/4] Generating report...")
    print_sample_answers(base_res, seal_res)  # Visual diff check
    print_report(base_res, seal_res)
    save_report(base_res, seal_res)


if __name__ == "__main__":
    main()
