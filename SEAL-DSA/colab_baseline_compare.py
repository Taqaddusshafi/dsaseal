#!/usr/bin/env python3
"""
=============================================================================
SEAL-DSA  ·  Google Colab Baseline vs SEAL Comparison
=============================================================================
PURPOSE:
    Run this notebook on Google Colab.  It compares:
      (A) A vanilla pre-trained model  (no fine-tuning)
      (B) Your SEAL-trained model      (uploaded .zip or Google Drive path)

STEPS:
    1.  Install deps   (Cell 1)
    2.  Upload your SEAL checkpoint zip OR mount Google Drive  (Cell 2)
    3.  Run comparison  (Cell 3 onwards)

HOW TO GET YOUR CHECKPOINT INTO COLAB:
    Option A - Upload zip:
        zip -r seal_checkpoint.zip checkpoints/epoch_5/
        # Then upload via Colab file browser or Files > Upload

    Option B - Google Drive (set DRIVE_CHECKPOINT_PATH below):
        from google.colab import drive
        drive.mount('/content/drive')

=============================================================================
"""

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
INSTALL_CMD = """
pip install -q torch transformers peft accelerate bitsandbytes tqdm tabulate
"""

# ── CELL 2: Configuration — EDIT THESE ───────────────────────────────────────
CONFIG = {
    # Base model (same one used during SEAL training)
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",

    # Path to your SEAL checkpoint inside Colab filesystem
    # - If you uploaded a zip: extract first, set path to extracted folder
    # - If using Drive:  "/content/drive/MyDrive/seal_checkpoints/epoch_5"
    # - Set to None to skip SEAL evaluation (base-only run)
    "seal_checkpoint_path": "/content/seal_checkpoint",

    # Evaluation data — paste the raw GitHub URL or upload the file
    # Default: uses the built-in mini eval set defined below
    "eval_json_path": None,          # e.g. "/content/dsa_eval_set.json"

    # Generation settings
    "max_new_tokens": 200,
    "use_4bit": True,                 # Set False if you have a Colab Pro+ with A100

    # Limit questions per topic for speed (None = all)
    "limit_per_topic": 3,
}

# ── Mini eval set (fallback when no JSON uploaded) ────────────────────────────
MINI_EVAL = {
    "arrays_strings": [
        {"question": "Implement Kadane's algorithm for maximum subarray sum.",
         "type": "coding", "function_name": "max_subarray_sum",
         "test_cases": [{"input": "([-2,1,-3,4,-1,2,1,-5,4],)", "expected": 6},
                        {"input": "([1],)", "expected": 1},
                        {"input": "([-1,-2,-3],)", "expected": -1}]},
        {"question": "Explain the two-pointer technique with two examples.",
         "type": "conceptual"},
        {"question": "Write a function to check if two strings are anagrams.",
         "type": "coding", "function_name": "is_anagram",
         "test_cases": [{"input": "('anagram','nagaram')", "expected": True},
                        {"input": "('rat','car')", "expected": False}]},
    ],
    "trees": [
        {"question": "Explain the difference between BFS and DFS traversal in trees.",
         "type": "conceptual"},
        {"question": "What is the time complexity of search in a balanced BST?",
         "type": "analytical"},
        {"question": "Write level-order (BFS) traversal of a binary tree.",
         "type": "coding"},
    ],
    "dynamic_programming": [
        {"question": "Explain memoization vs tabulation with the Fibonacci example.",
         "type": "conceptual"},
        {"question": "Solve the coin change problem: minimum coins to make a target.",
         "type": "coding", "function_name": "coin_change",
         "test_cases": [{"input": "([1,5,10,25],30)", "expected": 2},
                        {"input": "([2],3)", "expected": -1}]},
        {"question": "What are the two conditions for a problem to be solvable by DP?",
         "type": "conceptual"},
    ],
    "graphs": [
        {"question": "Find the number of islands in a 2D grid using DFS.",
         "type": "coding"},
        {"question": "When does Dijkstra's algorithm fail?",
         "type": "analytical"},
        {"question": "Explain adjacency list vs adjacency matrix tradeoffs.",
         "type": "conceptual"},
    ],
    "sorting_searching": [
        {"question": "Implement binary search. Return -1 if not found.",
         "type": "coding", "function_name": "binary_search",
         "test_cases": [{"input": "([1,3,5,7,9],5)", "expected": 2},
                        {"input": "([1,3,5,7,9],4)", "expected": -1}]},
        {"question": "Compare merge sort and quicksort on stability and complexity.",
         "type": "conceptual"},
        {"question": "Search for a target in a rotated sorted array.",
         "type": "coding"},
    ],
}

# =============================================================================
#  IMPLEMENTATION  (no need to edit below)
# =============================================================================

import json, os, sys, time, textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def install_deps():
    os.system("pip install -q torch transformers peft accelerate bitsandbytes tqdm tabulate")
    print("✅ Dependencies ready")


def load_eval_data(cfg: dict) -> Dict[str, List[dict]]:
    if cfg["eval_json_path"] and Path(cfg["eval_json_path"]).exists():
        with open(cfg["eval_json_path"]) as f:
            data = json.load(f)
        topics = data.get("topics", {})
        print(f"✅ Loaded eval JSON: {sum(len(v) for v in topics.values())} questions")
    else:
        topics = MINI_EVAL
        print(f"ℹ️  Using built-in mini eval set ({sum(len(v) for v in topics.values())} questions)")

    limit = cfg.get("limit_per_topic")
    if limit:
        topics = {k: v[:limit] for k, v in topics.items()}

    return topics


def load_model(model_name: str, use_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def load_seal_model(base_model, checkpoint_path: str):
    from peft import PeftModel
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if (p / "adapter_config.json").exists():
        print(f"  → Loading LoRA adapter from {p.name}")
        model = PeftModel.from_pretrained(base_model, str(p))
        model = model.merge_and_unload()
    else:
        from transformers import AutoModelForCausalLM
        import torch
        print(f"  → Loading full model from {p.name}")
        model = AutoModelForCausalLM.from_pretrained(
            str(p), device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
    return model


def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> Tuple[str, float]:
    import torch
    prompt = (
        "You are an expert in Data Structures and Algorithms. "
        "Answer the following DSA question clearly.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
    elapsed = time.time() - t0
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer, elapsed


def try_run_tests(answer: str, q: dict) -> Optional[float]:
    """Execute extracted code against test cases. Returns pass rate or None."""
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


def score_answer(answer: str, q: dict) -> float:
    """Heuristic scorer combining length, keywords, code presence, test pass rate."""
    words = len(answer.split())
    length_s = 0.0 if words == 0 else (0.3 if words < 10 else (0.6 if words < 30 else 1.0))

    kw_map = {
        "arrays_strings": ["array", "string", "index", "pointer", "window", "subarray"],
        "trees": ["tree", "root", "node", "traversal", "height", "bst", "leaf"],
        "graphs": ["graph", "vertex", "edge", "bfs", "dfs", "visited", "cycle"],
        "sorting_searching": ["sort", "binary", "search", "pivot", "merge", "complexity"],
        "dynamic_programming": ["dp", "memoization", "subproblem", "state", "transition"],
        "linked_lists": ["node", "next", "head", "pointer", "list"],
        "stacks_queues": ["stack", "queue", "push", "pop", "fifo", "lifo"],
    }
    topic = q.get("topic", "")
    kws = kw_map.get(topic, [])
    al = answer.lower()
    kw_s = min(sum(1 for k in kws if k in al) / max(len(kws) * 0.4, 1), 1.0) if kws else 0.5

    q_type = q.get("type", "conceptual")
    code_s = 1.0
    if q_type == "coding":
        code_s = 1.0 if any(k in answer for k in ["def ", "return ", "for ", "while ", "```"]) else 0.0

    test_s = try_run_tests(answer, q)

    overall = length_s * 0.2 + kw_s * 0.4 + code_s * 0.4
    if test_s is not None:
        overall = overall * 0.5 + test_s * 0.5
    return round(overall, 4)


def evaluate_model(model, tokenizer, eval_data: dict, cfg: dict, label: str) -> dict:
    model.eval()
    results = {"label": label, "topics": {}, "overall_score": 0.0, "total_time_s": 0.0}
    all_scores = []

    for topic, questions in eval_data.items():
        print(f"  📌 {topic} ({len(questions)} Qs)...", end=" ", flush=True)
        scores = []
        total_t = 0.0
        for q in questions:
            q["topic"] = topic
            ans, t = generate_answer(model, tokenizer, q["question"], cfg["max_new_tokens"])
            s = score_answer(ans, q)
            scores.append(s)
            total_t += t
        avg = sum(scores) / len(scores) if scores else 0.0
        results["topics"][topic] = {"avg_score": round(avg, 4), "scores": scores}
        results["total_time_s"] += total_t
        all_scores.extend(scores)
        print(f"score={avg:.3f}")

    results["overall_score"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    return results


def print_report(base_res: dict, seal_res: Optional[dict]):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 65)
    print("  📊  COMPARISON RESULTS")
    print("=" * 65)

    topics = list(base_res["topics"].keys())
    col = 18

    header = f"  {'Topic':<22} {'Base Model':>{col}}"
    if seal_res:
        header += f" {'SEAL Model':>{col}} {'Δ':>8}"
    print(header)
    print("  " + "─" * (22 + col + (col + 9 if seal_res else 0)))

    for topic in topics:
        b = base_res["topics"][topic]["avg_score"]
        row = f"  {topic:<22} {b:>{col}.3f}"
        if seal_res:
            s = seal_res["topics"].get(topic, {}).get("avg_score", 0.0)
            delta = s - b
            sign = "+" if delta >= 0 else ""
            arrow = "▲" if delta >= 0 else "▼"
            row += f" {s:>{col}.3f} {arrow}{sign}{delta:.3f}"
        print(row)

    print("  " + "─" * (22 + col + (col + 9 if seal_res else 0)))
    bo = base_res["overall_score"]
    row = f"  {'⭐ OVERALL':<22} {bo:>{col}.3f}"
    if seal_res:
        so = seal_res["overall_score"]
        delta = so - bo
        sign = "+" if delta >= 0 else ""
        arrow = "▲" if delta >= 0 else "▼"
        row += f" {so:>{col}.3f} {arrow}{sign}{delta:.3f}"
    print(row)
    print("=" * 65)

    if seal_res:
        delta = seal_res["overall_score"] - base_res["overall_score"]
        pct = (delta / max(base_res["overall_score"], 0.001)) * 100
        print(f"\n  🎯 SEAL improvement: {'+' if delta >= 0 else ''}{delta:.3f} ({pct:+.1f}%)")
        print(f"  ⏱  Base inference:  {base_res['total_time_s']:.1f}s")
        print(f"  ⏱  SEAL inference:  {seal_res['total_time_s']:.1f}s")


def save_report(base_res: dict, seal_res: Optional[dict], out_dir: str = "/content"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"comparison_{ts}.json"
    payload = {"base": base_res, "seal": seal_res, "generated_at": ts}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Report saved → {path}")
    return str(path)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run_comparison(cfg: dict = CONFIG):
    print("=" * 65)
    print("  🔬 SEAL-DSA  ·  Colab Baseline vs SEAL Comparison")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 1. Eval data
    print("\n[1/4] Loading evaluation data...")
    eval_data = load_eval_data(cfg)

    # 2. Base model
    print(f"\n[2/4] Loading base model: {cfg['base_model']}")
    base_model, tokenizer = load_model(cfg["base_model"], use_4bit=cfg["use_4bit"])
    print("  ✓ Base model ready")

    # 3. Evaluate base
    print("\n[3/4] Evaluating BASE model (no fine-tuning)...")
    base_res = evaluate_model(base_model, tokenizer, eval_data, cfg, label="base_model")
    print(f"  ✓ Base overall score: {base_res['overall_score']:.3f}")

    # 4. SEAL model
    seal_res = None
    ckpt = cfg.get("seal_checkpoint_path")
    if ckpt:
        print(f"\n[4/4] Loading SEAL checkpoint: {ckpt}")
        try:
            seal_model = load_seal_model(base_model, ckpt)
            print("  ✓ SEAL model ready — evaluating...")
            seal_res = evaluate_model(seal_model, tokenizer, eval_data, cfg, label="seal_model")
            print(f"  ✓ SEAL overall score: {seal_res['overall_score']:.3f}")
        except Exception as e:
            print(f"  ⚠️  Could not load SEAL checkpoint: {e}")
            print("  Continuing with base-only report.")
    else:
        print("\n[4/4] No SEAL checkpoint configured — skipping SEAL evaluation.")

    # 5. Report
    print_report(base_res, seal_res)
    save_report(base_res, seal_res)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # When running as a script (not in Colab cells), just run directly
    run_comparison(CONFIG)
