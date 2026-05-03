#!/usr/bin/env python3
"""
=============================================================================
SEAL-DSA Local Checkpoint Comparison Script
=============================================================================
PURPOSE:
    Run this script WHERE YOUR CHECKPOINTS ARE SAVED (locally / on the machine
    that ran SEAL training).

    It loads every saved checkpoint from your `checkpoints/` directory,
    runs them through the held-out evaluation set, and produces a full
    comparison table + JSON report.

USAGE:
    python compare_with_checkpoints.py [--checkpoint-dir checkpoints] \
                                       [--eval-data data/evaluation_sets/dsa_eval_set.json] \
                                       [--output-dir results] \
                                       [--model-name Qwen/Qwen2.5-1.5B-Instruct]

REQUIREMENTS:
    pip install torch transformers peft accelerate bitsandbytes tqdm tabulate

=============================================================================
"""

import argparse
import json
import os
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Graceful import handling ────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with:  pip install torch transformers peft accelerate bitsandbytes tqdm")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ─── Constants ────────────────────────────────────────────────────────────────
DIVIDER = "=" * 70
TOPIC_EMOJI = {
    "arrays_strings": "📊",
    "linked_lists": "🔗",
    "stacks_queues": "📚",
    "trees": "🌳",
    "graphs": "🕸️",
    "sorting_searching": "🔍",
    "dynamic_programming": "🧮",
}


# ─── Argument Parser ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SEAL checkpoints on the DSA evaluation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory containing SEAL checkpoints (default: checkpoints/)"
    )
    parser.add_argument(
        "--eval-data", default="data/evaluation_sets/dsa_eval_set.json",
        help="Path to the evaluation JSON file"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save comparison results (default: results/)"
    )
    parser.add_argument(
        "--model-name", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name (must match what was used during training)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum tokens to generate per answer"
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Disable 4-bit quantization (uses more VRAM but is more accurate)"
    )
    parser.add_argument(
        "--topics", nargs="+", default=None,
        help="Evaluate only specific topics (e.g. --topics trees graphs)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit questions per topic for quick tests"
    )
    return parser.parse_args()


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_eval_data(path: str, topics_filter: Optional[List[str]] = None,
                   limit: Optional[int] = None) -> Dict[str, List[dict]]:
    """Load and optionally filter the evaluation dataset."""
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Eval data not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    topics = data.get("topics", {})
    if topics_filter:
        topics = {k: v for k, v in topics.items() if k in topics_filter}

    if limit:
        topics = {k: v[:limit] for k, v in topics.items()}

    total = sum(len(v) for v in topics.values())
    print(f"[INFO] Loaded {total} questions across {len(topics)} topics")
    return topics


# ─── Checkpoint Discovery ─────────────────────────────────────────────────────
def find_checkpoints(checkpoint_dir: str) -> List[Path]:
    """
    Find all valid checkpoint directories.
    Supports both:
      - PEFT/LoRA checkpoints  (contains adapter_config.json)
      - Full saved models      (contains config.json or pytorch_model.bin)
    """
    base = Path(checkpoint_dir)
    if not base.exists():
        print(f"[ERROR] Checkpoint directory not found: {base}")
        print("  Make sure you're running this script from the SEAL-DSA root directory.")
        sys.exit(1)

    checkpoints = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        # LoRA adapter
        if (p / "adapter_config.json").exists():
            checkpoints.append(p)
        # Full model
        elif (p / "config.json").exists() or (p / "pytorch_model.bin").exists():
            checkpoints.append(p)

    if not checkpoints:
        print(f"[WARN] No checkpoints found in: {base}")
        print("  Directories found:")
        for d in base.iterdir():
            print(f"    {d.name}")
    else:
        print(f"[INFO] Found {len(checkpoints)} checkpoint(s):")
        for c in checkpoints:
            print(f"    • {c.name}")

    return checkpoints


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_base_model(model_name: str, use_quantize: bool = True):
    """Load the base model (optionally with 4-bit quantization)."""
    print(f"\n[MODEL] Loading base model: {model_name}")

    bnb_config = None
    if use_quantize and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[INFO] Using 4-bit quantization (QLoRA)")
    elif not torch.cuda.is_available():
        print("[WARN] No GPU detected — running on CPU (will be slow)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def load_checkpoint_model(base_model, checkpoint_path: Path):
    """Load a PEFT/LoRA checkpoint on top of the base model."""
    is_lora = (checkpoint_path / "adapter_config.json").exists()
    if is_lora:
        print(f"[MODEL] Loading LoRA adapter from: {checkpoint_path.name}")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        model = model.merge_and_unload()  # Merge LoRA weights for clean inference
    else:
        print(f"[MODEL] Loading full model from: {checkpoint_path.name}")
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return model


# ─── Answer Generation & Scoring ─────────────────────────────────────────────
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 256) -> Tuple[str, float]:
    """Generate an answer and return (answer_text, generation_time_s)."""
    prompt = (
        "You are an expert in Data Structures and Algorithms. "
        "Answer the following DSA question clearly and concisely.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
    elapsed = time.time() - t0

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, elapsed


def score_answer(answer: str, question_data: dict) -> Dict[str, float]:
    """
    Score an answer using heuristic metrics since we don't have a judge model.

    Metrics:
        length_score    — penalises empty / very short answers
        keyword_score   — checks presence of expected DSA concepts
        code_score      — rewards answers that include code (for coding questions)
        overall         — weighted combination
    """
    q_type = question_data.get("type", "conceptual")
    answer_lower = answer.lower()
    scores = {}

    # 1. Length score (0–1)
    words = len(answer.split())
    if words == 0:
        scores["length"] = 0.0
    elif words < 10:
        scores["length"] = 0.3
    elif words < 30:
        scores["length"] = 0.6
    elif words < 300:
        scores["length"] = 1.0
    else:
        scores["length"] = 0.9  # Slight penalty for very long rambling answers

    # 2. Keyword score — topic-specific keywords
    topic = question_data.get("topic", "")
    topic_keywords = {
        "arrays_strings": ["array", "string", "index", "pointer", "subarray", "window"],
        "linked_lists": ["node", "next", "pointer", "head", "tail", "list"],
        "stacks_queues": ["stack", "queue", "push", "pop", "fifo", "lifo", "top"],
        "trees": ["tree", "node", "root", "leaf", "traversal", "height", "depth", "bst"],
        "graphs": ["graph", "vertex", "edge", "bfs", "dfs", "visited", "cycle"],
        "sorting_searching": ["sort", "search", "binary", "pivot", "merge", "complexity"],
        "dynamic_programming": ["dp", "memoization", "subproblem", "state", "transition", "cache"],
    }
    keywords = topic_keywords.get(topic, [])

    # Also pick up words from the question itself
    question_words = question_data.get("question", "").lower().split()
    important_words = [w for w in question_words if len(w) > 4]
    all_kws = list(set(keywords + important_words))

    if all_kws:
        hit = sum(1 for kw in all_kws if kw in answer_lower)
        scores["keyword"] = min(hit / max(len(all_kws) * 0.4, 1), 1.0)
    else:
        scores["keyword"] = 0.5

    # 3. Code score (for coding questions)
    if q_type == "coding":
        has_code = (
            "def " in answer or
            "return " in answer or
            "for " in answer or
            "while " in answer or
            "```" in answer
        )
        scores["code"] = 1.0 if has_code else 0.0
    else:
        scores["code"] = 1.0  # Not applicable — full marks

    # 4. Testcase execution (only if function_name + test_cases present)
    if "function_name" in question_data and "test_cases" in question_data:
        pass_count, total_count = run_test_cases(answer, question_data)
        scores["test_pass_rate"] = pass_count / max(total_count, 1)
    else:
        scores["test_pass_rate"] = None  # N/A

    # 5. Overall weighted score
    code_w = 0.4 if q_type == "coding" else 0.0
    remaining_w = 1.0 - code_w
    overall = (
        scores["length"] * (remaining_w * 0.3) +
        scores["keyword"] * (remaining_w * 0.7) +
        scores["code"] * code_w
    )
    if scores["test_pass_rate"] is not None:
        # Blend test pass rate with overall
        overall = overall * 0.5 + scores["test_pass_rate"] * 0.5

    scores["overall"] = round(overall, 4)
    return scores


def run_test_cases(answer: str, question_data: dict) -> Tuple[int, int]:
    """
    Attempt to extract a Python function from the answer and run test cases.
    Returns (passed, total).
    """
    function_name = question_data.get("function_name", "")
    test_cases = question_data.get("test_cases", [])
    if not function_name or not test_cases:
        return 0, 0

    # Extract code block from answer
    code = ""
    if "```python" in answer:
        code = answer.split("```python")[1].split("```")[0]
    elif "```" in answer:
        code = answer.split("```")[1].split("```")[0]
    elif "def " + function_name in answer:
        # Try to grab function definition
        idx = answer.find("def " + function_name)
        code = answer[idx:]

    if not code.strip():
        return 0, 0

    passed = 0
    namespace = {}
    try:
        exec(code, namespace)
        func = namespace.get(function_name)
        if func is None:
            return 0, 0

        for tc in test_cases:
            try:
                args = eval(tc["input"])
                if not isinstance(args, tuple):
                    args = (args,)
                result = func(*args)
                if result == tc["expected"]:
                    passed += 1
            except Exception:
                pass  # Test case failed
    except Exception:
        pass  # Code failed to compile/execute

    return passed, len(test_cases)


# ─── Evaluation Runner ────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, eval_data: Dict[str, List[dict]],
                   max_new_tokens: int, label: str) -> Dict:
    """Evaluate a model on all topics and return structured results."""
    model.eval()
    results = {
        "label": label,
        "topics": {},
        "overall_score": 0.0,
        "total_questions": 0,
        "total_time_s": 0.0,
        "avg_time_per_q": 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    all_scores = []
    total_time = 0.0

    for topic, questions in eval_data.items():
        emoji = TOPIC_EMOJI.get(topic, "📌")
        print(f"\n  {emoji} Evaluating: {topic} ({len(questions)} questions)")
        topic_scores = []
        topic_answers = []

        for q_data in tqdm(questions, desc=f"    {topic}", leave=False):
            q_data["topic"] = topic  # Inject topic for keyword scoring
            question = q_data.get("question", "")
            answer, gen_time = generate_answer(model, tokenizer, question, max_new_tokens)
            scores = score_answer(answer, q_data)

            topic_scores.append(scores["overall"])
            total_time += gen_time
            topic_answers.append({
                "question": question,
                "answer": answer[:300] + "..." if len(answer) > 300 else answer,
                "scores": scores,
                "gen_time_s": round(gen_time, 3),
            })

        avg = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
        results["topics"][topic] = {
            "avg_score": round(avg, 4),
            "num_questions": len(questions),
            "scores": topic_scores,
            "answers": topic_answers,
        }
        all_scores.extend(topic_scores)
        print(f"    ✓ Avg score: {avg:.3f}")

    results["overall_score"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    results["total_questions"] = len(all_scores)
    results["total_time_s"] = round(total_time, 2)
    results["avg_time_per_q"] = round(total_time / max(len(all_scores), 1), 3)
    return results


# ─── Report Generation ────────────────────────────────────────────────────────
def print_comparison_table(all_results: List[Dict]):
    """Print a rich comparison table to stdout."""
    print(f"\n{DIVIDER}")
    print("  📊 SEAL-DSA CHECKPOINT COMPARISON RESULTS")
    print(DIVIDER)

    # Get all topics
    topics = list(all_results[0]["topics"].keys()) if all_results else []

    # Build table rows
    headers = ["Topic"] + [r["label"] for r in all_results]
    rows = []

    for topic in topics:
        emoji = TOPIC_EMOJI.get(topic, "📌")
        row = [f"{emoji} {topic}"]
        for res in all_results:
            score = res["topics"].get(topic, {}).get("avg_score", 0.0)
            row.append(f"{score:.3f}")
        rows.append(row)

    # Overall row
    overall_row = ["⭐ OVERALL"]
    for res in all_results:
        overall_row.append(f"{res['overall_score']:.3f}")
    rows.append(["—" * 20] + ["—" * 10] * len(all_results))
    rows.append(overall_row)

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        # Fallback plain text table
        col_w = 20
        print("  " + " | ".join(h.ljust(col_w) for h in headers))
        print("  " + "-+-".join("-" * col_w for _ in headers))
        for row in rows:
            print("  " + " | ".join(str(c).ljust(col_w) for c in row))

    # Timing summary
    print(f"\n{'─'*70}")
    print(f"  {'Model':<30} {'Qs':>5} {'Time(s)':>10} {'Avg/Q(s)':>10} {'Score':>8}")
    print(f"  {'─'*30} {'─'*5} {'─'*10} {'─'*10} {'─'*8}")
    for res in all_results:
        print(
            f"  {res['label']:<30} "
            f"{res['total_questions']:>5} "
            f"{res['total_time_s']:>10.1f} "
            f"{res['avg_time_per_q']:>10.3f} "
            f"{res['overall_score']:>8.3f}"
        )
    print(DIVIDER)


def save_results(all_results: List[Dict], output_dir: str):
    """Save full results to JSON and a compact summary to text."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON
    json_path = out / f"comparison_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "generated_at": ts}, f, indent=2)
    print(f"\n[SAVED] Full results → {json_path}")

    # Summary JSON (compact, no per-answer text)
    summary = []
    for r in all_results:
        summary.append({
            "label": r["label"],
            "overall_score": r["overall_score"],
            "total_questions": r["total_questions"],
            "avg_time_per_q": r["avg_time_per_q"],
            "per_topic": {t: d["avg_score"] for t, d in r["topics"].items()},
        })

    summary_path = out / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Summary       → {summary_path}")

    return json_path, summary_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(DIVIDER)
    print("  🔬 SEAL-DSA  ·  Local Checkpoint Comparison")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIVIDER)

    # 1. Load evaluation data
    print("\n[STEP 1/4] Loading evaluation data...")
    eval_data = load_eval_data(args.eval_data, args.topics, args.limit)
    if not eval_data:
        print("[ERROR] No evaluation data loaded. Exiting.")
        sys.exit(1)

    # 2. Discover checkpoints
    print("\n[STEP 2/4] Discovering checkpoints...")
    checkpoints = find_checkpoints(args.checkpoint_dir)

    # 3. Load base model once (shared backbone for LoRA models)
    print("\n[STEP 3/4] Loading base model...")
    base_model, tokenizer = load_base_model(args.model_name, use_quantize=not args.no_quantize)

    all_results = []

    # 4A. Evaluate BASE model first (no LoRA — this is the baseline)
    print(f"\n{'─'*70}")
    print("  Evaluating: BASE MODEL (no fine-tuning)")
    print(f"{'─'*70}")
    base_result = evaluate_model(
        base_model, tokenizer, eval_data, args.max_new_tokens,
        label="base_model"
    )
    all_results.append(base_result)

    # 4B. Evaluate each checkpoint
    for ckpt_path in checkpoints:
        print(f"\n{'─'*70}")
        print(f"  Evaluating checkpoint: {ckpt_path.name}")
        print(f"{'─'*70}")

        # Load LoRA on top of base (re-use base weights)
        try:
            ckpt_model = load_checkpoint_model(base_model, ckpt_path)
            result = evaluate_model(
                ckpt_model, tokenizer, eval_data, args.max_new_tokens,
                label=ckpt_path.name
            )
            all_results.append(result)
            del ckpt_model  # Free memory
        except Exception as e:
            print(f"  [ERROR] Failed to load {ckpt_path.name}: {e}")
            continue

    # 5. Print and save results
    print("\n[STEP 4/4] Generating report...")
    print_comparison_table(all_results)

    json_path, summary_path = save_results(all_results, args.output_dir)

    # Delta table: improvement over base
    if len(all_results) > 1:
        base_score = all_results[0]["overall_score"]
        print(f"\n  📈 Improvement over Base Model ({base_score:.3f}):")
        for res in all_results[1:]:
            delta = res["overall_score"] - base_score
            sign = "+" if delta >= 0 else ""
            bar = "█" * int(abs(delta) * 50)
            direction = "▲" if delta >= 0 else "▼"
            print(f"    {direction} {res['label']:<30} {sign}{delta:.3f}  {bar}")

    print(f"\n✅ Done! Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
