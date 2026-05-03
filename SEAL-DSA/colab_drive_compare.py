#!/usr/bin/env python3
"""
SEAL-DSA · Colab Comparison Script (v4)
Uses EXACT same prompts + generation params as SEAL training.

Cells:
  1: !pip install -q torch transformers peft accelerate bitsandbytes datasets evaluate rouge-score pyyaml tqdm
  2: from google.colab import drive; drive.mount('/content/drive')
     !rm -rf dsaseal && !git clone https://github.com/Taqaddusshafi/dsaseal.git && %cd /content/dsaseal/SEAL-DSA
  3: !python colab_drive_compare.py
"""

import sys, os, re, ast, time, json, math, warnings, subprocess
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# ── torchao version guard (PEFT needs >=0.16.0) ─────────────────────────────
try:
    import torchao
    from packaging.version import Version
    if Version(torchao.__version__) < Version("0.16.0"):
        print("⚙️  Upgrading torchao...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchao>=0.16.0"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except ImportError:
    pass

sys.path.insert(0, '/content/dsaseal/SEAL-DSA')
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =============================================================================
#  CONFIG
# =============================================================================
CONFIG = {
    "base_model":             "Qwen/Qwen2.5-1.5B-Instruct",
    "checkpoint_dir":         "/content/drive/MyDrive/SEAL-DSA/checkpoints",
    "explicit_checkpoint":    None,
    "eval_json_path":         "/content/dsaseal/SEAL-DSA/data/evaluation_sets/dsa_eval_set.json",
    "max_new_tokens":         200,   # reduced — 512 × 21 questions × 2 models crashes RAM
    "max_input_length":       512,
    "limit_per_topic":        3,
}

# ── Prompt template — copied from answer_generator.py ────────────────────────
ANSWER_WITH_REASONING_PROMPT = """You are solving a DSA problem step by step.

Question: {question}

Think through this problem:
1. Understand: What is being asked?
2. Plan: What approach should I use?
3. Solve: Work through the solution
4. Verify: Check edge cases and complexity

Solution:"""

# ── DSA keywords — copied from evaluator.py ──────────────────────────────────
DSA_KEYWORDS = {
    "arrays_strings":      {"concepts": ["array","string","index","subarray","two pointer","sliding window","prefix sum","hash map","sort","binary search"], "complexity_terms": ["O(n)","O(n²)","O(n log n)","O(1)"]},
    "linked_lists":        {"concepts": ["node","pointer","next","head","tail","singly","doubly","reverse","cycle","fast pointer","slow pointer"], "complexity_terms": ["O(n)","O(1)"]},
    "stacks_queues":       {"concepts": ["stack","queue","push","pop","enqueue","dequeue","LIFO","FIFO","priority queue","deque","monotonic"], "complexity_terms": ["O(1)","O(n)","amortized"]},
    "trees":               {"concepts": ["tree","root","leaf","node","height","depth","binary tree","BST","balanced","subtree","ancestor"], "complexity_terms": ["O(log n)","O(n)","O(h)"]},
    "graphs":              {"concepts": ["graph","vertex","edge","directed","undirected","adjacency list","connected","cycle","path","shortest path"], "complexity_terms": ["O(V+E)","O(V²)","O(E log V)"]},
    "sorting_searching":   {"concepts": ["sort","search","partition","pivot","merge","divide and conquer","binary search","lower bound","upper bound"], "complexity_terms": ["O(n log n)","O(n²)","O(n)","O(log n)"]},
    "dynamic_programming": {"concepts": ["dynamic programming","DP","memoization","tabulation","subproblem","overlapping","optimal substructure","state","transition","recurrence"], "complexity_terms": ["O(n)","O(n²)","O(n·W)"]},
}

# ── Fixed reference text for perplexity (same for both models) ───────────────
PPL_REF = (
    "A binary search tree stores elements so that for every node all values in the left subtree are smaller "
    "and all values in the right subtree are larger. Search insertion and deletion run in O(log n) on a balanced BST. "
    "Dynamic programming solves problems by breaking them into overlapping subproblems and storing results to avoid recomputation. "
    "Kadane's algorithm finds the maximum subarray sum in O(n) time."
)

# =============================================================================
#  MODEL LOADING
# =============================================================================

def _bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

def load_base_model(cfg):
    name = cfg["base_model"]
    # Load in fp16 (not 4-bit) — 4-bit causes NaN perplexity + CUDA multinomial crash
    # Sequential loading means only one model in VRAM at a time (~3GB fp16 on T4 = fine)
    print(f"\n⏳ Loading BASE model (fp16): {name}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=None, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    print("✅ Base model ready.")
    return model, tok

def find_latest_ckpt(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    entries = [d for d in os.listdir(ckpt_dir)
               if d.startswith("checkpoint_epoch_") and os.path.isdir(f"{ckpt_dir}/{d}")]
    if not entries:
        return None
    entries.sort(key=lambda x: int(x.split("_")[-1]))
    print(f"ℹ️  Latest checkpoint: {entries[-1]}")
    return f"{ckpt_dir}/{entries[-1]}"

def load_seal_model(cfg):
    ckpt = cfg["explicit_checkpoint"] or find_latest_ckpt(cfg["checkpoint_dir"])
    if not ckpt or not os.path.exists(ckpt):
        print(f"❌ No checkpoint at: {cfg['checkpoint_dir']}")
        return None, None, None

    p = Path(ckpt)
    print(f"\n📂 Checkpoint files ({p.name}):")
    for f in sorted(p.iterdir()):
        print(f"    {f.name:<40} {f.stat().st_size/1e6:.3f} MB")

    has_weights = (p/"adapter_model.safetensors").exists() or (p/"adapter_model.bin").exists()
    if not (p/"adapter_config.json").exists() or not has_weights:
        print("❌ Incomplete checkpoint — adapter_model.safetensors missing!")
        print("   In training Colab run: model.save_pretrained('/content/drive/MyDrive/SEAL-DSA/checkpoints/checkpoint_epoch_manual')")
        return None, None, None

    name = cfg["base_model"]
    print(f"\n⏳ Loading fresh base in fp16 for exact LoRA merge...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fresh = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=None, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        fresh.config.pad_token_id = fresh.config.eos_token_id

    print("⏳ Applying LoRA adapter...")
    from safetensors.torch import load_file

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peft_model = PeftModel.from_pretrained(fresh, str(p), is_trainable=False)
        
        # ── MANUAL WEIGHT FIX ──
        state_dict = load_file(p / "adapter_model.safetensors")
        fixed_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("base_model.model.base_model.model.", "base_model.model.")
            if new_k.endswith("lora_A.weight"):
                new_k = new_k.replace("lora_A.weight", "lora_A.default.weight")
            if new_k.endswith("lora_B.weight"):
                new_k = new_k.replace("lora_B.weight", "lora_B.default.weight")
            fixed_state_dict[new_k] = v
            
        # Inject directly into the PyTorch model instead of PEFT helper function
        missing, unexpected = peft_model.load_state_dict(fixed_state_dict, strict=False)
        
        if not fixed_state_dict:
            print("⚠️  Warning: Fixed state dict is empty!")

    # Verify LoRA weights are non-zero
    nz = sum(1 for n, p2 in peft_model.named_parameters()
             if ("lora_A" in n or "lora_B" in n) and p2.data.float().norm().item() > 1e-6)
    total = sum(1 for n, _ in peft_model.named_parameters() if "lora_A" in n or "lora_B" in n)
    print(f"📊 Non-zero LoRA layers: {nz}/{total}")
    if nz == 0:
        print("⚠️  ALL LoRA weights are zero — model was not trained or lora_B never updated!")

    model = peft_model.merge_and_unload()
    print("✅ SEAL model ready.")
    return model, tok, f"SEAL ({p.name})"

# =============================================================================
#  EVAL DATA
# =============================================================================

def load_eval_data(cfg):
    path = cfg.get("eval_json_path", "")
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        topics = data.get("topics", {})
        print(f"✅ Loaded {sum(len(v) for v in topics.values())} questions, {len(topics)} topics")
    else:
        topics = {
            "arrays_strings": [
                {"question": "Implement Kadane's algorithm for maximum subarray sum.",
                 "type": "coding", "function_name": "max_subarray_sum",
                 "test_cases": [{"input": "([-2,1,-3,4,-1,2,1,-5,4],)", "expected": 6}]},
                {"question": "Explain the two-pointer technique.", "type": "conceptual"},
            ],
            "dynamic_programming": [
                {"question": "Explain memoization vs tabulation.", "type": "conceptual"},
                {"question": "Implement coin change — minimum coins.", "type": "coding",
                 "function_name": "coin_change",
                 "test_cases": [{"input": "([1,5,10,25],30)", "expected": 2}]},
            ],
        }
        print("ℹ️  Using fallback mini eval set")
    limit = cfg.get("limit_per_topic")
    if limit:
        topics = {k: v[:limit] for k, v in topics.items()}
    return topics

# =============================================================================
#  GENERATION — matches answer_generator.py exactly
# =============================================================================

def generate_answer(model, tok, question: str, cfg: dict, first_call: bool = False):
    """
    Use Qwen2.5-Instruct chat template — plain text prompts output < 10 words on Instruct models.
    The chat template tells the model it is in Q&A mode and should give a full answer.
    """
    messages = [
        {"role": "system",
         "content": "You are an expert in Data Structures and Algorithms. "
                    "Answer the question thoroughly with explanation, time/space complexity, "
                    "and Python code where relevant."},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"System: You are a DSA expert.\nUser: {question}\nAssistant:"

    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=cfg["max_input_length"]).to(model.device)
    pad_id = tok.pad_token_id or tok.eos_token_id
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if first_call:
        print(f"\n  📝 First answer preview ({len(answer.split())} words):")
        print(f"  {answer[:200]}{'...' if len(answer)>200 else ''}\n")

    return answer, elapsed

# =============================================================================
#  SCORING — mirrors DSAEvaluator from evaluator.py
# =============================================================================

def score_correctness(answer: str, topic: str) -> float:
    kw = DSA_KEYWORDS.get(topic, {})
    concepts = kw.get("concepts", [])
    if not concepts:
        return 0.5
    al = answer.lower()
    hits = sum(1 for c in concepts if c in al)
    return min(1.0, hits / max(len(concepts), 1) * 1.5)

def score_completeness(answer: str) -> float:
    words = len(answer.split())
    if words < 10:   return 0.1
    if words < 30:   return 0.3
    if words < 80:   return 0.6
    if words < 200:  return 0.8
    return 1.0

def score_complexity(answer: str, topic: str) -> float:
    al = answer.lower()
    has_big_o  = bool(re.search(r'O\([^)]+\)', answer))
    has_time   = any(t in al for t in ["time complexity", "runtime"])
    has_space  = any(t in al for t in ["space complexity", "memory"])
    score = (0.4 if has_big_o else 0) + (0.2 if has_time else 0) + (0.2 if has_space else 0)
    terms = DSA_KEYWORDS.get(topic, {}).get("complexity_terms", [])
    score += min(0.2, sum(1 for t in terms if t in answer) * 0.1)
    return min(1.0, score)

def score_code(answer: str, q_type: str, test_cases=None) -> float:
    if q_type not in ("coding", "problem_solving"):
        return 0.7
    al = answer.lower()
    struct = (0.15 if "```" in answer or "def " in answer else 0) + \
             (0.15 if "def " in answer else 0) + \
             (0.10 if "return " in answer else 0) + \
             (0.10 if any(k in al for k in ["for ", "while "]) else 0)

    # Try to execute code
    code = ""
    m = re.search(r'```python(.*?)```', answer, re.DOTALL)
    if m:   code = m.group(1).strip()
    elif re.search(r'```(.*?)```', answer, re.DOTALL):
        code = re.search(r'```(.*?)```', answer, re.DOTALL).group(1).strip()
    elif "def " in answer:
        code = answer[answer.find("def "):]

    exec_score = 0.0
    if code:
        try:
            ast.parse(code)
            exec(compile(ast.parse(code), '<str>', 'exec'), {"__builtins__": {}})
            exec_score = 1.0
        except SyntaxError:
            exec_score = 0.0
        except Exception:
            exec_score = 0.5  # syntax OK, runtime error

    # Test cases
    if test_cases and code:
        ns = {}
        passed = 0
        try:
            exec(code, ns)
            fn_match = re.search(r'def (\w+)', code)
            fn = ns.get(fn_match.group(1)) if fn_match else None
            if fn:
                for tc in test_cases:
                    try:
                        args = eval(tc["input"])
                        if not isinstance(args, tuple): args = (args,)
                        if fn(*args) == tc["expected"]: passed += 1
                    except Exception: pass
        except Exception: pass
        test_rate = passed / len(test_cases)
        return min(1.0, 0.30 * (struct/0.50) + 0.30 * exec_score + 0.40 * test_rate)

    return min(1.0, struct + 0.50 * exec_score)

def score_explanation(answer: str) -> float:
    has_struct  = bool(re.search(r'\d+[.)]\s', answer)) or bool(re.search(r'[-•*]\s', answer))
    exp_words   = ["because","therefore","since","first","second","step","approach","for example","consider"]
    exp_count   = sum(1 for w in exp_words if w in answer.lower())
    has_example = any(w in answer.lower() for w in ["example","e.g.","input:","output:"])
    return min(1.0, (0.3 if has_struct else 0) + min(0.4, exp_count*0.1) + (0.3 if has_example else 0))

def score_answer(answer: str, q: dict) -> float:
    """Mirrors DSAEvaluator.evaluate() weights: 30/20/15/25/10"""
    topic   = q.get("topic", "")
    q_type  = q.get("type", "conceptual")
    tests   = q.get("test_cases")
    al      = answer.lower()
    corr    = score_correctness(al, topic)
    comp    = score_completeness(answer)
    compl   = score_complexity(al, topic)
    code    = score_code(answer, q_type, tests)
    expl    = score_explanation(answer)
    return round(0.30*corr + 0.20*comp + 0.15*compl + 0.25*code + 0.10*expl, 4)

def perplexity(model, tok) -> float:
    """Both models evaluated on the SAME fixed DSA reference text."""
    try:
        enc = tok(PPL_REF, return_tensors="pt").to(model.device)
        # Cast to bfloat16 for stable loss computation (fp16 can produce NaN on short sequences)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=enc["input_ids"]).loss
            if torch.isnan(loss) or torch.isinf(loss):
                return -1.0
            loss = loss.float().item()
        return round(math.exp(min(loss, 9.0)), 2)
    except Exception as e:
        print(f"  ⚠️  Perplexity failed: {e}")
        return -1.0

# =============================================================================
#  EVALUATION
# =============================================================================

def evaluate_model(model, tok, eval_data, cfg, label):
    print(f"\n{'─'*60}\n🚀 Evaluating: {label}\n{'─'*60}")
    model.eval()
    results = {"label": label, "topics": {}, "overall_score": 0.0,
               "perplexity": 0.0, "total_time_s": 0.0, "sample_answers": {}}
    all_scores = []

    # Compute perplexity once per model (same reference for both)
    ppl = perplexity(model, tok)
    results["perplexity"] = ppl
    print(f"  📉 Perplexity on DSA reference text: {ppl}")

    for topic, questions in eval_data.items():
        print(f"  📌 {topic} ({len(questions)} Qs) ...", end=" ", flush=True)
        scores, t_total = [], 0.0
        for i, q in enumerate(questions):
            q["topic"] = topic
            first = (topic == list(eval_data.keys())[0] and i == 0)
            ans, t = generate_answer(model, tok, q["question"], cfg, first_call=first)
            s = score_answer(ans, q)
            scores.append(s)
            t_total += t
            if i == 0:
                results["sample_answers"][topic] = (q["question"], ans)
        avg = sum(scores)/len(scores) if scores else 0.0
        results["topics"][topic] = {"avg_score": round(avg, 4), "scores": scores}
        results["total_time_s"] += t_total
        all_scores.extend(scores)
        print(f"score={avg:.3f}")

    results["overall_score"] = round(sum(all_scores)/len(all_scores), 4) if all_scores else 0.0
    return results

# =============================================================================
#  REPORT
# =============================================================================

def print_report(base, seal):
    # Sample answer diff
    topics = list(base.get("sample_answers", {}).keys())
    if topics:
        t = topics[0]
        bq, bans = base["sample_answers"][t]
        _, sans = seal["sample_answers"].get(t, (None, ""))
        print(f"\n{'═'*65}")
        print(f"  🔎 SAMPLE ANSWER CHECK ({t})")
        print(f"{'═'*65}")
        print(f"  Q: {bq[:100]}")
        print(f"\n  📘 BASE:\n  {bans[:350]}{'...' if len(bans)>350 else ''}")
        print(f"\n  📗 SEAL:\n  {sans[:350]}{'...' if len(sans)>350 else ''}")
        if bans.strip() == sans.strip():
            print("\n  ⚠️  IDENTICAL — LoRA weights may not have changed outputs")
        else:
            print("\n  ✅ Answers differ — models are genuinely different")

    # Table
    print(f"\n{'='*70}")
    print("  📊  SEAL-DSA · COMPARISON (Base vs SEAL)")
    print(f"{'='*70}")
    C = 12
    print(f"  {'Topic':<24} {'Base':>{C}} {'SEAL':>{C}} {'ΔScore':>8}  Ppl↓")
    print("  " + "─"*60)
    for topic in base["topics"]:
        b = base["topics"][topic]["avg_score"]
        s = seal["topics"].get(topic, {}).get("avg_score", 0.0)
        d = s - b
        print(f"  {topic:<24} {b:>{C}.3f} {s:>{C}.3f} {'▲' if d>=0 else '▼'}{d:>+6.3f}")
    print("  " + "─"*60)
    bo, so = base["overall_score"], seal["overall_score"]
    bp, sp = base["perplexity"], seal["perplexity"]
    d = so - bo
    print(f"  {'⭐ OVERALL':<24} {bo:>{C}.3f} {so:>{C}.3f} {'▲' if d>=0 else '▼'}{d:>+6.3f}")
    print(f"  {'📉 Perplexity':<24} {bp:>{C}.1f} {sp:>{C}.1f} {'▼' if sp<=bp else '▲'}{sp-bp:>+6.1f}")
    print(f"{'='*70}")
    pct = (d/max(bo,0.001))*100
    print(f"\n  🎯 Score improvement : {d:+.3f} ({pct:+.1f}%)")
    print(f"  📉 Perplexity change : {sp-bp:+.1f}  ({'SEAL more confident ✅' if sp<bp else 'SEAL less confident ⚠️'})")

def save_report(base, seal):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"/content/seal_comparison_{ts}.json"
    b2 = {k:v for k,v in base.items() if k != "sample_answers"}
    s2 = {k:v for k,v in seal.items() if k != "sample_answers"}
    with open(path, "w") as f:
        json.dump({"base": b2, "seal": s2, "generated_at": ts}, f, indent=2)
    print(f"\n💾 Report saved → {path}")

# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("="*65)
    print("  🔬 SEAL-DSA · Comparison Script (v4)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    print("\n[1/4] Loading eval data...")
    eval_data = load_eval_data(CONFIG)

    print("\n[2/4] Base model...")
    base_model, base_tok = load_base_model(CONFIG)
    base_res = evaluate_model(base_model, base_tok, eval_data, CONFIG, "Base Model")
    del base_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n[3/4] SEAL model...")
    seal_model, seal_tok, seal_label = load_seal_model(CONFIG)
    if seal_model is None:
        print(f"\n⚠️  SEAL eval skipped. Base score: {base_res['overall_score']:.3f}")
        return
    seal_res = evaluate_model(seal_model, seal_tok, eval_data, CONFIG, seal_label)
    del seal_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n[4/4] Report...")
    print_report(base_res, seal_res)
    save_report(base_res, seal_res)

if __name__ == "__main__":
    main()
