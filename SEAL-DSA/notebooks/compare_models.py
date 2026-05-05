"""
SEAL-DSA Quick Comparison — manual LoRA merge (no PeftModel issues).
"""
import torch, warnings, os, gc, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CKPT_DIR = "/content/drive/MyDrive/SEAL-DSA/checkpoints"

QUESTIONS = [
    "Implement Kadane's algorithm for maximum subarray sum in Python.",
    "Explain BFS vs DFS traversal in trees.",
    "What is dynamic programming? Explain memoization vs tabulation.",
]

def load_model():
    print(f"⏳ Loading {MODEL} (float32)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32, trust_remote_code=True
        ).to("cuda")
    t = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    print(f"✅ Loaded ({torch.cuda.memory_allocated()/1e9:.1f}GB)")
    return m, t

def ask(model, tok, question):
    msgs = [
        {"role": "system", "content": "You are a DSA expert. Give clear answers with code."},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    ids = inputs["input_ids"].to("cuda")
    attn = inputs["attention_mask"].to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=attn,
            max_new_tokens=200, do_sample=True,
            temperature=0.3, top_p=0.9,
            repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def find_ckpt():
    if not os.path.exists(CKPT_DIR): return None
    ents = [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint_epoch_")]
    if not ents: return None
    ents.sort(key=lambda x: int(x.split("_")[-1]))
    return f"{CKPT_DIR}/{ents[-1]}"

def manual_merge(model, ckpt_path):
    """Manually merge LoRA weights into base model — no PeftModel needed."""
    with open(f"{ckpt_path}/adapter_config.json") as f:
        cfg = json.load(f)
    alpha = cfg.get("lora_alpha", 16)
    r = cfg.get("r", 8)
    scale = alpha / r

    wt_file = f"{ckpt_path}/adapter_model.safetensors"
    if os.path.exists(wt_file):
        raw = load_file(wt_file, device="cuda")
    else:
        raw = torch.load(f"{ckpt_path}/adapter_model.bin", map_location="cuda", weights_only=True)

    # Group lora_A and lora_B pairs
    pairs = {}
    for key, val in raw.items():
        # Extract: model.layers.X.self_attn.Y_proj
        clean = key.replace("base_model.model.base_model.model.", "")
        clean = clean.replace("base_model.model.", "")
        if "lora_A" in clean:
            base_key = clean.replace(".lora_A.weight", "")
            pairs.setdefault(base_key, {})["A"] = val.float()
        elif "lora_B" in clean:
            base_key = clean.replace(".lora_B.weight", "")
            pairs.setdefault(base_key, {})["B"] = val.float()

    merged = 0
    sd = dict(model.named_parameters())
    for layer_name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        # Find matching parameter in model
        for param_name, param in sd.items():
            if layer_name in param_name and "lora" not in param_name:
                delta = (ab["B"] @ ab["A"]) * scale
                param.data += delta.to(param.dtype)
                merged += 1
                break

    print(f"📊 Manually merged {merged} LoRA layers (α={alpha}, r={r}, scale={scale})")
    norms = []
    for layer_name, ab in pairs.items():
        if "A" in ab and "B" in ab:
            delta = (ab["B"] @ ab["A"]) * scale
            norms.append(delta.norm().item())
    if norms:
        print(f"   LoRA delta norms: min={min(norms):.4f}, max={max(norms):.4f}, avg={sum(norms)/len(norms):.4f}")
    return model

# ── Main ──────────────────────────────────────────────────
print("=" * 60)
print("  🔬 SEAL-DSA Quick Comparison")
print("=" * 60)

base, tok = load_model()

print("\n📘 BASE MODEL ANSWERS:")
base_answers = []
for i, q in enumerate(QUESTIONS):
    print(f"\n--- Q{i+1}: {q}")
    a = ask(base, tok, q)
    base_answers.append(a)
    print(a[:500])

del base; gc.collect(); torch.cuda.empty_cache()

ckpt = find_ckpt()
if not ckpt:
    print("\n❌ No checkpoint found!")
else:
    print(f"\n⏳ Loading SEAL (manual merge from {os.path.basename(ckpt)})...")
    seal, tok2 = load_model()
    seal = manual_merge(seal, ckpt)
    print(f"✅ SEAL ready ({torch.cuda.memory_allocated()/1e9:.1f}GB)")

    print("\n📗 SEAL MODEL ANSWERS:")
    seal_answers = []
    for i, q in enumerate(QUESTIONS):
        print(f"\n--- Q{i+1}: {q}")
        a = ask(seal, tok2, q)
        seal_answers.append(a)
        print(a[:500])

    print(f"\n{'='*60}")
    print("  📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    for i, q in enumerate(QUESTIONS):
        bw, sw = len(base_answers[i].split()), len(seal_answers[i].split())
        same = "⚠️ SAME" if base_answers[i].strip()==seal_answers[i].strip() else "✅ DIFFERENT"
        print(f"\n  Q{i+1}: {q[:55]}...")
        print(f"    Base: {bw} words | SEAL: {sw} words | {same}")
