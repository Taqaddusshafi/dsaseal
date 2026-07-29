#!/usr/bin/env python3
"""
SEAL-DSA — learning-curve collector.

Evaluates the base model and EVERY checkpoint on the held-out set, producing a
proper base -> epoch1 -> epoch2 -> ... trajectory.

IMPORTANT — why this exists rather than using compare_with_checkpoints.py:
that script loads all adapters onto a single base_model object, and
merge_and_unload() merges in place, so checkpoint N is evaluated as
base + ckpt_0 + ... + ckpt_N (adapters accumulate). This script reloads the
base model from scratch before each checkpoint, so each row is that checkpoint
alone on top of the untouched base.

Cost: one full base-model load per checkpoint (~30 s each) plus 70 generations
per model state. With 5 checkpoints expect roughly 2 hours. Set LIMIT for a
quick smoke test first.

Outputs, in results_trajectory/:
    trajectory.json      per-question records for every model state
    trajectory.csv       topic x checkpoint score matrix
    trajectory.md        ready-to-paste table
    learning_curve.png   overall + per-topic score against epoch
"""

import json, csv, sys, statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_with_checkpoints import (
    load_eval_data, find_checkpoints, load_base_model,
    generate_answer, score_answer,
)
from peft import PeftModel
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_DATA  = "data/evaluation_sets/dsa_eval_set.json"
CKPT_DIR   = "checkpoints"
OUT_DIR    = Path("results_trajectory"); OUT_DIR.mkdir(exist_ok=True)
MAX_TOKENS = 256
LIMIT      = None       # e.g. 2 for a smoke test


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def run(model, tokenizer, eval_data, label):
    rows = []
    model.eval()
    for topic, questions in eval_data.items():
        for i, q in enumerate(tqdm(questions, desc=f"{label}:{topic}", leave=False)):
            ans, secs = generate_answer(model, tokenizer, q["question"], MAX_TOKENS)
            sc = score_answer(ans, {**q, "topic": topic})
            rows.append({
                "state": label, "topic": topic, "qid": f"{topic}_{i}",
                "answer_words": len(ans.split()), "gen_time_s": round(secs, 2),
                "length": sc["length"], "keyword": sc["keyword"], "code": sc["code"],
                "test_pass_rate": sc["test_pass_rate"], "overall": sc["overall"],
            })
    return rows


eval_data = load_eval_data(EVAL_DATA, limit=LIMIT)
ckpts = find_checkpoints(CKPT_DIR)
topics = list(eval_data.keys())

states = []          # [(label, rows), ...] in order
all_rows = []

# ── base model ──────────────────────────────────────────────────────────────
print("\n=== Evaluating: base ===")
model, tokenizer = load_base_model(MODEL_NAME)
rows = run(model, tokenizer, eval_data, "base")
states.append(("base", rows)); all_rows += rows
del model
torch.cuda.empty_cache()

# ── each checkpoint, on a FRESH base each time ──────────────────────────────
for ckpt in ckpts:
    label = ckpt.name
    print(f"\n=== Evaluating: {label} (fresh base) ===")
    model, tokenizer = load_base_model(MODEL_NAME)
    model = PeftModel.from_pretrained(model, str(ckpt)).merge_and_unload()
    rows = run(model, tokenizer, eval_data, label)
    states.append((label, rows)); all_rows += rows
    del model
    torch.cuda.empty_cache()

labels = [lbl for lbl, _ in states]

# ── topic x state matrix ────────────────────────────────────────────────────
matrix = []
for t in topics:
    row = {"topic": t}
    for lbl, rows in states:
        row[lbl] = mean([r["overall"] for r in rows if r["topic"] == t])
    matrix.append(row)

overall_row = {"topic": "OVERALL"}
pass_row    = {"topic": "TEST_PASS_RATE"}
for lbl, rows in states:
    overall_row[lbl] = mean([r["overall"] for r in rows])
    pass_row[lbl]    = mean([r["test_pass_rate"] for r in rows])
matrix += [overall_row, pass_row]

with open(OUT_DIR / "trajectory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["topic"] + labels); w.writeheader(); w.writerows(matrix)

# ── markdown ────────────────────────────────────────────────────────────────
hdr = "| Topic | " + " | ".join(labels) + " |"
sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
body = "\n".join("| " + r["topic"] + " | " + " | ".join(str(r[l]) for l in labels) + " |"
                 for r in matrix)
(OUT_DIR / "trajectory.md").write_text(
    f"# SEAL-DSA learning trajectory — generated {datetime.now():%Y-%m-%d %H:%M}\n\n"
    f"Base model `{MODEL_NAME}`, {sum(len(v) for v in eval_data.values())} held-out questions.\n"
    f"Each checkpoint evaluated on a freshly loaded base model, so adapters do not accumulate.\n\n"
    + hdr + "\n" + sep + "\n" + body + "\n")

# ── learning curve ──────────────────────────────────────────────────────────
x = range(len(labels))
plt.figure(figsize=(10, 5.5))
for r in matrix[:-2]:
    plt.plot(list(x), [r[l] for l in labels], marker="o", alpha=0.55, linewidth=1,
             label=r["topic"].replace("_", " "))
plt.plot(list(x), [overall_row[l] for l in labels], marker="s", linewidth=3,
         color="black", label="OVERALL")
plt.xticks(list(x), [l.replace("checkpoint_epoch_", "ep") for l in labels])
plt.xlabel("Training state"); plt.ylabel("Average score"); plt.ylim(0, 1.05)
plt.title("SEAL-DSA learning trajectory across training")
plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "learning_curve.png", dpi=200); plt.close()

plt.figure(figsize=(8, 4.5))
plt.plot(list(x), [pass_row[l] for l in labels], marker="o", linewidth=2.5, color="#c0392b")
plt.xticks(list(x), [l.replace("checkpoint_epoch_", "ep") for l in labels])
plt.xlabel("Training state"); plt.ylabel("Test-case pass rate")
plt.title("Functional correctness across training"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "pass_rate_curve.png", dpi=200); plt.close()

json.dump({"model": MODEL_NAME, "generated": datetime.now().isoformat(),
           "states": labels, "matrix": matrix, "records": all_rows},
          open(OUT_DIR / "trajectory.json", "w"), indent=2)

print("\n" + "=" * 60)
for l in labels:
    print(f"  {l:<24} overall {overall_row[l]}   pass {pass_row[l]}")
print(f"Artifacts written to {OUT_DIR.resolve()}")
print("=" * 60)
