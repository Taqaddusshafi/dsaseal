#!/usr/bin/env python3
"""
SEAL-DSA — Phase-level result collector (Base vs SEAL-adapted).

Reuses the scoring/generation code from compare_with_checkpoints.py, but
evaluates only TWO models:
    (a) the frozen base model            -> "Before self-adaptation"
    (b) the LAST available checkpoint    -> "After self-adaptive training"

Produces, in results_final/:
    final_report.json     raw per-question records for both models
    per_question.csv      one row per question (both models + delta)
    per_topic.csv         topic-wise aggregate table
    summary.md            ready-to-paste tables for the dissertation
    topic_comparison.png  grouped bar chart
    metric_breakdown.png  component-metric bar chart
    samples.md            side-by-side sample answers

Run from the SEAL-DSA root:  python colab_final_report.py
"""

import json, csv, sys, statistics, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_with_checkpoints import (
    load_eval_data, find_checkpoints, load_base_model,
    load_checkpoint_model, generate_answer, score_answer,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="SEAL-DSA — Phase-level result collector")
parser.add_argument("--epoch", type=int, default=None, help="Specific epoch number to evaluate (e.g. 2)")
parser.add_argument("--checkpoint", "--ckpt", type=str, default=None, help="Name or path of specific checkpoint")
parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory containing checkpoints")
parser.add_argument("--out-dir", type=str, default="results_final", help="Output directory")
parser.add_argument("--limit", type=int, default=None, help="Limit questions per topic")
args, _ = parser.parse_known_args()

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_DATA   = "data/evaluation_sets/dsa_eval_set.json"
CKPT_DIR    = args.ckpt_dir
OUT_DIR     = Path(args.out_dir); OUT_DIR.mkdir(exist_ok=True)
MAX_TOKENS  = 256
LIMIT       = args.limit


def run(model, tokenizer, eval_data, label):
    """Evaluate one model; return a flat list of per-question records."""
    rows = []
    model.eval()
    for topic, questions in eval_data.items():
        for i, q in enumerate(tqdm(questions, desc=f"{label}:{topic}", leave=False)):
            ans, secs = generate_answer(model, tokenizer, q["question"], MAX_TOKENS)
            sc = score_answer(ans, {**q, "topic": topic})
            rows.append({
                "model": label, "topic": topic, "qid": f"{topic}_{i}",
                "type": q.get("type", "conceptual"),
                "difficulty": q.get("difficulty", "NA"),
                "question": q["question"],
                "answer": ans,
                "answer_words": len(ans.split()),
                "gen_time_s": round(secs, 2),
                "length": sc["length"], "keyword": sc["keyword"], "code": sc["code"],
                "test_pass_rate": sc["test_pass_rate"],
                "overall": sc["overall"],
            })
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


# ── 1. data + models ────────────────────────────────────────────────────────
eval_data = load_eval_data(EVAL_DATA, limit=LIMIT)
ckpts = find_checkpoints(CKPT_DIR)
if not ckpts:
    sys.exit(f"No checkpoints found in '{CKPT_DIR}' — nothing to compare against.")

selected_ckpt = None
if args.checkpoint:
    for c in ckpts:
        if args.checkpoint == str(c) or args.checkpoint == c.name:
            selected_ckpt = c
            break
    if not selected_ckpt:
        for c in ckpts:
            if args.checkpoint in c.name:
                selected_ckpt = c
                break
    if not selected_ckpt:
        sys.exit(f"Checkpoint '{args.checkpoint}' not found in {CKPT_DIR}. Available: {[c.name for c in ckpts]}")
elif args.epoch is not None:
    patterns = [f"epoch_{args.epoch}", f"epoch-{args.epoch}", f"epoch{args.epoch}", f"ep{args.epoch}", f"-{args.epoch}"]
    for c in ckpts:
        if any(p in c.name.lower() for p in patterns):
            selected_ckpt = c
            break
    if not selected_ckpt:
        if 1 <= args.epoch <= len(ckpts):
            selected_ckpt = ckpts[args.epoch - 1]
        else:
            sys.exit(f"Epoch {args.epoch} not found. Available checkpoints: {[c.name for c in ckpts]}")
else:
    selected_ckpt = ckpts[-1]

final_ckpt = selected_ckpt
print(f"[INFO] Using checkpoint: {final_ckpt.name}")

base_model, tokenizer = load_base_model(MODEL_NAME)

# base FIRST — merge_and_unload() mutates the base weights in place
base_rows = run(base_model, tokenizer, eval_data, "Base")
seal_model = load_checkpoint_model(base_model, final_ckpt)
seal_rows = run(seal_model, tokenizer, eval_data, "SEAL")

all_rows = base_rows + seal_rows
by_qid = {"Base": {r["qid"]: r for r in base_rows},
          "SEAL": {r["qid"]: r for r in seal_rows}}

# ── 2. per-question CSV + paired stats ──────────────────────────────────────
paired = []
for qid, b in by_qid["Base"].items():
    s = by_qid["SEAL"][qid]
    paired.append({
        "qid": qid, "topic": b["topic"], "type": b["type"],
        "base_overall": b["overall"], "seal_overall": s["overall"],
        "delta": round(s["overall"] - b["overall"], 4),
        "base_words": b["answer_words"], "seal_words": s["answer_words"],
        "base_test_pass": b["test_pass_rate"], "seal_test_pass": s["test_pass_rate"],
    })

with open(OUT_DIR / "per_question.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0].keys())); w.writeheader(); w.writerows(paired)

improved = sum(1 for p in paired if p["delta"] > 1e-6)
degraded = sum(1 for p in paired if p["delta"] < -1e-6)
unchanged = len(paired) - improved - degraded

# ── 3. per-topic aggregation ────────────────────────────────────────────────
topics = list(eval_data.keys())
per_topic = []
for t in topics:
    b = [r for r in base_rows if r["topic"] == t]
    s = [r for r in seal_rows if r["topic"] == t]
    per_topic.append({
        "topic": t, "n": len(b),
        "base": mean([r["overall"] for r in b]),
        "seal": mean([r["overall"] for r in s]),
        "delta": round(mean([r["overall"] for r in s]) - mean([r["overall"] for r in b]), 4),
        "base_kw": mean([r["keyword"] for r in b]), "seal_kw": mean([r["keyword"] for r in s]),
        "base_test": mean([r["test_pass_rate"] for r in b]),
        "seal_test": mean([r["test_pass_rate"] for r in s]),
        "base_words": mean([r["answer_words"] for r in b]),
        "seal_words": mean([r["answer_words"] for r in s]),
    })

with open(OUT_DIR / "per_topic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_topic[0].keys())); w.writeheader(); w.writerows(per_topic)

base_overall = mean([r["overall"] for r in base_rows])
seal_overall = mean([r["overall"] for r in seal_rows])
rel_gain = round((seal_overall - base_overall) / base_overall * 100, 2)

# ── 4. markdown summary ─────────────────────────────────────────────────────
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)

lines = [
    f"# SEAL-DSA Results — generated {datetime.now():%Y-%m-%d %H:%M}",
    "",
    f"Base model: `{MODEL_NAME}`  ·  Adapted model: full self-adaptive training run "
    f"(`{final_ckpt.name}`)  ·  {len(paired)} held-out questions across {len(topics)} topics.",
    "",
    "## Table 1 — Overall performance",
    md_table(["Metric", "Before self-adaptation", "After self-adaptation", "Change"], [
        ["Overall score", base_overall, seal_overall, f"+{round(seal_overall-base_overall,4)}"],
        ["Relative gain (%)", "—", "—", f"{rel_gain}%"],
        ["Keyword coverage", mean([r['keyword'] for r in base_rows]), mean([r['keyword'] for r in seal_rows]), ""],
        ["Answer completeness", mean([r['length'] for r in base_rows]), mean([r['length'] for r in seal_rows]), ""],
        ["Code presence", mean([r['code'] for r in base_rows]), mean([r['code'] for r in seal_rows]), ""],
        ["Test-case pass rate", mean([r['test_pass_rate'] for r in base_rows]), mean([r['test_pass_rate'] for r in seal_rows]), ""],
        ["Avg answer length (words)", mean([r['answer_words'] for r in base_rows]), mean([r['answer_words'] for r in seal_rows]), ""],
        ["Avg generation time (s)", mean([r['gen_time_s'] for r in base_rows]), mean([r['gen_time_s'] for r in seal_rows]), ""],
    ]),
    "",
    "## Table 2 — Question-level outcome (paired)",
    md_table(["Outcome", "Count", "Share (%)"], [
        ["Improved", improved, round(improved / len(paired) * 100, 1)],
        ["Unchanged", unchanged, round(unchanged / len(paired) * 100, 1)],
        ["Degraded", degraded, round(degraded / len(paired) * 100, 1)],
    ]),
    "",
    "## Table 3 — Topic-wise comparison",
    md_table(["Topic", "N", "Before", "After", "Δ", "Test pass (before)", "Test pass (after)"],
             [[r["topic"], r["n"], r["base"], r["seal"], r["delta"], r["base_test"], r["seal_test"]]
              for r in per_topic]),
    "",
    f"Catastrophic forgetting check: {degraded} of {len(paired)} questions "
    f"({round(degraded/len(paired)*100,1)}%) scored lower after adaptation.",
]
(OUT_DIR / "summary.md").write_text("\n".join(lines))

# ── 5. charts ───────────────────────────────────────────────────────────────
x = range(len(per_topic))
plt.figure(figsize=(11, 5))
plt.bar([i - 0.2 for i in x], [r["base"] for r in per_topic], 0.4, label="Before self-adaptation")
plt.bar([i + 0.2 for i in x], [r["seal"] for r in per_topic], 0.4, label="After self-adaptation")
plt.xticks(list(x), [r["topic"].replace("_", "\n") for r in per_topic], fontsize=8)
plt.ylabel("Average score"); plt.ylim(0, 1.05)
plt.title("Topic-wise performance before and after SEAL self-adaptive training")
plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR / "topic_comparison.png", dpi=200); plt.close()

metrics = ["length", "keyword", "code", "test_pass_rate", "overall"]
bvals = [mean([r[m] for r in base_rows]) or 0 for m in metrics]
svals = [mean([r[m] for r in seal_rows]) or 0 for m in metrics]
plt.figure(figsize=(8, 4.5))
plt.bar([i - 0.2 for i in range(len(metrics))], bvals, 0.4, label="Before")
plt.bar([i + 0.2 for i in range(len(metrics))], svals, 0.4, label="After")
plt.xticks(range(len(metrics)), ["Length", "Keyword", "Code", "Test pass", "Overall"])
plt.ylabel("Score"); plt.ylim(0, 1.05); plt.title("Evaluation metric breakdown")
plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR / "metric_breakdown.png", dpi=200); plt.close()

# ── 6. sample answers (biggest gains) ───────────────────────────────────────
top = sorted(paired, key=lambda p: -p["delta"])[:5]
samp = ["# Sample answers — largest improvement after self-adaptation\n"]
for p in top:
    b, s = by_qid["Base"][p["qid"]], by_qid["SEAL"][p["qid"]]
    samp += [f"## {p['qid']}  (Δ = {p['delta']})", f"**Question:** {b['question']}\n",
             f"**Before ({b['overall']}):**\n\n```\n{b['answer'][:900]}\n```\n",
             f"**After ({s['overall']}):**\n\n```\n{s['answer'][:900]}\n```\n"]
(OUT_DIR / "samples.md").write_text("\n".join(samp))

json.dump({"model": MODEL_NAME, "checkpoint": final_ckpt.name,
           "generated": datetime.now().isoformat(),
           "overall": {"base": base_overall, "seal": seal_overall, "relative_gain_pct": rel_gain},
           "paired": {"improved": improved, "unchanged": unchanged, "degraded": degraded},
           "per_topic": per_topic, "records": all_rows},
          open(OUT_DIR / "final_report.json", "w"), indent=2)

print("\n" + "=" * 60)
print(f"Base : {base_overall}\nSEAL : {seal_overall}   ({rel_gain:+}%)")
print(f"improved {improved} | unchanged {unchanged} | degraded {degraded}")
print(f"Artifacts written to {OUT_DIR.resolve()}")
print("=" * 60)
