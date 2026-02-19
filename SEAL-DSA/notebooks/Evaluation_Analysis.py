"""
SEAL-DSA Evaluation & Visualization Notebook
================================================
Run after training to analyze results and generate plots.

This script:
  1. Loads training metrics
  2. Generates learning curves
  3. Creates per-topic comparison charts
  4. Visualizes forgetting rates
  5. Produces thesis-ready figures
"""

import json
import os
import sys

# Add project to path
if os.path.exists("seal_dsa"):
    sys.path.insert(0, ".")

# ============================================================
# Cell 1: Load Metrics
# ============================================================

metrics_path = "results/training_metrics.json"

if not os.path.exists(metrics_path):
    print(f"Metrics file not found: {metrics_path}")
    print("Run SEAL_DSA_Main.py first to generate training data.")
    sys.exit(1)

with open(metrics_path, 'r') as f:
    data = json.load(f)

summary = data["summary"]
topic_summary = data.get("topic_summary", {})
records = data.get("records", [])

print("=" * 50)
print("EXPERIMENT SUMMARY")
print("=" * 50)
for key, value in summary.items():
    print(f"  {key}: {value}")

# ============================================================
# Cell 2: Setup Visualization
# ============================================================

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    print("matplotlib not available. Install with: pip install matplotlib")
    HAS_MATPLOTLIB = False

if HAS_MATPLOTLIB:
    # Style configuration for thesis-quality figures
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

os.makedirs("results/figures", exist_ok=True)

# ============================================================
# Cell 3: Learning Curve Plot
# ============================================================

if HAS_MATPLOTLIB and records:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(len(records))
    scores = [r["avg_score"] for r in records]
    losses = [r["loss"] for r in records]

    # Score over time
    ax1.plot(steps, scores, 'b-', linewidth=2, label='Avg Score')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Learning Curve: Quality Score')
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Loss over time
    ax2.plot(steps, losses, 'r-', linewidth=2, label='Training Loss')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/figures/learning_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/learning_curve.png")
    plt.close()

# ============================================================
# Cell 4: Per-Topic Performance Bar Chart
# ============================================================

if HAS_MATPLOTLIB and topic_summary:
    topics = list(topic_summary.keys())
    first_scores = [topic_summary[t].get("first_score", 0) for t in topics]
    latest_scores = [topic_summary[t].get("latest_score", 0) for t in topics]

    x = range(len(topics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], first_scores, width,
                    label='Before SEAL', color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], latest_scores, width,
                    label='After SEAL', color='#51cf66', alpha=0.8)

    ax.set_xlabel('DSA Topic')
    ax.set_ylabel('Quality Score')
    ax.set_title('Per-Topic Performance: Before vs After SEAL Training')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', '\n') for t in topics], fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend()

    # Add improvement annotations
    for i, (first, latest) in enumerate(zip(first_scores, latest_scores)):
        improvement = latest - first
        ax.annotate(f'+{improvement:.2f}',
                    xy=(i + width/2, latest),
                    ha='center', va='bottom',
                    fontsize=8, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/topic_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/topic_comparison.png")
    plt.close()

# ============================================================
# Cell 5: Per-Topic Learning Curves
# ============================================================

if HAS_MATPLOTLIB and records:
    from collections import defaultdict

    topic_curves = defaultdict(list)
    for r in records:
        topic_curves[r["topic"]].append(r["avg_score"])

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e64980', '#7950f2', '#1c7ed6', '#12b886',
              '#fab005', '#fd7e14', '#868e96']

    for idx, (topic, scores) in enumerate(topic_curves.items()):
        color = colors[idx % len(colors)]
        ax.plot(range(len(scores)), scores, '-o', linewidth=2,
                markersize=4, label=topic.replace('_', ' ').title(),
                color=color)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Quality Score')
    ax.set_title('Learning Curves by Topic')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('results/figures/topic_learning_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/topic_learning_curves.png")
    plt.close()

# ============================================================
# Cell 6: Improvement Summary Table
# ============================================================

if topic_summary:
    print("\n" + "=" * 70)
    print("PER-TOPIC IMPROVEMENT TABLE")
    print("=" * 70)
    print(f"{'Topic':<25} {'First':>8} {'Latest':>8} {'Best':>8} {'Δ':>8}")
    print("-" * 70)

    total_improvement = 0
    for topic, data in topic_summary.items():
        first = data.get("first_score", 0)
        latest = data.get("latest_score", 0)
        best = data.get("best_score", 0)
        delta = data.get("improvement", 0)
        total_improvement += delta

        print(f"{topic:<25} {first:>8.3f} {latest:>8.3f} {best:>8.3f} {delta:>+8.3f}")

    avg_improvement = total_improvement / max(len(topic_summary), 1)
    print("-" * 70)
    print(f"{'Average':<25} {'':>8} {'':>8} {'':>8} {avg_improvement:>+8.3f}")
    print(f"\nTarget: 15-25% improvement → Achieved: {avg_improvement*100:.1f}%")

# ============================================================
# Cell 7: Latex Table for Thesis
# ============================================================

if topic_summary:
    print("\n% LaTeX table for thesis (copy-paste into .tex)")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Per-Topic Performance Improvement with SEAL-DSA}")
    print("\\label{tab:results}")
    print("\\begin{tabular}{lcccr}")
    print("\\toprule")
    print("Topic & Initial & Final & Best & Improvement \\\\")
    print("\\midrule")

    for topic, data in topic_summary.items():
        name = topic.replace('_', ' ').title()
        first = data.get("first_score", 0)
        latest = data.get("latest_score", 0)
        best = data.get("best_score", 0)
        delta = data.get("improvement", 0)
        print(f"{name} & {first:.3f} & {latest:.3f} & {best:.3f} & +{delta:.3f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

print("\n✅ Evaluation & Visualization complete!")
