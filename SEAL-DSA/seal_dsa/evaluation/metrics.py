"""
Metrics Tracker
=================
Tracks and visualizes training metrics across epochs and topics.

Metrics Tracked:
  - Average evaluation score per topic per epoch
  - Correct answer ratio
  - Training loss
  - Gradient norms
  - Learning rate schedule
  - Per-topic improvement over time
  - Forgetting rates
"""

import json
import logging
import os
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional wandb integration
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class MetricsTracker:
    """
    Comprehensive metrics tracking for SEAL-DSA experiments.
    
    Tracks metrics at multiple granularities:
      - Per-step (each SEAL loop iteration)
      - Per-topic (aggregated by topic)
      - Per-epoch (aggregated by epoch)
      - Overall (experiment-level)
    
    Optionally logs to Weights & Biases (wandb) for
    experiment dashboards and visualisation.
    """
    
    def __init__(self, use_wandb: bool = False, wandb_project: str = "SEAL-DSA"):
        self.records = []
        self.topic_history = defaultdict(list)
        self.epoch_history = defaultdict(list)
        self.start_time = datetime.now()
        
        # Setup wandb if requested and available
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            try:
                wandb.init(project=wandb_project, resume="allow")
                logger.info(f"WandB logging enabled (project: {wandb_project})")
            except Exception as e:
                logger.warning(f"WandB init failed: {e}. Continuing without WandB.")
                self.use_wandb = False
        elif use_wandb and not HAS_WANDB:
            logger.info("WandB not installed. pip install wandb to enable.")
    
    def record(
        self,
        epoch: int,
        topic: str,
        avg_score: float,
        correct_ratio: float,
        loss: float,
        grad_norm: float = 0.0,
        lr: float = 0.0,
        **kwargs,
    ):
        """Record a single training step's metrics."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "topic": topic,
            "avg_score": avg_score,
            "correct_ratio": correct_ratio,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": lr,
            **kwargs,
        }
        self.records.append(entry)
        self.topic_history[topic].append(entry)
        self.epoch_history[epoch].append(entry)
        
        # Log to wandb if enabled
        if self.use_wandb:
            try:
                wandb.log({
                    "epoch": epoch,
                    "avg_score": avg_score,
                    "correct_ratio": correct_ratio,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "learning_rate": lr,
                    f"topic/{topic}/score": avg_score,
                    f"topic/{topic}/correct_ratio": correct_ratio,
                })
            except Exception:
                pass  # Don't crash training for wandb issues
    
    def get_summary(self) -> Dict:
        """Get overall experiment summary."""
        if not self.records:
            return {"status": "no data"}
        
        all_scores = [r["avg_score"] for r in self.records]
        all_losses = [r["loss"] for r in self.records]
        
        # Compute improvement
        first_scores = [r["avg_score"] for r in self.records[:5]]
        last_scores = [r["avg_score"] for r in self.records[-5:]]
        
        first_avg = sum(first_scores) / len(first_scores) if first_scores else 0
        last_avg = sum(last_scores) / len(last_scores) if last_scores else 0
        improvement = last_avg - first_avg
        
        return {
            "total_steps": len(self.records),
            "total_epochs": len(self.epoch_history),
            "topics_trained": len(self.topic_history),
            "avg_score": sum(all_scores) / len(all_scores),
            "best_score": max(all_scores),
            "worst_score": min(all_scores),
            "avg_loss": sum(all_losses) / len(all_losses),
            "improvement": improvement,
            "improvement_pct": improvement / max(first_avg, 0.01) * 100,
            "duration": str(datetime.now() - self.start_time),
        }
    
    def get_topic_summary(self) -> Dict[str, Dict]:
        """Get per-topic performance summary."""
        summary = {}
        for topic, records in self.topic_history.items():
            scores = [r["avg_score"] for r in records]
            correct_ratios = [r["correct_ratio"] for r in records]
            
            summary[topic] = {
                "num_iterations": len(records),
                "avg_score": sum(scores) / len(scores),
                "best_score": max(scores),
                "latest_score": scores[-1],
                "first_score": scores[0],
                "improvement": scores[-1] - scores[0],
                "avg_correct_ratio": sum(correct_ratios) / len(correct_ratios),
            }
        return summary
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curves for plotting."""
        curves = {}
        
        # Overall learning curve
        curves["overall_score"] = [r["avg_score"] for r in self.records]
        curves["overall_loss"] = [r["loss"] for r in self.records]
        
        # Per-topic curves
        for topic, records in self.topic_history.items():
            curves[f"{topic}_score"] = [r["avg_score"] for r in records]
        
        return curves
    
    def print_summary(self):
        """Print a formatted summary to the logger."""
        summary = self.get_summary()
        topic_summary = self.get_topic_summary()
        
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total training steps: {summary['total_steps']}")
        logger.info(f"  Total epochs: {summary['total_epochs']}")
        logger.info(f"  Topics trained: {summary['topics_trained']}")
        logger.info(f"  Average score: {summary['avg_score']:.3f}")
        logger.info(f"  Best score: {summary['best_score']:.3f}")
        logger.info(f"  Average loss: {summary['avg_loss']:.4f}")
        logger.info(f"  Improvement: {summary['improvement']:.3f} "
                     f"({summary['improvement_pct']:.1f}%)")
        logger.info(f"  Duration: {summary['duration']}")
        
        logger.info("\n  Per-Topic Results:")
        logger.info(f"  {'Topic':<25} {'First':>8} {'Latest':>8} {'Best':>8} {'Δ':>8}")
        logger.info(f"  {'-'*57}")
        for topic, data in topic_summary.items():
            logger.info(
                f"  {topic:<25} {data['first_score']:>8.3f} "
                f"{data['latest_score']:>8.3f} {data['best_score']:>8.3f} "
                f"{data['improvement']:>+8.3f}"
            )
        logger.info("=" * 60)
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            "summary": self.get_summary(),
            "topic_summary": self.get_topic_summary(),
            "records": self.records,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Metrics saved to {path}")
    
    def load(self, path: str):
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.records = data.get("records", [])
        # Rebuild dictionaries
        for r in self.records:
            self.topic_history[r["topic"]].append(r)
            self.epoch_history[r["epoch"]].append(r)
