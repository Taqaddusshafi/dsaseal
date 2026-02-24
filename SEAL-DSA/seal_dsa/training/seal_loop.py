"""
SEAL Training Loop
====================
Orchestrates the complete Self-Adapting Learning loop.

The SEAL Loop (per epoch, per topic):
═══════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  ┌──────────────────┐                                   │
  │  │  1. GENERATE      │  "Create 20 questions about       │
  │  │     QUESTIONS     │   Binary Search Trees"            │
  │  └────────┬─────────┘                                   │
  │           │ [List[Question]]                            │
  │           ▼                                             │
  │  ┌──────────────────┐                                   │
  │  │  2. GENERATE      │  "Answer each question using      │
  │  │     ANSWERS       │   current model weights"          │
  │  └────────┬─────────┘                                   │
  │           │ [List[(Question, Answer)]]                  │
  │           ▼                                             │
  │  ┌──────────────────┐                                   │
  │  │  3. EVALUATE      │  "Score each answer on            │
  │  │     ANSWERS       │   correctness, completeness, etc" │
  │  └────────┬─────────┘                                   │
  │           │ [List[EvaluationResult]]                    │
  │           ▼                                             │
  │  ┌──────────────────┐                                   │
  │  │  4. UPDATE        │  "Fine-tune LoRA weights using    │
  │  │     PARAMETERS    │   evaluated answers as signal"    │
  │  └────────┬─────────┘                                   │
  │           │ [UpdateResult]                              │
  │           ▼                                             │
  │  ┌──────────────────┐                                   │
  │  │  5. CHECK         │  "Have we forgotten earlier       │
  │  │     FORGETTING    │   topics?"                        │
  │  └────────┬─────────┘                                   │
  │           │                                             │
  │           ▼                                             │
  │  ┌──────────────────┐                                   │
  │  │  6. SAVE          │  "Checkpoint model to Drive"      │
  │  │     CHECKPOINT    │                                   │
  │  └──────────────────┘                                   │
  │                                                         │
  │  ─── Repeat for next topic / epoch ──────────────────── │
  └─────────────────────────────────────────────────────────┘

Pseudo-code:
  for epoch in range(num_epochs):
    topics = curriculum.get_topics(epoch)
    for topic in topics:
      questions = question_gen.generate(topic, n=20)
      answers = answer_gen.answer(questions)
      evaluations = evaluator.evaluate(answers)
      updater.update(evaluations, ewc_loss)
      forgetting = detector.check(model, previous_topics)
      if epoch % save_freq == 0:
        checkpoint.save(model)
"""

import logging
import time
from typing import List, Optional, Dict

import torch

from seal_dsa.config import SEALDSAConfig
from seal_dsa.modules.question_generator import QuestionGenerator
from seal_dsa.modules.answer_generator import AnswerGenerator
from seal_dsa.modules.evaluator import DSAEvaluator
from seal_dsa.modules.parameter_updater import ParameterUpdater
from seal_dsa.curriculum.scheduler import CurriculumScheduler
from seal_dsa.training.checkpoint import CheckpointManager
from seal_dsa.training.ewc import EWC
from seal_dsa.evaluation.metrics import MetricsTracker
from seal_dsa.evaluation.forgetting_detector import ForgettingDetector

logger = logging.getLogger(__name__)


class SEALTrainingLoop:
    """
    Orchestrates the complete SEAL training loop.
    
    This is the main training driver that combines all modules
    into the self-improving learning cycle.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        question_generator: QuestionGenerator,
        answer_generator: AnswerGenerator,
        evaluator: DSAEvaluator,
        parameter_updater: ParameterUpdater,
        curriculum: CurriculumScheduler,
        checkpoint_manager: CheckpointManager,
        metrics_tracker: MetricsTracker,
        forgetting_detector: ForgettingDetector,
        config: SEALDSAConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.question_gen = question_generator
        self.answer_gen = answer_generator
        self.evaluator = evaluator
        self.updater = parameter_updater
        self.curriculum = curriculum
        self.checkpoint_mgr = checkpoint_manager
        self.metrics = metrics_tracker
        self.forgetting = forgetting_detector
        self.config = config
        self.device = device
        
        # Initialize EWC if enabled
        self.ewc = None
        if config.ewc.enabled:
            self.ewc = EWC(model, config.ewc)
            logger.info(f"EWC enabled (λ={config.ewc.lambda_})")
    
    def run(
        self,
        start_epoch: int = 0,
        num_epochs: Optional[int] = None,
        topics: Optional[List[str]] = None,
    ):
        """
        Run the complete SEAL training loop.
        
        Args:
            start_epoch: Starting epoch (for resuming)
            num_epochs: Total epochs to run
            topics: Override curriculum with specific topics
        """
        if num_epochs is None:
            num_epochs = self.config.seal.num_epochs
        
        total_start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # ── Get topics for this epoch ──────────────────────
            if topics:
                epoch_topics = topics
            else:
                epoch_topics = self.curriculum.get_topics_for_epoch(epoch)
            
            # Add review topics (forgetting prevention)
            review_topics = self.curriculum.get_review_topics()
            if review_topics:
                logger.info(f"Adding review topics: {review_topics}")
                epoch_topics = list(set(epoch_topics + review_topics))
            
            epoch_results = []
            
            # ── Process each topic ─────────────────────────────
            for topic_idx, topic in enumerate(epoch_topics):
                logger.info(f"\n--- Topic {topic_idx+1}/{len(epoch_topics)}: {topic} ---")
                
                topic_result = self._process_topic(epoch, topic)
                epoch_results.append(topic_result)
                
                # Record performance for adaptive scheduling
                self.curriculum.record_performance(
                    topic, topic_result["avg_score"]
                )
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # ── Update EWC after each epoch ────────────────────
            if self.ewc is not None:
                logger.info("Updating Fisher Information Matrix for EWC...")
                self.ewc.update_fisher(self.model, self.tokenizer)
            
            # ── Check forgetting across all topics ─────────────
            forgetting_report = self.forgetting.check_all_topics(
                self.model, self.tokenizer
            )
            if forgetting_report["max_forgetting"] > self.config.seal.forgetting_threshold:
                logger.warning(
                    f"⚠️ Forgetting detected: {forgetting_report['max_forgetting']:.2%} "
                    f"on topic '{forgetting_report['worst_topic']}'"
                )
            
            # ── Save Checkpoint ────────────────────────────────
            if (epoch + 1) % self.config.checkpoint.save_every_n_epochs == 0:
                self.checkpoint_mgr.save(
                    model=self.model,
                    optimizer=self.updater.optimizer,
                    epoch=epoch,
                    metrics=self.metrics.get_summary(),
                )
            
            # ── Log Epoch Summary ──────────────────────────────
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_summary(epoch, epoch_results, epoch_time)
        
        # ── Final Summary ──────────────────────────────────────
        total_time = time.time() - total_start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"SEAL Training Complete!")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"{'='*60}")
    
    def _process_topic(
        self,
        epoch: int,
        topic: str,
    ) -> Dict:
        """
        Process a single topic through the SEAL loop.
        
        Steps:
          1. Generate questions
          2. Generate answers  
          3. Evaluate answers
          4. Update parameters
          
        Returns dict with performance metrics.
        """
        result = {
            "topic": topic,
            "epoch": epoch,
            "avg_score": 0.0,
            "correct_ratio": 0.0,
            "loss": 0.0,
        }
        
        # ── Step 1: Generate Questions ─────────────────────────
        logger.info(f"  Step 1: Generating questions...")
        questions = self.question_gen.generate_questions(
            topic=topic,
            num_questions=self.config.seal.questions_per_topic,
        )
        
        if not questions:
            logger.warning(f"  No questions generated for '{topic}'. Skipping.")
            return result
        
        logger.info(f"  Generated {len(questions)} questions")
        
        # ── Step 2: Generate Answers ───────────────────────────
        logger.info(f"  Step 2: Generating answers...")
        answers = self.answer_gen.generate_answers_batch(questions)
        logger.info(f"  Generated {len(answers)} answers")
        
        # ── Step 3: Evaluate Answers ───────────────────────────
        logger.info(f"  Step 3: Evaluating answers...")
        evaluations = self.evaluator.evaluate_batch(answers)
        
        scores = [e.overall_score for e in evaluations]
        correct_count = sum(1 for e in evaluations if e.is_correct)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        logger.info(f"  Avg score: {avg_score:.3f}, "
                     f"Correct: {correct_count}/{len(evaluations)}")
        
        # ── Step 3.5: Self-Refinement (for low-scoring answers) ──
        refined_count = 0
        refinement_threshold = self.config.seal.quality_threshold
        
        for i, eval_result in enumerate(evaluations):
            if eval_result.overall_score < refinement_threshold:
                # Re-generate with evaluator feedback
                refined_answer = self.answer_gen.generate_refined_answer(
                    question=eval_result.answer.question,
                    previous_answer=eval_result.answer.answer,
                    feedback=eval_result.feedback,
                )
                # Re-evaluate the refined answer
                refined_eval = self.evaluator.evaluate(refined_answer)
                
                # Keep the better version
                if refined_eval.overall_score > eval_result.overall_score:
                    evaluations[i] = refined_eval
                    refined_count += 1
                    logger.debug(
                        f"  Refined answer improved: "
                        f"{eval_result.overall_score:.3f} → "
                        f"{refined_eval.overall_score:.3f}"
                    )
        
        if refined_count > 0:
            scores = [e.overall_score for e in evaluations]
            correct_count = sum(1 for e in evaluations if e.is_correct)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            logger.info(
                f"  After refinement: {refined_count} answers improved, "
                f"new avg score: {avg_score:.3f}"
            )
        
        # ── Step 4: Update Parameters ──────────────────────────
        logger.info(f"  Step 4: Updating parameters...")
        ewc_loss_fn = self.ewc.compute_loss if self.ewc else None
        update_result = self.updater.update(evaluations, ewc_loss_fn)
        
        logger.info(f"  Loss: {update_result.loss:.4f}, "
                     f"Grad norm: {update_result.grad_norm:.4f}")
        
        # ── Record Metrics ─────────────────────────────────────
        result["avg_score"] = avg_score
        result["correct_ratio"] = correct_count / max(len(evaluations), 1)
        result["loss"] = update_result.loss
        result["num_questions"] = len(questions)
        result["num_correct"] = correct_count
        result["num_refined"] = refined_count
        
        self.metrics.record(
            epoch=epoch,
            topic=topic,
            avg_score=avg_score,
            correct_ratio=result["correct_ratio"],
            loss=update_result.loss,
            grad_norm=update_result.grad_norm,
            lr=update_result.learning_rate,
        )
        
        return result
    
    def _log_epoch_summary(
        self,
        epoch: int,
        results: List[Dict],
        elapsed_time: float,
    ):
        """Log a summary of the epoch."""
        avg_score = sum(r["avg_score"] for r in results) / max(len(results), 1)
        avg_correct = sum(r["correct_ratio"] for r in results) / max(len(results), 1)
        avg_loss = sum(r["loss"] for r in results) / max(len(results), 1)
        
        logger.info(f"\n{'─'*40}")
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Topics trained: {len(results)}")
        logger.info(f"  Avg score: {avg_score:.3f}")
        logger.info(f"  Avg correct ratio: {avg_correct:.1%}")
        logger.info(f"  Avg loss: {avg_loss:.4f}")
        logger.info(f"  Time: {elapsed_time:.1f}s")
        logger.info(f"  GPU Memory: {self._get_gpu_memory()}")
        logger.info(f"{'─'*40}")
    
    def _get_gpu_memory(self) -> str:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return "N/A (CPU)"
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.1f}GB allocated / {reserved:.1f}GB reserved"
