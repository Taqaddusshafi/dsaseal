"""
Curriculum Scheduler
======================
Manages the progression of DSA topics during SEAL training.

Supports three scheduling strategies:
  1. Progressive: Follow the predefined curriculum order
  2. Random: Randomly sample topics each epoch
  3. Adaptive: Focus on topics with lowest performance

Curriculum learning theory suggests that training on easier topics
first and progressively increasing difficulty leads to better
convergence and generalization.

Reference: Bengio et al. (2009) "Curriculum Learning"
"""

import random
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from seal_dsa.config import CurriculumConfig
from seal_dsa.curriculum.dsa_topics import DSA_TOPICS, get_topic_names

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Manages the topic schedule for SEAL training.
    
    The scheduler determines which DSA topics to train on and
    in what order, adapting based on performance metrics.
    
    Scheduling Strategies:
    ┌──────────────────────────────────────────────┐
    │  Progressive Schedule                        │
    │  Week 1-2: Arrays ──▶ Week 3-4: Linked      │
    │  Lists ──▶ Week 5-6: Stacks ──▶ ...         │
    │  (Fixed order, follows curriculum)           │
    ├──────────────────────────────────────────────┤
    │  Adaptive Schedule                           │
    │  Epoch N: Evaluate all topics                │
    │  Focus on lowest-performing topics           │
    │  Re-evaluate → Adjust focus                  │
    │  (Performance-driven ordering)               │
    ├──────────────────────────────────────────────┤
    │  Random Schedule                             │
    │  Each epoch: randomly sample topics          │
    │  (Baseline comparison)                       │
    └──────────────────────────────────────────────┘
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.strategy = config.strategy
        self.current_week = 1
        self.current_epoch = 0
        
        # Performance tracking per topic
        self.topic_scores: Dict[str, List[float]] = defaultdict(list)
        self.topic_attempts: Dict[str, int] = defaultdict(int)
        
        # Build week-to-topic mapping
        self.week_topics = config.weeks
        
        logger.info(f"Curriculum scheduler initialized: strategy={self.strategy}")
    
    def get_topics_for_epoch(
        self,
        epoch: int,
        num_topics: Optional[int] = None,
    ) -> List[str]:
        """
        Get the list of topics to train on for a given epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            num_topics: Max number of topics (None = all topics for this epoch)
            
        Returns:
            List of topic keys to train on
        """
        self.current_epoch = epoch
        
        if self.strategy == "progressive":
            topics = self._progressive_schedule(epoch)
        elif self.strategy == "adaptive":
            topics = self._adaptive_schedule(epoch)
        elif self.strategy == "random":
            topics = self._random_schedule(epoch)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using progressive")
            topics = self._progressive_schedule(epoch)
        
        if num_topics:
            topics = topics[:num_topics]
        
        logger.info(f"Epoch {epoch}: Training on topics: {topics}")
        return topics
    
    def _progressive_schedule(self, epoch: int) -> List[str]:
        """
        Progressive curriculum: introduce topics gradually.
        
        Epoch 0: Only first topic
        Epoch 1: First two topics
        Epoch 2: First three topics
        ...
        
        This mimics the weekly curriculum progression.
        """
        all_topics = get_topic_names()
        
        # Each epoch introduces one more topic
        # But also reviews previous topics (with decreasing frequency)
        num_visible = min(epoch + 1, len(all_topics))
        
        topics = all_topics[:num_visible]
        
        # For later epochs, focus more on newer topics
        if epoch >= len(all_topics):
            # All topics unlocked, cycle through with emphasis on weak ones
            topics = self._prioritize_weak_topics(all_topics)
        
        return topics
    
    def _adaptive_schedule(self, epoch: int) -> List[str]:
        """
        Adaptive curriculum: focus on weakest topics.
        
        After the first epoch, sort topics by performance (ascending)
        and focus training on the lowest-performing topics.
        """
        all_topics = get_topic_names()
        
        if epoch == 0 or not self.topic_scores:
            # First epoch: start with fundamentals
            return all_topics[:3]
        
        # Sort topics by average score (ascending = weakest first)
        scored_topics = []
        for topic in all_topics:
            scores = self.topic_scores.get(topic, [])
            avg = sum(scores) / len(scores) if scores else 0.0
            scored_topics.append((topic, avg))
        
        scored_topics.sort(key=lambda x: x[1])
        
        # Focus on bottom half + always include newest topic
        n = max(3, len(scored_topics) // 2)
        focus_topics = [t[0] for t in scored_topics[:n]]
        
        # Ensure the "next" topic in curriculum is included
        next_topic_idx = min(epoch, len(all_topics) - 1)
        next_topic = all_topics[next_topic_idx]
        if next_topic not in focus_topics:
            focus_topics.append(next_topic)
        
        return focus_topics
    
    def _random_schedule(self, epoch: int) -> List[str]:
        """Random topic selection (baseline comparison)."""
        all_topics = get_topic_names()
        n = min(epoch + 2, len(all_topics))
        return random.sample(all_topics, n)
    
    def _prioritize_weak_topics(self, topics: List[str]) -> List[str]:
        """Re-order topics so weakest ones come first."""
        if not self.topic_scores:
            return topics
        
        def avg_score(topic):
            scores = self.topic_scores.get(topic, [])
            return sum(scores) / len(scores) if scores else 0.0
        
        return sorted(topics, key=avg_score)
    
    def record_performance(self, topic: str, score: float):
        """
        Record performance on a topic for adaptive scheduling.
        
        Args:
            topic: Topic key
            score: Average evaluation score for this topic
        """
        self.topic_scores[topic].append(score)
        self.topic_attempts[topic] += 1
        
        logger.debug(f"Recorded score {score:.3f} for topic '{topic}' "
                     f"(attempt #{self.topic_attempts[topic]})")
    
    def get_topic_performance(self) -> Dict[str, Dict]:
        """Get performance summary for all topics."""
        summary = {}
        for topic in get_topic_names():
            scores = self.topic_scores.get(topic, [])
            summary[topic] = {
                "attempts": self.topic_attempts.get(topic, 0),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "best_score": max(scores) if scores else 0.0,
                "latest_score": scores[-1] if scores else 0.0,
                "improvement": (scores[-1] - scores[0]) if len(scores) > 1 else 0.0,
            }
        return summary
    
    def should_review_topic(self, topic: str) -> bool:
        """
        Determine if a previously learned topic needs review.
        
        This helps prevent catastrophic forgetting by revisiting
        topics whose scores are declining.
        """
        scores = self.topic_scores.get(topic, [])
        if len(scores) < 2:
            return False
        
        # Check if score is declining
        recent_avg = sum(scores[-3:]) / min(3, len(scores[-3:]))
        overall_avg = sum(scores) / len(scores)
        
        return recent_avg < overall_avg * 0.9  # 10% decline triggers review
    
    def get_review_topics(self) -> List[str]:
        """Get list of topics that need review."""
        return [
            topic for topic in get_topic_names()
            if self.should_review_topic(topic)
        ]
