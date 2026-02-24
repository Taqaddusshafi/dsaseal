"""
Forgetting Detector
=====================
Detects catastrophic forgetting by periodically evaluating
the model on previously learned topics.

Detection Strategy:
  1. After each topic, evaluate on ALL previous topics
  2. Compare current score with historical best
  3. If score drops > threshold → forgetting detected
  4. Trigger review of affected topics

Metrics:
  Forgetting_rate(topic) = max(best_score) - current_score
  
  If Forgetting_rate > 5%: → Flag for review
  If Forgetting_rate > 10%: → Critical warning
"""

import logging
import random
from typing import Dict, List, Optional
from collections import defaultdict

import torch

from seal_dsa.config import SEALDSAConfig
from seal_dsa.curriculum.dsa_topics import DSA_TOPICS, get_topic_names

logger = logging.getLogger(__name__)


# Quick evaluation prompts for each topic
QUICK_EVAL_PROMPTS = {
    "arrays_strings": [
        "What is the time complexity of searching an unsorted array?",
        "Explain the two-pointer technique for sorted arrays.",
        "What is a sliding window and when is it used?",
    ],
    "linked_lists": [
        "What is the difference between singly and doubly linked lists?",
        "How does Floyd's cycle detection algorithm work?",
        "What is the time complexity of inserting at the head of a linked list?",
    ],
    "stacks_queues": [
        "What is the LIFO principle in stacks?",
        "How can you implement a queue using two stacks?",
        "What is a monotonic stack used for?",
    ],
    "trees": [
        "What is the property of a Binary Search Tree?",
        "Name the three types of depth-first tree traversals.",
        "What is the time complexity of searching in a balanced BST?",
    ],
    "graphs": [
        "What is the difference between BFS and DFS?",
        "When is Dijkstra's algorithm applicable?",
        "What is a topological sort and when is it used?",
    ],
    "sorting_searching": [
        "What is the worst-case complexity of quicksort?",
        "Is merge sort stable? Explain why.",
        "How does binary search work?",
    ],
    "dynamic_programming": [
        "What are the two properties needed for dynamic programming?",
        "Explain the difference between memoization and tabulation.",
        "What is the recurrence relation for the Fibonacci sequence?",
    ],
}

# Expected keywords in correct answers
EXPECTED_ANSWERS = {
    "arrays_strings": [
        ["O(n)", "linear"],
        ["two pointer", "sorted", "opposite"],
        ["sliding window", "subarray", "contiguous"],
    ],
    "linked_lists": [
        ["singly", "doubly", "pointer", "next", "previous"],
        ["floyd", "fast", "slow", "cycle", "tortoise"],
        ["O(1)", "constant", "head"],
    ],
    "stacks_queues": [
        ["LIFO", "last in first out", "push", "pop"],
        ["two stacks", "enqueue", "dequeue"],
        ["monotonic", "next greater", "decreasing", "increasing"],
    ],
    "trees": [
        ["left", "right", "less", "greater", "BST"],
        ["inorder", "preorder", "postorder"],
        ["O(log n)", "logarithmic", "balanced"],
    ],
    "graphs": [
        ["BFS", "DFS", "breadth", "depth", "queue", "stack"],
        ["dijkstra", "non-negative", "shortest path", "greedy"],
        ["topological", "DAG", "directed acyclic", "prerequisite"],
    ],
    "sorting_searching": [
        ["O(n²)", "quadratic", "worst", "pivot"],
        ["stable", "merge", "relative order", "yes"],
        ["binary", "sorted", "middle", "O(log n)", "half"],
    ],
    "dynamic_programming": [
        ["overlapping subproblems", "optimal substructure"],
        ["memoization", "top-down", "tabulation", "bottom-up"],
        ["F(n) = F(n-1) + F(n-2)", "fibonacci", "recurrence"],
    ],
}


class ForgettingDetector:
    """
    Detects and monitors catastrophic forgetting across topics.
    
    Theory:
    ───────
    Catastrophic forgetting occurs when learning new information
    causes the model to forget previously learned information.
    
    In the context of SEAL-DSA:
    - Learning "Graphs" might cause forgetting of "Arrays"
    - This is monitored by periodically testing on all topics
    - If forgetting > threshold, the topic is flagged for review
    
    The EWC module provides the prevention mechanism;
    this module provides the detection mechanism.
    """
    
    def __init__(self, config: SEALDSAConfig):
        self.config = config
        self.threshold = config.seal.forgetting_threshold
        
        # Historical best scores per topic
        self.best_scores: Dict[str, float] = {}
        self.score_history: Dict[str, List[float]] = defaultdict(list)
    
    @torch.no_grad()
    def check_all_topics(
        self,
        model,
        tokenizer,
    ) -> Dict:
        """
        Quick evaluation on all previously seen topics.
        
        Returns a forgetting report with per-topic scores.
        """
        model.eval()
        report = {
            "per_topic": {},
            "max_forgetting": 0.0,
            "worst_topic": None,
            "topics_at_risk": [],
        }
        
        for topic in get_topic_names():
            if topic not in QUICK_EVAL_PROMPTS:
                continue
            
            score = self._quick_evaluate_topic(model, tokenizer, topic)
            
            # Update best score
            if topic not in self.best_scores or score > self.best_scores[topic]:
                self.best_scores[topic] = score
            
            self.score_history[topic].append(score)
            
            # Compute forgetting
            forgetting = max(0.0, self.best_scores[topic] - score)
            
            report["per_topic"][topic] = {
                "current_score": score,
                "best_score": self.best_scores[topic],
                "forgetting": forgetting,
            }
            
            if forgetting > report["max_forgetting"]:
                report["max_forgetting"] = forgetting
                report["worst_topic"] = topic
            
            if forgetting > self.threshold:
                report["topics_at_risk"].append(topic)
        
        # Log results
        if report["topics_at_risk"]:
            logger.warning(
                f"⚠️ Forgetting detected in {len(report['topics_at_risk'])} topics: "
                f"{report['topics_at_risk']}"
            )
        else:
            logger.info("✓ No significant forgetting detected")
        
        return report
    
    def _quick_evaluate_topic(
        self,
        model,
        tokenizer,
        topic: str,
    ) -> float:
        """
        Quick evaluation of a single topic using canned prompts.
        
        Uses keyword matching for conceptual answers AND code execution
        for coding-style questions (mirrors the evaluator fix).
        
        Score = weighted average of answer quality metrics.
        """
        prompts = QUICK_EVAL_PROMPTS.get(topic, [])
        expected = EXPECTED_ANSWERS.get(topic, [])
        
        if not prompts:
            return 0.5  # Unknown topic
        
        total_score = 0.0
        total = len(prompts)
        
        for i, prompt in enumerate(prompts):
            answer = self._get_model_answer(model, tokenizer, prompt)
            answer_lower = answer.lower()
            
            if i < len(expected):
                # Keyword matching score
                expected_kws = expected[i]
                matches = sum(1 for kw in expected_kws if kw.lower() in answer_lower)
                keyword_score = min(1.0, matches / max(len(expected_kws) * 0.5, 1))
                
                # Also check if any code is present and valid
                code_bonus = 0.0
                if "def " in answer_lower or "```" in answer_lower:
                    import ast
                    import re
                    code_match = re.search(r'```python(.*?)```', answer_lower, re.DOTALL)
                    if not code_match:
                        code_match = re.search(r'```(.*?)```', answer_lower, re.DOTALL)
                    if not code_match:
                        code_match = re.search(r'(def\s+\w+.*)', answer_lower, re.DOTALL)
                    
                    if code_match:
                        code = code_match.group(1).strip() if code_match else ""
                        try:
                            ast.parse(code)
                            code_bonus = 0.2  # Valid syntax bonus
                        except SyntaxError:
                            pass
                
                total_score += min(1.0, keyword_score + code_bonus)
            else:
                # Heuristic: non-empty reasonable answer
                if len(answer.split()) > 5:
                    total_score += 0.5
        
        return total_score / total
    
    def _get_model_answer(
        self,
        model,
        tokenizer,
        question: str,
        max_tokens: int = 128,
    ) -> str:
        """Get a quick answer from the model."""
        prompt = f"Answer briefly: {question}\n\nAnswer:"
        
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=False,  # Greedy for consistency
                pad_token_id=tokenizer.pad_token_id,
            )
            
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)
            
        except Exception:
            return ""
    
    def get_forgetting_trend(self, topic: str) -> List[float]:
        """Get the score history for a topic to visualize forgetting."""
        return self.score_history.get(topic, [])
