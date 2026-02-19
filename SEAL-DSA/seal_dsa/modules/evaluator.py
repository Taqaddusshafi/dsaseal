"""
Evaluator Module (Rule-Based)
================================
Part 3 of the SEAL Loop: Self-Evaluate

This module evaluates the quality of generated answers using a 
rule-based approach. In the full SEAL framework, the model 
evaluates its own answers, but for our simplified version we use
deterministic rubrics for reliability on small models.

Evaluation Dimensions:
  1. Correctness: Does the answer contain correct DSA concepts?
  2. Completeness: Are all parts of the question addressed?
  3. Complexity Analysis: Is time/space complexity mentioned?
  4. Code Quality: Is the code syntactically correct? (for coding Qs)
  5. Explanation Quality: Is the reasoning clear?

Scoring:
  Total score ∈ [0, 1] = weighted sum of dimension scores
  
  Score = 0.35 × Correctness + 0.25 × Completeness + 
          0.20 × Complexity + 0.10 × Code + 0.10 × Explanation

Training Signal:
  - Score > threshold → Positive example (reinforce this behavior)
  - Score < threshold → Negative example (learn from this mistake)
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from seal_dsa.config import SEALDSAConfig
from seal_dsa.modules.answer_generator import GeneratedAnswer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Detailed evaluation of a single answer."""
    answer: GeneratedAnswer
    overall_score: float  # [0, 1]
    correctness_score: float
    completeness_score: float
    complexity_score: float
    code_score: float
    explanation_score: float
    feedback: str
    is_correct: bool  # Above threshold
    details: Dict[str, any] = field(default_factory=dict)


# ── DSA Concept Keywords by Topic ───────────────────────────────

DSA_KEYWORDS = {
    "arrays_strings": {
        "concepts": [
            "array", "string", "index", "element", "subarray", "substring",
            "traverse", "iterate", "slice", "two pointer", "sliding window",
            "prefix sum", "hash map", "hash set", "sort", "binary search",
            "in-place", "contiguous", "buffer", "mutable", "immutable",
        ],
        "complexity_terms": ["O(n)", "O(n²)", "O(n log n)", "O(1)", "linear", "quadratic"],
        "algorithms": [
            "two pointer", "sliding window", "kadane", "prefix sum",
            "dutch national flag", "moore voting",
        ],
    },
    "linked_lists": {
        "concepts": [
            "node", "pointer", "next", "head", "tail", "singly", "doubly",
            "circular", "insert", "delete", "reverse", "merge", "cycle",
            "fast pointer", "slow pointer", "sentinel", "dummy node",
        ],
        "complexity_terms": ["O(n)", "O(1)", "linear", "constant"],
        "algorithms": [
            "floyd", "tortoise and hare", "merge sort", "reverse",
            "detect cycle", "find middle",
        ],
    },
    "stacks_queues": {
        "concepts": [
            "stack", "queue", "push", "pop", "enqueue", "dequeue", "peek",
            "top", "front", "rear", "LIFO", "FIFO", "overflow", "underflow",
            "priority queue", "deque", "circular queue", "monotonic",
        ],
        "complexity_terms": ["O(1)", "O(n)", "amortized", "constant"],
        "algorithms": [
            "balanced parentheses", "next greater element", "stock span",
            "queue using stacks", "min stack", "infix to postfix",
        ],
    },
    "trees": {
        "concepts": [
            "tree", "root", "leaf", "node", "parent", "child", "sibling",
            "height", "depth", "level", "binary tree", "BST", "balanced",
            "complete", "full", "perfect", "subtree", "ancestor", "descendant",
        ],
        "complexity_terms": ["O(log n)", "O(n)", "O(h)", "height", "balanced"],
        "algorithms": [
            "inorder", "preorder", "postorder", "level order", "BFS", "DFS",
            "AVL rotation", "red-black", "insert", "delete", "search",
            "lowest common ancestor", "diameter", "serialize",
        ],
    },
    "graphs": {
        "concepts": [
            "graph", "vertex", "edge", "node", "directed", "undirected",
            "weighted", "adjacency list", "adjacency matrix", "connected",
            "component", "cycle", "path", "shortest path", "spanning tree",
            "degree", "in-degree", "out-degree", "bipartite", "DAG",
        ],
        "complexity_terms": ["O(V+E)", "O(V²)", "O(E log V)", "O(V·E)"],
        "algorithms": [
            "BFS", "DFS", "dijkstra", "bellman-ford", "floyd-warshall",
            "kruskal", "prim", "topological sort", "union-find",
            "tarjan", "kosaraju", "articulation point",
        ],
    },
    "sorting_searching": {
        "concepts": [
            "sort", "search", "comparison", "stable", "in-place", "partition",
            "pivot", "merge", "divide and conquer", "binary search",
            "lower bound", "upper bound", "order statistics",
        ],
        "complexity_terms": [
            "O(n log n)", "O(n²)", "O(n)", "O(log n)",
            "best case", "worst case", "average case",
        ],
        "algorithms": [
            "merge sort", "quick sort", "heap sort", "counting sort",
            "radix sort", "bucket sort", "insertion sort", "selection sort",
            "binary search", "interpolation search", "exponential search",
        ],
    },
    "dynamic_programming": {
        "concepts": [
            "dynamic programming", "DP", "memoization", "tabulation",
            "subproblem", "overlapping", "optimal substructure", "state",
            "transition", "base case", "recurrence", "bottom-up", "top-down",
        ],
        "complexity_terms": ["O(n)", "O(n²)", "O(n·W)", "O(n·m)", "polynomial"],
        "algorithms": [
            "fibonacci", "knapsack", "LCS", "LIS", "edit distance",
            "coin change", "matrix chain", "rod cutting", "subset sum",
            "longest palindromic subsequence", "catalan",
        ],
    },
}


class DSAEvaluator:
    """
    Rule-based evaluator for DSA answers.
    
    Design Rationale:
    ─────────────────
    Why rule-based instead of model-based evaluation?
    
    1. Small models (1-4B) are unreliable self-evaluators
    2. Rule-based evaluation is deterministic and reproducible
    3. No additional GPU memory needed for a separate evaluator
    4. Rubric-based scoring provides interpretable feedback
    5. Can be validated against human judgments
    
    Limitation: Cannot evaluate semantic correctness deeply.
    This is acknowledged as a project limitation.
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │           DSA Evaluator                 │
    │                                         │
    │  (Question, Answer) Pair                │
    │        │                                │
    │        ├──▶ Correctness Check           │
    │        │    (keyword matching)           │
    │        ├──▶ Completeness Check           │
    │        │    (coverage analysis)          │
    │        ├──▶ Complexity Check             │
    │        │    (O-notation detection)       │
    │        ├──▶ Code Quality Check           │
    │        │    (syntax validation)          │
    │        └──▶ Explanation Check            │
    │             (structure analysis)         │
    │                                         │
    │        ▼                                │
    │  ┌──────────────┐                       │
    │  │ Weighted      │                      │
    │  │ Score         │ → Training Signal     │
    │  │ Computation   │                      │
    │  └──────────────┘                       │
    └─────────────────────────────────────────┘
    """
    
    # Dimension weights
    WEIGHTS = {
        "correctness": 0.35,
        "completeness": 0.25,
        "complexity": 0.20,
        "code": 0.10,
        "explanation": 0.10,
    }
    
    def __init__(self, config: SEALDSAConfig):
        self.config = config
        self.threshold = config.seal.correctness_threshold
        self.evaluations_done = 0
    
    def evaluate(self, answer: GeneratedAnswer) -> EvaluationResult:
        """
        Evaluate a single generated answer.
        
        Args:
            answer: GeneratedAnswer to evaluate
            
        Returns:
            EvaluationResult with detailed scoring
        """
        topic = answer.question.topic
        answer_text = answer.answer.lower()
        question_text = answer.question.question.lower()
        
        # ── Compute individual dimension scores ────────────────
        correctness = self._score_correctness(answer_text, topic)
        completeness = self._score_completeness(answer_text, question_text, topic)
        complexity = self._score_complexity(answer_text, topic)
        code = self._score_code(answer_text, answer.question.question_type)
        explanation = self._score_explanation(answer_text)
        
        # ── Weighted overall score ─────────────────────────────
        overall = (
            self.WEIGHTS["correctness"] * correctness +
            self.WEIGHTS["completeness"] * completeness +
            self.WEIGHTS["complexity"] * complexity +
            self.WEIGHTS["code"] * code +
            self.WEIGHTS["explanation"] * explanation
        )
        
        # ── Generate Feedback ──────────────────────────────────
        feedback = self._generate_feedback(
            correctness, completeness, complexity, code, explanation
        )
        
        self.evaluations_done += 1
        
        return EvaluationResult(
            answer=answer,
            overall_score=overall,
            correctness_score=correctness,
            completeness_score=completeness,
            complexity_score=complexity,
            code_score=code,
            explanation_score=explanation,
            feedback=feedback,
            is_correct=overall >= self.threshold,
        )
    
    def evaluate_batch(
        self, answers: List[GeneratedAnswer]
    ) -> List[EvaluationResult]:
        """Evaluate a batch of answers."""
        results = [self.evaluate(ans) for ans in answers]
        
        # Log summary statistics
        scores = [r.overall_score for r in results]
        correct_count = sum(1 for r in results if r.is_correct)
        
        logger.info(
            f"Evaluation: {correct_count}/{len(results)} correct, "
            f"Avg score: {sum(scores)/len(scores):.3f}"
        )
        
        return results
    
    def _score_correctness(self, answer: str, topic: str) -> float:
        """
        Score based on presence of correct DSA concepts.
        
        Checks:
        - Relevant concept keywords present
        - Known algorithms mentioned where appropriate
        - No obviously wrong statements detected
        """
        topic_data = DSA_KEYWORDS.get(topic, {})
        concepts = topic_data.get("concepts", [])
        algorithms = topic_data.get("algorithms", [])
        
        if not concepts:
            return 0.5  # Unknown topic, neutral score
        
        # Count concept matches
        concept_matches = sum(1 for c in concepts if c in answer)
        concept_ratio = concept_matches / max(len(concepts), 1)
        
        # Check for algorithm mentions
        algo_matches = sum(1 for a in algorithms if a in answer)
        algo_bonus = min(0.3, algo_matches * 0.1)
        
        # Penalize for common wrong patterns
        wrong_patterns = [
            "i don't know", "i'm not sure", "cannot answer",
            "this is impossible", "undefined behavior",
        ]
        penalty = sum(0.2 for p in wrong_patterns if p in answer)
        
        score = min(1.0, concept_ratio * 1.5 + algo_bonus - penalty)
        return max(0.0, score)
    
    def _score_completeness(
        self, answer: str, question: str, topic: str
    ) -> float:
        """
        Score based on how completely the question is addressed.
        
        Checks:
        - Answer length relative to question complexity
        - Key parts of the question addressed
        - Multiple aspects covered
        """
        # Length heuristic (longer answers tend to be more complete)
        words = answer.split()
        if len(words) < 10:
            length_score = 0.1
        elif len(words) < 30:
            length_score = 0.3
        elif len(words) < 80:
            length_score = 0.6
        elif len(words) < 200:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Check if answer addresses question keywords
        question_words = set(question.split()) - {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'to', 'of', 'and', 'in', 'that', 'have', 'for', 'it',
            'with', 'as', 'on', 'at', 'by', 'this', 'from',
        }
        
        if question_words:
            coverage = sum(1 for w in question_words if w in answer)
            coverage_ratio = coverage / len(question_words)
        else:
            coverage_ratio = 0.5
        
        return 0.5 * length_score + 0.5 * coverage_ratio
    
    def _score_complexity(self, answer: str, topic: str) -> float:
        """
        Score based on complexity analysis presence.
        
        Checks:
        - Big-O notation present
        - Time complexity mentioned
        - Space complexity mentioned
        - Correct complexity for the topic
        """
        topic_data = DSA_KEYWORDS.get(topic, {})
        expected_terms = topic_data.get("complexity_terms", [])
        
        # Check for Big-O notation
        has_big_o = bool(re.search(r'O\([^)]+\)', answer, re.IGNORECASE))
        
        # Check for time/space keywords
        has_time = any(t in answer for t in ["time complexity", "time:", "runtime"])
        has_space = any(t in answer for t in ["space complexity", "space:", "memory"])
        
        # Check for specific complexity terms
        term_matches = sum(1 for t in expected_terms if t.lower() in answer)
        
        score = 0.0
        if has_big_o:
            score += 0.4
        if has_time:
            score += 0.2
        if has_space:
            score += 0.2
        if term_matches > 0:
            score += min(0.2, term_matches * 0.1)
        
        return min(1.0, score)
    
    def _score_code(self, answer: str, question_type: str) -> float:
        """
        Score code quality (for coding questions).
        
        Checks:
        - Code block present (for coding questions)
        - Python/pseudocode syntax
        - Function definition present
        - Return statement present
        """
        if question_type not in ["coding", "problem_solving"]:
            return 0.7  # Neutral for non-coding questions
        
        # Check for code indicators
        has_code_block = "```" in answer or "def " in answer
        has_function = "def " in answer or "function " in answer
        has_return = "return " in answer
        has_loop = any(kw in answer for kw in ["for ", "while ", "foreach"])
        has_conditional = "if " in answer
        
        score = 0.0
        if has_code_block:
            score += 0.3
        if has_function:
            score += 0.3
        if has_return:
            score += 0.2
        if has_loop or has_conditional:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_explanation(self, answer: str) -> float:
        """
        Score explanation quality.
        
        Checks:
        - Structured response (numbered steps, bullet points)
        - Transitional words
        - Example presence
        """
        # Structural elements
        has_numbering = bool(re.search(r'\d+[\.\)]\s', answer))
        has_bullets = bool(re.search(r'[-•*]\s', answer))
        has_structure = has_numbering or has_bullets
        
        # Explanation markers
        explanation_words = [
            "because", "therefore", "since", "thus", "hence",
            "first", "second", "third", "step", "approach",
            "the idea is", "we can", "this works because",
            "for example", "consider", "let's",
        ]
        explanation_count = sum(1 for w in explanation_words if w in answer)
        
        # Example presence
        has_example = any(w in answer for w in [
            "example", "for instance", "e.g.", "consider",
            "input:", "output:", "test case",
        ])
        
        score = 0.0
        if has_structure:
            score += 0.3
        score += min(0.4, explanation_count * 0.1)
        if has_example:
            score += 0.3
        
        return min(1.0, score)
    
    def _generate_feedback(
        self,
        correctness: float,
        completeness: float,
        complexity: float,
        code: float,
        explanation: float,
    ) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []
        
        if correctness < 0.5:
            feedback_parts.append("Include more relevant DSA concepts and terminology.")
        if completeness < 0.5:
            feedback_parts.append("Address all parts of the question more thoroughly.")
        if complexity < 0.5:
            feedback_parts.append("Add time and space complexity analysis with Big-O notation.")
        if code < 0.5:
            feedback_parts.append("Include working code implementation.")
        if explanation < 0.5:
            feedback_parts.append("Improve explanation structure with steps and examples.")
        
        if not feedback_parts:
            return "Good answer! All dimensions scored well."
        
        return " ".join(feedback_parts)
    
    def get_stats(self) -> Dict:
        """Return evaluation statistics."""
        return {
            "total_evaluations": self.evaluations_done,
        }
