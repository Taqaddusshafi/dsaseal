"""
Evaluator Module (Rule-Based + Code Execution)
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
  4. Code Quality: Is the code syntactically correct + executable?
  5. Explanation Quality: Is the reasoning clear?

Scoring:
  Total score ∈ [0, 1] = weighted sum of dimension scores
  
  Score = 0.30 × Correctness + 0.20 × Completeness + 
          0.15 × Complexity + 0.25 × Code + 0.10 × Explanation

Training Signal:
  - Score > threshold → Positive example (reinforce this behavior)
  - Score < threshold → Negative example (learn from this mistake)
"""

import re
import ast
import logging
from typing import List, Dict, Tuple, Optional, Any
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
    details: Dict[str, Any] = field(default_factory=dict)


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
    Rule-based evaluator for DSA answers with code execution checking.
    
    Novel: Adaptive Evaluation Weights
    ===================================
    Unlike fixed-weight evaluators, SEAL uses adaptive dimension weights
    that shift emphasis toward dimensions where the model is weakest.
    
    This creates a co-evolving evaluation function where the model and
    evaluator form a tight feedback loop:
      - Model improves → evaluation shifts focus to remaining weaknesses
      - Prevents the model from "gaming" the evaluator by only optimising
        easy dimensions while ignoring hard ones
    
    Mathematical formulation:
      w_d^{t+1} = α · w_d^{t} + (1-α) · softmax(-s_d^{t})
    
    where s_d^{t} is the avg score on dimension d at time t,
    and α is the EMA smoothing factor (default 0.9).
    """

    # Base weights (used as initialisation and fallback)
    BASE_WEIGHTS = {
        "correctness": 0.30,
        "completeness": 0.20,
        "complexity": 0.15,
        "code": 0.25,
        "explanation": 0.10,
    }

    def __init__(self, config: SEALDSAConfig):
        self.config = config
        self.threshold = config.seal.correctness_threshold
        self.evaluations_done = 0
        
        # ── Adaptive Weights State ─────────────────────────────
        # Current weights (will evolve during training)
        self.WEIGHTS = dict(self.BASE_WEIGHTS)
        
        # Running averages per dimension (for adaptive weight computation)
        self._dimension_scores: Dict[str, List[float]] = {
            d: [] for d in self.BASE_WEIGHTS
        }
        self._weight_ema_alpha = 0.9  # Smoothing factor
        self._adapt_every_n = 20      # Re-compute weights every N evals

    def evaluate(self, answer: GeneratedAnswer) -> EvaluationResult:
        topic = answer.question.topic
        answer_text = answer.answer.lower()
        question_text = answer.question.question.lower()

        correctness = self._score_correctness(answer_text, topic)
        completeness = self._score_completeness(answer_text, question_text, topic)
        complexity = self._score_complexity(answer_text, topic)
        code = self._score_code(answer_text, answer.question.question_type)
        explanation = self._score_explanation(answer_text)

        # Use adaptive weights
        overall = (
            self.WEIGHTS["correctness"] * correctness +
            self.WEIGHTS["completeness"] * completeness +
            self.WEIGHTS["complexity"] * complexity +
            self.WEIGHTS["code"] * code +
            self.WEIGHTS["explanation"] * explanation
        )

        feedback = self._generate_feedback(
            correctness, completeness, complexity, code, explanation
        )

        self.evaluations_done += 1

        # Track per-dimension scores for adaptive weight computation
        self._dimension_scores["correctness"].append(correctness)
        self._dimension_scores["completeness"].append(completeness)
        self._dimension_scores["complexity"].append(complexity)
        self._dimension_scores["code"].append(code)
        self._dimension_scores["explanation"].append(explanation)

        # Periodically adapt weights
        if self.evaluations_done % self._adapt_every_n == 0:
            self._adapt_weights()

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
            details={"weights": dict(self.WEIGHTS)},
        )

    def _adapt_weights(self):
        """
        Adapt evaluation weights based on model's weakness profile.
        
        Algorithm (Novel Contribution):
        ================================
        1. Compute average score per dimension over recent evaluations
        2. Apply inverse-softmax: dimensions with LOWER avg scores
           receive HIGHER weights (focusing training on weaknesses)
        3. Smooth with EMA to prevent oscillation
        4. Normalise so weights sum to 1.0
        
        This implements Equation (3) from the SEAL paper:
          w_d^{t+1} = α · w_d^{t} + (1-α) · softmax(-s_d / τ)
        
        where τ=0.5 is a temperature controlling sharpness.
        """
        import math
        
        # Use recent scores (last 50 evaluations)
        window = 50
        avg_scores = {}
        for dim, scores in self._dimension_scores.items():
            recent = scores[-window:] if scores else [0.5]
            avg_scores[dim] = sum(recent) / len(recent)
        
        # Inverse-softmax with temperature
        temperature = 0.5
        neg_scores = {d: -s / temperature for d, s in avg_scores.items()}
        max_neg = max(neg_scores.values())  # For numerical stability
        exp_scores = {d: math.exp(s - max_neg) for d, s in neg_scores.items()}
        total_exp = sum(exp_scores.values())
        target_weights = {d: v / total_exp for d, v in exp_scores.items()}
        
        # EMA smoothing: blend with current weights
        alpha = self._weight_ema_alpha
        for dim in self.WEIGHTS:
            self.WEIGHTS[dim] = (
                alpha * self.WEIGHTS[dim] + 
                (1 - alpha) * target_weights[dim]
            )
        
        # Normalise to sum to 1.0
        total = sum(self.WEIGHTS.values())
        self.WEIGHTS = {d: w / total for d, w in self.WEIGHTS.items()}
        
        logger.debug(
            f"Adapted weights: " +
            ", ".join(f"{d}={w:.3f}" for d, w in self.WEIGHTS.items())
        )

    def evaluate_batch(
        self, answers: List[GeneratedAnswer]
    ) -> List[EvaluationResult]:
        """Evaluate a batch of answers."""
        results = [self.evaluate(ans) for ans in answers]

        scores = [r.overall_score for r in results]
        correct_count = sum(1 for r in results if r.is_correct)

        logger.info(
            f"Evaluation: {correct_count}/{len(results)} correct, "
            f"Avg score: {sum(scores)/len(scores):.3f}"
        )

        return results

    def _score_correctness(self, answer: str, topic: str) -> float:
        topic_data = DSA_KEYWORDS.get(topic, {})
        concepts = topic_data.get("concepts", [])
        algorithms = topic_data.get("algorithms", [])

        if not concepts:
            return 0.5

        concept_matches = sum(1 for c in concepts if c in answer)
        concept_ratio = concept_matches / max(len(concepts), 1)

        algo_matches = sum(1 for a in algorithms if a in answer)
        algo_bonus = min(0.3, algo_matches * 0.1)

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
        topic_data = DSA_KEYWORDS.get(topic, {})
        expected_terms = topic_data.get("complexity_terms", [])

        has_big_o = bool(re.search(r'O\([^)]+\)', answer, re.IGNORECASE))
        has_time = any(t in answer for t in ["time complexity", "time:", "runtime"])
        has_space = any(t in answer for t in ["space complexity", "space:", "memory"])
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

    # ── Safe builtins for sandboxed code execution ────────────
    SAFE_BUILTINS = {
        "len": len, "range": range, "int": int, "str": str,
        "float": float, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "print": lambda *a, **kw: None,  # silence prints
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "sorted": sorted, "reversed": reversed, "bool": bool,
        "isinstance": isinstance, "type": type, "None": None,
        "True": True, "False": False, "hash": hash,
        "ord": ord, "chr": chr, "hex": hex, "bin": bin,
        "pow": pow, "divmod": divmod, "round": round,
        "any": any, "all": all, "iter": iter, "next": next,
        "ValueError": ValueError, "TypeError": TypeError,
        "IndexError": IndexError, "KeyError": KeyError,
        "StopIteration": StopIteration, "Exception": Exception,
    }

    def _extract_code(self, answer: str) -> Optional[str]:
        """
        Extract Python code from an answer string.

        Tries (in order):
          1. ```python ... ``` fenced block
          2. ``` ... ``` generic fenced block
          3. Raw function definition (def ...)

        Returns:
            Extracted code string or None if no code found.
        """
        # Try to extract code from markdown block first
        code_match = re.search(r'```python(.*?)```', answer, re.DOTALL)
        if not code_match:
            # Try plain markdown block
            code_match = re.search(r'```(.*?)```', answer, re.DOTALL)

        if code_match:
            return code_match.group(1).strip()

        # Try to extract raw function definition
        code_match = re.search(r'(def\s+\w+.*)', answer, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return None

    # ✅ NEW METHOD - extracts and executes code to check correctness
    def _score_code_execution(self, answer: str) -> float:
        """
        Actually execute the code to check if it runs without errors.

        Returns:
          1.0 - Code runs successfully
          0.5 - Code has valid syntax but fails at runtime
          0.0 - Code has syntax errors or no code found
        """
        code = self._extract_code(answer)
        if not code:
            return 0.0

        # Step 1: Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.debug(f"Syntax error in generated code: {e}")
            return 0.0

        # Step 2: Runtime execution check (safe sandbox)
        try:
            safe_globals = {"__builtins__": self.SAFE_BUILTINS.copy()}
            exec(compile(tree, '<string>', 'exec'), safe_globals)
            logger.debug("Code executed successfully")
            return 1.0
        except Exception as e:
            logger.debug(f"Runtime error in generated code: {e}")
            return 0.5  # Syntax OK but runtime error

    # ✅ NEW METHOD - run code against test cases for true correctness
    def _score_code_with_tests(
        self, answer: str, test_cases: List[Dict[str, Any]],
    ) -> float:
        """
        Run extracted code against provided test cases.

        This is the key improvement over keyword matching: we actually
        verify that the function produces correct outputs.

        Args:
            answer: The model's answer text (may contain code blocks)
            test_cases: List of dicts with keys:
                - 'input': the arguments to pass (as a string expression)
                - 'expected': the expected return value
                - 'function': name of the function to call

        Returns:
            Score in [0, 1] = fraction of test cases passed.
            Returns -1.0 if no code or no test cases (caller should
            fall back to execution-only scoring).
        """
        if not test_cases:
            return -1.0  # Signal: no test cases, use fallback

        code = self._extract_code(answer)
        if not code:
            return 0.0

        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        # Execute to define functions
        try:
            safe_globals = {"__builtins__": self.SAFE_BUILTINS.copy()}
            exec(compile(tree, '<string>', 'exec'), safe_globals)
        except Exception:
            return 0.0

        # Run test cases
        passed = 0
        for tc in test_cases:
            fn_name = tc.get("function", "")
            if fn_name not in safe_globals:
                # Try to find any defined function
                fn_name = self._find_function_name(code)
                if not fn_name or fn_name not in safe_globals:
                    continue

            fn = safe_globals[fn_name]
            try:
                # Evaluate input expression in the safe sandbox
                args = eval(tc["input"], {"__builtins__": self.SAFE_BUILTINS.copy()})
                if not isinstance(args, tuple):
                    args = (args,)
                result = fn(*args)
                expected = tc["expected"]
                if result == expected:
                    passed += 1
                    logger.debug(f"  Test PASSED: {fn_name}({tc['input']}) == {expected}")
                else:
                    logger.debug(
                        f"  Test FAILED: {fn_name}({tc['input']}) "
                        f"returned {result}, expected {expected}"
                    )
            except Exception as e:
                logger.debug(f"  Test ERROR: {fn_name}({tc['input']}): {e}")

        return passed / len(test_cases)

    @staticmethod
    def _find_function_name(code: str) -> Optional[str]:
        """Find the first function name defined in code."""
        match = re.search(r'def\s+(\w+)', code)
        return match.group(1) if match else None

    # ✅ UPDATED _score_code - now calls _score_code_execution + test cases
    def _score_code(
        self, answer: str, question_type: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Score code quality (for coding questions).

        Scoring breakdown:
        - Structural checks:   0.30 (code block, function, return, loops)
        - Execution check:     0.30 (code runs without errors)
        - Test case check:     0.40 (code produces correct outputs)

        If no test cases are provided, execution check gets 0.50 weight
        and structural checks get 0.50 weight (original behaviour).
        """
        if question_type not in ["coding", "problem_solving"]:
            return 0.7  # Neutral for non-coding questions

        # ── Structural checks ─────────────────────────────────
        has_code_block = "```" in answer or "def " in answer
        has_function = "def " in answer or "function " in answer
        has_return = "return " in answer
        has_loop = any(kw in answer for kw in ["for ", "while ", "foreach"])
        has_conditional = "if " in answer

        structural = 0.0
        if has_code_block:
            structural += 0.15
        if has_function:
            structural += 0.15
        if has_return:
            structural += 0.10
        if has_loop or has_conditional:
            structural += 0.10

        # ── Execution check ───────────────────────────────────
        execution_score = self._score_code_execution(answer)

        # ── Test-case check (if available) ────────────────────
        test_score = self._score_code_with_tests(answer, test_cases or [])

        if test_score >= 0.0:  # -1 means no test cases available
            # Full scoring: structural 0.30 + execution 0.30 + tests 0.40
            score = 0.30 * (structural / 0.50) + 0.30 * execution_score + 0.40 * test_score
        else:
            # Fallback: structural 0.50 + execution 0.50
            score = structural + 0.50 * execution_score

        return min(1.0, score)

    def _score_explanation(self, answer: str) -> float:
        has_numbering = bool(re.search(r'\d+[\.\)]\s', answer))
        has_bullets = bool(re.search(r'[-•*]\s', answer))
        has_structure = has_numbering or has_bullets

        explanation_words = [
            "because", "therefore", "since", "thus", "hence",
            "first", "second", "third", "step", "approach",
            "the idea is", "we can", "this works because",
            "for example", "consider", "let's",
        ]
        explanation_count = sum(1 for w in explanation_words if w in answer)

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
        feedback_parts = []

        if correctness < 0.5:
            feedback_parts.append("Include more relevant DSA concepts and terminology.")
        if completeness < 0.5:
            feedback_parts.append("Address all parts of the question more thoroughly.")
        if complexity < 0.5:
            feedback_parts.append("Add time and space complexity analysis with Big-O notation.")
        if code < 0.5:
            feedback_parts.append("Include working, executable code implementation.")
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