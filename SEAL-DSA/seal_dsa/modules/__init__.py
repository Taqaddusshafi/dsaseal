"""
SEAL-DSA Modules
=================
Core components of the SEAL self-improving learning loop:
  1. QuestionGenerator — self-generates DSA questions
  2. AnswerGenerator — attempts to answer them
  3. DSAEvaluator — scores answers with code execution
  4. ParameterUpdater — updates LoRA weights based on scores
"""

from seal_dsa.modules.evaluator import DSAEvaluator, EvaluationResult
from seal_dsa.modules.question_generator import QuestionGenerator, GeneratedQuestion
from seal_dsa.modules.answer_generator import AnswerGenerator, GeneratedAnswer
from seal_dsa.modules.parameter_updater import ParameterUpdater, UpdateResult

__all__ = [
    "DSAEvaluator", "EvaluationResult",
    "QuestionGenerator", "GeneratedQuestion",
    "AnswerGenerator", "GeneratedAnswer",
    "ParameterUpdater", "UpdateResult",
]
