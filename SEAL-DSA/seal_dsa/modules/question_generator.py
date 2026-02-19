"""
Question Generator Module
============================
Part 1 of the SEAL Loop: Self-Generate Questions

This module prompts the language model to generate DSA questions
for a given topic. The questions serve as the self-training data
that drives the autonomous learning loop.

Data Flow:
  Topic (e.g., "Binary Trees") 
    → Prompt Template 
    → LLM Generation 
    → Question Parsing 
    → Quality Filtering 
    → Question Set

Question Types Generated:
  1. Conceptual: "What is the time complexity of..."
  2. Coding: "Write a function to..."
  3. Analytical: "Compare and contrast..."
  4. Problem-solving: "Given an array of N integers..."
"""

import re
import json
import logging
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from seal_dsa.config import SEALDSAConfig
from seal_dsa.curriculum.dsa_topics import DSA_TOPICS

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """Represents a single generated question."""
    question: str
    topic: str
    subtopic: str
    difficulty: str  # easy, medium, hard
    question_type: str  # conceptual, coding, analytical, problem_solving
    expected_concepts: List[str]
    quality_score: float = 0.0


# ── Prompt Templates ────────────────────────────────────────────

QUESTION_GENERATION_PROMPT = """You are an expert Data Structures and Algorithms instructor creating exam questions.

Topic: {topic}
Subtopic: {subtopic}
Difficulty: {difficulty}
Question Type: {question_type}

Generate a clear, specific DSA question that tests understanding of the given topic.

Requirements:
- The question should be self-contained
- Include specific constraints where applicable (e.g., time/space complexity requirements)
- For coding questions, specify input/output format
- Target {difficulty} difficulty level

Generate ONLY the question, nothing else.

Question:"""


BATCH_GENERATION_PROMPT = """You are an expert DSA instructor. Generate {num_questions} diverse questions about {topic}.

Requirements:
- Mix of conceptual, coding, and analytical questions
- Vary difficulty from easy to hard
- Each question should be clearly separated
- Questions should cover different subtopics within {topic}

Format each question as:
Q1: [question text]
Q2: [question text]
...

Topic details: {topic_description}

Questions:"""


class QuestionGenerator:
    """
    Generates DSA questions using the language model.
    
    This is the first step in the SEAL loop. The model generates
    questions about DSA topics, which it will then attempt to answer.
    This creates a self-supervised learning signal.
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │         Question Generator              │
    │                                         │
    │  Topic + Difficulty                     │
    │        │                                │
    │        ▼                                │
    │  ┌──────────────┐                       │
    │  │ Prompt        │                      │
    │  │ Construction  │                      │
    │  └──────┬───────┘                       │
    │         ▼                               │
    │  ┌──────────────┐                       │
    │  │ LLM          │ ← Temperature=0.8     │
    │  │ Generation   │ ← Top-p=0.9          │
    │  └──────┬───────┘                       │
    │         ▼                               │
    │  ┌──────────────┐                       │
    │  │ Parsing &    │                       │
    │  │ Filtering    │ ← Quality threshold   │
    │  └──────┬───────┘                       │
    │         ▼                               │
    │  [Question Set]                         │
    └─────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: SEALDSAConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.generated_count = 0
        
    @torch.no_grad()
    def generate_questions(
        self,
        topic: str,
        num_questions: int = 20,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> List[GeneratedQuestion]:
        """
        Generate a set of DSA questions for a given topic.
        
        Args:
            topic: DSA topic key (e.g., "arrays_strings")
            num_questions: Number of questions to generate
            difficulty_distribution: Dict mapping difficulty to proportion
                Default: {"easy": 0.3, "medium": 0.5, "hard": 0.2}
                
        Returns:
            List of GeneratedQuestion objects
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
        
        topic_info = DSA_TOPICS.get(topic, {})
        if not topic_info:
            logger.warning(f"Unknown topic: {topic}. Using generic prompts.")
            topic_info = {"name": topic, "subtopics": [topic], "description": topic}
        
        questions = []
        question_types = ["conceptual", "coding", "analytical", "problem_solving"]
        
        for difficulty, proportion in difficulty_distribution.items():
            n = max(1, int(num_questions * proportion))
            
            for i in range(n):
                subtopic = random.choice(topic_info.get("subtopics", [topic]))
                q_type = random.choice(question_types)
                
                question = self._generate_single_question(
                    topic=topic_info.get("name", topic),
                    subtopic=subtopic,
                    difficulty=difficulty,
                    question_type=q_type,
                )
                
                if question and self._passes_quality_check(question):
                    questions.append(GeneratedQuestion(
                        question=question,
                        topic=topic,
                        subtopic=subtopic,
                        difficulty=difficulty,
                        question_type=q_type,
                        expected_concepts=topic_info.get("key_concepts", []),
                    ))
        
        self.generated_count += len(questions)
        logger.info(f"Generated {len(questions)}/{num_questions} questions for '{topic}'")
        
        return questions
    
    def _generate_single_question(
        self,
        topic: str,
        subtopic: str,
        difficulty: str,
        question_type: str,
    ) -> Optional[str]:
        """Generate a single question using the LLM."""
        prompt = QUESTION_GENERATION_PROMPT.format(
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            question_type=question_type,
        )
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.model.max_length,
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,  # Slightly higher for diversity
                top_p=self.config.model.top_p,
                top_k=self.config.model.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Decode only the generated portion
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            question = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            question = question.strip()
            
            return question if len(question) > 20 else None
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return None
    
    def generate_batch(
        self,
        topic: str,
        num_questions: int = 10,
    ) -> List[GeneratedQuestion]:
        """Generate multiple questions in a single LLM call (more efficient)."""
        topic_info = DSA_TOPICS.get(topic, {})
        
        prompt = BATCH_GENERATION_PROMPT.format(
            num_questions=num_questions,
            topic=topic_info.get("name", topic),
            topic_description=topic_info.get("description", ""),
        )
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.model.max_length,
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return self._parse_batch_questions(text, topic)
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return []
    
    def _parse_batch_questions(
        self, text: str, topic: str,
    ) -> List[GeneratedQuestion]:
        """Parse multiple questions from batch-generated text."""
        questions = []
        
        # Try to split by Q1:, Q2:, etc. pattern
        pattern = r'Q\d+:\s*(.+?)(?=Q\d+:|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # Fallback: split by newlines
            matches = [q.strip() for q in text.split('\n') if len(q.strip()) > 20]
        
        topic_info = DSA_TOPICS.get(topic, {})
        difficulties = ["easy", "medium", "hard"]
        q_types = ["conceptual", "coding", "analytical", "problem_solving"]
        
        for i, q_text in enumerate(matches):
            q_text = q_text.strip()
            if len(q_text) > 20 and self._passes_quality_check(q_text):
                questions.append(GeneratedQuestion(
                    question=q_text,
                    topic=topic,
                    subtopic=random.choice(topic_info.get("subtopics", [topic])),
                    difficulty=difficulties[i % len(difficulties)],
                    question_type=q_types[i % len(q_types)],
                    expected_concepts=topic_info.get("key_concepts", []),
                ))
        
        return questions
    
    def _passes_quality_check(self, question: str) -> bool:
        """
        Basic quality check for generated questions.
        
        Checks:
        1. Minimum length (at least 20 characters)
        2. Contains a question mark or imperative verb
        3. No excessive repetition
        4. Contains DSA-relevant keywords
        """
        # Length check
        if len(question) < 20 or len(question) > 1000:
            return False
        
        # Must look like a question or instruction
        question_indicators = ['?', 'write', 'implement', 'explain', 'what',
                              'how', 'why', 'compare', 'design', 'analyze',
                              'find', 'given', 'determine', 'calculate']
        has_indicator = any(ind in question.lower() for ind in question_indicators)
        if not has_indicator:
            return False
        
        # No excessive repetition (ratio of unique words)
        words = question.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too many repeated words
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Return generation statistics."""
        return {
            "total_generated": self.generated_count,
        }
