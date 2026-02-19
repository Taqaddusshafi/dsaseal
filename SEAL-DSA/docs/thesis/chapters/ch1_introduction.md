# Chapter 1: Introduction

## 1.1 Background and Motivation

The rapid advancement of Large Language Models (LLMs) has revolutionized natural language processing and AI-assisted education. Models such as GPT-4, Gemini, and LLaMA have demonstrated remarkable capabilities in understanding and generating human-like text across various domains. However, a fundamental limitation persists: **these models are frozen after deployment**. Once trained, they cannot learn from new interactions, adapt to changing knowledge, or correct their own mistakes without expensive retraining.

This limitation is particularly detrimental in educational applications where:

1. **Knowledge evolves**: New algorithms, best practices, and teaching methodologies emerge continuously
2. **Student needs vary**: Different learners require different explanations and difficulty levels
3. **Mistakes compound**: A model that cannot learn from its errors will consistently produce the same incorrect outputs
4. **Retraining costs are prohibitive**: Full fine-tuning of even moderately-sized LLMs requires expensive GPU infrastructure that is inaccessible to most educational institutions

The **core research problem** addressed in this thesis is:

> _How can a language model continuously improve its domain-specific knowledge through autonomous self-evaluation and micro-parameter updates, while operating within the constraints of free cloud infrastructure?_

## 1.2 Problem Statement

Current approaches to maintaining LLM quality face several challenges:

| Challenge                   | Description                                         | Impact                      |
| --------------------------- | --------------------------------------------------- | --------------------------- |
| **Static Knowledge**        | Models cannot learn post-deployment                 | Outdated responses          |
| **Expensive Retraining**    | Full fine-tuning requires significant GPU resources | Cost barrier (>$1000)       |
| **Catastrophic Forgetting** | Learning new information destroys old knowledge     | Unreliable performance      |
| **No Self-Assessment**      | Models cannot evaluate their own responses          | No improvement signal       |
| **Infrastructure Costs**    | A100/H100 GPUs required for training                | Inaccessible to researchers |

## 1.3 Proposed Solution: SEAL-DSA

We propose **SEAL-DSA** (Simplified Self-Adapting Language model for DSA education), a lightweight implementation of the MIT CSAIL (2025) SEAL framework, specifically adapted for Data Structures and Algorithms education. Our system implements a **four-stage autonomous learning loop**:

```
┌──────────────────────────────────────────────────────────────┐
│                     SEAL Learning Loop                        │
│                                                               │
│   ┌─────────────┐     ┌─────────────┐     ┌──────────────┐  │
│   │  1. GENERATE │────▶│ 2. ATTEMPT  │────▶│ 3. EVALUATE  │  │
│   │   Questions  │     │   Answers   │     │   Quality    │  │
│   └─────────────┘     └─────────────┘     └──────┬───────┘  │
│          ▲                                        │          │
│          │              ┌──────────────┐          │          │
│          └──────────────│ 4. UPDATE    │◀─────────┘          │
│                         │  Parameters  │                     │
│                         │  (via LoRA)  │                     │
│                         └──────────────┘                     │
│                                                               │
│   Continuous Loop ─── Each iteration improves the model      │
└──────────────────────────────────────────────────────────────┘
```

### Key Innovations:

1. **Self-Supervised Learning Loop**: The model generates its own training data through question generation, answer attempts, and self-evaluation
2. **LoRA Micro-Updates**: Parameter-efficient fine-tuning that updates only 0.1-0.5% of model parameters
3. **Curriculum Learning**: Progressive topic introduction following a 16-week DSA syllabus
4. **EWC-Based Forgetting Prevention**: Elastic Weight Consolidation protects previously learned knowledge
5. **Free Infrastructure**: Entire system runs on Google Colab Free Tier (T4 GPU)

## 1.4 Research Objectives

1. **Primary**: Implement a simplified SEAL framework that achieves 15-25% accuracy improvement on DSA questions through autonomous self-improvement
2. **Secondary**: Demonstrate that LoRA-based micro-updates enable continuous learning without catastrophic forgetting (<5% degradation)
3. **Practical**: Achieve all objectives within Google Colab Free Tier constraints (T4 GPU, 15GB VRAM, <$20 total cost)
4. **Educational**: Create a reusable framework for educational AI that can be adapted to other domains

## 1.5 Scope and Limitations

### In Scope:

- Data Structures and Algorithms domain (7 major topics)
- Small open-source models (1-4B parameters)
- LoRA fine-tuning with PEFT
- Rule-based evaluation (no separate evaluator model)
- Google Colab Free Tier deployment

### Out of Scope:

- Multi-domain generalization
- Human-in-the-loop evaluation
- Models larger than 4B parameters
- Production deployment
- Real student interaction studies

## 1.6 Thesis Organization

| Chapter | Title                 | Description                                        |
| ------- | --------------------- | -------------------------------------------------- |
| 1       | Introduction          | Problem statement, motivation, and overview        |
| 2       | Literature Review     | Related work and theoretical foundations           |
| 3       | Methodology           | System architecture and mathematical formulations  |
| 4       | Implementation        | Technical details, code structure, and Colab setup |
| 5       | Results & Evaluation  | Experimental results and analysis                  |
| 6       | Conclusion            | Summary, limitations, and future work              |
| A       | Appendix: Mathematics | Detailed mathematical formulations                 |
| B       | Appendix: Viva Q&A    | Potential viva questions and answers               |
