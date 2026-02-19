# SEAL-DSA: Simplified Self-Adapting Language Model for DSA Education

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Google%20Colab-Free%20Tier-orange.svg)](https://colab.research.google.com/)

## 📋 Abstract

**SEAL-DSA** implements a simplified version of the MIT CSAIL (2025) SEAL framework, adapted for Data Structures and Algorithms (DSA) education. The system creates an autonomous learning loop where a small language model (1–4B parameters) continuously improves its DSA knowledge through self-generated questions, self-evaluation, and micro-parameter updates using Low-Rank Adaptation (LoRA).

Key innovations:
- **Self-Improving Loop**: Generate → Attempt → Evaluate → Update cycle
- **LoRA Micro-Updates**: Efficient parameter updates without full retraining
- **Curriculum Learning**: Progressive 16-week DSA topic mastery
- **Free Infrastructure**: Runs entirely on Google Colab Free Tier
- **Catastrophic Forgetting Mitigation**: EWC-based regularization

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SEAL-DSA Framework                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Question    │───▶│   Answer     │───▶│  Evaluator   │  │
│  │  Generator    │    │  Generator   │    │   Module     │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│         ▲                                        │          │
│         │            ┌──────────────┐            │          │
│         └────────────│  Parameter   │◀───────────┘          │
│                      │   Updater    │                       │
│                      │  (LoRA)      │                       │
│                      └──────────────┘                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Curriculum   │    │  Checkpoint  │    │  Metrics     │  │
│  │  Scheduler    │    │   Manager    │    │  Tracker     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Open `notebooks/SEAL_DSA_Main.ipynb` in Google Colab
2. Run all cells sequentially
3. Results are saved to Google Drive automatically

### Option 2: Local Setup
```bash
git clone https://github.com/yourusername/SEAL-DSA.git
cd SEAL-DSA
pip install -r requirements.txt
python -m seal_dsa.main --config configs/default.yaml
```

## 📁 Repository Structure

```
SEAL-DSA/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── configs/
│   ├── default.yaml              # Default training configuration
│   └── colab_optimized.yaml      # Colab-specific configuration
├── seal_dsa/
│   ├── __init__.py
│   ├── main.py                   # Main entry point
│   ├── config.py                 # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Model loading with LoRA
│   │   └── lora_config.py        # LoRA configuration
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── question_generator.py # Self-question generation
│   │   ├── answer_generator.py   # Answer generation
│   │   ├── evaluator.py          # Rule-based evaluation
│   │   └── parameter_updater.py  # LoRA micro-updates
│   ├── curriculum/
│   │   ├── __init__.py
│   │   ├── scheduler.py          # Curriculum progression
│   │   └── dsa_topics.py         # DSA topic definitions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── seal_loop.py          # Main SEAL training loop
│   │   ├── ewc.py                # Elastic Weight Consolidation
│   │   └── checkpoint.py         # Checkpoint management
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── baseline.py           # Baseline comparisons
│   │   └── forgetting_detector.py# Catastrophic forgetting detection
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # Logging utilities
│       └── colab_utils.py        # Colab-specific utilities
├── data/
│   ├── dsa_seed_questions.json   # Seed questions for bootstrapping
│   └── evaluation_sets/
│       └── dsa_eval_set.json     # Held-out evaluation set
├── notebooks/
│   ├── SEAL_DSA_Main.ipynb       # Main Colab notebook
│   ├── Evaluation_Analysis.ipynb # Results analysis notebook
│   └── Visualization.ipynb      # Visualization notebook
├── docs/
│   ├── thesis/
│   │   ├── chapters/
│   │   │   ├── ch1_introduction.md
│   │   │   ├── ch2_literature_review.md
│   │   │   ├── ch3_methodology.md
│   │   │   ├── ch4_implementation.md
│   │   │   ├── ch5_results.md
│   │   │   └── ch6_conclusion.md
│   │   └── appendices/
│   │       ├── mathematical_formulations.md
│   │       └── viva_questions.md
│   └── architecture.md
├── results/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_question_generator.py
    ├── test_evaluator.py
    └── test_seal_loop.py
```

## 📊 Expected Results

| Metric | Baseline (Static) | SEAL-DSA | Improvement |
|--------|-------------------|----------|-------------|
| DSA Accuracy | ~35% | ~50-55% | 15-25% |
| Question Quality | N/A | 3.5/5.0 | - |
| Forgetting Rate | N/A | <5% | - |
| Training Cost | $0 | <$20 | - |
| GPU Hours | 0 | ~10-15h | - |

## 🔬 Key Technologies

- **Base Model**: Qwen2.5-1.5B / Phi-2 (2.7B) / TinyLlama-1.1B
- **Fine-tuning**: LoRA (rank 8-16) via PEFT library
- **Framework**: PyTorch + HuggingFace Transformers
- **Infrastructure**: Google Colab Free Tier (T4 GPU)
- **Forgetting Prevention**: Elastic Weight Consolidation (EWC)

## 📝 Citation

```bibtex
@mastersthesis{seal_dsa_2025,
  title={SEAL: A Simplified Self-Adapting Language Model for DSA Education using LoRA on Google Colab},
  author={Your Name},
  year={2025},
  school={Your University},
  type={M.Tech Thesis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MIT CSAIL SEAL Framework (2025)
- LoRA by Hu et al. (2022)
- HuggingFace PEFT library
- Google Colab platform
