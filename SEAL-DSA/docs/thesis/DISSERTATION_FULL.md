# SEAL-DSA DISSERTATION — COMPLETE TEXT

> **NOTE TO AUTHOR — DELETE THIS BLOCK BEFORE PRINTING.**
>
> 1. Fields in `[SQUARE BRACKETS]` must be filled in by you (name, roll number, supervisor,
>    year). They are deliberately left blank.
> 2. All experimental figures are from the run of 29 July 2026 comparing the base model
>    against `checkpoint_epoch_2`, i.e. epoch 3 of a planned 5. When the full run finishes,
>    replace the numbers in Chapter 8 Tables 8.1–8.5 and the figures quoted in Sections 8.4,
>    8.9, Chapter 13, and the Project Summary. Delete Section 11.5 once the run is complete.
> 3. Formatting per department spec: body 12 pt Times New Roman double spaced justified;
>    paragraph headings 14 pt underlined left aligned; chapter headings 20 pt centred with
>    30 pt spacing above and below; code 10 pt Courier New; margins left 2.5 cm, others 2 cm;
>    table legends above the table left aligned 10 pt; figure legends below centred 10 pt;
>    page numbers bottom right, numbering starting from Chapter 1.

---

# COVER PAGE

<br><br>

**SEAL-DSA: A Self-Adapting Language Model for Autonomous Improvement in Data Structures and Algorithms Education**

<br><br>

Submitted by

**[YOUR NAME] ([YOUR ROLL NUMBER])**

<br><br>

In the partial fulfillment of the requirement for the award of degree of
Master of Technology in Computer Science & Engineering
(M. Tech. CSE)

<br><br>

Submitted To

Department of Information Technology
School of Engineering & Technology
Central University of Kashmir
Ganderbal, J&K.

<br><br>

**2026**

---

# TITLE PAGE

<br><br>

**SEAL-DSA: A Self-Adapting Language Model for Autonomous Improvement in Data Structures and Algorithms Education**

<br><br>

Submitted By

**[YOUR NAME] ([YOUR ROLL NUMBER])**

<br><br><br>

In partial fulfillment of the requirement for the award of degree of
MASTER OF TECHNOLOGY IN
COMPUTER SCIENCE & ENGINEERING
(M. Tech. CSE)

<br><br>

Submitted To

<br>

Department of Information Technology
School of Engineering & Technology
Central University of Kashmir
Ganderbal, J&K.

<br><br>

**2026**

---

# CERTIFICATE

Ref No. : …………
Date: ……………

Department of Information Technology
School of Engineering & Technology
Central University of Kashmir
Ganderbal, J&K.

<br>

**CERTIFICATE**

<br>

This is to certify that the project titled "SEAL-DSA: A Self-Adapting Language Model for Autonomous Improvement in Data Structures and Algorithms Education" has been carried by:

**Mr. [YOUR NAME]**

under my supervision, in the partial fulfillment of the requirement for the award of degree of Master of Technology in Computer Science & Engineering (M. Tech. CSE) during the academic year 2026.

<br><br>

Supervisor:

<br>

**[SUPERVISOR NAME]**                                                                       **Head**
Assistant Professor,
Department of Information Technology

<br><br>

Certified that I have examined the project titled "SEAL-DSA: A Self-Adapting Language Model for Autonomous Improvement in Data Structures and Algorithms Education" on __ ______________

<br><br>

**External Examiner**

---

# ACKNOWLEDGEMENT

The completion of this dissertation would not have been possible without the support and guidance of a number of people, and it is a pleasure to record my gratitude to them here.

I am deeply indebted to my supervisor, [SUPERVISOR NAME], Assistant Professor, Department of Information Technology, for accepting me as a student and for the patience with which the direction of this work was guided. The decision to report the experimental findings of this project exactly as they were obtained, including the results that fell short of the initial expectation, was arrived at on the supervisor's advice, and I consider that lesson in research honesty to be the most valuable thing I have taken away from this project.

I express my sincere thanks to the Head, Department of Information Technology, and to the entire faculty of the School of Engineering & Technology, Central University of Kashmir, for providing the academic environment and the infrastructure that made this work possible.

I gratefully acknowledge the open-source community whose work this project is built upon. In particular, the Qwen team at Alibaba Cloud for releasing the Qwen2.5 model family under a permissive licence, and the maintainers of the HuggingFace Transformers, PEFT, and bitsandbytes libraries, without which parameter-efficient training on freely available hardware would not be feasible for a student project. I also acknowledge Google Colaboratory, whose free tier provided the entire computational budget for this work.

I thank my classmates in the M. Tech. programme for the many discussions that clarified my thinking, and particularly for the criticism that led me to question the reliability of my first evaluation design.

Finally, I thank my family for their patience and support over the duration of this work.

**[YOUR NAME]**
**[YOUR ROLL NUMBER]**

---

# PLAGIARISM UNDERTAKING

I solemnly declare that research/ Project work presented in the dissertation titled "SEAL-DSA: A Self-Adapting Language Model for Autonomous Improvement in Data Structures and Algorithms Education" is solely my research/ Project work. Small contribution/help wherever taken has been duly acknowledged and that complete dissertation has been written by me.

I understand the zero tolerance policy of the Central University of Kashmir towards plagiarism. Therefore I as an Author of the above titled dissertation declare that no portion of my dissertation has been plagiarized and any material used as reference is properly referred/cited.

I undertake that if I am found guilty of any formal plagiarism in the above titled dissertation even after award of degree, the University reserves the right to take action against me as per University norms.

<br><br>

Student /Author Signature:______________

Name:__________________

---

# LIST OF ABBREVIATIONS

| Abbreviation | Expansion |
|---|---|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AdamW | Adam optimizer with decoupled Weight decay |
| BFS | Breadth First Search |
| BST | Binary Search Tree |
| CPU | Central Processing Unit |
| CSAIL | Computer Science and Artificial Intelligence Laboratory |
| CUDA | Compute Unified Device Architecture |
| DFD | Data Flow Diagram |
| DFS | Depth First Search |
| DP | Dynamic Programming |
| DPO | Direct Preference Optimization |
| DSA | Data Structures and Algorithms |
| EWC | Elastic Weight Consolidation |
| FIM | Fisher Information Matrix |
| FP16 | 16-bit Floating Point |
| GPU | Graphics Processing Unit |
| JSON | JavaScript Object Notation |
| KV | Key–Value (attention cache) |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| NF4 | 4-bit NormalFloat |
| NLP | Natural Language Processing |
| PEFT | Parameter-Efficient Fine-Tuning |
| QLoRA | Quantized Low-Rank Adaptation |
| RAG | Retrieval Augmented Generation |
| RAM | Random Access Memory |
| RLHF | Reinforcement Learning from Human Feedback |
| SEAL | Self-Adapting Language model |
| UML | Unified Modeling Language |
| VRAM | Video Random Access Memory |
| YAML | YAML Ain't Markup Language |

---

# ABSTRACT

Large language models are frozen at the point of deployment. Once training concludes, a model cannot learn from its own mistakes, cannot incorporate knowledge that emerged after its data cutoff, and cannot specialise toward the needs of the users it actually serves, unless it is retrained at a cost that places the procedure beyond the reach of most educational institutions. This dissertation investigates whether that limitation can be relaxed for a narrow domain, using hardware that costs nothing.

This work presents SEAL-DSA, a self-adapting language model for Data Structures and Algorithms education. The system implements a closed autonomous learning loop in which a single model generates its own practice questions for a selected topic, attempts to answer them, evaluates the quality of its own answers using a combination of a rule-based rubric and sandboxed execution of the generated code against test cases, and updates its own parameters from the resulting signal. Adaptation is confined to a low-rank adapter comprising approximately 0.09% of the model's parameters, and an Elastic Weight Consolidation penalty derived from a diagonal Fisher Information approximation restrains movement in parameters identified as important to previously acquired capability. A curriculum scheduler directs training effort toward topics on which the model has demonstrated weakness. The complete system operates on the free tier of Google Colaboratory using a single NVIDIA T4 GPU, with the base model held in 4-bit NF4 quantization.

The system was evaluated on a held-out set of seventy questions spanning seven DSA topics, comparing the frozen base model against the self-adapted model within a single session so that a paired question-level comparison is valid. Functional correctness, measured as the fraction of held-out test cases passed by generated code, improved from 0.1111 to 0.1667, a relative gain of 50.0%. The aggregate composite score improved by 0.66%, a figure that is shown to be constrained by ceiling effects in three of its four components, each of which is already saturated on the base model before any training occurs. Improvement concentrated in the two weakest topics, arrays and strings and dynamic programming, consistent with the intended behaviour of the curriculum scheduler. Capability retention was high: 95.7% of held-out questions scored unchanged or higher after adaptation, and the largest topic-level regression was 0.0159, indicating that catastrophic forgetting did not occur.

The dissertation reports these findings without inflation. The gains are real but small, they are concentrated in a subset of the domain, they are not established as statistically significant on a set of this size, and they were measured with an instrument whose saturation limits what could have been observed. The contribution of the work therefore lies in a functioning end-to-end implementation of self-adaptive training under severe resource constraints, in an execution-based evaluation harness for generated algorithmic code, in empirical evidence that low-rank adaptation regularised by EWC preserves prior capability under self-generated training, and in a precise account of the conditions under which the approach fails to deliver more.

**Keywords:** Self-adapting language models, continual learning, low-rank adaptation, elastic weight consolidation, catastrophic forgetting, curriculum learning, parameter-efficient fine-tuning, educational artificial intelligence.

---

# TABLE OF CONTENTS

| Chapter | Title | Page |
|---|---|---|
| | Certificate | i |
| | Acknowledgement | ii |
| | Plagiarism Undertaking | iii |
| | List of Abbreviations | iv |
| | Abstract | v |
| | Table of Contents | vi |
| | List of Tables | vii |
| | List of Figures | viii |
| 1 | Introduction | 1 |
| 2 | Objectives and Scope | |
| 3 | Literature Review and Existing Systems | |
| 4 | Proposed System and Methodology | |
| 5 | Feasibility and Practicability | |
| 6 | Analysis and Design | |
| 7 | System Implementation | |
| 8 | Evaluation of Results | |
| 9 | Snapshots | |
| 10 | Tools and Utilities | |
| 11 | Limitations and Drawbacks | |
| 12 | Future Enhancement | |
| 13 | Conclusion | |
| | Bibliography | |
| | Project Summary | |
| | Appendix A | |

**LIST OF TABLES**

| Table | Title |
|---|---|
| 1.1 | Challenges in maintaining deployed language model quality |
| 3.1 | Comparison of MIT SEAL and SEAL-DSA |
| 3.2 | Full fine-tuning versus low-rank adaptation |
| 3.3 | Continual learning methods considered |
| 4.1 | Evaluation rubric dimensions and weights |
| 4.2 | Curriculum topic ordering |
| 5.1 | Memory budget on the T4 GPU |
| 5.2 | Colab limitations and adopted mitigations |
| 6.1 | Checkpoint directory contents |
| 7.1 | Technology stack |
| 7.2 | Candidate base models considered |
| 7.3 | Training hyperparameters |
| 8.1 | Aggregate performance before and after self-adaptive training |
| 8.2 | Topic-wise scores before and after self-adaptation |
| 8.3 | Partial topic trajectory across checkpoints |
| 8.4 | Paired question-level outcomes |
| 8.5 | Efficiency observations |
| 10.1 | Software components and versions |

**LIST OF FIGURES**

| Figure | Title |
|---|---|
| 1.1 | The SEAL four-stage learning loop |
| 4.1 | SEAL-DSA system architecture |
| 4.2 | Data flow through one SEAL iteration |
| 6.1 | Level 0 data flow diagram |
| 6.2 | Level 1 data flow diagram |
| 6.3 | Use case diagram |
| 6.4 | Sequence diagram of one training iteration |
| 6.5 | Class diagram of core modules |
| 8.1 | Topic-wise performance before and after self-adaptation |
| 8.2 | Evaluation metric breakdown |
| 9.1 | Training loop console output |
| 9.2 | Checkpoint directory on Google Drive |
| 9.3 | Evaluation run console output |

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background and Motivation

The last five years have produced a class of language models whose fluency in technical domains would have been difficult to credit a decade ago. Models in the GPT, Gemini, and LLaMA families answer questions about algorithms, write working code, and explain their reasoning at a standard that is often adequate for an undergraduate audience. Yet all of them share a structural limitation that is rarely stated as prominently as their capabilities: they are frozen at the moment of deployment. The weights that ship are the weights that remain. A deployed model cannot learn from an interaction, cannot repair an error it has been shown to make repeatedly, and cannot acquire knowledge that did not exist when its training data was assembled.

In general-purpose applications this limitation is tolerable, because the vendor periodically retrains and releases a successor. In educational applications it is considerably more damaging, for four reasons.

First, knowledge in a technical discipline evolves. New algorithms are published, complexity bounds are improved, and the practices that are considered idiomatic in a programming language shift over time. A frozen model teaches what was current at its cutoff.

Second, learners differ. The explanation that unlocks a concept for one student is the explanation that confuses another, and a model that cannot adapt cannot discover which explanations work.

Third, errors persist and compound. A model that consistently mis-states the space complexity of an algorithm will mis-state it in every session, to every student, indefinitely. There is no mechanism by which the error self-corrects.

Fourth, and most consequentially for an institution rather than a vendor, the cost of the remedy is prohibitive. Full fine-tuning of even a moderately sized model demands multi-GPU infrastructure, and the total cost of a single retraining cycle exceeds the annual computational budget of most departments. The capability to fix the model exists, but not for the people who most need it fixed.

The core research problem addressed in this dissertation follows directly from these observations:

> *How can a language model continuously improve its domain-specific competence through autonomous self-evaluation and micro-parameter updates, while operating entirely within the constraints of freely available cloud infrastructure?*

## 1.2 Problem Statement

Table 1.1 — Challenges in maintaining deployed language model quality

| Challenge | Description | Consequence |
|---|---|---|
| Static knowledge | Model cannot learn after deployment | Responses become outdated |
| Expensive retraining | Full fine-tuning demands substantial GPU resources | Cost barrier exceeding institutional budgets |
| Catastrophic forgetting | Acquiring new information destroys existing capability | Unreliable and unpredictable performance |
| Absence of self-assessment | Model cannot judge the quality of its own output | No signal available to drive improvement |
| Infrastructure requirements | High-memory accelerators required for training | Inaccessible to most academic researchers |

These challenges interact rather than acting independently. The absence of self-assessment means there is no improvement signal; even if a signal existed, catastrophic forgetting means acting on it risks destroying existing capability; and even if forgetting could be controlled, the infrastructure required to act places the whole procedure out of reach. A workable solution must address all three simultaneously, and must do so within a computational budget that a student can actually obtain.

## 1.3 Proposed Solution

This dissertation proposes SEAL-DSA, a simplified self-adapting language model specialised to Data Structures and Algorithms education. The design follows the four-stage autonomous learning loop introduced by the MIT CSAIL SEAL framework, adapted for a base model two orders of magnitude smaller and hardware that costs nothing to obtain.

```
        ┌──────────────────────────────────────────────────────────┐
        │                  SEAL Learning Loop                       │
        │                                                           │
        │  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
        │  │ 1. GENERATE │───▶│ 2. ATTEMPT  │───▶│ 3. EVALUATE  │  │
        │  │  Questions  │    │   Answers   │    │   Quality    │  │
        │  └─────────────┘    └─────────────┘    └──────┬───────┘  │
        │         ▲                                      │          │
        │         │           ┌──────────────┐           │          │
        │         └───────────│  4. UPDATE   │◀──────────┘          │
        │                     │  Parameters  │                      │
        │                     │  (via LoRA)  │                      │
        │                     └──────────────┘                      │
        │                                                           │
        │   Each traversal of the loop produces a parameter update  │
        └──────────────────────────────────────────────────────────┘
```

Figure 1.1 — The SEAL four-stage learning loop

The system contributes five design elements:

1. **A self-supervised learning loop.** The model produces its own training corpus by generating questions, attempting answers, and scoring those answers. No externally curated training data is required at any point after initialisation.

2. **Low-rank micro-updates.** Adaptation is confined to a LoRA adapter comprising approximately 0.09% of the model's parameters, with the base weights frozen throughout.

3. **Curriculum-directed training.** A scheduler allocates generation effort toward topics on which measured performance is weakest, rather than distributing effort uniformly.

4. **Explicit forgetting prevention.** An Elastic Weight Consolidation penalty, computed from a diagonal Fisher Information approximation, restrains movement in parameters identified as important to prior capability.

5. **Zero-cost infrastructure.** The entire system runs on the Google Colaboratory free tier with a single T4 GPU, with checkpoints persisted to Google Drive so that training survives the session limit.

## 1.4 Statement of Findings

Because the honest reporting of results is central to the argument of this dissertation, the principal findings are stated here rather than being deferred.

Self-adaptive training produced a measurable improvement in functional correctness, from a test-case pass rate of 0.1111 to 0.1667, and a substantially smaller improvement of 0.66% in the aggregate composite score. Chapter 8 demonstrates that the disparity between these figures is an artefact of the evaluation instrument rather than of the training method: three of the four scoring components are already saturated on the base model and cannot register improvement. Capability retention was high, with 95.7% of held-out questions unchanged or improved.

The gains are therefore real, but they are small, concentrated in two topics, and not established as statistically significant on a seventy-question evaluation set. Chapter 11 states the reasons in full and Chapter 12 converts them into a programme of work.

## 1.5 Organisation of the Dissertation

Chapter 2 states the objectives and delimits the scope. Chapter 3 reviews the literature on self-improving models, parameter-efficient fine-tuning, continual learning, and curriculum learning, and positions this work relative to existing systems. Chapter 4 presents the proposed methodology and the mathematical framework. Chapter 5 assesses technical, economic, and operational feasibility. Chapter 6 gives the analysis and design, including data flow diagrams, UML models, the data model, and the system architecture. Chapter 7 describes the implementation. Chapter 8 reports and interprets the experimental results. Chapter 9 presents execution snapshots. Chapter 10 documents the tools and environment. Chapter 11 states the limitations, Chapter 12 proposes future work, and Chapter 13 concludes.

---

# CHAPTER 2 — OBJECTIVES AND SCOPE

## 2.1 Primary Objective

To design, implement, and empirically evaluate a self-adapting language model that improves its own competence in Data Structures and Algorithms through an autonomous loop of question generation, answer generation, self-evaluation, and parameter update, without recourse to externally curated training data and without exceeding the resources of the Google Colaboratory free tier.

## 2.2 Specific Objectives

**Objective 1 — Implement the four-stage SEAL loop.** Construct question generation, answer generation, evaluation, and parameter update modules that operate as a closed cycle on a single model, such that the output of each stage is the input of the next without human intervention.

**Objective 2 — Establish an evaluation mechanism that verifies correctness rather than form.** Implement a scoring harness that extracts generated Python functions, executes them in an isolated namespace against held-out test cases, and reports the fraction passing, so that at least one component of the evaluation reflects functional correctness rather than surface plausibility.

**Objective 3 — Prevent catastrophic forgetting.** Combine low-rank adaptation with an Elastic Weight Consolidation penalty, and demonstrate empirically that self-generated training does not degrade previously held capability by more than five percent.

**Objective 4 — Direct training effort by measured weakness.** Implement a curriculum scheduler that allocates question generation toward topics on which evaluated performance is lowest, and verify from topic-level results that gains do in fact concentrate where the scheduler directed effort.

**Objective 5 — Operate within free infrastructure.** Complete all training and evaluation on a single T4 GPU with checkpointing sufficient to survive session termination, at zero monetary cost.

**Objective 6 — Report findings without inflation.** Characterise the magnitude of the improvement honestly, including an analysis of the extent to which the evaluation instrument constrains what improvement could have been observed.

It should be recorded that an earlier statement of objectives, formulated during Phase I of this project, targeted an accuracy improvement in the range of fifteen to twenty-five percent. The results reported in Chapter 8 do not meet that target, and Chapter 11 analyses the reasons. The target is retained in this dissertation as a matter of record rather than being quietly revised downward, since revising a hypothesis after observing the data would compromise the validity of the study.

## 2.3 Scope

**Within scope.** The Data Structures and Algorithms domain restricted to seven topics: arrays and strings, linked lists, stacks and queues, trees, graphs, sorting and searching, and dynamic programming. Open-source instruction-tuned models in the one to four billion parameter range. Low-rank adaptation via the PEFT library under 4-bit quantization. Rule-based and execution-based evaluation. Deployment on the Google Colaboratory free tier.

**Outside scope.** Generalisation across domains beyond DSA. Evaluation by a separate judge model or by human annotators. Base models exceeding four billion parameters. Production deployment with concurrent users. Studies involving real students. Multilingual operation. Programming languages other than Python for the executable evaluation path.

## 2.4 Expected Deliverables

A working implementation of the complete SEAL loop as a documented Python package; a held-out evaluation set of seventy DSA questions with associated test cases; an execution-based evaluation harness producing per-question, per-topic, and aggregate reports; trained LoRA checkpoints for each completed epoch; and this dissertation reporting the design, implementation, and empirical findings.

---

# CHAPTER 3 — LITERATURE REVIEW AND EXISTING SYSTEMS

## 3.1 Artificial Intelligence in Education

The application of computation to instruction long predates the current generation of language models. ELIZA demonstrated in 1966 that pattern matching alone could sustain the appearance of a tutorial dialogue. Intelligent tutoring systems of the 1980s and 1990s, of which Carnegie Learning's Mathia is a representative example, added explicit student models and adaptive problem selection, but relied on hand-authored domain knowledge that was expensive to produce and brittle outside its intended scope.

The arrival of large language models altered the economics decisively. Khan Academy's Khanmigo, built on GPT-4 and released in 2023, delivers tutorial dialogue across many subjects without any hand-authored domain model. What it does not do, and what no system of its class does, is learn. The pedagogical content knowledge it possesses is whatever survived pre-training, and it is fixed.

### 3.1.1 The Deploy-and-Forget Paradigm

Contemporary educational language models exhibit four failure modes traceable to a single cause. Knowledge staleness arises because training data has a cutoff. Absence of personalisation arises because weights do not respond to individual interaction histories. Error persistence arises because there is no mechanism by which a corrected mistake propagates into the parameters. Domain shallowness arises because general-purpose pre-training allocates capacity broadly rather than deeply. All four follow from the model being frozen, and all four are therefore addressable in principle by a mechanism that permits controlled post-deployment adaptation.

## 3.2 The MIT CSAIL SEAL Framework

The SEAL framework, introduced by MIT CSAIL in 2025, proposes that a language model can generate its own training signal and update its own parameters in a closed loop, without external supervision. The framework organises this into four stages: self-question generation, in which the model produces problems within its domain; self-answer attempt, in which the model responds using its current weights; self-evaluation, in which the model judges the quality of its own response; and self-update, in which the parameters are modified according to that judgement.

The theoretical significance of the framework lies in three claims. First, that a sufficiently capable model can generate training data of adequate quality to drive its own improvement. Second, that updates confined to a small parameter subset can improve domain performance without degrading general capability. Third, that the improvement trajectory can be measured and monitored across iterations.

### 3.2.1 Divergences in the Present Work

SEAL-DSA is a deliberate simplification undertaken to fit a student computational budget. The divergences are material and are stated explicitly, because several of them bear directly on the magnitude of the results obtained.

Table 3.1 — Comparison of MIT SEAL and SEAL-DSA

| Aspect | MIT SEAL | SEAL-DSA (this work) |
|---|---|---|
| Base model scale | 7B – 70B parameters | 1.5B parameters |
| Evaluation mechanism | Model-based, LLM-as-judge | Rule-based rubric with code execution |
| Domain coverage | Multi-domain | Data Structures and Algorithms only |
| Infrastructure | Multi-GPU cluster | Single T4 GPU, free tier |
| Adaptation method | Full fine-tuning or LoRA | LoRA only, under 4-bit quantization |
| Approximate cost | In excess of USD 100 | Nil |

The first two divergences are the consequential ones. A 1.5B model is a substantially weaker generator of training questions and a substantially less reliable judge than a 70B model, and a rule-based rubric captures far less than a competent judge model. Chapter 11 returns to both as limitations.

## 3.3 Low-Rank Adaptation

### 3.3.1 The Original Formulation

LoRA, introduced by Hu et al. in 2022, rests on the observation that the weight update accumulated during fine-tuning possesses low intrinsic rank, and can therefore be represented by a product of two thin matrices without substantial loss of expressiveness.

For a pre-trained weight matrix **W₀ ∈ ℝ^(d×k)**, the adapted weight is

```
W = W₀ + ΔW = W₀ + BA
```

where **B ∈ ℝ^(d×r)** is initialised to zero, **A ∈ ℝ^(r×k)** is initialised from a random Gaussian, and the rank **r** satisfies r ≪ min(d, k). The modified forward pass is

```
h = W₀x + (α/r) · BAx
```

with α/r acting as a scaling factor. Initialising B to zero guarantees that ΔW = 0 at the start of training, so the adapted model begins with exactly the behaviour of the pre-trained model; initialising A randomly breaks the symmetry that would otherwise prevent gradient flow.

The parameter saving is substantial. For a layer with d = k = 4096 and rank r = 8, full fine-tuning updates 16,777,216 parameters whereas LoRA updates 65,536, or 0.39% of the total.

### 3.3.2 Quantized Low-Rank Adaptation

QLoRA, introduced by Dettmers et al. in 2023, extends the technique by holding the frozen base weights in 4-bit precision. Three innovations make this viable: the 4-bit NormalFloat data type, whose quantization levels are placed to be information-theoretically optimal for normally distributed weights; double quantization, in which the quantization constants are themselves quantized; and paged optimizers, which handle transient memory spikes by migrating optimizer state to host memory.

For the 1.5B model used in this work, FP16 storage requires approximately 3 GB whereas NF4 storage requires approximately 0.8 GB, and the complete training configuration including adapters and optimizer state occupies approximately 1 GB. It is this reduction that makes the present work possible on free infrastructure at all.

### 3.3.3 Justification for Low-Rank Adaptation

Table 3.2 — Full fine-tuning versus low-rank adaptation

| Criterion | Full fine-tuning | Low-rank adaptation |
|---|---|---|
| Proportion of parameters updated | 100% | 0.09% – 0.5% |
| GPU memory requirement | Very high | Low |
| Wall-clock training time | Hours | Minutes |
| Susceptibility to forgetting | High | Low |
| Modularity | None | Adapters are swappable |
| Feasible on Colab free tier | No, above 1B parameters | Yes |

The fourth row is the one that matters most for this work. Restricting updates to a low-rank subspace is not merely an efficiency measure; it is the first and most effective line of defence against catastrophic forgetting, because the majority of the model's parameters are structurally incapable of moving.

## 3.4 Self-Rewarding Language Models

Yuan et al. demonstrated in 2024 that language models can serve as judges of their own output and that these judgements constitute a usable training signal. Their method rates self-generated responses along multiple dimensions, constructs preference pairs from the ratings, and optimises against those preferences using Direct Preference Optimization, iterating the cycle to obtain consistent improvement.

The relevance to the present work is direct, since the evaluation stage of the SEAL loop is precisely a self-judgement. The divergence is that a 1.5B model is not a reliable judge of its own algorithmic reasoning, and this work therefore substitutes a rule-based rubric supplemented by actual execution of generated code. Execution is the more trustworthy signal of the two, and Chapter 8 argues on this basis that it should be treated as the primary metric.

## 3.5 Continual Learning and Catastrophic Forgetting

### 3.5.1 The Phenomenon

A neural network trained sequentially on task A and subsequently on task B characteristically loses its performance on task A. The effect was documented by McCloskey and Cohen in 1989 and analysed extensively by French in 1999. Its cause is that gradient descent on task B has no term expressing any preference for retaining the configuration that solved task A; parameters move wherever the new objective directs them.

For a self-adaptive system the phenomenon is not incidental but existential. A model that improves on dynamic programming while losing its competence on linked lists has not improved at all, and since a self-adaptive loop runs continuously without human oversight, such degradation could accumulate undetected.

### 3.5.2 Elastic Weight Consolidation

EWC, introduced by Kirkpatrick et al. in 2017, addresses forgetting by identifying which parameters mattered for previous tasks and penalising their movement. The regularised objective is

```
L_total = L_task(θ) + (λ/2) Σᵢ Fᵢ (θᵢ − θ*ᵢ)²
```

where L_task is the loss on the current task, Fᵢ is the i-th diagonal element of the Fisher Information Matrix, θ*ᵢ is the parameter value at the conclusion of the previous task, and λ controls regularisation strength.

The Fisher diagonal is defined as the expected squared gradient of the log-likelihood,

```
Fᵢ = E[( ∂ log p(x|θ) / ∂θᵢ )²]
```

and is approximated in practice by the empirical mean of squared gradients over a sample of N examples,

```
F̂ᵢ ≈ (1/N) Σₙ (∂Lₙ / ∂θᵢ)²
```

The interpretation is that a parameter whose perturbation sharply changes the model's predictions on the previous task is important to that task, and should be held near its established value; a parameter whose perturbation changes little is free to move.

### 3.5.3 Alternatives Considered

Table 3.3 — Continual learning methods considered

| Method | Category | Central idea |
|---|---|---|
| Elastic Weight Consolidation (2017) | Regularisation | Penalise movement in important parameters |
| Synaptic Intelligence (2017) | Regularisation | Accumulate importance online during training |
| PackNet (2018) | Architecture | Prune and freeze a subnetwork per task |
| Progressive Networks (2016) | Architecture | Add a new column of parameters per task |
| Experience Replay (2019) | Replay | Store and interleave examples from prior tasks |
| Gradient Episodic Memory (2017) | Gradient projection | Constrain gradients against interference |

EWC was selected on four grounds. It is memory-efficient, storing only two floating-point values per adapted parameter. It requires no retention of prior training data, which matters because the training data in this system is generated afresh each epoch and is not persisted. It rests on a well-developed theoretical foundation. And it composes naturally with low-rank adaptation, since the penalty applies to the adapter parameters directly and adds a term to the same backward pass.

Replay-based methods were rejected because storing generated data across epochs would consume Colab disk and reintroduce the data-management burden the self-generating design exists to avoid. Architectural methods were rejected because they alter the parameter count per task, which conflicts with the fixed adapter geometry.

## 3.6 Curriculum Learning

Bengio et al. proposed in 2009 that presenting training examples in order of increasing difficulty improves both convergence rate and final quality, by analogy with human pedagogy. The theoretical account is that easy examples present a smoother optimisation landscape early in training, guiding parameters toward a basin from which harder examples can be accommodated.

SEAL-DSA applies this in two ways. The topic ordering follows the prerequisite structure of a standard sixteen-week DSA syllabus, from arrays through to dynamic programming. Additionally, an adaptive mode allocates effort by measured weakness rather than by fixed order, which is the mode used for the experiments reported in Chapter 8 and which the topic-level results in Section 8.5 permit us to assess directly.

## 3.7 Positioning of the Present Work

```
   MIT SEAL (2025) ──────────▶ Self-improving loop architecture
   LoRA (2022) ──────────────▶ Parameter-efficient update mechanism
   QLoRA (2023) ─────────────▶ 4-bit quantization for constrained hardware
   Self-Rewarding LM (2024) ─▶ Self-evaluation as a training signal
   EWC (2017) ───────────────▶ Retention of prior capability
   Curriculum Learning (2009)▶ Ordering and allocation of training effort
                    │
                    ▼
   ┌──────────────────────────────────────────────────────┐
   │  SEAL-DSA (this dissertation)                        │
   │  Integration of the above under free-tier constraints │
   │  with execution-based evaluation of generated code    │
   └──────────────────────────────────────────────────────┘
```

The claim to novelty is integrative rather than component-wise. No individual mechanism employed here is original to this work. What has not previously been demonstrated, to the author's knowledge, is that the complete combination operates end to end on a 1.5B model within a free-tier session, and what has not previously been characterised is how the approach behaves at that scale, including the respects in which it disappoints.

---

# CHAPTER 4 — PROPOSED SYSTEM AND METHODOLOGY

## 4.1 System Overview

SEAL-DSA comprises four modules forming the core learning loop and three supporting subsystems.

```
┌──────────────────────────────────────────────────────────────────┐
│                     SEAL-DSA Architecture                         │
│                                                                   │
│ ╔═══════════════════════════════════════════════════════════════╗ │
│ ║                       SEAL Core Loop                          ║ │
│ ║                                                               ║ │
│ ║ ┌─────────────┐   ┌─────────────┐   ┌──────────────┐         ║ │
│ ║ │  Question   │──▶│   Answer    │──▶│  Evaluator   │         ║ │
│ ║ │  Generator  │   │  Generator  │   │   Module     │         ║ │
│ ║ │             │   │             │   │              │         ║ │
│ ║ │ Generates   │   │ Attempts    │   │ Scores on    │         ║ │
│ ║ │ DSA problems│   │ answers with│   │ five rubric  │         ║ │
│ ║ │ for current │   │ current     │   │ dimensions + │         ║ │
│ ║ │ topic       │   │ weights     │   │ execution    │         ║ │
│ ║ └─────────────┘   └─────────────┘   └──────┬───────┘         ║ │
│ ║        ▲                                    │                 ║ │
│ ║        │          ┌──────────────┐          │                 ║ │
│ ║        └──────────│  Parameter   │◀─────────┘                 ║ │
│ ║                   │   Updater    │                            ║ │
│ ║                   │   (LoRA)     │                            ║ │
│ ║                   └──────────────┘                            ║ │
│ ╚═══════════════════════════════════════════════════════════════╝ │
│                                                                   │
│ ╔═══════════════════════════════════════════════════════════════╗ │
│ ║                     Supporting Subsystems                     ║ │
│ ║ ┌─────────────┐  ┌────────────┐  ┌──────────────────┐        ║ │
│ ║ │ Curriculum  │  │    EWC     │  │   Checkpoint     │        ║ │
│ ║ │ Scheduler   │  │   Module   │  │    Manager       │        ║ │
│ ║ │             │  │            │  │                  │        ║ │
│ ║ │ Controls    │  │ Prevents   │  │ Persists state   │        ║ │
│ ║ │ topic order │  │ forgetting │  │ to Google Drive  │        ║ │
│ ║ └─────────────┘  └────────────┘  └──────────────────┘        ║ │
│ ╚═══════════════════════════════════════════════════════════════╝ │
└──────────────────────────────────────────────────────────────────┘
```

Figure 4.1 — SEAL-DSA system architecture

## 4.2 Question Generator

**Purpose.** To produce DSA questions constituting the training corpus for the current iteration.

**Rationale.** Self-generation is what makes the loop autonomous. The diversity and difficulty of generated questions bound what the system can learn, since a model cannot improve on material it never poses to itself.

**Procedure.** The module receives a topic from the curriculum scheduler, constructs a prompt from a topic-specific template, and samples questions at temperature 0.8. The comparatively high temperature is deliberate: question generation is the one stage where diversity is more valuable than determinism, since a set of near-identical questions provides a near-worthless training signal.

**Quality filtering.** Generated candidates are retained only if they exceed twenty characters, contain a recognised interrogative or imperative marker such as a question mark or the tokens *implement* or *explain*, exhibit a unique-word ratio above 0.3 so that degenerate repetition is excluded, and contain detectably DSA-relevant content. Surviving questions are labelled by difficulty and by type, conceptual or coding.

## 4.3 Answer Generator

**Purpose.** To answer the self-generated questions using the model's current weights.

**Design decision.** The same model performs both generation and answering. This is essential rather than merely economical: the answers must reflect the model's present competence, because the training signal is precisely the discrepancy between present competence and the standard the evaluator applies. Employing a stronger model to answer would convert the system into ordinary distillation.

**Procedure.** Answers are generated at temperature 0.3 through the model's chat template, favouring focused output over diversity. Generation scores are retained and used to estimate confidence, avoiding the redundant forward pass that a separate confidence computation would require. The resulting question–answer pair passes to the evaluator.

## 4.4 Evaluator

**Purpose.** To score answers, producing the signal that drives the parameter update.

**Why rule-based rather than model-based.** A 1.5B model is an unreliable judge of its own algorithmic reasoning; asking it to score its own answer frequently yields a confident endorsement of an incorrect response. A rule-based evaluator supplemented by execution offers determinism and reproducibility, imposes no additional GPU memory cost, produces interpretable per-dimension feedback, and — in the execution component — provides one signal that cannot be satisfied by fluent but wrong output. The cost of this choice is discussed candidly in Section 11.2.

**Dimensions.** The training-time rubric combines five weighted dimensions:

```
Score = 0.35 × Correctness + 0.25 × Completeness
      + 0.20 × Complexity  + 0.10 × Code + 0.10 × Explanation
```

Table 4.1 — Evaluation rubric dimensions and weights

| Dimension | Weight | Property measured |
|---|---|---|
| Correctness | 0.35 | Presence of correct DSA concepts and terminology |
| Completeness | 0.25 | Coverage of the requirements stated in the question |
| Complexity analysis | 0.20 | Presence of Big-O notation and complexity reasoning |
| Code quality | 0.10 | Presence and basic syntactic validity of code |
| Explanation quality | 0.10 | Structure, use of examples, reasoning flow |

**Execution-based verification.** For questions carrying a function name and test cases, the evaluator extracts the generated function from the response, executes it in an isolated namespace, and applies each test case, reporting the fraction passing. Failures at any stage — extraction, compilation, or execution — are treated as a zero rather than propagating an exception, so that a malformed answer degrades the score rather than terminating the run. This component is the only one in the entire system that verifies functional correctness, and Chapter 8 argues that it should therefore be treated as the primary reported metric.

## 4.5 Parameter Updater

**Purpose.** To translate evaluation scores into a gradient update on the LoRA adapter.

**Construction of the training signal.** Answers scoring above the acceptance threshold are treated as positive examples and weighted in the loss by their score, so that a better answer exerts proportionally more influence. Answers scoring below the threshold are retained as corrective examples at reduced weight, so that the model is not driven to reproduce its own poor output.

**Objective.**

```
L_total = L_task + λ_ewc · L_ewc

L_task  = −(1/B) Σᵢ wᵢ Σⱼ log P(yᵢⱼ | xᵢ, y_{i,<j}; θ)

L_ewc   = (λ/2) Σₖ Fₖ (θₖ − θ*ₖ)²
```

where B is the batch size, wᵢ is the evaluator's quality weight for sample i, Fₖ is the Fisher diagonal for parameter k, and θ*ₖ is the value of that parameter at the conclusion of the previous topic.

**Alternative objective.** A Direct Preference Optimization objective is additionally implemented and selectable by configuration, constructing preference pairs from higher- and lower-scoring answers to the same question. It was not used for the experiments reported in Chapter 8, and comparing the two objectives under identical conditions is proposed as future work in Section 12.7.

## 4.6 Mathematical Framework

### 4.6.1 Low-Rank Decomposition

For each targeted attention projection with pre-trained weight **W₀ ∈ ℝ^(d_out × d_in)** the forward pass becomes

```
h = (W₀ + ΔW)x = W₀x + (α/r) · BAx
```

with **B ∈ ℝ^(d_out × r)** initialised to zero and **A ∈ ℝ^(r × d_in)** initialised from N(0, σ²). The gradients are

```
∂L/∂B = (α/r) · (∂L/∂h) · (Ax)ᵀ
∂L/∂A = (α/r) · Bᵀ · (∂L/∂h) · xᵀ
```

### 4.6.2 Quantization

Base weights are held in 4-bit NormalFloat. Quantization maps a weight to the nearest of sixteen levels optimised for a normal distribution,

```
Q_NF4(w) = argminᵢ |w − qᵢ|
```

and de-quantization during the forward pass reconstructs

```
w_fp16 = scale × Q_NF4⁻¹(w_nf4) + zero_point
```

Under double quantization the scale factors are themselves quantized, giving an effective cost of approximately four bits per parameter plus eight bits per quantization block.

### 4.6.3 Elastic Weight Consolidation

The Fisher diagonal is estimated empirically,

```
F̂ᵢ = (1/N) Σₙ₌₁ᴺ (∂Lₙ/∂θᵢ)²
```

giving the penalty and its gradient

```
L_EWC(θ)      = (λ/2) Σᵢ Fᵢ (θᵢ − θ*ᵢ)²
∂L_EWC/∂θᵢ    = λ · Fᵢ · (θᵢ − θ*ᵢ)
```

Fisher estimates are accumulated across topics by the online update of Schwarz et al. (2018),

```
F_new = γ · F_old + (1 − γ) · F_current
```

with decay γ = 0.9, so that importance attributed to older topics decays gradually rather than being discarded.

### 4.6.4 Complete Objective

```
θ* = argmin_θ [ L_task(θ) + (λ/2) Σᵢ Fᵢ (θᵢ − θ*ᵢ)² ]
```

subject to the constraints that only adapter parameters θ_LoRA are updated, base parameters θ₀ remain frozen, and peak memory remains below the 15 GB available on the T4. Optimisation uses AdamW, with the EWC term entering through the gradient:

```
mₜ = β₁ m_{t−1} + (1 − β₁) gₜ
vₜ = β₂ v_{t−1} + (1 − β₂) gₜ²
m̂ₜ = mₜ / (1 − β₁ᵗ)
v̂ₜ = vₜ / (1 − β₂ᵗ)
θₜ = θ_{t−1} − η ( m̂ₜ / (√v̂ₜ + ε) + λ_wd θ_{t−1} )

with  gₜ = ∂L_task/∂θ + λ_ewc · Fᵢ · (θᵢ − θ*ᵢ)
```

## 4.7 Curriculum Design

Table 4.2 — Curriculum topic ordering

| Weeks | Topic | Stratum |
|---|---|---|
| 1–2 | Arrays and strings | Foundation |
| 3–4 | Linked lists | Foundation |
| 5–6 | Stacks and queues | Foundation |
| 7–8 | Trees | Foundation |
| 9–10 | Graphs | Advanced |
| 11–12 | Sorting and searching | Advanced |
| 13–14 | Dynamic programming | Advanced |
| 15–16 | Review and consolidation | Advanced |

Three scheduling strategies are implemented. The progressive strategy introduces one new topic per epoch while revisiting all previous topics. The adaptive strategy allocates effort to the topics with the lowest evaluated scores and is the strategy used for the reported experiments. The random strategy serves as a control.

An anti-forgetting review mechanism supplements these: when the forgetting detector observes a degradation exceeding five percent on any previously trained topic, that topic is reinserted into the current epoch's training set irrespective of the active strategy.

## 4.8 Data Flow Through One Iteration

```
Topic = "Trees"
     │
     ▼
[Question Generator]  temperature 0.8, quality filter
     │
     ▼
Questions = [ "What is the time complexity of BST search?",
              "Implement inorder traversal recursively.",
              "Find the lowest common ancestor in a BST.", ... ]
     │
     ▼
[Answer Generator]  temperature 0.3, chat template
     │
     ▼
Answers = [ (q₁, "The time complexity is O(h) where h ..."),
            (q₂, "def inorder(root): ..."), ... ]
     │
     ▼
[Evaluator]  rubric scoring + sandboxed execution
     │
     ▼
Evaluations = [ (q₁, a₁, 0.72, "good concept coverage"),
                (q₂, a₂, 0.55, "complexity analysis absent"), ... ]
     │
     ▼
[Parameter Updater]  weighted loss + EWC penalty
     │
     ▼
B ← B − η ∇_B (L_task + L_ewc)
A ← A − η ∇_A (L_task + L_ewc)
     │
     ▼
Fisher updated, checkpoint written, forgetting check performed
```

Figure 4.2 — Data flow through one SEAL iteration

## 4.9 Evaluation Methodology

Five metrics are defined. DSA accuracy is the proportion of held-out answers scoring above threshold. Quality score is the mean rubric score in [0, 1]. Forgetting rate is the maximum over topics of the decline from the best historical score. Training efficiency is GPU-hours per percentage point of improvement. Inference speed is measured as generation time per answer.

The experimental protocol evaluates the frozen base model, executes the SEAL loop for the configured number of epochs with a checkpoint written after each, and evaluates the adapted model on the identical held-out set. Because 4-bit generation is not bit-reproducible across sessions, both model states are evaluated within a single session against an identical question ordering, permitting a valid paired comparison at question level. This design decision was made after an early run produced base-model figures differing in the third decimal place between sessions, and it is the reason the paired analysis in Section 8.7 can be reported with confidence.

---

# CHAPTER 5 — FEASIBILITY AND PRACTICABILITY

## 5.1 Technical Feasibility

The binding technical constraint is GPU memory. An NVIDIA T4 provides 15 GB of VRAM, against which the following budget was computed and subsequently verified during execution.

Table 5.1 — Memory budget on the T4 GPU

| Component | Estimated requirement |
|---|---|
| Base model, 4-bit quantized | ~0.8 GB |
| LoRA adapter parameters | ~0.01 GB |
| AdamW optimizer state | ~0.04 GB |
| Gradient buffers | ~0.04 GB |
| Key–value cache during generation | 1–2 GB |
| Tokenised batch | ~0.1 GB |
| EWC Fisher diagonal and θ* | ~0.02 GB |
| PyTorch and CUDA context overhead | 1–2 GB |
| **Total** | **~3–5 GB** |
| **Available** | **15 GB** |
| **Headroom** | **~10 GB** |

The margin is comfortable, and in practice the constraint that bound first was wall-clock time rather than memory: generation of an answer averages approximately fifteen seconds, so a seventy-question evaluation of two model states consumes roughly thirty-five minutes, and a full training epoch across seven topics at twenty questions per topic consumes considerably more.

One technical obstacle was encountered and resolved during implementation. Mixed-precision training was initially enabled, but the gradient scaler interacts incorrectly with 4-bit quantized weights, causing the scaler to skip optimizer updates and leaving the adapter effectively untrained. Mixed precision is disabled in the final configuration, at some cost in throughput. This is documented in Section 7.8.

## 5.2 Economic Feasibility

The monetary cost of the project is nil. The base model is released under Apache 2.0, all libraries are open source, computation is provided by the Google Colaboratory free tier, and checkpoint storage uses the free allocation of Google Drive. Total checkpoint storage across all epochs is under 50 MB, since only adapter weights and metadata are persisted.

For comparison, full fine-tuning of a 1.5B model on rented cloud infrastructure would require an accelerator of substantially greater capacity for several hours, and the SEAL loop's repeated generation phases would multiply that requirement. The economic argument for parameter-efficient adaptation is therefore not incidental to this work; it is the reason the work exists in this form.

## 5.3 Operational Feasibility

Table 5.2 — Colab limitations and adopted mitigations

| Limitation | Mitigation adopted |
|---|---|
| Twelve-hour session ceiling | Checkpoint written to Google Drive after every epoch |
| Approximately 12.7 GB system RAM | 4-bit quantization with batch size 2 |
| Unpredictable session termination | Automatic resume from the most recent checkpoint |
| Constrained ephemeral disk | Only adapter weights persisted, base model refetched |
| Non-deterministic GPU allocation | All comparative evaluation performed within one session |

The resume path required particular attention. An early defect caused the experiment summary to fail when a run resumed past the final configured epoch, producing an empty summary rather than an informative message; this was corrected. A related configuration defect, in which the epoch count had been left at one from an earlier debugging session, caused a resumed run to terminate immediately without training; this too was corrected, and the incident is the reason the results in Chapter 8 reflect three epochs rather than five.

## 5.4 Schedule Feasibility

The work was organised across two phases. Phase I established the problem, surveyed the literature, and produced a prototype of the loop. Phase II completed the implementation, built the evaluation harness, executed training, and produced this dissertation. The principal schedule risk realised in practice was the wall-clock cost of evaluation, which is dominated by autoregressive generation and cannot be reduced without either shortening responses or reducing the question count, both of which would weaken the study.

## 5.5 Assessment

The project is feasible on all four dimensions, and the artefact exists and runs. It should be recorded, however, that feasibility of execution is a weaker property than sufficiency of resources: the system runs within the free tier, but Chapter 11 argues that several of the limitations in the results follow directly from the scale that the free tier permits.

---

# CHAPTER 6 — ANALYSIS AND DESIGN

## 6.1 Data Flow Diagrams

### 6.1.1 Level 0 — Context Diagram

```
                    ┌────────────────────────┐
   Configuration    │                        │   Trained adapter
   ────────────────▶│                        │──────────────────▶
                    │                        │
   Seed topics      │      SEAL-DSA          │   Evaluation report
   ────────────────▶│       System           │──────────────────▶
                    │                        │
   Evaluation set   │                        │   Checkpoints
   ────────────────▶│                        │──────────────────▶
                    └────────────────────────┘
```

Figure 6.1 — Level 0 data flow diagram

### 6.1.2 Level 1 — Decomposition

```
                        ┌──────────────────┐
   Config ─────────────▶│ 1.0 Curriculum   │
                        │     Scheduler    │
                        └────────┬─────────┘
                                 │ topic
                                 ▼
                        ┌──────────────────┐
                        │ 2.0 Question     │
                        │     Generator    │
                        └────────┬─────────┘
                                 │ questions
                                 ▼
                        ┌──────────────────┐
                        │ 3.0 Answer       │
                        │     Generator    │
                        └────────┬─────────┘
                                 │ (q, a) pairs
                                 ▼
                        ┌──────────────────┐      ┌─────────────┐
                        │ 4.0 Evaluator    │◀────▶│ D1: Test    │
                        │                  │      │     cases   │
                        └────────┬─────────┘      └─────────────┘
                                 │ scored pairs
                                 ▼
                        ┌──────────────────┐      ┌─────────────┐
                        │ 5.0 Parameter    │◀────▶│ D2: Fisher  │
                        │     Updater      │      │     matrix  │
                        └────────┬─────────┘      └─────────────┘
                                 │ updated adapter
                                 ▼
                        ┌──────────────────┐      ┌─────────────┐
                        │ 6.0 Checkpoint   │─────▶│ D3: Drive   │
                        │     Manager      │      │  checkpoints│
                        └────────┬─────────┘      └─────────────┘
                                 │ scores
                                 ▼
                        ┌──────────────────┐
                        │ 7.0 Forgetting   │───▶ review flags to 1.0
                        │     Detector     │
                        └──────────────────┘
```

Figure 6.2 — Level 1 data flow diagram

## 6.2 UML Models

### 6.2.1 Use Case Diagram

```
                    SEAL-DSA System
   ┌────────────────────────────────────────────────┐
   │                                                 │
   │   ( Configure training run )                    │
   │   ( Initiate training )                         │
   │   ( Resume from checkpoint )                    │
   │   ( Evaluate model )                            │
   │   ( Compare checkpoints )                       │
   │   ( Inspect forgetting report )                 │
   │                                                 │
   │   ( Generate questions )      «internal»        │
   │   ( Generate answers )        «internal»        │
   │   ( Score answers )           «internal»        │
   │   ( Update parameters )       «internal»        │
   │   ( Persist checkpoint )      «internal»        │
   │                                                 │
   └────────────────────────────────────────────────┘
        ▲                                    ▲
        │                                    │
   ┌────┴─────┐                     ┌────────┴───────┐
   │ Researcher│                     │ Google Drive   │
   │  (human)  │                     │  (external)    │
   └───────────┘                     └────────────────┘
```

Figure 6.3 — Use case diagram

### 6.2.2 Sequence Diagram

```
Researcher  Main   Scheduler  QGen   AGen   Eval   Updater  EWC   Ckpt
    │        │         │        │      │      │       │      │      │
    │ start  │         │        │      │      │       │      │      │
    ├───────▶│         │        │      │      │       │      │      │
    │        │ topic   │        │      │      │       │      │      │
    │        ├────────▶│        │      │      │       │      │      │
    │        │◀────────┤        │      │      │       │      │      │
    │        │ generate(topic)  │      │      │       │      │      │
    │        ├─────────────────▶│      │      │       │      │      │
    │        │◀─ questions ─────┤      │      │       │      │      │
    │        │ answer(questions)│      │      │       │      │      │
    │        ├────────────────────────▶│      │       │      │      │
    │        │◀─ answers ───────────────┤      │       │      │      │
    │        │ evaluate(pairs)  │      │      │       │      │      │
    │        ├──────────────────────────────▶│       │      │      │
    │        │◀─ scores ─────────────────────┤       │      │      │
    │        │ update(scores)   │      │      │       │      │      │
    │        ├──────────────────────────────────────▶│      │      │
    │        │                  │      │      │  ewc_loss()  │      │
    │        │                  │      │      │       ├─────▶│      │
    │        │                  │      │      │       │◀─────┤      │
    │        │◀─ updated ────────────────────────────┤      │      │
    │        │ update_fisher()  │      │      │       │      │      │
    │        ├─────────────────────────────────────────────▶│      │
    │        │ save(epoch)      │      │      │       │      │      │
    │        ├────────────────────────────────────────────────────▶│
    │◀─ summary ────────────────┤      │      │       │      │      │
```

Figure 6.4 — Sequence diagram of one training iteration

### 6.2.3 Class Diagram

```
┌────────────────────────┐        ┌────────────────────────┐
│ SEALTrainingLoop       │        │ CurriculumScheduler    │
├────────────────────────┤        ├────────────────────────┤
│ - model                │───────▶│ - strategy: str        │
│ - tokenizer            │        │ - topic_scores: dict   │
│ - config: Config       │        ├────────────────────────┤
├────────────────────────┤        │ + get_topics(epoch)    │
│ + run(num_epochs)      │        │ + update_scores(...)   │
│ + run_epoch(epoch)     │        └────────────────────────┘
│ + resume(checkpoint)   │
└───────┬────────────────┘        ┌────────────────────────┐
        │                         │ QuestionGenerator      │
        │────────────────────────▶├────────────────────────┤
        │                         │ + generate(topic, n)   │
        │                         │ + filter_quality(qs)   │
        │                         └────────────────────────┘
        │                         ┌────────────────────────┐
        │────────────────────────▶│ AnswerGenerator        │
        │                         ├────────────────────────┤
        │                         │ + answer(questions)    │
        │                         │ + estimate_confidence()│
        │                         └────────────────────────┘
        │                         ┌────────────────────────┐
        │────────────────────────▶│ DSAEvaluator           │
        │                         ├────────────────────────┤
        │                         │ + evaluate(pairs)      │
        │                         │ + run_test_cases(...)  │
        │                         └────────────────────────┘
        │                         ┌────────────────────────┐
        │────────────────────────▶│ ParameterUpdater       │
        │                         ├────────────────────────┤
        │                         │ - optimizer: AdamW     │
        │                         │ + update(evals, ewc)   │
        │                         └───────┬────────────────┘
        │                                 │
        │                                 ▼
        │                         ┌────────────────────────┐
        │                         │ EWC                    │
        │                         ├────────────────────────┤
        │                         │ - fisher: dict         │
        │                         │ - theta_star: dict     │
        │                         ├────────────────────────┤
        │                         │ + compute_loss(model)  │
        │                         │ + update_fisher(model) │
        │                         └────────────────────────┘
        │                         ┌────────────────────────┐
        └────────────────────────▶│ CheckpointManager      │
                                  ├────────────────────────┤
                                  │ + save(model, epoch)   │
                                  │ + load(path)           │
                                  │ + sync_to_drive()      │
                                  └────────────────────────┘
```

Figure 6.5 — Class diagram of core modules

## 6.3 Data Model

The system uses no relational database. Persistence is file-based, using JSON for structured records and the safetensors format for adapter weights. This choice reflects the deployment environment: a Colab session has no database server, and introducing one would add an operational dependency without benefit, since the data volume is small and access patterns are strictly sequential.

### 6.3.1 Seed Question Record

```json
{
  "topic": "arrays_strings",
  "difficulty": "medium",
  "type": "coding",
  "question": "Given an array of integers, find the maximum sum of any contiguous subarray.",
  "function_name": "max_subarray_sum",
  "test_cases": [
    { "input": "[[-2,1,-3,4,-1,2,1,-5,4]]", "expected": 6 },
    { "input": "[[1]]", "expected": 1 }
  ]
}
```

### 6.3.2 Evaluation Set Structure

```json
{
  "topics": {
    "arrays_strings":       [ { ...question record... }, ... ],
    "linked_lists":         [ ... ],
    "stacks_queues":        [ ... ],
    "trees":                [ ... ],
    "graphs":               [ ... ],
    "sorting_searching":    [ ... ],
    "dynamic_programming":  [ ... ]
  }
}
```

Ten question records occupy each topic, giving seventy in total.

### 6.3.3 Per-Question Result Record

```json
{
  "model": "SEAL",
  "topic": "arrays_strings",
  "qid": "arrays_strings_3",
  "type": "coding",
  "answer_words": 164,
  "gen_time_s": 14.7,
  "length": 1.0,
  "keyword": 0.94,
  "code": 1.0,
  "test_pass_rate": 0.1667,
  "overall": 0.7135
}
```

### 6.3.4 Checkpoint Contents

Table 6.1 — Checkpoint directory contents

| Artefact | Format | Approximate size |
|---|---|---|
| Adapter weights | safetensors | 7–15 MB |
| Adapter configuration | JSON | < 1 KB |
| Optimizer state | PyTorch | ~15 MB |
| Training metadata | JSON | < 1 KB |
| Metrics history | JSON | < 10 KB |
| EWC Fisher diagonal and θ* | PyTorch | ~20 MB |

Base model weights and the tokenizer are deliberately not persisted, being refetched from the model hub, which reduces each checkpoint by approximately 3 GB.

## 6.4 Design Rationale for Absence of a Database

Three considerations dictated file-based persistence. The Colab environment provides no persistent database service and the session filesystem is destroyed on termination, so any database would need to be reconstituted from files at each resume, which is the file-based design with additional steps. The data volume is trivially small. And the access pattern is a sequential write of results followed by a single aggregation pass, for which the relational model offers no advantage.

---

# CHAPTER 7 — SYSTEM IMPLEMENTATION

## 7.1 Technology Stack

Table 7.1 — Technology stack

| Component | Technology | Version | Role |
|---|---|---|---|
| Language | Python | 3.10+ | Core implementation |
| Numerical framework | PyTorch | 2.1+ | Training and inference |
| Model library | HuggingFace Transformers | 4.36+ | Model and tokenizer loading |
| Adaptation | PEFT | 0.7+ | Low-rank adapter management |
| Quantization | bitsandbytes | 0.41+ | 4-bit NF4 weights |
| Configuration | OmegaConf / PyYAML | 2.3+ / 6.0+ | Layered configuration |
| Metrics | scikit-learn, NumPy, SciPy | current | Aggregation and statistics |
| Visualisation | Matplotlib, Seaborn | 3.7+ / 0.12+ | Result figures |
| Infrastructure | Google Colaboratory | Free tier | T4 GPU |
| Persistence | Google Drive | Free tier | Checkpoint storage |

## 7.2 Selection of the Base Model

Table 7.2 — Candidate base models considered

| Model | Parameters | VRAM at 4-bit | Instruction quality | Suitability |
|---|---|---|---|---|
| TinyLlama-1.1B | 1.1B | ~2 GB | Basic | Comfortable but weak |
| Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | Good | Selected |
| Microsoft Phi-2 | 2.7B | ~4 GB | Very good | Feasible |
| Qwen2.5-3B-Instruct | 3.0B | ~4.5 GB | Very good | Memory-tight |

Qwen2.5-1.5B-Instruct was selected because it is already instruction-tuned and therefore follows the generation prompts without further alignment work, it fits the T4 with substantial headroom for the key–value cache during generation, it offers the best quality-to-size ratio among the candidates for instruction following, and it is released under Apache 2.0.

The choice is revisited critically in Section 11.1. A larger model would generate better questions and evaluate more reliably, and it is not established that the effect sizes reported in Chapter 8 would remain unchanged at greater scale.

## 7.3 Adapter Configuration

```python
lora_config = LoraConfig(
    r=4,                     # rank
    lora_alpha=8,            # scaling, alpha/r = 2
    lora_dropout=0.1,        # higher dropout, small data regime
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

The parameter budget for Qwen2.5-1.5B, with hidden size 1536 and 28 layers, follows:

```
Parameters per adapted module = 2 x 1536 x 4          =    12,288
Adapted modules per layer      = 4 (q, k, v, o)
Parameters per layer           = 4 x 12,288           =    49,152
Total adapter parameters       = 28 x 49,152          = 1,376,256

Total model parameters                                 ~ 1,540,000,000
Adapter proportion             = 1,376,256 / 1.54e9   ~ 0.089%
```

Rank 4 rather than 8 was adopted in the Colab-optimised configuration to reduce training time. The trade-off is real: a lower rank constrains the expressiveness of the update, and Section 11.7 identifies rank as a parameter deserving systematic study.

## 7.4 Training Configuration

Table 7.3 — Training hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Learning rate | 2 × 10⁻⁴ | Stable across a long schedule |
| Batch size | 2 | Bounded by Colab system RAM |
| Gradient accumulation | 8 | Yields an effective batch of 16 |
| Epochs | 5 | Full schedule; three completed at time of writing |
| Questions per topic | 20 | Balances diversity against wall-clock cost |
| Maximum sequence length | 1024 | Accommodates full coding solutions |
| Weight decay | 0.01 | Standard AdamW regularisation |
| Gradient clipping | 1.0 | Guards against loss spikes |
| Scheduler | Cosine | Smooth decay over the schedule |
| EWC λ | 0.3 | Moderate retention pressure |
| Fisher sample size | 100 | Bounded by session time |
| Mixed precision | Disabled | Conflicts with 4-bit quantization |

## 7.5 Colab Environment Setup

```python
# Cell 1 — dependencies
!pip install -q torch transformers peft accelerate bitsandbytes
!pip install -q datasets evaluate rouge-score tabulate

# Cell 2 — persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — source
!git clone <repository-url> /content/dsaseal
%cd /content/dsaseal/SEAL-DSA

# Cell 4 — verify accelerator
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 5 — restore checkpoints from Drive if present
!cp -r /content/drive/MyDrive/seal_checkpoints/* checkpoints/ 2>/dev/null

# Cell 6 — train
!python -m seal_dsa.main --config configs/colab_optimized.yaml
```

## 7.6 Principal Algorithms

### 7.6.1 Training Loop

```
Algorithm 7.1: SEAL Training Loop
─────────────────────────────────
Input : base model M, curriculum C, configuration K
Output: adapted model M*

 1  M  <- LoadModel(K.model_name) with LoRA(r=4, alpha=8)
 2  E  <- InitialiseEWC()
 3
 4  FOR epoch = 1 TO K.num_epochs DO
 5      topics <- C.get_topics(epoch)
 6      FOR EACH topic T IN topics DO
 7          Q <- GenerateQuestions(M, T, n=20, temperature=0.8)
 8          Q <- FilterByQuality(Q)
 9          A <- []
10          FOR EACH q IN Q DO
11              a <- GenerateAnswer(M, q, temperature=0.3)
12              A.append((q, a))
13          END FOR
14          S <- []
15          FOR EACH (q, a) IN A DO
16              score <- Evaluate(a, rubric, test_cases)
17              S.append((q, a, score))
18          END FOR
19          L_task  <- WeightedLanguageModellingLoss(M, S)
20          L_ewc   <- E.compute_loss(M)
21          L_total <- L_task + lambda * L_ewc
22          theta_LoRA <- theta_LoRA - eta * grad(L_total)
23      END FOR
24      E.update_fisher(M)
25      f <- CheckForgetting(M, previously_trained_topics)
26      IF f > 0.05 THEN C.add_to_review(affected_topics)
27      SaveCheckpoint(M, epoch)  ; sync to Drive
28  END FOR
29  RETURN M
```

### 7.6.2 Evaluation Protocol

```
Algorithm 7.2: Evaluation Protocol
──────────────────────────────────
Input : model M, held-out set S (70 questions, 7 topics)
Output: per-question, per-topic, and aggregate metrics

 1  results <- {}
 2  FOR EACH topic T IN S.topics DO
 3      scores_T <- []
 4      FOR EACH question q IN S[T] DO
 5          a     <- M.generate(q, greedy=True, max_new_tokens=256)
 6          comp  <- ScoreComponents(a, q)         ; length, keyword, code
 7          IF q has function_name AND test_cases THEN
 8              comp.test_pass <- ExecuteTestCases(a, q)
 9          END IF
10          scores_T.append(Combine(comp))
11      END FOR
12      results[T] <- Aggregate(scores_T)
13  END FOR
14  results.overall <- Aggregate(all scores)
15  RETURN results
```

### 7.6.3 Sandboxed Execution of Generated Code

```
Algorithm 7.3: Test Case Execution
──────────────────────────────────
Input : answer text a, question record q
Output: (passed, total)

 1  code <- ExtractCodeBlock(a)        ; ```python fence, bare fence,
 2                                     ; or "def <function_name>" onward
 3  IF code is empty THEN RETURN (0, 0)
 4  namespace <- {}
 5  TRY
 6      exec(code, namespace)
 7      f <- namespace[q.function_name]
 8      IF f is undefined THEN RETURN (0, 0)
 9      passed <- 0
10      FOR EACH tc IN q.test_cases DO
11          TRY
12              args <- parse(tc.input)
13              IF result of f(*args) equals tc.expected THEN
14                  passed <- passed + 1
15              END IF
16          CATCH  ; individual test failure is not fatal
17          END TRY
18      END FOR
19      RETURN (passed, |q.test_cases|)
20  CATCH                              ; code failed to compile
21      RETURN (0, 0)
22  END TRY
```

The nested exception handling is deliberate. Generated code fails in many ways — syntax errors, undefined names, infinite recursion on an edge case, returning a list where a tuple is expected — and every one of these must degrade the score rather than terminate a training run that may already have consumed hours of session time.

## 7.7 Checkpointing and Recovery

Persisted after each epoch: adapter weights, adapter configuration, optimizer state, training metadata including the epoch index, metrics history, and the EWC Fisher diagonal with the reference parameters θ*. Not persisted: base weights and tokenizer, both refetched from the model hub.

Recovery mounts Drive, loads the base model, applies the most recent adapter, restores optimizer and EWC state, and resumes from the recorded epoch index. Two defects in this path were identified and corrected during development, as described in Section 5.3.

## 7.8 Implementation Difficulties Encountered

**Mixed precision against 4-bit weights.** With mixed precision enabled, the gradient scaler repeatedly detected non-finite values and skipped optimizer steps, so the adapter received almost no updates while training appeared superficially to proceed. The symptom was a loss curve that moved plausibly alongside an adapter whose weights were nearly unchanged. Disabling mixed precision resolved it.

**Migration to chat templates.** Answer generation initially used a hand-constructed prompt string. Migrating to the tokenizer's chat template improved instruction adherence, since the model was fine-tuned against that format, and simultaneously permitted confidence to be estimated from generation scores rather than by a redundant second forward pass.

**Cross-session non-determinism.** Base-model scores differed in the third decimal place between sessions, traced to non-determinism in 4-bit generation and to rounding introduced when merging an adapter into quantized weights. The evaluation protocol was consequently revised to require both model states to be evaluated within a single session, which is what makes the paired analysis of Section 8.7 sound.

**Configuration drift.** The epoch count in the working configuration had been reduced to one during debugging and was not restored, so a resumed run terminated without training. The configuration was corrected, but the incident cost a session and is the proximate reason the results in Chapter 8 reflect three epochs rather than five.

---

# CHAPTER 8 — EVALUATION OF RESULTS

## 8.1 Experimental Setup

Evaluation used a held-out set of seventy questions distributed uniformly across seven topics, ten per topic: arrays and strings, linked lists, stacks and queues, trees, graphs, sorting and searching, and dynamic programming. The base model is Qwen2.5-1.5B-Instruct in 4-bit NF4 quantization with double quantization and bfloat16 compute dtype. Generation was greedy with a repetition penalty of 1.1 and a limit of 256 new tokens. All runs used a single NVIDIA T4 GPU on Google Colaboratory.

Two model states were compared: the frozen base model, denoted *before self-adaptation*, and the model after self-adaptive training with the adapter merged into the base weights, denoted *after self-adaptation*. Endpoint comparison isolates the effect of the loop as a whole rather than of any single pass over the curriculum. Both states were evaluated within one session against an identical question ordering, for the reasons given in Section 7.8.

The results reported here reflect three completed epochs of a five-epoch schedule. Section 11.5 states the consequence.

## 8.2 Metrics

Four components are scored per answer. *Answer completeness* penalises empty or truncated output. *Keyword coverage* measures the presence of topic-appropriate technical vocabulary drawn from a per-topic lexicon augmented with salient terms from the question. *Code presence* checks, for coding questions, whether an executable Python construct appears. *Test-case pass rate* extracts the generated function, executes it in an isolated namespace, and reports the fraction of held-out test cases satisfied. The four combine into a weighted composite in [0, 1].

Only the fourth verifies functional correctness. Section 8.4 argues that it should be read as the primary result.

## 8.3 Aggregate Performance

Table 8.1 — Aggregate performance before and after self-adaptive training (n = 70)

| Metric | Before | After | Change |
|---|---|---|---|
| Overall composite score | 0.8634 | 0.8691 | +0.0057 (+0.66%) |
| Keyword coverage | 0.9511 | 0.9491 | −0.0020 |
| Answer completeness | 1.0000 | 1.0000 | 0.0000 |
| Code presence | 1.0000 | 1.0000 | 0.0000 |
| Test-case pass rate | 0.1111 | 0.1667 | +0.0556 (+50.0%) |
| Mean answer length (words) | 165.67 | 163.04 | −2.63 |
| Mean generation time (s) | 15.09 | 14.70 | −0.39 |

The composite improved by 0.66%. Taken alone this is a modest movement, and the following section explains why the composite understates what changed.

## 8.4 Ceiling Effects in the Composite Metric

Three of the four components exhibit ceiling effects on this evaluation set. Answer completeness is 1.0000 for both states, indicating that the base model already produces responses of adequate length for every question in the set. Code presence is likewise 1.0000 for both, indicating that the base model already emits code constructs for every coding question. Keyword coverage begins at 0.9511, leaving under five percentage points of headroom, and moves marginally negative.

Since these three enter the weighted composite and none can improve, the composite is structurally constrained toward a near-zero delta regardless of what self-adaptation achieves. The composite therefore measures the wrong property on this evaluation set: it confirms that the base model produces well-formed DSA answers, which was never in question, while diluting the one signal indicating whether those answers are correct.

Test-case pass rate is the sole component with meaningful headroom, beginning at 0.1111. On this metric the model improved to 0.1667, a relative gain of 50.0%. This measures whether generated code compiles and returns expected outputs on unseen inputs, and it is therefore the metric most directly reflecting the objective of the SEAL loop.

Two qualifications are necessary. In absolute terms the change corresponds to a small number of additional questions passing their test cases, and a seventy-question set is not large enough to establish statistical significance for a difference of this magnitude. The finding is accordingly reported as directional, not as a significance claim. Furthermore, the decision to treat pass rate as primary was taken on the argument set out above, concerning what each component is capable of measuring, and not on the basis of which component produced the more favourable number; the ceiling in the other three components is a property of the base model and the question set that would have been present whatever the training outcome.

## 8.5 Topic-wise Analysis

Table 8.2 — Topic-wise scores before and after self-adaptation

| Topic | N | Before | After | Δ | Pass rate before | Pass rate after |
|---|---|---|---|---|---|---|
| Arrays and strings | 10 | 0.6635 | 0.7135 | +0.0500 | 0.0000 | 0.1667 |
| Linked lists | 10 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A |
| Stacks and queues | 10 | 0.9500 | 0.9500 | 0.0000 | 0.5000 | 0.5000 |
| Trees | 10 | 0.9820 | 0.9661 | −0.0159 | N/A | N/A |
| Graphs | 10 | 0.9286 | 0.9286 | 0.0000 | 0.0000 | 0.0000 |
| Sorting and searching | 10 | 0.7950 | 0.7950 | 0.0000 | 0.2500 | 0.2500 |
| Dynamic programming | 10 | 0.7247 | 0.7306 | +0.0059 | 0.0000 | 0.0000 |

Figure 8.1 — Topic-wise performance before and after self-adaptation *(insert `topic_comparison.png`)*

Figure 8.2 — Evaluation metric breakdown *(insert `metric_breakdown.png`)*

Three observations follow.

Gains concentrate in the weakest topics. Arrays and strings, weakest before adaptation at 0.6635, records the largest improvement at +0.0500, with its pass rate rising from 0.0000 to 0.1667. Dynamic programming, second weakest at 0.7247, records the second largest gain. This is the behaviour the adaptive curriculum scheduler was designed to produce, and it constitutes direct evidence bearing on Objective 4 of Chapter 2.

Topics beginning at or near the ceiling do not move. Linked lists remains at 1.0000, and stacks and queues, graphs, and sorting and searching are unchanged to four decimal places, consistent with the analysis of Section 8.4 rather than with a failure of training.

Graphs and dynamic programming retain a pass rate of 0.0000 in both states. These are the two topics demanding the longest multi-step implementations, and neither state produces executable solutions for them. Section 11.6 discusses this as a limitation of the approach rather than of its configuration.

## 8.6 Learning Trajectory

A partial trajectory across successive checkpoints is available for the topics evaluated before an earlier comparison run was interrupted by session termination.

Table 8.3 — Partial topic trajectory across checkpoints

| Topic | Base | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|---|
| Arrays and strings | 0.657 | 0.663 | 0.675 | 0.714 |
| Dynamic programming | 0.733 | 0.783 | — | 0.731 |
| Sorting and searching | 0.821 | 0.811 | — | 0.795 |
| Linked lists | 1.000 | 1.000 | 1.000 | 1.000 |
| Stacks and queues | 0.946 | 0.950 | 0.946 | 0.950 |

Arrays and strings rises monotonically from 0.657 to 0.714, with the largest single increment between the second and third checkpoints. This suggests that training had not converged at the point of measurement, and that further passes over the curriculum would be expected to yield additional improvement on this topic.

The dynamic programming column is non-monotonic. On a ten-question topic this is more plausibly sampling variation than genuine regression, and it is reported without a causal claim. Sorting and searching declines slightly across the same interval, and the same caution applies.

Base-model figures in Table 8.3 differ marginally from Table 8.2 because the two runs executed in separate sessions and 4-bit generation is not bit-reproducible across sessions. Figures from the two runs are consequently not combined within any single table, and no comparison in this dissertation depends on doing so.

## 8.7 Paired Analysis and Capability Retention

Because both states were evaluated on identical questions within one session, a paired comparison at question level is valid.

Table 8.4 — Paired question-level outcomes (n = 70)

| Outcome | Count | Share (%) |
|---|---|---|
| Improved | 3 | 4.3 |
| Unchanged | 64 | 91.4 |
| Degraded | 3 | 4.3 |

This distribution is the principal evidence regarding catastrophic forgetting. Three questions of seventy, 4.3% of the set, scored lower after adaptation, and the largest topic-level regression is −0.0159 on trees. No topic collapsed, and no previously passing functional solution was lost. Objective 3 of Chapter 2, requiring degradation below five percent, is therefore met.

The retention profile is the intended effect of the EWC penalty combined with the low-rank constraint. Restricting adaptation to a rank-4 adapter and penalising movement in parameters with high Fisher values confines the effect of self-generated training to a small subset of behaviours.

The cost of that conservatism is visible in the same table. A 91.4% unchanged rate indicates that the update is narrow as well as safe, and it is not possible from these data to separate the contribution of EWC from that of the low-rank constraint, since both were active throughout and no ablation was performed. Section 11.7 records this as a limitation and Section 12.5 proposes the ablation.

## 8.8 Efficiency

Table 8.5 — Efficiency observations

| Measure | Before | After | Change |
|---|---|---|---|
| Mean generation time per answer (s) | 15.09 | 14.70 | −0.39 |
| Mean answer length (words) | 165.67 | 163.04 | −2.63 |
| Checkpoint size (adapter only) | — | ~15 MB | — |
| Monetary cost of full training run | — | Nil | — |

Neither timing difference is large enough to attribute to the adapter with confidence; both are consistent with ordinary variation in greedy decoding length across seventy questions. The relevant conclusion is negative rather than positive: merging the adapter imposes no measurable inference-time penalty, supporting the practicability claim of Chapter 5.

## 8.9 Summary of Findings

The evaluation supports three conclusions.

First, self-adaptive training produced a 50.0% relative improvement in functional correctness, from a pass rate of 0.1111 to 0.1667, this being the only component with meaningful headroom. The absolute magnitude is small and significance is not established on a set of this size.

Second, improvement concentrates in the weakest topics, arrays and strings and dynamic programming, consistent with the adaptive scheduler's design and satisfying Objective 4.

Third, retention is high, with 95.7% of questions unchanged or improved and a maximum topic-level regression of 0.0159, satisfying Objective 3 and indicating that EWC-regularised low-rank adaptation did not induce catastrophic forgetting.

The composite improvement of 0.66% is reported for completeness and should not be read as the effect size, since three of its four components are saturated on this evaluation set. Against the Phase I target of a fifteen to twenty-five percent improvement stated in Section 2.2, the result on the composite metric plainly falls short; the result on functional correctness exceeds it in relative terms while remaining small in absolute terms. Both statements are true and the dissertation declines to present only the more favourable one. Redesign of the evaluation set to remove ceiling effects is identified in Section 12.1 as the highest-priority item of future work.

---

# CHAPTER 9 — SNAPSHOTS

This chapter presents captured output from the execution of the system. Terminal transcripts are reproduced rather than reconstructed.

## 9.1 Training Loop Execution

```
======================================================================
  SEAL-DSA  ·  Self-Adaptive Training
======================================================================
[STEP 1/4] Loading configuration: configs/colab_optimized.yaml
[MODEL] Loading base model: Qwen/Qwen2.5-1.5B-Instruct
[INFO] Using 4-bit quantization (QLoRA)
Loading weights: 100%|██████████| 338/338 [00:12<00:00, 27.55it/s]
[LORA] trainable params: 1,376,256 || all params: 1,543,714,304 || trainable%: 0.0892
[CURRICULUM] Epoch 1 topics: arrays_strings, linked_lists
[QGEN] Generating 20 questions for topic: arrays_strings
[AGEN] Answering 20 questions ...
[EVAL] Mean rubric score: 0.68
[UPDATE] L_task = 1.842   L_ewc = 0.000   L_total = 1.842
[CKPT] Saved checkpoint_epoch_0 -> /content/drive/MyDrive/seal_checkpoints/
```

Figure 9.1 — Training loop console output

## 9.2 Checkpoint Persistence

```
/content/drive/MyDrive/seal_checkpoints/
├── checkpoint_epoch_0/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── ewc_state.pt
│   └── metadata.json
├── checkpoint_epoch_1/
└── checkpoint_epoch_2/
```

Figure 9.2 — Checkpoint directory on Google Drive

## 9.3 Evaluation Run

```
[INFO] Loaded 70 questions across 7 topics
[INFO] Found 3 checkpoint(s):
    • checkpoint_epoch_0
    • checkpoint_epoch_1
    • checkpoint_epoch_2
[INFO] Using final checkpoint: checkpoint_epoch_2
[MODEL] Loading base model: Qwen/Qwen2.5-1.5B-Instruct
[INFO] Using 4-bit quantization (QLoRA)
Base:arrays_strings: 100%|██████████| 10/10 [02:19<00:00, 13.95s/it]
Base:linked_lists:   100%|██████████| 10/10 [02:05<00:00, 12.51s/it]
...
[MODEL] Loading LoRA adapter from: checkpoint_epoch_2
SEAL:arrays_strings: 100%|██████████| 10/10 [02:10<00:00, 12.63s/it]
...
============================================================
Base : 0.8634
SEAL : 0.8691   (+0.66%)
improved 3 | unchanged 64 | degraded 3
Artifacts written to /content/dsaseal/SEAL-DSA/results_final
============================================================
```

Figure 9.3 — Evaluation run console output

## 9.4 Generated Result Artefacts

The evaluation run produces `final_report.json` containing per-question records for both states, `per_question.csv` and `per_topic.csv` for tabular analysis, `summary.md` containing the tables reproduced in Chapter 8, `topic_comparison.png` and `metric_breakdown.png` reproduced as Figures 8.1 and 8.2, and `samples.md` containing side-by-side before-and-after answers for the questions exhibiting the largest improvement.

---

# CHAPTER 10 — TOOLS AND UTILITIES

## 10.1 Hardware

The single computational resource is an NVIDIA Tesla T4 GPU allocated by the Google Colaboratory free tier, providing 15 GB VRAM and Turing-generation tensor cores. The host provides approximately 12.7 GB of system RAM and roughly 100 GB of ephemeral disk destroyed at session end. No local hardware was used for training; development and authorship were performed on a personal workstation.

## 10.2 Software Components

Table 10.1 — Software components and versions

| Component | Version | Function |
|---|---|---|
| Python | 3.12 | Runtime |
| PyTorch | 2.11 (CUDA 12.8) | Tensors, autograd, optimisation |
| Transformers | 5.12 | Model and tokenizer loading, generation |
| PEFT | 0.19 | LoRA adapter creation, merging, persistence |
| bitsandbytes | 0.49 | 4-bit NF4 quantization |
| Accelerate | 1.14 | Device placement |
| Datasets | 4.0 | Dataset handling |
| NumPy | 2.0 | Numerical support |
| SciPy | 1.16 | Statistical routines |
| scikit-learn | 1.6 | Metric computation |
| Matplotlib | 3.10 | Figure generation |
| Seaborn | 0.13 | Figure styling |
| pandas | 2.2 | Tabular aggregation |
| PyYAML / OmegaConf | 6.0 / 2.3 | Layered configuration |
| tqdm | 4.67 | Progress reporting |
| pytest | 8.4 | Unit testing |

## 10.3 Online Services

The HuggingFace Hub supplies the base model and tokenizer. Google Colaboratory supplies computation. Google Drive supplies checkpoint persistence across sessions. Git and GitHub provide version control. All services were used within their free allocations.

## 10.4 Repository Organisation

```
SEAL-DSA/
├── seal_dsa/
│   ├── main.py                     entry point
│   ├── config.py                   configuration handling
│   ├── curriculum/                 topic definitions and scheduler
│   ├── models/                     model loading and LoRA setup
│   ├── modules/                    question gen, answer gen, evaluator, updater
│   ├── training/                   SEAL loop, EWC, checkpointing
│   ├── evaluation/                 baseline, metrics, forgetting detector
│   └── utils/                      logging, Colab helpers
├── configs/
│   ├── default.yaml
│   └── colab_optimized.yaml
├── data/
│   ├── dsa_seed_questions.json
│   └── evaluation_sets/dsa_eval_set.json
├── tests/                          unit tests
├── docs/                           thesis chapters and appendices
├── notebooks/                      Colab-oriented scripts
├── compare_with_checkpoints.py     multi-checkpoint comparison
├── colab_final_report.py           base-versus-final report generator
└── requirements.txt
```

## 10.5 Testing

Unit tests cover configuration loading and override resolution, curriculum scheduling behaviour, evaluator rubric scoring, and sandboxed code execution including the failure paths where generated code does not compile or a test case raises. The execution tests were the most valuable during development, since that component must degrade gracefully rather than terminate a long-running session.

---

# CHAPTER 11 — LIMITATIONS AND DRAWBACKS

## 11.1 Scale of the Base Model

The system uses Qwen2.5-1.5B-Instruct, the largest model that trains reliably under 4-bit quantization within the memory and session limits of the free Colab tier. Self-adaptive training depends on the model's capacity to generate questions worth learning from and to assess its own answers usefully, and both capacities improve substantially with scale. The modest effect sizes of Chapter 8 are therefore not independent of this choice, and no claim is made that they transfer unchanged to larger models.

## 11.2 Heuristic Evaluation Without a Judge Model

Answers are scored by a heuristic combining length, keyword overlap, code presence, and test-case execution. Only the last verifies correctness. Keyword overlap in particular can reward an answer that deploys expected vocabulary while reasoning wrongly, and cannot distinguish a correct explanation from a fluent incorrect one. A stronger design would employ a separate larger judge model, or human annotation on a sampled subset. Neither was feasible within the computational budget.

## 11.3 Ceiling Effects in the Evaluation Set

As established in Section 8.4, three of four components are saturated before training begins. The instrument is therefore unable to register improvement on most of its own metrics, placing a structural upper bound on the measurable effect of any training procedure. This is a defect of the evaluation instrument rather than of the training method, and it is the single most consequential limitation of the present work.

## 11.4 Size of the Evaluation Set

Seventy questions across seven topics yields ten per topic. At this size a change affecting one or two questions produces visible movement in a topic mean, and the study is underpowered for significance testing at the observed effect magnitudes. Topic-level results should be read as indicative rather than conclusive.

## 11.5 Incomplete Training Run

The results of Chapter 8 reflect three epochs of a planned five-epoch schedule, the shortfall arising from the configuration defect described in Section 7.8. Table 8.3 indicates that arrays and strings was still improving at the point of measurement, so the reported figures are properly read as a lower bound on what the configured schedule would produce rather than as a converged result. *(Delete this section once the full run is complete.)*

## 11.6 Failure on Complex Code Generation

Graphs and dynamic programming record a pass rate of 0.0000 in both states. The system neither improved nor degraded on the two topics demanding the longest multi-step implementations. This exposes a structural property of self-adaptive training: the loop can only learn from answers the model is already capable of producing, and therefore cannot bootstrap a capability absent at initialisation. Where the base model produces no executable solution, self-generated training has no correct output to reinforce.

## 11.7 Conservatism of the Update and Absence of Ablation

The 91.4% unchanged rate demonstrates that the mechanisms delivering strong retention also restrict how much the model can change. The configuration adopted favours safety over plasticity. Moreover, EWC and the low-rank constraint were active simultaneously throughout, and no ablation was performed, so their individual contributions to the retention result cannot be separated from these data. A sweep of EWC strength against adapter rank was not conducted and would be required to locate a useful operating point.

## 11.8 Reproducibility Under Quantization

Generation under 4-bit quantization is not bit-reproducible across sessions, and merging an adapter into quantized weights introduces further rounding. Differences between runs of an identical configuration are therefore expected, as observed between Tables 8.2 and 8.3. All comparative claims in this dissertation derive from paired within-session evaluation, which controls the effect but does not eliminate it.

## 11.9 Absence of Human and Multi-Seed Validation

No human expert reviewed the generated answers, and no student used the system. The experiments were executed once per configuration rather than across multiple random seeds, so run-to-run variance is uncharacterised. The protocol of Section 4.9 anticipated three seeded trials with a paired significance test; the wall-clock cost of evaluation on the free tier made this impracticable within the project schedule. This is a genuine shortfall against the stated methodology and is recorded as such.

---

# CHAPTER 12 — FUTURE ENHANCEMENT

## 12.1 Redesign of the Evaluation Set

The highest-priority improvement is an evaluation set the base model does not already saturate: a higher proportion of coding questions carrying executable test cases, retirement of conceptual questions answered perfectly before training, and a larger count per topic to reduce sampling variance. Without this, no training improvement can be measured reliably regardless of its true magnitude.

## 12.2 Model-Based Evaluation

Replacing the keyword heuristic with a larger judge model scoring correctness, completeness, and reasoning quality would remove the principal weakness of Section 11.2. Validating the judge against human annotation on a sampled subset would permit correctness to be reported directly rather than approximated.

## 12.3 Completion and Extension of the Schedule

Completing the five-epoch schedule and extending it while monitoring for convergence would establish whether the upward trajectory on arrays and strings continues, plateaus, or reverses. A learning curve over a full schedule is a materially stronger result than a single endpoint comparison.

## 12.4 Scaling the Base Model

Repeating the experiment at seven billion parameters would test whether effect size grows with the capability of the self-generating and self-scoring components, which is the central open question raised by Chapter 8.

## 12.5 Ablation and Hyperparameter Study

Separate ablation of EWC and of the low-rank constraint would apportion the retention result of Section 8.7 between them. A sweep over EWC strength λ and adapter rank r would map the plasticity–retention frontier and determine whether higher gains are available at acceptable cost in retention.

## 12.6 Difficulty-Aware Curriculum

The scheduler currently targets weak topics. Extending it to target weak difficulty bands within a topic, and to escalate difficulty as competence is demonstrated, would provide a more informative signal, particularly for graphs and dynamic programming where no functional capability presently exists to reinforce.

## 12.7 Preference-Based Optimisation

The Direct Preference Optimization objective described in Section 4.5 is implemented but unevaluated. Comparing it against the supervised objective under otherwise identical conditions is a direct and inexpensive extension.

## 12.8 Multi-Seed Validation with Significance Testing

Executing three or more seeded trials per configuration and applying a paired significance test would supply the statistical foundation absent from the present results, addressing Section 11.9 and permitting the effect to be reported as significant or not on evidence rather than left undetermined.

## 12.9 Extension Beyond Python and Beyond DSA

The execution harness is Python-specific. Extending it to additional languages would broaden applicability, and applying the whole loop to an adjacent technical domain would test whether the architecture generalises or whether it depends on properties peculiar to algorithmic problems, notably the availability of cheap automatic verification through test cases.

---

# CHAPTER 13 — CONCLUSION

This project set out to determine whether a small language model can improve its own competence in Data Structures and Algorithms through an autonomous loop of self-generated questions, self-generated answers, self-evaluation, and self-directed parameter update, without catastrophically forgetting what it already knew, and within infrastructure available at no cost.

The system was built and it operates end to end. Question generation, answer generation, rubric and execution-based evaluation, curriculum scheduling, EWC-regularised low-rank parameter updating, and checkpointed resumable training function as an integrated pipeline on a single T4 GPU. Objectives 1, 2, and 5 of Chapter 2 are met without qualification: the loop is closed, evaluation includes genuine execution of generated code against held-out test cases, and the whole system runs at zero monetary cost with recovery across session termination.

On the evidence collected, the loop produces a measurable but small improvement. Functional correctness, measured as the fraction of held-out test cases passed by generated code, rose from 0.1111 to 0.1667, a relative improvement of 50.0%, while the aggregate composite moved by 0.66%. The disparity between these figures is explained by ceiling effects in three of the four scoring components, and the composite is accordingly not treated as the primary result. Gains concentrated in the weakest topics, which is the behaviour the curriculum scheduler was designed to produce, satisfying Objective 4.

The retention result is the clearest positive finding. Across seventy held-out questions, 95.7% scored unchanged or higher after self-adaptation, with a maximum topic-level regression of 0.0159. Low-rank adaptation combined with an Elastic Weight Consolidation penalty confined the effect of self-generated training without measurable damage to prior capability, satisfying Objective 3, and merging the adapter imposed no inference-time cost.

Against the Phase I target of a fifteen to twenty-five percent improvement in accuracy, the outcome falls short on the composite metric and exceeds it in relative terms on functional correctness while remaining small in absolute terms. Both statements are recorded. The dissertation does not select the more flattering of the two, and does not revise the original target retrospectively to accommodate the result.

The limitations are equally clear and are stated in Chapter 11. The base model is small, the evaluation is heuristic rather than judged, the evaluation set saturates before training begins, the question count is too small for significance testing, the reported schedule is incomplete, no ablation separates the contributions of the two retention mechanisms, no multi-seed validation was performed, and the system produces no executable solutions for graph or dynamic programming problems in either state.

The honest conclusion is therefore qualified. Self-adaptive training with explicit retention regularisation is workable at this scale and does not destroy prior capability, but the gains obtained here are small, concentrated, and measured with an instrument that limited what could be observed. The value of the work lies in a functioning implementation under severe resource constraints, in an execution-based evaluation harness for generated algorithmic code, in empirical evidence that EWC-regularised low-rank adaptation preserves prior capability under self-generated training, and in a precise account of where the approach and its evaluation fall short. Chapter 12 converts that account into a concrete programme of work, of which the redesign of the evaluation instrument is the first and most consequential step.

A final observation may be permitted. The most instructive outcome of this project was not the improvement it measured but the discovery that the measurement itself was the binding constraint. A system can only improve as far as its evaluator can perceive, and an evaluator saturated before training begins renders even a successful method indistinguishable from an unsuccessful one. That lesson generalises well beyond the present work.

---

# BIBLIOGRAPHY

[1] Massachusetts Institute of Technology Computer Science and Artificial Intelligence Laboratory, "SEAL: Self-Adapting Language Models," MIT CSAIL Technical Report, Cambridge, MA, 2025.

[2] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.

[3] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023, pp. 10088–10115.

[4] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, "Overcoming catastrophic forgetting in neural networks," *Proc. National Academy of Sciences*, vol. 114, no. 13, pp. 3521–3526, 2017.

[5] W. Yuan, R. Y. Pang, K. Cho, S. Sukhbaatar, J. Xu, and J. Weston, "Self-Rewarding Language Models," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024.

[6] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning," in *Proc. 26th Annual Int. Conf. Machine Learning (ICML)*, 2009, pp. 41–48.

[7] M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: The sequential learning problem," *Psychology of Learning and Motivation*, vol. 24, pp. 109–165, 1989.

[8] R. M. French, "Catastrophic forgetting in connectionist networks," *Trends in Cognitive Sciences*, vol. 3, no. 4, pp. 128–135, 1999.

[9] J. Schwarz, W. Czarnecki, J. Luketina, A. Grabska-Barwinska, Y. W. Teh, R. Pascanu, and R. Hadsell, "Progress & Compress: A scalable framework for continual learning," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018, pp. 4528–4537.

[10] F. Zenke, B. Poole, and S. Ganguli, "Continual learning through synaptic intelligence," in *Proc. Int. Conf. Machine Learning (ICML)*, 2017, pp. 3987–3995.

[11] A. Mallya and S. Lazebnik, "PackNet: Adding multiple tasks to a single network by iterative pruning," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7765–7773.

[12] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, "Progressive Neural Networks," *arXiv preprint arXiv:1606.04671*, 2016.

[13] D. Lopez-Paz and M. Ranzato, "Gradient Episodic Memory for continual learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 6467–6476.

[14] A. Yang et al., "Qwen2.5 Technical Report," *arXiv preprint arXiv:2412.15115*, 2024.

[15] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

[16] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, "Training language models to follow instructions with human feedback," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022, pp. 27730–27744.

[17] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, and J. Brew, "Transformers: State-of-the-Art Natural Language Processing," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations*, 2020, pp. 38–45.

[18] S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan, "PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods," GitHub repository, 2022. [Online]. Available: https://github.com/huggingface/peft

[19] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.

[20] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[21] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.

[22] M. Chen et al., "Evaluating Large Language Models Trained on Code," *arXiv preprint arXiv:2107.03374*, 2021.

[23] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, *Introduction to Algorithms*, 4th ed. Cambridge, MA: MIT Press, 2022.

[24] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, "Continual lifelong learning with neural networks: A review," *Neural Networks*, vol. 113, pp. 54–71, 2019.

[25] Z. Zheng, Y. Ning, Y. Zhang, J. Chen, and Y. Wang, "Towards an Understanding of Large Language Models in Software Engineering Tasks," *Empirical Software Engineering*, vol. 30, 2025.

---

# PROJECT SUMMARY

This project implements SEAL-DSA, a self-adapting language model that improves its own competence in Data Structures and Algorithms without externally supplied training data. The system builds on Qwen2.5-1.5B-Instruct under 4-bit QLoRA quantization and operates entirely within the limits of a single freely available GPU session.

The pipeline is a closed loop. A curriculum scheduler selects a topic according to measured weakness. The model generates practice questions for that topic, generates candidate answers, and evaluates those answers using a weighted rubric combined with sandboxed execution of generated code against held-out test cases. Scored question–answer pairs drive an update to a low-rank adapter comprising approximately 0.089% of the model's parameters, with an Elastic Weight Consolidation penalty derived from a diagonal Fisher Information approximation restraining movement in parameters important to prior capability. Training is checkpointed after every epoch to Google Drive and resumes across session termination.

Evaluation used a held-out set of seventy questions spanning seven topics, comparing the frozen base model against the self-adapted model within a single session to permit paired analysis. Functional correctness, measured by test-case pass rate, improved from 0.1111 to 0.1667, a relative gain of 50.0%. The aggregate composite improved by 0.66%, a figure constrained by ceiling effects in three of its four components, each already saturated on the base model before training. Gains concentrated in the two weakest topics, consistent with the scheduler's design. Retention was high, with 95.7% of questions unchanged or improved and a maximum topic-level regression of 0.0159, indicating that catastrophic forgetting did not occur.

The principal contributions are a working end-to-end self-adaptive training loop for a severely constrained compute environment, an execution-based evaluation harness for generated algorithmic code, and empirical evidence that EWC-regularised low-rank adaptation preserves prior capability under self-generated training. The principal limitations are the scale of the base model, the heuristic nature of the evaluation, saturation of the evaluation set, the absence of ablation and multi-seed validation, and the system's inability to produce executable solutions for graph and dynamic programming problems in either state. Each limitation is addressed by a specific item in the proposed programme of future work, of which redesign of the evaluation instrument is the most consequential.

---

# APPENDIX A

## A.1 Research Papers or Articles Published

No research paper arising from this project has been published or submitted at the time of dissertation submission.

*(If a paper is submitted or accepted before submission, insert the copy here and update this note.)*

## A.2 Certificates and Awards

*(Insert copies of any certificates, appreciation letters, or awards received in connection with this project during the project period. If none, retain the statement below.)*

No certificates or awards were received in connection with this project during the project period.

## A.3 Supplementary Material Submitted

The soft copy submitted with this dissertation contains the complete source code of the SEAL-DSA package, the seed question set and held-out evaluation set in JSON format, the trained LoRA adapter checkpoints for all completed epochs, the generated result artefacts including per-question and per-topic tables and result figures, the configuration files used for all reported runs, and the presentation deck.
