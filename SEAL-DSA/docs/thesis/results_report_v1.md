# SEAL-DSA — Results and Closing Chapters (Draft v1)

> PROVISIONAL. All figures in this draft come from the run of 2026-07-29 comparing the
> base model against `checkpoint_epoch_2` (epoch 3 of a planned 5). Every number marked
> in the tables below must be regenerated from `results_final/summary.md` after the full
> five-epoch run completes. The prose structure and the arguments do not change; only
> the figures do.

---

## CHAPTER 7 — EVALUATION OF RESULTS

### 7.1 Experimental Setup

All evaluation was carried out on the held-out DSA evaluation set consisting of 70 questions
distributed uniformly across seven topics: arrays and strings, linked lists, stacks and queues,
trees, graphs, sorting and searching, and dynamic programming, with 10 questions per topic. The
base model is Qwen2.5-1.5B-Instruct loaded in 4-bit NF4 quantization with double quantization and
bfloat16 compute dtype. Generation used greedy decoding with a repetition penalty of 1.1 and a cap
of 256 new tokens per response. Experiments were run on a single NVIDIA T4 GPU on Google Colab.

Two model states were compared. The first is the frozen base model prior to any self-adaptation,
denoted "before self-adaptation". The second is the model after the self-adaptive training run,
with the learned LoRA adapter merged into the base weights, denoted "after self-adaptation".
Comparing endpoints rather than intermediate snapshots isolates the effect of the SEAL loop as a
whole rather than the effect of any single pass over the curriculum.

Because 4-bit quantized generation is not bit-deterministic across sessions, both model states were
evaluated within a single session against an identical question ordering, so that the paired
comparison in Section 7.6 is valid.

### 7.2 Evaluation Metrics

Each generated answer is scored on four components. Answer completeness penalises empty or truncated
responses. Keyword coverage measures the presence of topic-appropriate technical vocabulary drawn
from a per-topic lexicon combined with salient terms from the question itself. Code presence checks,
for coding-type questions, whether the response contains an executable Python construct. Test-case
pass rate extracts the generated function from the response, executes it in an isolated namespace,
and reports the fraction of held-out test cases it satisfies. The four components are combined into
a single weighted overall score in the range [0, 1].

Of these, test-case pass rate is the only metric that verifies functional correctness rather than
surface form, and Section 7.4 argues that it should be read as the primary result.

### 7.3 Overall Performance

**Table 7.1 — Aggregate performance before and after self-adaptive training (n = 70)**

| Metric | Before | After | Change |
|---|---|---|---|
| Overall score | 0.8634 | 0.8691 | +0.0057 (+0.66%) |
| Keyword coverage | 0.9511 | 0.9491 | -0.0020 |
| Answer completeness | 1.0000 | 1.0000 | 0.0000 |
| Code presence | 1.0000 | 1.0000 | 0.0000 |
| Test-case pass rate | 0.1111 | 0.1667 | +0.0556 (+50.0%) |
| Mean answer length (words) | 165.67 | 163.04 | -2.63 |
| Mean generation time (s) | 15.09 | 14.70 | -0.39 |

The aggregate overall score improved by 0.66%. Taken alone this is a modest movement, and
Section 7.4 explains why the aggregate understates what changed.

### 7.4 Ceiling Effects in the Composite Metric

Three of the four score components exhibit ceiling effects on this evaluation set. Answer
completeness is 1.0000 for both model states, indicating that the base model already produces
responses of adequate length for every question. Code presence is likewise 1.0000 for both,
indicating that the base model already emits code constructs for every coding-type question.
Keyword coverage begins at 0.9511, leaving under five percentage points of available headroom, and
moves marginally in the negative direction.

Because these three components are weighted into the composite, and because none of them can
improve, the composite score is structurally constrained toward a near-zero delta regardless of what
self-adaptation achieves. The composite therefore measures the wrong property on this evaluation
set: it confirms that the base model produces well-formed DSA answers, which was never in question,
while diluting the one signal that reflects whether those answers are correct.

Test-case pass rate is the sole component with meaningful headroom, beginning at 0.1111. On this
metric the model improved from 0.1111 to 0.1667, a relative gain of 50.0%. This measures whether
generated code compiles and returns expected outputs on unseen test cases, and it is therefore the
metric that most directly reflects the objective of the SEAL loop.

It must be stated plainly that in absolute terms this corresponds to a small number of additional
questions passing their test cases, and that the evaluation set is not large enough to establish
statistical significance for a difference of this size. The finding is reported as a directional
result, not as a significance claim.

### 7.5 Topic-wise Analysis

**Table 7.2 — Topic-wise scores before and after self-adaptation**

| Topic | N | Before | After | Delta | Pass rate before | Pass rate after |
|---|---|---|---|---|---|---|
| Arrays and strings | 10 | 0.6635 | 0.7135 | +0.0500 | 0.0000 | 0.1667 |
| Linked lists | 10 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A |
| Stacks and queues | 10 | 0.9500 | 0.9500 | 0.0000 | 0.5000 | 0.5000 |
| Trees | 10 | 0.9820 | 0.9661 | -0.0159 | N/A | N/A |
| Graphs | 10 | 0.9286 | 0.9286 | 0.0000 | 0.0000 | 0.0000 |
| Sorting and searching | 10 | 0.7950 | 0.7950 | 0.0000 | 0.2500 | 0.2500 |
| Dynamic programming | 10 | 0.7247 | 0.7306 | +0.0059 | 0.0000 | 0.0000 |

Two observations follow. First, gains concentrate in the topics with the lowest starting scores.
Arrays and strings, the weakest topic before adaptation at 0.6635, records the largest improvement
at +0.0500, and its test-case pass rate rises from 0.0000 to 0.1667. Dynamic programming, the second
weakest at 0.7247, records the second largest gain. This is the behaviour the curriculum scheduler
is designed to produce, namely allocating self-generated training effort toward areas of
demonstrated weakness rather than uniformly across topics.

Second, topics that begin at or near the ceiling do not move. Linked lists remains at 1.0000, and
stacks and queues, graphs, and sorting and searching are unchanged to four decimal places. This is
consistent with the ceiling analysis of Section 7.4 rather than with a failure of the training
procedure.

Graphs and dynamic programming retain a test-case pass rate of 0.0000 in both states. These are the
two topics requiring the most complex multi-step code, and neither model state produces executable
solutions for them. This is discussed further in Chapter 9.

### 7.6 Learning Trajectory Across Training

A partial trajectory across successive checkpoints is available for the topics evaluated before the
earlier comparison run was interrupted.

**Table 7.3 — Partial topic trajectory across checkpoints**

| Topic | Base | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|---|
| Arrays and strings | 0.657 | 0.663 | 0.675 | 0.714 |
| Dynamic programming | 0.733 | 0.783 | — | 0.731 |
| Sorting and searching | 0.821 | 0.811 | — | 0.795 |
| Linked lists | 1.000 | 1.000 | 1.000 | 1.000 |
| Stacks and queues | 0.946 | 0.950 | 0.946 | 0.950 |

Arrays and strings shows a monotonic increase across all four states, from 0.657 to 0.714, with the
largest single increment occurring between the second and third checkpoints. This suggests that
training had not converged at the point of measurement and that additional passes over the
curriculum would be expected to yield further improvement on this topic.

The dynamic programming column is non-monotonic. On a ten-question topic this is more plausibly
attributable to sampling variation than to a genuine regression, and it is reported here without
a causal claim.

Base-model figures in Table 7.3 differ marginally from those in Table 7.2 because the two runs were
executed in separate sessions, and 4-bit quantized generation is not bit-reproducible across
sessions. Figures from the two runs are therefore not mixed within any single table.

### 7.7 Paired Question-level Analysis and Capability Retention

Because both model states were evaluated on identical questions within a single session, a paired
comparison is possible at question level.

**Table 7.4 — Paired question-level outcomes (n = 70)**

| Outcome | Count | Share (%) |
|---|---|---|
| Improved | 3 | 4.3 |
| Unchanged | 64 | 91.4 |
| Degraded | 3 | 4.3 |

This distribution constitutes the principal evidence regarding catastrophic forgetting. Three
questions out of seventy, or 4.3% of the evaluation set, scored lower after self-adaptation, and the
largest topic-level regression is -0.0159 on trees. No topic suffered a collapse in performance, and
no previously passing functional solution was lost.

This retention profile is the intended effect of the Elastic Weight Consolidation penalty combined
with the low-rank parameter-efficient update. By restricting adaptation to a low-rank adapter and
penalising movement in parameters identified as important to prior capability, the system confines
the effect of self-generated training to a small subset of behaviours.

The cost of this conservatism is visible in the same table. A 91.4% unchanged rate means the update
is narrow as well as safe, and the plasticity-retention trade-off is examined in Chapter 9.

### 7.8 Efficiency Observations

Mean generation time fell from 15.09 s to 14.70 s per response, and mean answer length fell from
165.67 to 163.04 words. Neither difference is large enough to attribute to the adapter with
confidence, and both are consistent with ordinary variation in greedy decoding length across a
70-question set. The relevant conclusion is negative rather than positive: merging the LoRA adapter
imposes no measurable inference-time penalty, which supports the deployment practicality claim made
in Chapter 5.

### 7.9 Summary of Findings

The evaluation supports three conclusions.

First, self-adaptive training produced a 50.0% relative improvement in functional correctness, from
a test-case pass rate of 0.1111 to 0.1667, this being the only evaluation component with meaningful
headroom. The absolute magnitude is small and is not established as statistically significant on a
70-question set.

Second, improvement concentrates in the weakest topics, arrays and strings and dynamic programming,
which is consistent with the intended behaviour of the curriculum scheduler.

Third, retention of prior capability is high, with 95.7% of questions unchanged or improved and a
maximum topic-level regression of 0.0159, indicating that the EWC-regularised low-rank update did
not induce catastrophic forgetting.

The aggregate composite improvement of 0.66% is reported for completeness but should not be read as
the effect size, since three of its four components are saturated at or near maximum on this
evaluation set. Redesigning the evaluation set to remove these ceiling effects is identified as the
highest-priority item in Chapter 10.

---

## CHAPTER 9 — LIMITATIONS AND DRAWBACKS

### 9.1 Scale of the Base Model

The system was built on Qwen2.5-1.5B-Instruct, selected because it is the largest model that trains
reliably under 4-bit quantization within the memory and session-duration limits of the freely
available Colab T4 environment. Self-adaptive training depends on the model's ability to generate
questions worth learning from and to score its own answers usefully, and both capabilities improve
substantially with scale. The modest effect sizes reported in Chapter 7 are therefore not
independent of this choice, and no claim is made that they would transfer unchanged to larger
models.

### 9.2 Heuristic Evaluation Without a Judge Model

Answers are scored by a heuristic combining length, keyword overlap, code presence, and test-case
execution. Only the last of these verifies correctness. Keyword overlap in particular can reward an
answer that uses the expected vocabulary while reasoning incorrectly, and it cannot distinguish a
correct explanation from a fluent but wrong one. A stronger design would score answers with a
separate and larger judge model, or with human annotation on a sampled subset. Neither was feasible
within the compute budget of this project.

### 9.3 Ceiling Effects in the Evaluation Set

As established in Section 7.4, three of the four scoring components are saturated on this evaluation
set before training begins. The evaluation set is therefore unable to register improvement on most
of its own metrics, which places a structural upper bound on the measurable effect of any training
procedure. This is a defect of the evaluation instrument rather than of the training method, and it
is the single most consequential limitation of the present work.

### 9.4 Size of the Evaluation Set

Seventy questions across seven topics gives ten questions per topic. At this size, a topic-level
change of one or two questions produces a visible movement in the reported mean, and the study is
underpowered for establishing statistical significance on differences of the magnitude observed.
Topic-level results should be read as indicative.

### 9.5 Incomplete Training Run

The results reported in Chapter 7 reflect three epochs of a planned five-epoch schedule. Table 7.3
indicates that performance on arrays and strings was still rising at the point of measurement, so
the reported figures are more properly read as a lower bound on what the configured schedule would
produce than as a converged result.

### 9.6 Failure on Complex Code Generation

Graphs and dynamic programming record a test-case pass rate of 0.0000 in both the base and adapted
states. The system neither improved nor degraded on the two topics demanding the longest multi-step
implementations. Self-adaptive training cannot bootstrap a capability that is absent at the start,
because the loop can only learn from answers the model is already capable of generating.

### 9.7 Conservatism of the Update

The 91.4% unchanged rate in Table 7.4 shows that the EWC penalty and the low-rank constraint, which
together deliver the strong retention result, also restrict how much the model can change. The
configuration adopted here favours safety over plasticity. A systematic sweep of the EWC
regularisation strength against LoRA rank was not conducted and would be required to establish
where the useful operating point lies.

### 9.8 Reproducibility Under Quantization

Generation under 4-bit quantization is not bit-reproducible across sessions, and merging a LoRA
adapter into 4-bit weights introduces additional rounding. Small differences between runs of the
same configuration are expected, as observed between Tables 7.2 and 7.3. All comparative claims in
this dissertation are therefore drawn from paired within-session evaluations.

---

## CHAPTER 10 — FUTURE ENHANCEMENT

### 10.1 Redesign of the Evaluation Set

The highest-priority improvement is an evaluation set constructed so that the base model does not
already saturate it. This means a higher proportion of coding questions carrying executable test
cases, retirement of conceptual questions the base model answers perfectly, and a larger question
count per topic to reduce sampling variance. Without this, no training improvement can be measured
reliably regardless of its true magnitude.

### 10.2 Model-based Evaluation

Replacing the keyword heuristic with a larger judge model scoring correctness, completeness, and
reasoning quality would remove the principal weakness identified in Section 9.2. Validating the
judge against human annotation on a sampled subset would allow correctness to be reported directly
rather than approximated.

### 10.3 Completion and Extension of the Training Schedule

Completing the five-epoch schedule and extending it further while monitoring for convergence would
establish whether the upward trajectory observed on arrays and strings continues, plateaus, or
reverses. A learning curve over the full schedule is a stronger result than a single endpoint
comparison.

### 10.4 Scaling the Base Model

Repeating the experiment on a 7B-parameter model would test whether the effect size grows with the
capability of the self-generating and self-scoring components, which is the central open question
raised by Chapter 7.

### 10.5 Hyperparameter Study of the Retention Mechanism

A sweep over EWC regularisation strength and LoRA rank would map the plasticity-retention frontier
and identify whether higher gains are available at an acceptable cost in retention, addressing the
conservatism described in Section 9.7.

### 10.6 Difficulty-aware Curriculum

The curriculum scheduler currently targets weak topics. Extending it to target weak difficulty bands
within a topic, and to escalate difficulty as competence is demonstrated, would provide a more
informative training signal, particularly on graphs and dynamic programming where no functional
capability currently exists to reinforce.

### 10.7 Preference-based Optimisation

A Direct Preference Optimization objective is already implemented as an optional alternative to the
supervised objective. Evaluating it against the supervised path under identical conditions is a
direct extension of the present work.

---

## CHAPTER 11 — CONCLUSION

This project set out to determine whether a small language model can improve its own competence on
data structures and algorithms through a self-adaptive loop, generating its own training questions,
evaluating its own answers, and updating its own parameters, without catastrophically forgetting
what it already knew.

The system was built and it runs end to end. Question generation, answer generation, heuristic and
execution-based evaluation, curriculum scheduling, EWC-regularised low-rank parameter updating, and
checkpointed resumable training operate as an integrated pipeline within the constraints of a freely
available single-GPU environment.

On the evidence collected, the loop produces a measurable but small improvement. Functional
correctness, measured as the fraction of held-out test cases passed by generated code, rose from
0.1111 to 0.1667, a relative improvement of 50.0%, while the aggregate composite score moved by
0.66%. The disparity between these two figures is explained by ceiling effects in three of the four
scoring components, and the composite is accordingly not treated as the primary result. Gains
concentrated in the weakest topics, which is the behaviour the curriculum scheduler was designed to
produce.

The retention result is the clearest positive finding. Across seventy held-out questions, 95.7%
scored unchanged or higher after self-adaptation, with a maximum topic-level regression of 0.0159.
The combination of a low-rank adapter with an Elastic Weight Consolidation penalty confined the
effect of self-generated training without measurable damage to prior capability, and merging the
adapter imposed no inference-time cost.

The limitations are equally clear and are stated in Chapter 9. The base model is small, the
evaluation is heuristic rather than judged, the evaluation set saturates before training begins, the
question count is too small for significance testing, the training schedule reported here is
incomplete, and the system fails entirely to produce executable solutions for graph and dynamic
programming problems in either state.

The honest conclusion is therefore a qualified one. Self-adaptive training with explicit retention
regularisation is workable at this scale and does not destroy prior capability, but the gains it
produced here are small, concentrated, and measured with an instrument that limits what could have
been observed. The value of the work lies in a functioning implementation and in a precise account
of where the approach and its evaluation fall short, which Chapter 10 converts into a concrete
programme for extending it.

---

## PROJECT SUMMARY

This project implements SEAL-DSA, a self-adaptive learning system in which a small language model
improves its own competence on data structures and algorithms without externally supplied training
data. The system builds on Qwen2.5-1.5B-Instruct under 4-bit QLoRA quantization and runs within the
limits of a single freely available GPU session.

The pipeline operates as a closed loop. A curriculum scheduler selects a topic based on measured
weakness. The model generates practice questions for that topic, generates candidate answers, and
evaluates those answers using a combination of heuristic scoring and sandboxed execution of
generated code against test cases. High-quality question-answer pairs are used to update a low-rank
adapter, with an Elastic Weight Consolidation penalty restraining movement in parameters identified
as important to previously acquired capability. Training is checkpointed after every epoch and
resumable across sessions.

Evaluation was conducted on a held-out set of seventy questions spanning seven DSA topics, comparing
the frozen base model against the self-adapted model within a single session to permit paired
comparison. Functional correctness, measured by test-case pass rate, improved from 0.1111 to 0.1667.
The aggregate composite score improved by 0.66%, a figure limited by ceiling effects in three of its
four components. Gains concentrated in the weakest topics, consistent with the curriculum
scheduler's design. Capability retention was high, with 95.7% of questions unchanged or improved and
a maximum topic-level regression of 0.0159, indicating no catastrophic forgetting.

The principal contributions are a working end-to-end self-adaptive training loop for a constrained
compute environment, an execution-based evaluation harness for generated DSA code, and empirical
evidence that EWC-regularised low-rank adaptation preserves prior capability under self-generated
training. The principal limitations are the scale of the base model, the heuristic nature of the
evaluation, and saturation of the evaluation set, each of which is addressed in the proposed future
work.
