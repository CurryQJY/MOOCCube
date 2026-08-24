# Cold-Consistent Knowledge Calibration (C3K) Design

**Status:** proposed; no implementation or experiment is authorized by this document alone.
**Scope:** replace the current CKG-RL main-method path for strict course-cold ranking with a single, query-conditioned scoring function whose training and full-ranking inference computations are aligned.

## 1. Problem and decision

The current CKG-RL path has a material train--inference mismatch:

- training executes a simulator on warm interactions, while the primary ranking loss normally receives the unrefined base course vector because strict-cold targets are absent from training and pseudo-cold training is disabled;
- inference refines each strict-cold course once with a global learner bank, then caches that course vector before user-specific ranking;
- the inference simulator is not conditioned on the actual queried learner and does not execute the same target-anchored trajectory used in training.

The revised method must not claim online sequential RL recommendation. It will instead learn a cold-consistent, learner-conditioned ranking score. The central paper claim becomes:

> A cold course must be trained and ranked using the same evidence boundary, while course knowledge must calibrate relevance for the particular learner--course pair rather than act as a course-global heuristic.

## 2. Objectives and non-goals

### Objectives

1. Use the same score function for training, validation, and strict full-ranking inference.
2. Make zero-interaction evidence available at training time through deterministic pseudo-cold views of warm courses.
3. Model prerequisite readiness, concept continuity, difficulty fit, and redundancy as learner-conditioned ranking evidence.
4. Improve strict cold course-macro ranking without reproducing the new-version hot/overall collapse.
5. Preserve full-catalog ranking: no candidate sampling, oracle candidate routing, test-time tuning, or target-course interaction leakage.

### Non-goals

- Do not retain PPO, actor-critic terminology, simulated learner actions, or an online sequential-decision claim.
- Do not claim pedagogical or learning-outcome causality from metadata-derived proxy signals.
- Do not introduce SAGE, CGRC reconstruction, LLM scoring, or other inactive experimental branches into the primary method.
- Do not reuse the current R@10=0.2863 checkpoint as an initialization or headline result; it is a diagnostic artifact from a mismatched path.

## 3. Method: C3K

### 3.1 Evidence-consistent course encoder

For each course \(c\), cache content-derived features \(\mathbf{x}_c\), structural metadata \(\mathbf{m}_c\), and, for warm courses only, an interaction-trained ID embedding \(\mathbf{v}_c\).

Define an evidence mask \(z_c\in\{0,1\}\), where \(z_c=1\) denotes a cold-style view. The course representation is

\[
\mathbf{e}_c(z_c)=\operatorname{Enc}_c\left(\mathbf{x}_c,\mathbf{m}_c,(1-z_c)\mathbf{v}_c\right).
\]

At strict cold evaluation, \(z_c=1\) only for strict-cold candidates. Warm catalog candidates retain their available warm evidence. During training, a seed-fixed subset of eligible warm courses is designated pseudo-cold before optimization begins. When one of these courses appears in a training batch, the implementation constructs both its full and masked view: the masked view participates in ranking training and the pair participates in consistency training. This creates masked training examples without changing the data split.

The pseudo-cold sampler must be deterministic under seed, use only train-derived popularity, and never use validation/test labels or interactions.

### 3.2 Learner-conditioned knowledge features

Encode a learner from train-history courses only:

\[
\mathbf{h}_u=\operatorname{Enc}_u(\mathcal{H}^{\mathrm{train}}_u).
\]

For every query--candidate pair, compute a normalized structural vector

\[
\mathbf{k}(u,c)=[C(u,c),P(u,c),D(u,c),R(u,c)],
\]

where \(C\) is concept continuity, \(P\) is prerequisite gap, \(D\) is difficulty gap, and \(R\) is redundancy. All values must be computable from train history plus course-side metadata. Their definitions and orientation must exactly match the evaluator/audit definitions or document any justified difference.

### 3.3 Knowledge calibration score

The only ranking score is

\[
s_\theta(u,c,z_c)=
\frac{\phi_u(\mathbf{h}_u)^\top\phi_c(\mathbf{e}_c(z_c))}{\tau}
\quad + \quad g_\theta(\mathbf{h}_u,\mathbf{e}_c(z_c))
\quad + \quad \boldsymbol{\rho}_\theta(\mathbf{h}_u,\mathbf{e}_c(z_c),\mathbf{k}(u,c))^\top\mathbf{k}(u,c).
\]

The second term is a small residual MLP over learned user/course embeddings only. The third term is a bounded, learner-conditioned gate over interpretable knowledge features. Structural evidence enters the score only through this third term. Its coefficients are sign-constrained: concept continuity can only receive a non-negative calibration, while prerequisite gap, difficulty gap, and redundancy can only receive non-positive calibration. For example,

\[
\boldsymbol{\rho}_\theta=[a_C,-a_P,-a_D,-a_R],\qquad
a_j=a_{\max}\,\sigma(\widetilde a_j).
\]

This prevents the method from reducing to a course-global scalar bias and permits an audit of how each structural signal changes a particular learner--course score.

The gate must be bounded (for example, via tanh or sigmoid-scaled coefficients) and regularized toward zero so that structural metadata calibrates relevance rather than overwhelms it.

### 3.4 Training objective

Each training batch contains a mixture of ordinary warm views and pseudo-cold views. The objective is

\[
\mathcal{L}=\mathcal{L}_{\mathrm{rank}}
\quad + \quad \lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}
\quad + \quad \lambda_{\mathrm{gate}}\mathcal{L}_{\mathrm{gate}}.
\]

- \(\mathcal{L}_{\mathrm{rank}}\): masked batch-softmax or pairwise ranking loss using the same \(s_\theta\) as inference. Known-positive and same-item false negatives must be masked.
- \(\mathcal{L}_{\mathrm{cons}}\): consistency between the full and pseudo-cold views of the same warm course. Concretely, it is a stop-gradient cosine alignment between their content-adapter projections, not an equality constraint on their final ranking scores; warm ID evidence can therefore remain useful.
- \(\mathcal{L}_{\mathrm{gate}}\): coefficient norm and smoothness regularization for the sign-constrained gate.

The structural terms are an inductive calibration prior, not pseudo-labels for pedagogical quality. No reward rollouts, simulated actions, PPO loss, or course-global cached refinement are part of this objective.

## 4. Unified data flow

    Train: (u, c+, c-, z_c) -> Enc_u / Enc_c / k(u,c) -> s_theta(u,c,z_c) -> losses
    Validate: same score over full catalog -> cold, hot, overall metrics
    Test: same score over full catalog -> frozen metrics

Only candidate-invariant \(\mathbf{x}_c\), \(\mathbf{m}_c\), and optionally \(\mathbf{e}_c(z_c)\) may be cached. The learner-conditioned terms are calculated in user and item blocks during full ranking. The evaluator must call the same score implementation used by the training loss, not reconstruct a parallel evaluation score.

## 5. Model selection and safeguards

### Validation selection

Select checkpoints only on validation data using a pre-registered constrained criterion:

1. maximize cold item-macro NDCG@10;
2. reject a checkpoint if hot item-macro NDCG@10 falls more than a fixed tolerance from the content baseline;
3. break ties with cold Recall@10, then overall NDCG@10.

The tolerance is fixed before validation screening. A recommended initial value is 0.003 absolute hot NDCG@10, subject to a pilot-based feasibility check and no test-set adjustment.

### Mandatory parity checks

1. For fixed \((u,c,z_c)\), the training helper and the full-ranking evaluator return the same score within numerical tolerance.
2. A strict-cold item has its ID branch masked in both pseudo-cold training and cold inference.
3. A warm candidate retains its ID branch in full-ranking inference.
4. Evaluation masks only train history under the declared train-only policy.
5. Candidate structural features never include target interaction, validation interaction, or test interaction evidence.
6. The candidate bank contains the entire catalog after history masking.

## 6. Experimental program

### Phase A: validation-only one-seed screen

Run in order, sharing split, seed, training budget, and candidate evaluator:

1. **Base:** content encoder with cold masking; no pseudo-cold training or knowledge calibration.
2. **Base + CC:** add pseudo-cold consistency.
3. **Feature-concat control:** Base + CC with an unconstrained MLP over \([\mathbf{h}_u,\mathbf{e}_c,\mathbf{k}(u,c)]\).
4. **Base + CC + KC:** add learner-conditioned, sign-constrained knowledge calibration.
5. **Full C3K:** add sign-constrained gate regularization.

For each row, record cold/hot/overall R@5/10/20 and N@5/10/20, training cost, inference cost, checkpoint epoch, source hash, and parity-test result. Do not inspect test metrics.

Proceed only if Full C3K exceeds Base on validation cold NDCG@10 while passing the hot-retention guard. If it fails, diagnose on validation and simplify the failed module rather than tuning against test data.

### Phase B: frozen three-seed final experiment

Freeze architecture, loss weights, pseudo-cold policy, validation selector, and all baselines. Run seeds 2025/2026/2027 and evaluate test exactly once per seed. Report cold, hot, and count-weighted overall course-macro metrics. The content-only Base and the feature-concatenation control are mandatory three-seed baselines.

### Phase C: direct ablations

Run only same-configuration, three-seed interventions:

1. w/o pseudo-cold consistency;
2. w/o learner-conditioned knowledge calibration;
3. w/o sign constraint and gate regularization;
4. content-only Base;
5. unconstrained feature-concatenation control.

Signal-level removals are supplementary diagnostics and must not be advertised as independently necessary unless direct paired evidence supports that claim.

## 7. Paper changes after successful validation

- Rename the method and title to remove “Reinforcement Learning”; a provisional name is **Cold-Consistent Knowledge Calibration (C3K)**.
- Replace simulated-learner and actor-critic diagrams with a user-history / candidate-course / knowledge-calibration diagram.
- State that the model is a strict cold-start ranker, not an online learning-path or pedagogical-outcome optimizer.
- Replace the current RQ3 table with direct C3K ablations only; do not shift old ablation deltas to a new Full score.
- Report the cold--hot trade-off and overall result rather than only cold gains.

## 8. Risk register and mitigations

| Risk | Why it matters | Mitigation / stop rule |
|---|---|---|
| The method looks like feature concatenation | Weak novelty and easy reviewer objection | Preserve the evidence-consistency problem and sign-constrained query-conditioned gate; compare against an MLP feature-concatenation baseline. |
| Knowledge features leak target/history data | Invalid cold-start claim | Compute features from train history only; hash and audit input sources per split. |
| Pseudo-cold does not resemble strict cold | Apparent alignment is cosmetic | Mask the exact ID pathway and test a fixed pseudo-cold rate on validation; document the remaining distribution gap. |
| Cold gains still damage hot/overall | The current failure repeats | Enforce validation hot-retention gate and report overall as a primary diagnostic. |
| Query-conditioned full ranking is too slow | Method cannot scale to COCO | Cache invariant course tensors; score user/item blocks; report actual latency and memory. |
| Structural calibration worsens real ranking | Over-constraining harms relevance | Use bounded sign-constrained gates, norm regularization, and validation-only stop rules; do not fabricate preference labels from proxies. |
| Signal definitions are circular | Pedagogical claims become self-validating | Frame them as objective-aligned proxies; retain external ranking metrics as primary evidence. |
| Three seeds do not resolve uncertainty | Overclaiming component effects | Report seed-level values and paired intervals; label non-significant or inconsistent effects as descriptive. |
| Code paths drift again | Reintroduces train--inference mismatch | Make parity tests required in CI and bind run manifests to source/config hashes. |

## 9. Acceptance criteria

The method may replace the paper's current CKG-RL headline only if all are true:

1. all parity and leakage checks pass;
2. validation-only Phase A selects Full C3K without violating its hot guard;
3. frozen three-seed test shows a credible cold improvement over Base and the strongest relevant baseline;
4. hot and count-weighted overall metrics do not conceal a material regression;
5. every main-paper ablation is a direct same-configuration run;
6. method text, code, and evaluator describe the same computation.
