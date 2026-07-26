# CKG-RL V3: USIM-Consistent Simulation Engine Design

## Status

Approved in conversation on 2026-07-22. This design applies the USIM interaction engine to the current CKG-RL method; it is not a standalone reproduction baseline.

## Goal

Replace the V2 fixed-length, candidate-local PPO probe with a deployable CKG-RL simulator whose train-time supervision follows the USIM paper while its inference path has no access to teacher item embeddings, target interactions, or target residuals. Preserve the current MOOCCube strict item-cold protocol, content initialization, full-ranking evaluator, and course-knowledge modules.

## Scope Boundary

V3 reproduces the *simulation semantics* of USIM, not the original paper's datasets, backbone, or reported benchmark. The original MOOCCube/Junyi evaluation and the CKG extensions remain the method under study.

The existing `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py` stays unchanged as the V2 reference route. V3 is a new wrapper module so that V2 results and checkpoints remain reproducible.

## Design Choice

Three implementation routes were considered:

1. Modify the V2 branch in place. This would entangle the `.2863`/V2 lineage with a new simulator and make regressions hard to audit.
2. Run the unmodified `USIM-main` implementation as the new method. This is a useful reference, but it cannot directly express the current CKG reward, strict-cold masking, or scalable MOOCCube candidate path.
3. **Chosen:** add a V3 wrapper around the current FAST3/CKG model. It retains the current model, evaluator, content encoder, and course artifacts while replacing only the user-sequence simulation core with explicit USIM-compatible components.

## V3 Data Flow

```text
warm checkpoint -> frozen teacher item/user space
                         |
train warm courses -> deterministic pseudo-cold masking -> h_0 = content-only course state
                         |
         train-only oracle: e_i, observed U_i
                         |
  USIM candidate mixture + a_end -> actor/critic -> h_{t+1}
                         |                         |
       R_emb + R_rec - step cost + CKG reward      replay / RecPPO
                         |
                h_T enters current ranking loss

real strict-cold course at inference:
content-only h_0 -> target-free retrieval + CKG observable features + actor/a_end -> h_T -> full ranking
```

## Core Contracts

### State, action, and transition

- State is `(h_t, remaining_steps)`.
- The action space is a candidate-user set plus a learned `a_end` action.
- An active user action applies `h_{t+1} = h_t + lambda * e_u`.
- `a_end` leaves `h_t` unchanged and marks the trajectory done. Later steps have zero reward and cannot change the state.
- Inference greedily selects among target-free retrieved users and `a_end`; it never uses `e_i`, a held-out positive user, or `e_i - h_t`.

### Training candidates

For pseudo-cold warm courses, training builds a bounded union of:

- residual-nearest teacher users from `e_i - h_t`;
- observed interaction users for the source warm course;
- state-nearest users, ensuring that inference-retrievable users occur during training;
- random users; and
- `a_end`.

The bounded candidate sizes are explicit V3 launcher parameters. They are a necessary scalability adaptation for MOOCCube, whose user bank is much larger than the original benchmarks. The reward remains aggregated over all observed training users for a pseudo-cold course, computed in chunks rather than reduced to the current V2 single-positive Monte Carlo estimate.

### Reward and RecPPO

For active rows, V3 uses:

`R = R_emb + R_rec - step_penalty + R_CKG`.

- `R_emb` is the reduction in teacher-item Euclidean distance.
- `R_rec` is the reduction in mean absolute user-score error across the pseudo-cold course's observed training users.
- `R_CKG` uses only course relations and the selected user's observed history: concept fit, prerequisite gap, difficulty mismatch, and redundancy, controlled by the existing course-reward settings.
- Transitions are stored in a bounded replay buffer with `(state, candidate IDs, action, reward, next_state, done, old_log_prob)`.
- RecPPO uses a frozen target actor/critic for TD targets and adds the original termination-state value anchor `V([e_i, l]) = 0`.

### CKG integration

The following current-method components remain enabled in the intended V3 run:

- content-derived cold-course initialization and strict ID masking;
- course-aware candidate bias using features available from course metadata and learner history;
- course reward terms; and
- prerequisite auxiliary loss.

The engine-only test configuration disables these terms solely to establish that the simulator behaves correctly. It is a development gate, not a separate paper method.

## Leakage and Compatibility Rules

- Teacher item vectors and observed interaction users may be used only for pseudo-cold training rewards and training candidate construction.
- They must not be consulted by `infer_refined_item_vectors` or evaluation.
- The V3 wrapper delegates static splitting, checkpoint loading, course artifact construction, ranking loss, and full-ranking evaluation to the existing candidate implementation.
- A V3 run must have isolated output/checkpoint roots and a manifest flag identifying the route.

## Tests and Acceptance Gates

Unit tests must prove:

1. `a_end` freezes the state and suppresses future reward/updates.
2. An active user action exactly follows the embedding transition.
3. the recommendation reward aggregates all supplied positive users, not one selected label;
4. inferred trajectories never receive teacher target data or observed positive users;
5. replayed PPO detaches old log probabilities, uses done masking, and includes the terminal value anchor; and
6. CKG reward is zero when disabled and uses only observable metadata/history when enabled.

Before a full run, a CPU trace/preflight must exercise a train pseudo-cold episode and a target-free inference episode. The first full experiment is MOOCCube seed 2025, compared against the same-split old full CKG-RL and V2 results on cold, hot, and overall item-macro Recall/NDCG. It must additionally report termination rate, average active steps, rollout displacement, and reward components. Only a non-degenerate simulator without the V2-scale hot/overall collapse proceeds to seeds 2026 and 2027.
