# CKG-RL V3.2 Clean Pipeline Design

## Status

Approved from the July 22 AAAI-method diagnosis. This design replaces the contaminated V3/V3.1 warm-start route with an isolated `T -> G -> V3` pipeline. Historical V2 and V3.1 files remain reference routes and must not be edited by this work.

## Problem Addressed

The former V3 route initialized from a checkpoint trained with legacy PPO, random ID masking, course losses, and an earlier simulator. It then mixed deterministic pseudo-cold rows (which entered V3) with random-ID-dropout rows (which did not), so train-time states did not match the V3 inference state. Its candidate set also used teacher residual and positive-user sources during training but only state retrieval at inference.

## Clean Protocol

The outer strict item-cold split is read from the existing shared static split and is never resampled:

```text
H_train = static_train
H_val   = warm rows in static_val
H_test  = warm rows in static_test
C_val   = strict-cold rows in static_val
C_test  = strict-cold rows in static_test
```

Before fitting a student module, eligible `H_train` item IDs are deterministically divided into disjoint item sets:

```text
H_G     = all remaining warm-training items
P_train = pseudo-cold policy-training items
P_val   = pseudo-cold policy-validation items
```

The split is item-level, stratified by `H_train` popularity, stored in `clean_partition.json`, and validated for disjointness. `H_val`, `H_test`, `C_val`, and `C_test` are never used to choose pseudo items.

### Stage T: clean behavioral teacher

`CleanTeacher` is a standard normalized user/item embedding recommender trained only on `H_train` interactions. It has no content encoder, PPO, rollout, course term, pseudo-cold mask, random ID dropout, or refined evaluation. Checkpoint selection uses `H_val` warm NDCG@10 only. Its frozen item/user tables are the sole behavioral oracle for later training.

### Stage G: content generator

`ContentGenerator` maps frozen item content to teacher item space. It receives targets only for `H_G`; it has no item-ID lookup or teacher user input. It is selected on a held-out item-level fraction of `H_G`, not on `P_val` or outer cold data. Pseudo items are deliberately excluded from this regression supervision so their V3 proxy state is content-only in the same sense as a strict-cold item.

### Stage V3: policy adaptation

The frozen teacher and generator produce `h_0 = G(c_i)` for `P_train`. Only the actor/critic is optimized. `e_i^T` and all `H_train` positive users for pseudo item `i` are used only in the reward:

```text
R_t = (||h_t - e_i^T|| - ||h_{t+1} - e_i^T||)
      + mean_{u in U_i}(|u^T h_t - u^T e_i^T| - |u^T h_{t+1} - u^T e_i^T|)
      - step_penalty + optional_observable_course_reward
```

The main route's action pool is exactly inference-legal: state-nearest teacher users plus a separate `END` action. Teacher residuals and known positives are not candidate sources. An optional `privileged` mode is recorded as an explicit non-main ablation and cannot be selected by the seed launcher.

Course features may be added as an action-logit bias and reward only from course metadata and the selected user's `H_train` history after removing the target pseudo item. This target exclusion is unconditional, even if a target appears in a historical user set.

The state change is trust-projected after the episode:

```text
h_out = h_0 + min(1, max_delta / ||h_T - h_0||) * (h_T - h_0)
```

where `max_delta` is learned-free and set in the run manifest. PPO stores detached row transitions and uses a frozen target critic for one-step TD targets.

### Inference and Evaluation

Hot items use the frozen teacher item bank exactly. Strict-cold item vectors are `G(c_i)` followed by target-free legal V3 rollout and trust projection. No strict-cold target, positive user, reward, residual, or target history enters inference. Full-ranking evaluation uses the shared train-only user history and exports hot/cold/overall interaction and item-macro metrics.

`P_val` and `C_val` can choose policy epochs under a stated cold-improvement/hot-retention gate. `H_test` and `C_test` are read only once after configuration and checkpoint selection are frozen.

## Artifacts

Each run writes:

- `clean_partition.json` with row/item counts and stable hashes;
- `teacher.pt`, `generator.pt`, and `policy.pt` with stage metadata;
- `clean_manifest.json` recording disallowed legacy components, candidate mode, frozen-stage hashes, and no-oracle inference;
- `validation_epochs.csv` and `final_metrics.json` with hot/cold/overall metrics;
- `smoke_report.json` for a target-free inference trace.

## Non-negotiable Guards

1. Pseudo partitions are deterministic, disjoint, and derived only from `H_train`.
2. Teacher fit cannot see non-warm or held-out rows.
3. Generator target IDs cannot intersect `P_train` or `P_val`.
4. Policy inference cannot receive target embeddings, positive users, residual candidates, or target history.
5. Training and inference action candidates are legal state retrieval candidates plus `END` in the main route.
6. Hot bank output is teacher-bank parity; policy never runs for hot items.
7. No global random ID masking exists in this route.
