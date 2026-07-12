# EdGCL MOOCCube Seed-2025 Feasibility Audit

Audit date: 2026-07-12

## Decision

**NO-GO for a formal EdGCL adaptation on the current strict item-cold protocol
and RTX 5070 12 GB hardware.** No EdGCL result is added to the main table.

This is a feasibility decision, not an accuracy result. The audit intentionally
does not replace EdGCL's social view with a co-enrolment proxy, drop its social
objective, truncate users, or run a sampled evaluator merely to obtain a number.

## Inputs Audited

- Paper: *EdGCL: Disentangling Social and Cognitive Homophily in Graph-Based
  Educational Recommender Systems*, AAAI 2026.
- Official repository: <https://github.com/DaSESmartEdu/EdGCL>, commit
  `7cb98e960a5845cde920474e4acaad6b1781c65d` at audit time.
- Strict split:
  `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025`.
- Hardware: NVIDIA GeForce RTX 5070, 12,227 MiB.

## Official Release Gate

The official `data.zip` contains only a `data/mooper` directory with 13 files;
it does not include the MOOCCubeX data described in the README.

Running the release in the project's CUDA environment failed before model
construction because `run_edgcl.py` requests:

```text
data/mooper/social_adj_train_dense_0.1.pt
data/mooper/social_adj_train_target_0.1.pt
```

while the archive provides only the corresponding filenames without `_0.1`.
Temporary hard links in the audit copy resolved only this packaging mismatch.
With those links and `-ep 1`, the official MOOPer smoke did not finish within a
304-second bounded window and produced no checkpoint or metric. The temporary
Python child was terminated after the timeout. This evidence does not establish
an accuracy failure; it establishes that the released pipeline is not a quick,
self-contained reproducibility starting point on the available hardware.

## Strict Evidence Boundary

The seed-2025 protocol manifest contains:

| Field | Value |
|---|---:|
| Users | 199,199 |
| Courses | 698 |
| Train interactions | 464,314 |
| Validation cold courses | 34 |
| Test cold courses | 68 |
| Test history policy | train only |

MOOCCube provides `user-course`, `user-video`, `course-video`,
`course-concept`, teacher, school, and prerequisite relations. The current
course artifact reports video-side relations for all 698 courses, so a cold
course can retain course-side structure after its training interactions are
removed.

However, no forum, reply, friendship, study-group, or other social relation is
available in the current strict inputs. EdGCL's paper explicitly separates
social homophily from cognitive homophily. Creating its social graph from
co-enrolment or train-history similarity would derive both views from the same
learning behavior and change the core mechanism. Reusing raw `user-video`
edges is also unsafe until every edge belonging to validation/test cold courses
is removed; otherwise it reveals held-out cold-course behavior.

Therefore the social-view gate fails for a protocol-faithful three-dataset
adaptation.

## Memory Gate

The release stores the social adjacency densely and computes a global user-user
attention matrix in `model/model.py`:

```python
user = self.softmax(q @ k.T / (self.config.hidden_dim ** 0.5)) @ v
```

For 199,199 strict-split users:

| Tensor | Float32 memory |
|---|---:|
| One dense user-by-user matrix | 147.82 GiB |
| Social adjacency plus one user-attention matrix | 295.64 GiB |
| One dense user-by-course matrix | 0.52 GiB |

These figures exclude graph-transformer activations, gradients, optimizer
state, the heterogeneous graph, and evaluator buffers. The dense user-by-user
components alone exceed the 12 GB GPU by more than an order of magnitude and
also exceed a single 80 GB GPU. Reducing batch size cannot solve this because
the matrices are constructed globally in each forward pass.

## Required Conditions To Reconsider

1. An official sparse/local-attention EdGCL implementation, or a separately
   justified research reimplementation that removes global `N x N` storage and
   attention without changing the intended social/cognitive objectives.
2. A true train-only social relation for every evaluated dataset, rather than a
   co-interaction proxy.
3. A strict conversion that removes all validation/test cold-course behavior
   from every user-resource relation, excludes cold courses from training
   negatives, and retains only course-side relations for their representations.
4. A fixed-checkpoint full-catalog, train-history-masked, item-macro evaluation
   smoke before any multi-seed training.

Until these conditions hold, EdGCL should remain a high-quality AAAI related-work
reference rather than a reported adapted baseline.
