# IDRMI Strict Single-Seed Design

Date: 2026-07-14

## Goal

Run a source-faithful feasibility experiment for IDRMI, the model from *An
Explainable Graph-Based Course Recommendation Model Based on Multiple Interest
Factors* (Expert Systems with Applications 2025), on the existing MOOCCube
strict item-cold seed-2025 protocol.

The experiment is a gate, not a paper result. It must first prove that the
released IDRMI components can train on a nonempty train-only graph and produce
finite, nonconstant full-catalog scores for strict cold courses.

## Chosen Approach

Create an external strict adapter and leave
`paper_aaai27/baseline_sources/IDRMI` unchanged as the author-source snapshot.
The adapter imports and reuses the released NGCF and KGCN modules. It implements
only the data, device, batching, checkpoint, and ranking repairs required to
execute the released scoring design under the current protocol.

A direct patch to the author snapshot was rejected because it would mix source
provenance with experiment-specific changes. A clean reimplementation was
rejected because it would create unnecessary equation-drift risk.

## Source Fidelity

The adapter preserves these released IDRMI elements:

- NGCF propagation over the user-course interaction graph;
- KGCN aggregation over course-side knowledge-graph neighbors;
- course-match, user-choice, and course-preference interest factors;
- the released equal-weight average of the three interest factors;
- min-max remapping to the released `[0.5, 1.5]` interval;
- the released learned score and interest-factor fusion before `tanh`;
- binary cross-entropy training on labeled user-course pairs.

The following changes are execution and protocol repairs rather than model
changes:

- populate the NGCF adjacency from positive training rows because the released
  `self.R[uid, item] = 1` block is commented out;
- derive cardinalities from exported data instead of hard-coded constants;
- replace hard-coded `.cuda()` calls with an explicit device;
- make neighbor sampling and batches deterministic under seed 2025;
- sample training negatives from warm training courses only;
- construct every interaction-derived factor from training positives only;
- select checkpoints on validation cold course-macro NDCG@10;
- replace balanced-label classification evaluation with full-catalog ranking.

The released interest-factor code derives neighbors inside each random batch.
Training will retain that batch-local behavior with deterministic batches.
Evaluation will score one user's complete 698-course catalog as one logical
batch, making the factor computation deterministic and independent of scoring
chunk size.

## Components

### Strict Adapter

Create `paper_aaai27/scripts/idrmi_strict_adapter.py` with focused units for:

- strict text-file loading and validation;
- train-only sparse NGCF adjacency construction and source-style mean
  normalization;
- deterministic KGCN neighbor sampling;
- train-only user/course history indices;
- source-faithful interest-factor calculation;
- warm-only negative sampling and BCE training;
- full-catalog scoring, train-history masking, and course-macro metrics;
- validation checkpoint selection, JSON/Markdown reporting, and CLI parsing.

The author-source directory is added to `sys.path` only while loading its
modules. The adapter records the source path and SHA-256 hashes of imported
source files in the report.

### Tests

Create `tests/test_idrmi_strict_adapter.py`. Tests use small synthetic graphs
and cover:

- adjacency contains positive train edges and no validation/test edges;
- negative sampling cannot choose strict cold courses;
- course-match, user-choice, and course-preference use train history only;
- source fusion produces finite scores and preserves pair order/shape;
- history masking still restores the current target score;
- course-macro Recall/NDCG matches a hand-calculated example;
- validation tracking restores the best state rather than the last state;
- CLI defaults select MOOCCube seed 2025 and CUDA automatically when present.

Every behavior is introduced through a failing test before implementation.

## Data Flow

1. Read the existing seed-2025 strict export under
   `paper_aaai27/baseline_sources/_prepared/mooccube_seed2025/idrmi/Data/moocCube`.
2. Separate positive and negative train rows. Reject any train row whose course
   belongs to the validation/test cold-course sets.
3. Build the bipartite NGCF graph and all heuristic histories from positive
   train rows only.
4. Build KGCN neighbors from the static `kg_index.tsv`; cold courses may retain
   permitted static KG edges.
5. Train IDRMI with deterministic batches and warm-only negatives.
6. At each evaluation epoch, rank all 698 courses for validation users, mask
   training history, and select the best checkpoint by validation cold
   course-macro `N@10`.
7. Restore the best checkpoint and evaluate the strict test split once.
8. Write metrics, per-course records, protocol manifest, source hashes, and
   diagnostics to a run-specific output directory.

## Staged Execution

The first run is a GPU smoke with approximately 1,024-2,048 training examples,
one epoch, and the complete course catalog. It may cap validation/test users but
must not cap or sample the 698-course candidate catalog.

The smoke passes only when all of the following hold:

- CUDA is actually selected;
- NGCF adjacency has nonzero user-course edges;
- all train negatives are warm courses;
- loss and scores are finite;
- score standard deviation is greater than `1e-8`;
- validation and test contain at least one evaluated cold course;
- output includes full-catalog course-macro Recall/NDCG;
- no validation/test interaction is used in any training-derived structure.

If the smoke passes, run the uncapped MOOCCube seed-2025 single seed with early
stopping. If it fails because source-faithful interest factors are infeasible or
degenerate, stop and report the gate failure instead of redesigning IDRMI.

## Output Policy

Write smoke outputs under
`paper_aaai27/baseline_sources/_idrmi_strict/mooccube_seed2025_smoke` and the
uncapped run under
`paper_aaai27/baseline_sources/_idrmi_strict/mooccube_seed2025_single`.

Do not edit `main_table.tex`, `main.tex`, or paper result tables during this
feasibility task. A result can enter the table only after protocol checks,
source-fidelity review, and explicit user approval.
