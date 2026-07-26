# Figure 1 Validation-Only Motivation Redesign

## Objective

Replace the held-out-test motivation analysis in Figure 1 with a validation-only,
read-only diagnosis of two frozen reference models. The revised figure must show
that course-specific PCGNN and generic cold-start CGRC leave complementary gaps
under strict course-cold full-catalog ranking, without using test results to justify
CKG-RL components.

## Evidence Role

Figure 1 is a descriptive motivation analysis, not a final model comparison and
not a mechanistic experiment. It may support only the bounded claim that the two
reference models do not jointly provide strong cold-course exposure and favorable
course-structure characteristics on MOOCCube validation data.

The figure must not claim that:

- CKG-RL improves any plotted outcome;
- a particular diagnostic caused the selection of a CKG-RL component;
- the structural proxies validate pedagogy, mastery, readiness, or learning outcomes;
- either baseline's behavior identifies a causal mechanism.

Final test performance remains in RQ1. Test-list structural analysis remains a
post-hoc, objective-aligned audit in Figure 3. Component attribution remains in RQ3
and must use controlled Full-versus-ablation interventions.

## Data Boundary

### Dataset and protocol

- Dataset: MOOCCube with MOOCCubeX course-side relations.
- Split: validation only.
- Protocol: strict course-cold, full-catalog ranking, train-history masking.
- Seeds: 2025, 2026, and 2027.
- Required validation cold-course count: 34 courses per seed and 102
  `(seed, target course)` units in total, as recorded by the three current split
  manifests. The implementation must reject incomplete coverage.

### Frozen reference models

- PCGNN: the validation-selected checkpoints from the retained strict MOOCCube
  three-seed runs under
  `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed<seed>_full_formal_kg_warm`.
- CGRC: the validation-selected repaired-run checkpoints under
  `checkpoints/content_delta_pop5/p1_motivation_cgrc_main_table_reproduction/strict_item_cold_balanced_thr1_seed_<seed>`.
- CKG-RL outputs, checkpoints, and test results are not inputs to Figure 1.

### Read-only evaluation contract

No retraining is required. Evaluation must:

1. restore the checkpoint state selected by validation cold course-macro NDCG@10;
2. evaluate only validation records and validation cold target courses;
3. use train-only learner histories and the same full catalog as the paper protocol;
4. export ranked Top-10 or Top-20 validation lists without modifying checkpoints;
5. record checkpoint hashes before and after export and require equality;
6. record split hashes, source hashes, seed, cutoff, and record counts in manifests;
7. reject any path, row, or metric identified as test data.

Because the checkpoints were selected on validation behavior, the figure is a
descriptive diagnosis of the selected reference models rather than an unbiased
estimate of their generalization. This limitation must be stated in the surrounding
text or supplement.

## Figure Design

### Figure title

`Validation diagnosis of complementary baseline gaps`

### Panel (a): Cold-course exposure

Show course-level validation NDCG@10 distributions for both PCGNN and CGRC.
Use compact ECDFs or aligned small-multiple histograms with direct model labels.
For each model, report:

- the median course-level NDCG@10;
- the fraction of validation cold courses with NDCG@10 no greater than 0.10;
- the mean cold-course proportion in Top-10 lists;
- effective cold-list coverage, defined as the fraction of validation lists with
  at least one recommended cold course;
- missingness, defined as one minus effective cold-list coverage.

These quantities are descriptive. The `0.10` cutoff is retained only as a
pre-existing diagnostic threshold and is not a success criterion or significance
test.

### Panel (b): Structure of recommended cold courses

Evaluate only courses that are both recommended in Top-10 and cold according to
training popularity. Show PCGNN and CGRC absolute means for:

- prerequisite gap, lower is better;
- concept continuity, higher is better;
- difficulty gap, lower is better;
- structural redundancy, lower is better.

Lists with no recommended cold course remain missing for the conditional analysis;
they must not be zero-filled. Coverage and missingness from Panel (a) accompany
the conditional results so that structural values cannot be read independently of
which lists are observable.

Use distinct markers and grayscale-safe line or hatch encodings. Do not include a
`CKG-RL response` strip, CKG-RL score, improvement arrow, significance star, or
component mapping inside the figure.

## Aggregation and Uncertainty

The data flow is:

1. compute exposure and structural quantities per validation recommendation list;
2. aggregate lists sharing the same `(seed, target cold course)`;
3. give each target course equal weight within a seed;
4. give each seed equal weight in the displayed mean.

Error bars are descriptive 95% percentile intervals from 10,000 seed-stratified
bootstrap resamples. Within each bootstrap replicate, retain the three seeds and
resample target cold courses with replacement inside each seed before averaging
seed-level means. The bootstrap random seed is 2027.

Figure 1 performs no null-hypothesis tests, carries no significance markers, and
therefore defines no multiplicity family. Exact point estimates, intervals, unit
counts, coverage, and missingness must be exported to a compact CSV.

## Manuscript Alignment

### Introduction

Replace language stating that held-out test observations motivate specific
components with a validation-bounded statement:

> Validation diagnostics on MOOCCube characterize complementary baseline gaps:
> the generic cold-start and course-specific references do not jointly provide
> reliable cold-course exposure and favorable course-structure characteristics
> under full-catalog ranking. This motivates studying both objectives together.

Specific choices such as educational rewards, knowledge-guided sampling,
prerequisite supervision, content anchoring, masking, and simulation must be
grounded in the method rationale, prior work, and validation ablations rather than
mapped one-to-one from Figure 1.

### Caption

Use a caption based on:

> Validation-only diagnosis of frozen PCGNN and CGRC on MOOCCube under the strict
> course-cold full-catalog protocol. Panel (a) reports course-level exposure and
> cold-list coverage. Panel (b) reports conditional structural proxies among
> recommended cold courses; lists without a cold recommendation remain missing
> and their frequency is reported as missingness. Points aggregate matched
> `(seed, target-course)` units, and bars are 95% intervals from 10,000
> seed-stratified bootstrap resamples. The diagnostics are descriptive and are not
> independent pedagogical or learning-outcome validation.

### Research questions

RQ2 must no longer ask which test patterns motivate method components. Its bounded
role is to distinguish validation motivation from the frozen-model test audit.

## Outputs

Generate stable artifacts under `paper_aaai27/figures/`:

- `mooccube_validation_motivation.pdf`
- `mooccube_validation_motivation.svg`
- `mooccube_validation_motivation.png`
- `mooccube_validation_motivation_summary.csv`
- `mooccube_validation_motivation_manifest.json`

Preserve the current test-based Figure 1 artifacts until the validation replacement
has passed all checks. Do not overwrite or delete existing user artifacts during
development.

## Verification

Automated checks must prove that:

- no Figure 1 input path or selected row belongs to the test split;
- all target courses belong to the validation cold set and have zero train interactions;
- all learner histories are train-only;
- the three expected seeds and exact manifest-derived cold-course counts are present;
- checkpoint hashes are unchanged by export;
- every Top-K list is finite, descending, duplicate-free, and history-masked;
- conditional cold-only missing values remain missing;
- CKG-RL is absent from the figure inputs and plotted data;
- the plotted CSV reproduces all displayed values and intervals;
- PDF, SVG, and PNG outputs are nonempty and visually legible at AAAI column width;
- the compiled paper has no undefined references, undefined citations, overfull
  boxes, clipping, or overlapping labels on the Figure 1 page.

## Acceptance Decision

The redesign is complete only when Figure 1 can be regenerated from validation
artifacts alone, its checkpoint and split provenance passes the read-only audit,
and all manuscript language treats it as descriptive motivation rather than test-set
model selection, pedagogical validation, or mechanistic evidence.
