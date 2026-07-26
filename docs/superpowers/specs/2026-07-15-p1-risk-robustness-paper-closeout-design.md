# P1 Risk Robustness And Paper Closeout Design

## Objective

Turn the completed P1 Top-20 recommendation export into a provenance-bound,
robustness-audited, paper-ready result without retraining CKG-RL or CGRC and
without changing the frozen recommendation lists.

## Scope

The closeout has four deliverables:

1. harden export provenance and coverage gates;
2. test whether the difficulty result depends on the readiness and structural
   complexity definitions;
3. diagnose rank- and history-dependent risk trade-offs;
4. generate publication artifacts and revise the paper claim.

Model training, checkpoint selection, reward reweighting, and a post-hoc
composite risk score are out of scope.

## Provenance And Coverage Gates

CKG-RL validation consumes each export-time manifest and checks model, seed,
Top-K, record count, export path, checkpoint hashes, and source split manifest.
CGRC receives a read-only checkpoint export entrypoint that hashes `best.pt`
before and after replay, records evaluator and split hashes, and atomically
writes an export manifest beside the JSONL. The analysis rejects any mismatch
between the manifest, replay result, designated checkpoint, split, JSONL, or
native cold-test count.

The original CGRC training-end result remains a separate comparison. Its small
replay drift is reported but is not used as the JSONL acceptance reference;
the same replay evaluator that emitted the JSONL is the acceptance reference.

## Risk Semantics

The four P1 risk signals remain model-neutral and unchanged. Learner readiness
is clarified as the mean structural complexity of the `min(k, |H_u|)` most
structurally advanced distinct courses available in the learner's training
history. Empty histories have readiness zero, although strict cold-test users
are expected to retain nonempty training history. The primary setting remains
`k=5` and P95 robust normalization.

Course artifacts store a binary prerequisite matrix once. Per-seed readiness
is precomputed for every sensitivity setting so recommendation records do not
reallocate the full course matrix or repeatedly sort the same learner history.

## Robustness Protocol

Difficulty sensitivity crosses:

- readiness depth `k in {3, 5, 10}`;
- structural-count scaling `{P90, P95, catalog maximum}`.

Every setting uses the frozen Top-10 lists. List means are macro-averaged by
`(model, seed, target course)`, and CKG-RL minus CGRC differences use the same
204 matched seed-course units, 10,000 paired bootstrap resamples, 100,000
paired sign permutations, and analysis seed 2027. Cold-only difficulty remains
missing for lists with no cold recommendation.

The analysis also reports:

- course-macro risk by rank 1--10;
- course-macro Top-10 risk within history-size bins `1--2`, `3--4`, `5--9`,
  and `10+` distinct courses;
- per-seed directions and matched-pair coverage.

The implementation must reproduce the primary P95/Top-5 difficulty output from
the frozen P1 analysis before other sensitivity settings are accepted.

## Paper Artifacts

The main one-column figure contains a favorable-effect forest plot for the four
Top-10 risks and a separate cold-course exposure comparison. Positive favorable
effect means lower CKG-RL values for prerequisite gap, difficulty gap, and
redundancy, but higher CKG-RL values for concept continuity. The caption states
this transformation explicitly.

A supplementary robustness figure summarizes the nine difficulty settings and
rank-wise effects. CSV outputs retain raw CKG-RL minus CGRC differences so no
directional transformation is hidden from readers.

## Manuscript Claim

The paper must not claim that CKG-RL uniformly lowers pedagogical risk. The
defensible result is that CKG-RL substantially increases cold-course exposure
and improves prerequisite coverage and structural redundancy, while concept
continuity decreases and cold-only difficulty can increase. Top-20 wording is
limited to the metrics whose directions are stable across cutoffs; difficulty
is explicitly described as cutoff-dependent.

The limitations text records the CGRC repaired-rerun status, seed-2026 RNG
resume limitation, proxy nature of all four signals, three-fit inference scope,
and absence of a preregistered composite risk weight.

## Acceptance Criteria

- Six export manifests pass checkpoint, split, JSONL, and native-count checks.
- No exported record fails seed, sequence, Top-20, sorting, or leakage checks.
- P95/Top-5 recomputation matches the existing course-macro difficulty result.
- All nine sensitivity settings contain 204 matched overall pairs; missing
  cold-only coverage is reported rather than zero-filled.
- Figures render to PDF, SVG, and PNG without clipping or overlapping text.
- The paper compiles and its claims match the observed metric directions.
- Focused P1 tests and relevant evaluator/export tests pass.
