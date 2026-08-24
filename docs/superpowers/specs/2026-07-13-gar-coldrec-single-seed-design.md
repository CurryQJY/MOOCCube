# ColdRec GAR Strict Single-Seed Design

Date: 2026-07-13

## Goal

Run one MOOCCube seed-2025 GPU feasibility experiment for GAR under the
project's strict course-cold protocol while preserving the released ColdRec GAR
model implementation.

## Source-Fidelity Boundary

The experiment executes the GAR implementation from:

- Repository: `https://github.com/YuanchenBei/ColdRec`
- Audited commit: `18efd24ec79b0ac2b5b7b10ebc8703274fc117d1`
- Model file: `tmp/candidate_repos/ColdRec/model/GAR.py`

The adapter must not modify GAR's learner, generator architecture, loss terms,
optimizer, or pairwise batch sampler. Its MF teacher is also trained through
ColdRec's released `model/MF.py` path with the same strict training export and
embedding width.

The only permitted adaptation points are:

1. Export the established strict split to ColdRec's input layout.
2. Replace ColdRec's native overall user-macro validation callback with
   validation cold course-macro NDCG@10 over the full catalog.
3. Recompute final metrics with the project's full-catalog evaluator and export
   protocol evidence.

This produces `GAR (ColdRec source, strict adapter)`, not an official-protocol
reproduction. The older local `gar_static_hin.py` GAFC/GAN implementation is not
used.

## Inputs

- Processed data: `processed_data_hin_clean_pop5`
- Strict split:
  `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025`
- Seed: `2025`
- Cold threshold: loaded from the strict split manifest, expected to be `1`
- Course content: existing frozen 768-dimensional content tensor
- Device: CUDA device 0

## Components

### Strict Adapter

Create `gar_coldrec_static.py` as a focused adapter. It reuses the tested
ColdRec export, runtime, id-remapping, history-building, and external evaluation
helpers already used by `fsgnn_coldrec_static.py` and
`m2vae_coldrec_static.py`.

The adapter will:

1. Load `static_train.pkl`, `static_val.pkl`, and `static_test.pkl` directly.
2. Export only train interactions as `warm_train.csv`.
3. Export warm/cold validation and test views without changing item identities.
4. Verify that validation/test cold items have no training interactions.
5. Require matching MF user/item embedding files before constructing GAR.
6. Instantiate GAR through ColdRec `Config -> model_factory`.
7. Bind a strict validation callback to the GAR trainer instance. The callback
   builds the current generated cold-item embeddings, restores original IDs,
   evaluates full-catalog cold course-macro NDCG@10, and invokes GAR's own
   `save()` when the score improves.
8. Run GAR's released `train()` method, then externally evaluate the retained
   best embeddings.

The callback changes checkpoint selection only. It does not add a training
loss, alter gradients, or modify ColdRec source files.

### Serial Runner

Create `run_gar_coldrec_single_seed.ps1` to perform two stages:

1. Train or reuse a matching ColdRec MF teacher on the exported seed-2025
   strict dataset.
2. Launch `gar_coldrec_static.py` on CUDA and write all logs/results under
   `paper_aaai27/baseline_sources/_gar_coldrec_strict/mooccube_seed2025_single`.

The runner must verify the MF files, use explicit paths, stop on nonzero exit,
and avoid touching the main table.

## Evaluation Protocol

Validation and test ranking include all 698 catalog courses, including warm
competitors. Only each learner's training history and padding IDs are masked;
the target score is restored after masking. Validation selection uses cold
course-macro NDCG@10. Test exports both interaction-macro diagnostics and the
required course-macro Recall/NDCG at 5, 10, and 20.

Test history defaults to train-only. Validation and test interactions never
enter MF or GAR training, negative sampling, user profiles, or graph evidence.

## Outputs

The output directory contains:

- `gar_coldrec_strict_result.json`
- `gar_coldrec_strict_report.md`
- `per_item_full_cold_gar_coldrec.csv`
- `per_item_full_hot_gar_coldrec.csv`
- `mf_backbone.log`
- `gar_training.log`
- copied protocol/source manifests

The result JSON records the ColdRec commit/status, exact command arguments,
source and split paths, seed, device, epochs, best validation epoch, metric
counts, history policy, and source-fidelity note.

## First-Run Budget and Gate

The first run uses a short but real feasibility budget:

- MF teacher: 5 epochs
- GAR: 10 epochs
- GAR validation every epoch
- embedding width: 64
- batch size: 4096 unless memory evidence requires a smaller value

The run passes feasibility only when:

1. CUDA is actually used.
2. Training and strict full-catalog evaluation finish without non-finite loss.
3. Validation and test cold course counts are nonzero.
4. A retained checkpoint and per-course files exist.
5. The result manifest confirms train-only evidence, full-catalog candidates,
   history masking, and course-macro metrics.

No main-table edit or multi-seed expansion is performed in this task.

## Tests

Add unit tests for:

- GAR command construction and source-model selection.
- Strict split export and zero train overlap for held-out cold courses.
- Rejection when matching MF embeddings are absent.
- Strict validation callback selecting by cold course-macro NDCG@10.
- Result manifest protocol fields and per-course output paths.
- PowerShell runner defaults to MOOCCube seed 2025 and executes MF before GAR.

Tests follow red-green order before production code is added.

