# Static Prerequisite v2: Reproducible Cold-Baseline Design

## Status

Approved direction; implementation starts only after this specification is reviewed.

## Purpose

Establish a trustworthy, reproducible cold-start baseline before changing the
single-model dual-path method. The existing `seed*_prereq` artifacts are useful
historical evidence, but the source/configuration that produced them is not
captured in a manifest. The current clean scorer also has two failure modes that
must be guarded against:

1. historical runs failed because the prerequisite index attribute name was
   inconsistent (`prereq_index` versus `prereq_idx`);
2. the normal sampled-negative loss return path can omit the prerequisite term.

The new baseline must make activation of the prerequisite objective observable
and testable.

## Goals

1. Preserve the existing static architecture and evaluation protocol.
2. Make prerequisite loss wiring explicit, consistent, and unit-tested.
3. Produce a fresh seed-2025 smoke/full result, followed by a clean 3-seed panel.
4. Record enough provenance to reproduce every reported metric.
5. Never overwrite historical outputs.

## Non-goals

- No dual-path or hybrid architecture changes in this experiment.
- No broad hyperparameter search.
- No test-set model selection.
- No claim that the baseline is SOTA until the fresh 3-seed panel is complete.

## Implementation boundary

Create an isolated scorer entry point, `static_prereq_v2.py`, derived from the
current `static_content_scorer_clean.py`. Keep the original file and all existing
outputs unchanged.

The v2 entry point will:

- use one canonical field name, `prereq_idx`;
- expose `--prereq-weight`, `--aux-weight`, `--prereq-path`, and the existing
  dropout/epoch/batch controls through the CLI;
- compute `main_loss`, `aux_loss`, and `prereq_loss` separately, then return
  their sum in every batch-size/negative-sampling branch;
- log epoch-level averages for all three loss components;
- write a `run_manifest.json` containing source SHA-256, git HEAD and dirty
  state, command-line arguments, seed, data/split paths, Python/PyTorch/CUDA
  versions, and timestamp.

Add `tests/test_static_prereq_v2.py` with these checks:

1. the prerequisite index loads under the canonical attribute name;
2. a valid prerequisite fixture produces a finite, nonzero prerequisite loss;
3. the same deterministic batch has measurably different loss/gradient when
   `prereq_weight=0` versus `prereq_weight=1`;
4. the regular sampled-negative branch includes the prerequisite term;
5. batches with no valid prerequisite rows return a zero prerequisite term and
   remain finite.

## Experiment protocol

All runs use the existing strict item-cold split, full-catalog evaluator,
item-macro metrics, and validation cold N@10 for checkpoint selection.

### Stage 0: static checks

- run the new unit tests;
- run `--dry-run` with the real prerequisite index;
- verify `has_prereq=440/698` (or the value reported by the loaded index).

### Stage 1: activation smoke

Run seed 2025 for two epochs into
`outputs/static_prereq_v2/_smoke_seed2025/`. Require a nonzero logged
`prereq_loss`, a clean exit, and all expected provenance files.

### Stage 2: seed-2025 full run

Run the standard 60-epoch/early-stop protocol into
`outputs/static_prereq_v2/seed2025/`.

### Stage 3: 3-seed panel

Run seeds 2025, 2026, and 2027 into separate directories under
`outputs/static_prereq_v2/`. In the same protocol, run a v2 `prereq_weight=0`
control if the activation smoke confirms that the objective is wired. Compare
the fresh v2 control and v2 prerequisite runs against the historical artifacts,
but do not merge their statistics.

## Acceptance criteria

The baseline is considered valid only if:

- all v2 tests pass;
- the smoke run loads the index and reports a finite, nonzero prerequisite loss;
- the activation probe shows a deterministic loss/gradient difference between
  weights 0 and 1;
- every full seed writes `best.pt`, `val_history.json`, `test_metrics.json`, and
  `run_manifest.json`;
- the final table states explicitly that all reported metrics are test metrics
  selected by validation, with no test-time tuning.

If any criterion fails, stop before running the 3-seed panel and fix the issue.

## Reproducibility and safety

- Use new script/output paths; do not modify or delete historical checkpoints.
- Keep the existing dirty worktree intact; the manifest records its state.
- Use unbuffered logs and a bounded launcher so an interruption leaves a clear
  exit status rather than a partial result being treated as success.
- Report the fresh v2 baseline separately from the older `seed*_prereq`
  artifacts.

## Decision after the panel

- If v2 prerequisite materially improves cold over the v2 control and is stable
  across seeds, use it as the verified cold component and consider porting the
  same objective into dual-path.
- If it does not help, retain the verified static control as the cold reference
  and stop spending compute on prerequisite-weight sweeps.
- Only after this decision should dual-path changes be designed.
