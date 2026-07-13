# ColdRec GAR Source-Default Three-Seed Design

## Goal

Run a formal MOOCCube GAR baseline for seeds 2025, 2026, and 2027 under the
existing strict balanced item/course-cold protocol, using the released ColdRec
training implementation and source-default epoch ceilings.

## Scope

- Dataset: MOOCCube only.
- Seeds: 2025, 2026, and 2027.
- Cold split: `strict_item_cold_balanced_thr1_seed_<seed>`.
- Candidate set: the complete 698-course catalog.
- History masking and interaction evidence: training interactions only.
- Checkpoint metric: validation cold course-macro `N@10`.
- Execution: serial on CUDA GPU 0.
- Reporting: per-seed results plus three-seed mean and sample standard
  deviation for interaction-macro and course-macro Recall/NDCG.
- Excluded: editing the paper main table or launching another dataset.

## Source Fidelity

The run uses ColdRec commit
`18efd24ec79b0ac2b5b7b10ebc8703274fc117d1`. The released GAR architecture,
loss, optimizer, and pairwise sampler remain unchanged.

The formal ceilings follow the ColdRec defaults:

- MF teacher: 500 epochs maximum.
- GAR: 500 epochs maximum.
- Embedding size: 64.
- Batch size: 4096.
- Learning rate: 0.001.
- Regularization: 0.0001.
- GAR alpha: 0.05.
- GAR beta: 0.1.
- Early-stop patience: 5, evaluated every epoch.

MF retains ColdRec's native validation behavior. GAR validation is rebound by
the existing strict adapter so checkpoint selection uses full-catalog cold
course-macro `N@10` without changing training behavior.

## Components

### Serial Runner

A new PowerShell runner invokes `run_gar_coldrec_single_seed.ps1` for seeds
2025, 2026, and 2027 in that order. Each seed writes to an isolated directory
under a formal three-seed output root. A failed seed stops the queue and leaves
completed artifacts available for safe resume.

The runner passes the source-default epoch ceilings explicitly, requires CUDA,
records a timestamped queue log, skips an already complete seed unless forced,
and never launches two GAR jobs concurrently.

### Aggregator

A dedicated Python aggregator reads the three
`gar_coldrec_strict_result.json` files. Before computing statistics it requires:

- exactly the requested seeds;
- the expected ColdRec commit and unchanged source flag;
- CUDA execution;
- full-catalog candidate mode;
- train-only test history and interaction evidence;
- zero held-out-cold/train overlap;
- nonempty validation and test cold-course sets;
- finite metrics and a finite retained checkpoint score;
- per-course CSV row counts matching the JSON counts.

It writes a per-seed CSV, a summary CSV/JSON, and a Markdown report containing
means and sample standard deviations for all R/N metrics at 5, 10, and 20.

## Failure Handling

The queue stops immediately on a nonzero child exit, missing result file, or
aggregation gate failure. Existing completed seeds are retained. A subsequent
run without `-Force` resumes from the first missing seed; `-Force` intentionally
rebuilds all requested seeds and their MF teachers.

## Verification

Contract tests cover seed order, source-default ceilings, CUDA, serial
invocation, output isolation, and expected aggregate artifacts. Aggregator unit
tests cover successful three-seed statistics and rejection of missing seeds,
protocol violations, nonfinite values, and row-count mismatches.

The final run is accepted only when all three processes exit successfully, the
aggregator gates pass, and the generated summary reports three runs for both
interaction-macro and course-macro metrics.
