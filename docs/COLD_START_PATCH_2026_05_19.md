# Cold-start patch (2026-05-19) — Stage 2.1 + 2.3

This patch adds **two flag-gated changes** to `usim_feedback_fast3_content_delta.py`,
plus flips `run_sampled_eval` to off by default (the static main table now reports
full-ranking item-macro only). Every behavior change is reversible via env vars.

## Behavior changes & rollback

| Env var | New default | Legacy default | Rollback to legacy |
|---|---|---|---|
| `USIM_AUX_HOT_ONLY` | `0` (legacy) | `0` | unchanged — set `1` to enable hot-only aux InfoNCE |
| `USIM_EARLY_STOP_SCORE_MODE` | `cold_only` (legacy) | `cold_only` | unchanged — set `geometric` / `harmonic` / `sum` to balance cold + hot |
| `USIM_RUN_SAMPLED_EVAL` | **`0`** (was `1`) | `1` | set `1` to restore (1+200) sampled eval at test time |

The first two flags default to legacy values so re-running an old config
without env changes reproduces previous numbers bit-for-bit (verified by
`_test_aux_hot_only.py::test_legacy_path_bit_identical`). Only the
`run_sampled_eval` default is flipped — set `USIM_RUN_SAMPLED_EVAL=1` if you
need the legacy sampled rows.

## Where the changes live

1. **`BaseConfig.__init__`**
   - Adds `self.aux_hot_only` (env `USIM_AUX_HOT_ONLY`).
   - Adds `self.early_stop_score_mode` (env `USIM_EARLY_STOP_SCORE_MODE`,
     validated against `{cold_only, geometric, harmonic, sum}`).
   - Flips `self.run_sampled_eval` default to `"0"`.

2. **`FastFeedbackUSIM._compute_aux_loss(id_e_true, content_e, effective_cold)`** *(new method)*
   - Legacy branch (`aux_hot_only=False` or `effective_cold is None`): bit-
     identical InfoNCE over the full batch.
   - Hot-only branch: index-selects hot rows, runs InfoNCE only on them.
     Cold rows therefore receive **zero gradient** from the aux term.
   - All-cold or single-hot batches return a graph-connected zero scalar
     so `total_loss.backward()` never crashes.

3. **`FastFeedbackUSIM.forward()`** *(refactor)*
   - Replaces the inline 4-line aux-loss block with
     `aux_loss = self._compute_aux_loss(id_e_true, content_e, effective_cold)`.
     No other change to the loss assembly.

4. **`_compute_early_stop_score(cold_metrics, hot_metrics, k, mode)`** *(new helper)*
   - Located next to `_metric_or_zero`. Implements `cold_only / geometric /
     harmonic / sum`. Defaults / unknown modes fall back to `cold_only`.

5. **`run_static_experiment` (static path only)**
   - Replaces `cur_score = _metric_or_zero(...)` with `_compute_early_stop_score(...)`.
   - Updates per-epoch log to print Cold N@k, Hot N@k, Cold R@k, Hot R@k,
     and `score[<mode>]` so the chosen score is visible.
   - Updates startup banner and restore-best print to show `score_mode`.
   - Persists `aux_hot_only`, `early_stop_score_mode`, `early_stop_average_mode`,
     `early_stop_k`, `early_stop_patience` in the `static_protocol_manifest.json`
     for audit / reproducibility.

The streaming protocol (`run_streaming_experiment`) is intentionally untouched;
its `es_best` block already enforces a `hot_r10_drop_tol` floor and the static
patch should not silently change online-learning behavior.

## Verification commands

```powershell
.\py.bat _test_aux_hot_only.py        # 7 tests: legacy parity, hot-only masking, gradient flow, edge cases
.\py.bat _test_early_stop_score.py    # 11 tests: every mode + edge cases + config validation
.\py.bat _test_seen_index_fastpath.py # regression: seen-index fast path still bit-identical
```

All three suites must print "All ... tests passed".

## How to roll out the cold-start improvements

The patch *enables* the rollout but does not turn it on automatically. To
actually use the new branches in a run:

```powershell
$env:USIM_AUX_HOT_ONLY            = "1"
$env:USIM_EARLY_STOP_SCORE_MODE   = "geometric"   # or "harmonic"
$env:USIM_RUN_SAMPLED_EVAL        = "0"            # already default
```

To roll back any single change without code edits, simply unset / flip the
corresponding env var. The startup log prints both flags so the active
configuration is always traceable from `run.log`.

## A/B verification suggested

For a single seed (≈ same wall-clock as one ablation run):

1. Run with all three env vars at legacy values → baseline.
2. Run with `USIM_AUX_HOT_ONLY=1` only → isolates aux-loss contribution.
3. Run with `USIM_AUX_HOT_ONLY=1` + `USIM_EARLY_STOP_SCORE_MODE=geometric` →
   isolates the joint effect.

Compare `final_fullrank_usim_feedback_fast3_content_delta_static.csv` columns
`full_cold_item_macro_n10` and `full_hot_item_macro_n10` across the three
runs.
