# CGRC Runtime Repair Notes

This note records the low-risk path for filling missing CGRC timing entries in
the AAAI supplement. Do not infer CGRC latency from metric JSON files unless
they contain explicit timing fields.

## Current No-GPU Repair

The paper now reuses the retained matched MOOCCube profiling rerun without
starting another experiment:

- MOOCCube CGRC inference uses explicit `final_infer_s` values from three seeds
  under `p1_motivation_cgrc_main_table_reproduction`.
- MOOCCube CGRC train/epoch uses 100 unique epoch timers from retained seed-2026
  and seed-2027 logs. Seed 2025 remains unavailable and the table says so.
- Junyi CGRC train/epoch remains `--`; its retained logs contain no separable
  training timer. COCO CGRC training and inference timing are complete.
- `build_revision_tables.py` reads only the explicitly whitelisted matched
  profiling directory, so unrelated P1 Top-K export timings are not mixed in.

## When GPU Is Available

The full rebuild below is no longer required for the current MOOCCube inference
cell. Use it only if a three-seed MOOCCube training-time estimate is required or
if the retained matched profiling evidence is intentionally replaced. Run one
seed at a time if GPU memory is tight.

```powershell
$env:CGRC_PAPER_SAVE_CKPT = "1"
$env:CGRC_PAPER_AUTO_RESUME = "1"
$env:CGRC_PAPER_FORCE_FRESH = "0"

foreach ($seed in 2025, 2026, 2027) {
  $split = "strict_item_cold_balanced_thr1_seed_$seed"
  $outDir = "outputs\content_delta_pop5\static_item_cold_balanced\$split\runtime_cgrc_profile"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $env:CGRC_PAPER_CKPT_DIR = "checkpoints\mooccubex\runtime_cgrc_profile\$split"
  .\run_cgrc_paper_static.ps1 `
    -Seeds $seed `
    -ColdThreshold 1 `
    -Epochs 50 `
    -BatchSize 4096 `
    -DataDir "processed_data_hin_clean_pop5" `
    -OutputRoot "outputs\content_delta_pop5\static_item_cold_balanced" `
    -ResultSubdir "runtime_cgrc_profile" `
    -BestAverageMode "item_macro" `
    *> "$outDir\run.log"
}
```

After the runs finish, regenerate the paper tables:

```powershell
.\py.bat paper_aaai27\scripts\build_revision_tables.py
```

The generator reads:

- `[CGRC-TRAIN] Epoch ... Time: ...s` from the `run.log` files.
- Completed `[CGRC-TRAIN-PROGRESS] ... elapsed=...` epoch lines from retained
  CGRC logs, when full epoch timers were emitted before the new summary tag.
- `final_infer_s` or `total_s` from `cgrc_paper_static_result.json`.

## If Matching Checkpoints Already Exist

A dedicated eval-only CGRC profiler is preferable to full retraining, but only
when the checkpoint matches the exact split, model configuration, and evaluator.
If the retained checkpoint is incomplete or from a different result family, use
the rebuild path above rather than mixing timing evidence across protocols.
