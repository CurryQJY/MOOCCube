# PCGNN Strict Multi-Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete PCGNN under the shared strict item-cold protocol for seeds 2025, 2026, and 2027, then publish only verified full-catalog item-macro results.

**Architecture:** Reuse `paper_aaai27/scripts/pcgnn_strict_adapter.py` and its KG-joint, warm-candidate configuration. Each seed receives an isolated output/checkpoint directory and reads the corresponding existing strict split. Results are accepted only after the adapter report confirms full-catalog evaluation, train-history masking, and nonempty cold item-macro metrics.

**Tech Stack:** Python, PyTorch, PCGNN/RecBole adapter, shared pickle splits, JSON reports.

---

### Task 1: Audit the formal reference run

**Files:**
- Read: `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025_full_formal_kg_warm/pcgnn_strict_adapter_report.json`
- Read: `paper_aaai27/scripts/pcgnn_strict_adapter.py`

- [x] **Step 1: Verify the only report eligible as the 2025 reference**

Run:

```powershell
Get-Content -Raw paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025_full_formal_kg_warm/pcgnn_strict_adapter_report.json
```

Expected: `epochs` is `20`, `split_root` ends in `seed_2025`, and the report contains nonempty `full_cold_item_macro` validation and test metrics.

- [x] **Step 2: Record the formal adapter arguments**

Run:

```powershell
Get-Content paper_aaai27/baseline_sources/ADAPTATION_NOTES.md | Select-Object -Skip 384 -First 24
```

Expected: the command uses KG joint training, `--rs-candidate-mode warm`, strict split root, and full-catalog item-macro evaluation.

### Task 2: Launch seed 2026 in an isolated run directory

**Files:**
- Read: `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2026/static_split_summary.json`
- Create: `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2026_full_formal_kg_warm/`

- [x] **Step 1: Verify the 2026 split before launch**

Run:

```powershell
Get-Content -Raw outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2026/static_split_summary.json
```

Expected: `split_mode` is `strict_item_cold_balanced` and `true_item_cold_start` is `true`.

- [x] **Step 2: Queue the unchanged formal adapter with the 2026 split after the active GPU job finishes**

Run:

```powershell
.\py.bat paper_aaai27\scripts\pcgnn_strict_adapter.py --split-root outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2026 --seed 2026 --max-train-examples -1 --max-val-examples -1 --max-test-examples -1 --epochs 20 --early-stop-patience 5 --train-batch-size 32 --eval-batch-size 64 --kg-batch-size 256 --kg-loss-weight 1.0 --rs-candidate-mode warm --out-dir paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2026_full_formal_kg_warm --checkpoint-dir paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2026_full_formal_kg_warm\checkpoints
```

Expected: an isolated report and best checkpoint are written without changing the 2025 directory.

- [ ] **Step 3: Verify protocol fields after completion**

Run:

```powershell
Get-Content -Raw paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2026_full_formal_kg_warm/pcgnn_strict_adapter_report.json
```

Expected: the report identifies seed 2026 and has nonempty validation/test cold item-macro metrics.

### Task 3: Launch and verify seed 2027

**Files:**
- Read: `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2027/static_split_summary.json`
- Create: `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2027_full_formal_kg_warm/`

- [x] **Step 1: Verify the 2027 strict split**

Run:

```powershell
Get-Content -Raw outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2027/static_split_summary.json
```

Expected: `split_mode` is `strict_item_cold_balanced` and `true_item_cold_start` is `true`.

- [ ] **Step 2: Repeat the formal adapter for seed 2027**

Run:

```powershell
.\py.bat paper_aaai27\scripts\pcgnn_strict_adapter.py --split-root outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2027 --seed 2027 --max-train-examples -1 --max-val-examples -1 --max-test-examples -1 --epochs 20 --early-stop-patience 5 --train-batch-size 32 --eval-batch-size 64 --kg-batch-size 256 --kg-loss-weight 1.0 --rs-candidate-mode warm --out-dir paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2027_full_formal_kg_warm --checkpoint-dir paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2027_full_formal_kg_warm\checkpoints
```

Expected: an isolated report and checkpoint with nonempty cold item-macro metrics.

### Task 4: Aggregate only verified formal runs

**Files:**
- Read: `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025_full_formal_kg_warm/pcgnn_strict_adapter_report.json`
- Read: `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2026_full_formal_kg_warm/pcgnn_strict_adapter_report.json`
- Read: `paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2027_full_formal_kg_warm/pcgnn_strict_adapter_report.json`

- [ ] **Step 1: Extract cold item-macro test metrics and confirm all three seed IDs**

Run:

```powershell
rg -n '"seed"|"full_cold_item_macro"|"test"' paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed20*_full_formal_kg_warm/pcgnn_strict_adapter_report.json
```

Expected: exactly three formal reports, for seeds 2025, 2026, and 2027.

- [ ] **Step 2: Add PCGNN to the main-table aggregator only after metrics agree with the reports**

Expected: the aggregator source references the three verified report paths and no smoke directory.
