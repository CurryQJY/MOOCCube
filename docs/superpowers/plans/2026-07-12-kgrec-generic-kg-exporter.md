# KGRec Generic KG Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export Junyi and COCO strict splits into KGRec atomic data while preserving arbitrary KG triples and validating RecBole link coverage.

**Architecture:** Add a generic raw-KG builder and RecBole file exporter beside the existing MOOCCube-specific path. Course entities remain the first KG entities; all other head/tail entities follow. Existing writers and runner remain the output and execution surfaces.

**Tech Stack:** Python 3.12, pandas, NumPy, PyTorch, pytest, CUDA.

---

### Task 1: Add failing generic-KG builder tests

**Files:**
- Modify: `tests/test_kgrec_strict_adapter.py`
- Modify: `paper_aaai27/scripts/kgrec_strict_adapter.py`

- [ ] Add imports for `build_atomic_data_from_kg_triples` and
  `export_recbole_kgrec_dataset`.
- [ ] Add a test with one warm course, one cold course, and an
  external-to-external prerequisite triple. Assert courses occupy entity IDs
  0 and 1, the prerequisite triple is retained, and the cold course has KG
  degree.
- [ ] Add a temporary-file exporter test with `.link`, `.kg`, and strict split
  pickles. Assert source provenance, relation counts, contiguous IDs, and all
  cold items having KG edges.
- [ ] Add a missing-link test that expects `ValueError` naming the unmapped
  course.
- [ ] Run:

```powershell
D:\Anaconda3\envs\zw\python.exe -m pytest tests/test_kgrec_strict_adapter.py -q --basetemp D:\DeskTop\MOOCCube\outputs\kgrec_generic_exporter_red
```

Expected: import failure because the generic functions do not exist.

### Task 2: Implement the raw arbitrary-KG builder

**Files:**
- Modify: `paper_aaai27/scripts/kgrec_strict_adapter.py`
- Test: `tests/test_kgrec_strict_adapter.py`

- [ ] Implement `build_atomic_data_from_kg_triples`, accepting split pairs and
  raw string `(head, relation, tail)` triples.
- [ ] Build users and courses from split pairs only.
- [ ] Assign courses first in `entity_to_id`, followed by sorted non-course KG
  entities.
- [ ] Map every KG triple without filtering external-to-external edges.
- [ ] Count course degree when a course appears as head or tail.
- [ ] Run the focused tests and verify the builder test passes while exporter
  tests still fail for the missing exporter.

### Task 3: Implement RecBole link/KG parsing and export

**Files:**
- Modify: `paper_aaai27/scripts/kgrec_strict_adapter.py`
- Test: `tests/test_kgrec_strict_adapter.py`

- [ ] Add TSV readers that validate exact semantic columns for `.link` and
  `.kg`, independent of the RecBole type suffixes in headers.
- [ ] Validate one-to-one item/entity mappings and require all split courses in
  link entity IDs.
- [ ] Implement `export_recbole_kgrec_dataset` using the generic builder and
  existing atomic writer.
- [ ] Add source paths, `full_arbitrary_entity_graph`, included relations, and
  relation counts to the manifest.
- [ ] Run the focused adapter tests and verify all pass.

### Task 4: Run the full KGRec unit suite

**Files:**
- Test: `tests/test_kgrec_native_scatter.py`
- Test: `tests/test_kgrec_strict_adapter.py`
- Test: `tests/test_kgrec_strict_runner.py`

- [ ] Run:

```powershell
D:\Anaconda3\envs\zw\python.exe -m pytest tests/test_kgrec_native_scatter.py tests/test_kgrec_strict_adapter.py tests/test_kgrec_strict_runner.py -q --basetemp D:\DeskTop\MOOCCube\outputs\kgrec_generic_exporter_full
```

- [ ] Require zero failures and run `git diff --check` on the two modified
  source/test files.

### Task 5: Export canonical Junyi and COCO seed-2025 atomic data

**Files:**
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/junyi_seed2025_atomic/`
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/coco_seed2025_atomic/`

- [ ] Export Junyi from
  `outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2025`
  plus `junyi_strict_full.link/.kg`.
- [ ] Export COCO from
  `outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_2025`
  plus `coco_strict_full.link/.kg`.
- [ ] Verify all strict checks, exact course counts, cold-course KG coverage,
  and Junyi prerequisite relation retention.

### Task 6: Run full-configuration CUDA smokes

**Files:**
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/junyi_seed2025_fullconfig_smoke/`
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/coco_seed2025_fullconfig_smoke/`

- [ ] Run each dataset with seed 2025, `epochs=1`, `patience=1`, `lr=1e-5`,
  `dim=64`, `context_hops=2`, `max_train_batches=1`, and CUDA.
- [ ] Use an evaluation batch size that fits the catalog: 2048 for Junyi and
  128 for COCO.
- [ ] Verify reports are complete, epoch 0/1 progress exists, loss components
  are finite, checkpoints are nonempty, cold metrics are nonempty, and no
  KGRec Python process remains.
