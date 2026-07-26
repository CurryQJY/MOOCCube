# KGRec MOOCCube Seed 2025 Adaptation Audit

Status: data-adaptation ready; GPU smoke not started.

This note records the strict item-cold KGRec adaptation prepared for the MOOCCube
seed 2025 split. It is not an official KGRec reproduction result and must be
reported, if used later, as `KGRec (adapted)`.

## Scope

- Source model: KGRec, KDD 2023.
- Local source: `paper_aaai27/baseline_sources/KGRec`.
- Exported atomic dataset:
  `paper_aaai27/baseline_sources/_kgrec_strict/mooccube_seed2025_atomic`.
- Strict split:
  `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025`.
- Included KG relations: `course_video`, `course_concept`, `course_teacher`,
  `course_school`.
- Excluded for now: `user-video`, to avoid leaking validation/test cold-course
  behavior through user-side activity edges.

## Export Summary

- Users: 197,104.
- Course items: 698.
- KG entities: 65,621.
- Relations: 4.
- CF train pairs: 464,314.
- Validation pairs: 90,501.
- Test pairs: 123,643.
- KG triples: 216,124.

Relation edge counts:

- `course_concept`: 166,835.
- `course_school`: 697.
- `course_teacher`: 2,329.
- `course_video`: 46,263.

## Strict Checks

- Cold items absent from CF train positives: pass.
- Item IDs are contiguous `0..n_items-1`: pass.
- KG entity IDs are contiguous: pass.
- Relation IDs are contiguous: pass.
- All strict cold items have at least one course-side KG edge: pass.

## Implementation Notes

- Added `paper_aaai27/scripts/kgrec_strict_adapter.py`.
- Added strict adapter tests in `tests/test_kgrec_strict_adapter.py`.
- Existing `torch_scatter` compatibility tests remain in
  `tests/test_kgrec_native_scatter.py`.
- Validation and test splits are exported separately as `validation.txt` and
  `test.txt`; the official KGRec-style files `train.txt` and `kg_final.txt` are
  also emitted.

## Verification

Command:

```powershell
D:\Anaconda3\envs\zw\python.exe -m pytest tests/test_kgrec_native_scatter.py tests/test_kgrec_strict_adapter.py -q --basetemp D:\DeskTop\MOOCCube\outputs\kgrec_adapter_pytest
```

Result: 6 passed.

## Next Gate

The next step is a bounded seed-2025 GPU smoke after the current GPU jobs finish
or one GPU job is stopped. A formal table entry requires a strict evaluator
report with CUDA device, warm-only CF negative sampling, train-history masking,
full-catalog ranking, and nonempty item-macro cold metrics.
