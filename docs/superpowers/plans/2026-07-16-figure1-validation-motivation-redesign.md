# Figure 1 Validation-Only Motivation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Figure 1's held-out-test motivation evidence with a validation-only, read-only diagnosis of frozen PCGNN and CGRC checkpoints.

**Architecture:** Extend the existing PCGNN and CGRC Top-K replay entrypoints with an explicit `validation` evaluation target while preserving their current `test` defaults. A focused validation analyzer consumes the six exported JSONL files, reuses the established course-structure proxy implementation, aggregates 102 `(seed, target-course)` units, and writes compact plotting data. A separate plotting module produces the new Figure 1, after which the manuscript is revised to give validation motivation, test audit, and component intervention distinct evidential roles.

**Tech Stack:** Python 3.12, PyTorch, pandas, NumPy, matplotlib, pytest, LaTeX/latexmk, Poppler.

---

### Task 1: Add validation-only read-only export support

**Files:**
- Modify: `export_p1_pcgnn_topk.py`
- Modify: `export_p1_cgrc_topk.py`
- Modify: `cgrc_paper_static_hin.py`
- Modify: `tests/test_p1_checkpoint_export_entrypoints.py`

- [ ] **Step 1: Write failing PCGNN split-selection tests**

Add tests requiring a pure helper and CLI option:

```python
def test_pcgnn_validation_view_uses_validation_examples_only():
    selected = select_pcgnn_analysis_view(
        "validation",
        validation_examples=[{"target": 11}],
        test_examples=[{"target": 22}],
    )
    assert selected == [{"target": 11}]


def test_pcgnn_analysis_split_parses_validation_target():
    args = parse_pcgnn_args(["--seed", "2025", "--analysis-split", "validation"])
    assert args.analysis_split == "validation"
```

- [ ] **Step 2: Write failing CGRC split-selection tests**

Require `build_cgrc_runtime_environment(..., analysis_split="validation")` to emit
`CGRC_PAPER_EVAL_SPLIT=validation`, and require a pure selector in
`cgrc_paper_static_hin.py`:

```python
def test_cgrc_validation_runtime_is_explicit():
    env = build_cgrc_runtime_environment(
        seed=2025,
        split_dir="split",
        checkpoint_dir="checkpoint",
        output_dir="output",
        topk_output="top20.jsonl",
        analysis_split="validation",
    )
    assert env["CGRC_PAPER_EVAL_SPLIT"] == "validation"


def test_cgrc_final_view_selects_validation_loader_and_seen_history():
    view = select_final_evaluation_view(
        "validation",
        val_loader="val-loader",
        test_loader="test-loader",
        train_seen="train-seen",
        test_seen="test-seen",
        val_cold_items="val-cold",
        test_cold_items="test-cold",
    )
    assert view.loader == "val-loader"
    assert view.seen_items == "train-seen"
    assert view.cold_items == "val-cold"
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
.\py.bat -m pytest tests\test_p1_checkpoint_export_entrypoints.py -q --basetemp=.pytest_tmp\validation_motivation_export_red
```

Expected: failures because `--analysis-split`, `select_pcgnn_analysis_view`,
`CGRC_PAPER_EVAL_SPLIT`, and `select_final_evaluation_view` do not exist.

- [ ] **Step 4: Implement PCGNN validation export**

Add:

```python
def select_pcgnn_analysis_view(analysis_split, *, validation_examples, test_examples):
    if analysis_split == "validation":
        return validation_examples
    if analysis_split == "test":
        return test_examples
    raise ValueError(f"unsupported analysis split: {analysis_split}")
```

Expose `parse_args()` as the existing parser entrypoint with:

```python
parser.add_argument("--analysis-split", choices=("validation", "test"), default="test")
```

When `validation` is selected:

- export `validation_examples`, never `test_examples`;
- set JSONL metadata `analysis_split="validation"`;
- compare replay metrics with `report["validation"]` and the checkpoint validation metadata;
- write `pcgnn_top20_validation.jsonl`, `pcgnn_validation_replay_result.json`, and `validation_export_manifest.json` unless an explicit output directory is supplied;
- record `analysis_split` in the replay result and manifest;
- keep current test filenames and behavior unchanged when the option is omitted.

- [ ] **Step 5: Implement CGRC validation export without retraining**

Add a frozen dataclass and selector:

```python
@dataclass(frozen=True)
class EvaluationView:
    name: str
    loader: object
    seen_items: object
    cold_items: object


def select_final_evaluation_view(
    analysis_split,
    *,
    val_loader,
    test_loader,
    train_seen,
    test_seen,
    val_cold_items,
    test_cold_items,
):
    if analysis_split == "validation":
        return EvaluationView("validation", val_loader, train_seen, val_cold_items)
    if analysis_split == "test":
        return EvaluationView("test", test_loader, test_seen, test_cold_items)
    raise ValueError(f"unsupported evaluation split: {analysis_split}")
```

Read `CGRC_PAPER_EVAL_SPLIT` in `CGRCConfig`, select the final loader/seen map/cold
items after restoring `best_state`, and use the selected view for all final full-ranking
metrics and Top-K export. Add `analysis_split` to Top-K metadata and
`evaluation_split` to `cgrc_paper_static_result.json`. Keep the default `test` path
byte-compatible in behavior.

In `export_p1_cgrc_topk.py`, add `--analysis-split`, pass it into the environment,
require the native result's `evaluation_split` to match, and record it in the export
manifest. Continue hashing both `latest.pt` and `best.pt` before and after replay.

- [ ] **Step 6: Run focused export tests and verify GREEN**

Run the command from Step 3.

Expected: all tests pass, including existing test-export behavior.

- [ ] **Step 7: Commit the exporter changes**

```powershell
git add export_p1_pcgnn_topk.py export_p1_cgrc_topk.py cgrc_paper_static_hin.py tests/test_p1_checkpoint_export_entrypoints.py
git commit -m "feat: add validation-only baseline Top-K replay"
```

### Task 2: Export the six frozen validation recommendation files

**Files:**
- Generate: `outputs/validation_motivation/pcgnn/strict_item_cold_balanced_thr1_seed_2025/`
- Generate: `outputs/validation_motivation/pcgnn/strict_item_cold_balanced_thr1_seed_2026/`
- Generate: `outputs/validation_motivation/pcgnn/strict_item_cold_balanced_thr1_seed_2027/`
- Generate: `outputs/validation_motivation/cgrc/strict_item_cold_balanced_thr1_seed_2025/`
- Generate: `outputs/validation_motivation/cgrc/strict_item_cold_balanced_thr1_seed_2026/`
- Generate: `outputs/validation_motivation/cgrc/strict_item_cold_balanced_thr1_seed_2027/`

- [ ] **Step 1: Record checkpoint hashes before replay**

Use `Get-FileHash -Algorithm SHA256` for the three PCGNN `best_model.pt` files and
the six CGRC `latest.pt`/`best.pt` files. Save the hashes in the generated manifests,
not in a hand-maintained table.

- [ ] **Step 2: Run PCGNN validation exports**

Run all three seeds:

```powershell
foreach ($seed in 2025, 2026, 2027) {
  .\py.bat export_p1_pcgnn_topk.py `
    --seed $seed `
    --analysis-split validation `
    --top-k 20 `
    --output-dir "outputs\validation_motivation\pcgnn\strict_item_cold_balanced_thr1_seed_$seed"
  if ($LASTEXITCODE -ne 0) { throw "PCGNN validation export failed for seed $seed" }
}
```

Expected: one complete Top-20 JSONL bound to 34 validation cold target courses,
a validation replay result, and a read-only manifest per seed.

- [ ] **Step 3: Run CGRC validation exports**

Run all three seeds:

```powershell
foreach ($seed in 2025, 2026, 2027) {
  .\py.bat export_p1_cgrc_topk.py `
    --seed $seed `
    --analysis-split validation `
    --split-dir "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_$seed" `
    --checkpoint-dir "checkpoints\content_delta_pop5\p1_motivation_cgrc_main_table_reproduction\strict_item_cold_balanced_thr1_seed_$seed" `
    --output-dir "outputs\validation_motivation\cgrc\strict_item_cold_balanced_thr1_seed_$seed\eval" `
    --topk-output "outputs\validation_motivation\cgrc\strict_item_cold_balanced_thr1_seed_$seed\top20_validation.jsonl" `
    --top-k 20
  if ($LASTEXITCODE -ne 0) { throw "CGRC validation export failed for seed $seed" }
}
```

Expected: no training epochs execute after checkpoint resume; the final evaluation
reports `evaluation_split=validation` and exports only strict validation-cold rows.

- [ ] **Step 4: Verify hashes and coverage**

Require before/after checkpoint hashes to match and require each manifest to report:

```json
{"analysis_split": "validation", "top_k": 20, "target_course_count": 34}
```

Reject any JSONL record with `analysis_split != "validation"` or a target outside
the split manifest's `strict_item_cold_val` rows.

### Task 3: Build validation motivation statistics

**Files:**
- Create: `paper_aaai27/scripts/analyze_validation_motivation.py`
- Create: `tests/test_validation_motivation_analysis.py`
- Generate: `paper_aaai27/figures/validation_motivation_analysis/course_macro.csv`
- Generate: `paper_aaai27/figures/mooccube_validation_motivation_summary.csv`
- Generate: `paper_aaai27/figures/mooccube_validation_motivation_manifest.json`

- [ ] **Step 1: Write failing analysis tests**

Use synthetic validation records for two models and three seeds. Tests must require:

- metadata `analysis_split=validation`;
- exact target membership in `strict_item_cold_val`;
- rejection of any `strict_item_cold_test` target;
- Top-20 completeness, finite descending scores, unique items, and no train-history leakage;
- list-before-course aggregation;
- cold-only missingness preservation;
- equal seed weighting;
- deterministic seed-stratified bootstrap intervals;
- exactly 102 course units per model for real-data validation.

The wished-for bootstrap API is:

```python
mean, low, high = seed_stratified_interval(
    course_rows,
    value_column="cold_prerequisite_gap",
    n_bootstrap=10_000,
    random_seed=2027,
)
```

- [ ] **Step 2: Run analysis tests and verify RED**

```powershell
.\py.bat -m pytest tests\test_validation_motivation_analysis.py -q --basetemp=.pytest_tmp\validation_motivation_analysis_red
```

Expected: import failure because the focused analyzer does not yet exist.

- [ ] **Step 3: Implement the focused analyzer**

Reuse these established functions rather than duplicating proxy definitions:

```python
from paper_aaai27.scripts.analyze_p1_topk_motivation import (
    CourseMacroAccumulator,
    RISK_COLUMNS,
    analyze_export_record,
    build_real_risk_artifacts,
    validate_export_record,
)
```

Implement `validation_seed_inputs()` to load `static_train.pkl` and
`static_val.pkl`, select only `_split_source == "strict_item_cold_val"`, construct
train-only histories/popularity, and return the exact ordered validation pairs.

For each model/seed JSONL:

1. validate each record against the ordered validation pair;
2. compute Top-10 list and cold-only structural proxies;
3. compute target rank, Recall@10, and NDCG@10;
4. aggregate first by `(model, seed, target_item_id)`;
5. calculate cold share, effective list coverage, and missingness;
6. calculate model-level equal-seed means and seed-stratified 95% intervals;
7. write only PCGNN and CGRC rows;
8. bind all six export manifests and source hashes into the analysis manifest.

- [ ] **Step 4: Run analysis tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Run the real validation analysis**

```powershell
.\py.bat paper_aaai27\scripts\analyze_validation_motivation.py
```

Expected: 204 total model-course rows, 102 for PCGNN and 102 for CGRC, with no
CKG-RL rows and no test-bound inputs.

- [ ] **Step 6: Commit analyzer and tests**

```powershell
git add paper_aaai27/scripts/analyze_validation_motivation.py tests/test_validation_motivation_analysis.py
git commit -m "feat: analyze validation-only motivation evidence"
```

### Task 4: Draw the new Figure 1

**Files:**
- Create: `paper_aaai27/scripts/draw_validation_motivation.py`
- Create: `tests/test_draw_validation_motivation.py`
- Generate: `paper_aaai27/figures/mooccube_validation_motivation.pdf`
- Generate: `paper_aaai27/figures/mooccube_validation_motivation.svg`
- Generate: `paper_aaai27/figures/mooccube_validation_motivation.png`

- [ ] **Step 1: Write failing drawing tests**

Require the drawing function to:

- accept only `pcgnn` and `cgrc` rows with `analysis_split=validation`;
- reject CKG-RL or test rows;
- render Panel (a) exposure distributions and direct labels;
- render Panel (b) four cold-only absolute proxy means with 95% intervals;
- display coverage/missingness near the conditional proxy panel;
- contain no `CKG-RL response`, improvement arrow, or significance marker;
- export nonempty PDF, SVG, and PNG files.

- [ ] **Step 2: Run drawing tests and verify RED**

```powershell
.\py.bat -m pytest tests\test_draw_validation_motivation.py -q --basetemp=.pytest_tmp\validation_motivation_draw_red
```

Expected: import failure because the drawing module does not yet exist.

- [ ] **Step 3: Implement the two-panel plot**

Use a 3.35-inch single-column figure. Panel (a) uses two directly labeled ECDFs
and annotates median NDCG@10, fraction at or below 0.10, Top-10 cold share, and
effective coverage. Panel (b) uses four metric rows with offset PCGNN/CGRC markers
and seed-stratified 95% intervals. Use orange circles for PCGNN, gray squares for
CGRC, distinct line styles, and no color-only distinction.

The x-axis and caption must state that Panel (b) is conditional on a cold course
being recommended. No significance language is generated.

- [ ] **Step 4: Run drawing tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Generate and inspect the real figure**

```powershell
.\py.bat paper_aaai27\scripts\draw_validation_motivation.py
```

Inspect the PNG in color and grayscale. Verify readable labels, complete intervals,
no overlap, and visible marker/line distinctions at final paper scale.

- [ ] **Step 6: Commit drawing code and tests**

```powershell
git add paper_aaai27/scripts/draw_validation_motivation.py tests/test_draw_validation_motivation.py
git commit -m "feat: draw validation-only motivation figure"
```

### Task 5: Align the manuscript and verify the paper

**Files:**
- Modify: `paper_aaai27/main.tex`
- Modify: `tests/test_method_motivation.py`
- Generate: `paper_aaai27/main.pdf`

- [ ] **Step 1: Write failing manuscript assertions**

Replace the current test-based assertions with checks requiring:

```python
assert "Validation diagnosis of complementary baseline gaps" in manuscript
assert "validation-only" in manuscript.lower()
assert "mooccube_validation_motivation.pdf" in manuscript
assert "These observations motivate" not in manuscript
assert "model-neutral pedagogical-risk audit" not in manuscript
```

Also require the caption to name MOOCCube, validation, strict course-cold,
full-catalog ranking, 102 `(seed, target-course)` units, coverage/missingness, and
10,000 seed-stratified bootstrap resamples.

- [ ] **Step 2: Run manuscript tests and verify RED**

```powershell
.\py.bat -m pytest tests\test_method_motivation.py -q --basetemp=.pytest_tmp\validation_motivation_manuscript_red
```

Expected: failures on the current held-out-test motivation wording and old figure path.

- [ ] **Step 3: Revise the Introduction, Figure 1 caption, and RQ2 role**

Change Figure 1 to `figures/mooccube_validation_motivation.pdf`. State that the
selected baseline checkpoints are described on validation data and that the figure
motivates jointly studying exposure and course structure, not particular components.

Retitle RQ2 as `Exposure and Objective-Aligned Structural Diagnostics`. Explicitly
separate:

- Figure 1: validation-only descriptive motivation;
- Figure 3: frozen held-out-test post-hoc audit;
- RQ3: component-level interventions.

Replace `model-neutral pedagogical-risk audit` with `post-hoc objective-aligned
structural-proxy audit` and state that no causal pedagogical or learning-outcome
claim is supported.

- [ ] **Step 4: Run the focused regression suite**

```powershell
.\py.bat -m pytest `
  tests\test_p1_checkpoint_export_entrypoints.py `
  tests\test_validation_motivation_analysis.py `
  tests\test_draw_validation_motivation.py `
  tests\test_method_motivation.py `
  tests\test_p1_topk_motivation_analysis.py `
  tests\test_draw_p1_topk_motivation.py `
  -q --basetemp=.pytest_tmp\validation_motivation_final
```

Expected: all tests pass.

- [ ] **Step 5: Compile the paper**

From `paper_aaai27`, run the repository's existing `latexmk` XeLaTeX recipe for
`main.tex`. Expected: exit code zero.

- [ ] **Step 6: Inspect the final PDF page**

Render the Figure 1 page with `pdftoppm -png -r 180`. Verify no clipping, overlap,
unreadable text, or accidental reference to test data in the Figure 1 caption.

- [ ] **Step 7: Check the LaTeX log**

Search `main.log` for `Undefined control sequence`, undefined citations/references,
and `Overfull \\hbox`. Expected: no matches attributable to this change.

- [ ] **Step 8: Commit the manuscript revision**

```powershell
git add paper_aaai27/main.tex tests/test_method_motivation.py paper_aaai27/figures/mooccube_validation_motivation.pdf paper_aaai27/figures/mooccube_validation_motivation.svg paper_aaai27/figures/mooccube_validation_motivation.png paper_aaai27/figures/mooccube_validation_motivation_summary.csv paper_aaai27/figures/mooccube_validation_motivation_manifest.json
git commit -m "docs: replace Figure 1 with validation motivation"
```

## Execution Note

Execute inline in the existing workspace. Do not reset, clean, or rewrite unrelated
user changes. Runtime JSONL exports remain uncommitted; commit only source, tests,
paper-ready figure artifacts, compact summary/provenance files, and manuscript text.
