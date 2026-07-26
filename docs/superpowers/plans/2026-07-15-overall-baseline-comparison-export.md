# Overall Baseline Comparison Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible exporter for Overall Item-Macro comparisons of the ten eligible main-table baselines and CKG-RL across MOOCCube, Junyi, and COCO.

**Architecture:** A focused Python exporter will reuse the locked main-table artifact mappings, normalize each source into seed-level cold/hot course-macro records, reconstruct Overall Item-Macro by target-course counts, and validate direct overall values when present. Pure calculation and summarization functions remain independently testable; the CLI writes CSV and formatted XLSX artifacts plus a coverage audit that retains PCGNN as unavailable.

**Tech Stack:** Python 3.12, pandas, NumPy, openpyxl, pytest, existing main-table audit modules.

---

## File Structure

- Create `paper_aaai27/scripts/export_overall_baseline_comparison.py`: source loading, validation, aggregation, ranking, CSV/XLSX writing, and CLI.
- Create `tests/test_export_overall_baseline_comparison.py`: unit and end-to-end tests for arithmetic, coverage, schemas, and workbook output.
- Generate `paper_aaai27/figures/overall_baseline_comparison/overall_baseline_comparison.xlsx`.
- Generate `paper_aaai27/figures/overall_baseline_comparison/overall_baseline_summary.csv`.
- Generate `paper_aaai27/figures/overall_baseline_comparison/overall_baseline_seed_detail.csv`.
- Generate `paper_aaai27/figures/overall_baseline_comparison/overall_baseline_wide.csv`.
- Generate `paper_aaai27/figures/overall_baseline_comparison/overall_baseline_coverage.csv`.

### Task 1: Overall reconstruction and direct-value validation

**Files:**
- Create: `tests/test_export_overall_baseline_comparison.py`
- Create: `paper_aaai27/scripts/export_overall_baseline_comparison.py`

- [ ] **Step 1: Write failing arithmetic tests**

Add tests that import the exporter by path and require count-weighted reconstruction and direct-value consistency:

```python
def test_weighted_overall_uses_course_counts() -> None:
    value = MODULE.weighted_overall(0.2, 2, 0.8, 8)
    assert value == pytest.approx(0.68)
    assert value != pytest.approx(0.5)


def test_validate_direct_overall_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="direct overall mismatch"):
        MODULE.validate_direct_overall(0.2, 0.25, tolerance=1e-8)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
.\py.bat -m pytest tests\test_export_overall_baseline_comparison.py -q --basetemp .pytest_tmp\overall_export_red1
```

Expected: import failure because `export_overall_baseline_comparison.py` does not exist.

- [ ] **Step 3: Implement the minimal arithmetic API**

Create the exporter with:

```python
METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")


def weighted_overall(cold: float, cold_count: int, hot: float, hot_count: int) -> float:
    values = (cold, hot)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("cold and hot metrics must be finite")
    if cold_count <= 0 or hot_count <= 0:
        raise ValueError("cold and hot course counts must be positive")
    return (float(cold) * cold_count + float(hot) * hot_count) / (cold_count + hot_count)


def validate_direct_overall(reconstructed: float, direct: float, tolerance: float = 5e-5) -> None:
    if not math.isfinite(float(direct)) or abs(float(reconstructed) - float(direct)) > tolerance:
        raise ValueError(
            f"direct overall mismatch: reconstructed={reconstructed:.12g}, direct={direct:.12g}"
        )
```

- [ ] **Step 4: Run GREEN**

Run the focused test command and expect both tests to pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add paper_aaai27/scripts/export_overall_baseline_comparison.py tests/test_export_overall_baseline_comparison.py
git commit -m "test: define overall baseline aggregation"
```

### Task 2: Normalize core, SEMCo, KGRec, and PCGNN artifacts

**Files:**
- Modify: `tests/test_export_overall_baseline_comparison.py`
- Modify: `paper_aaai27/scripts/export_overall_baseline_comparison.py`

- [ ] **Step 1: Write failing source-normalization tests**

Add synthetic tests for per-item sources, direct report sources, and unavailable PCGNN coverage:

```python
def test_build_seed_row_reconstructs_all_metrics() -> None:
    row = MODULE.build_seed_row(
        dataset="Toy",
        method="Baseline",
        seed=2025,
        cold={metric: 0.2 for metric in MODULE.METRICS},
        hot={metric: 0.8 for metric in MODULE.METRICS},
        cold_count=2,
        hot_count=8,
        cold_source="cold.csv",
        hot_source="hot.csv",
        direct=None,
    )
    assert row["status"] == "ready"
    assert row["R@10"] == pytest.approx(0.68)


def test_unavailable_row_contains_no_numeric_metrics() -> None:
    row = MODULE.unavailable_row("MOOCCube", "PCGNN", 2025, "missing warm targets")
    assert row["status"] == "unavailable_missing_warm_targets"
    assert all(pd.isna(row[metric]) for metric in MODULE.METRICS)
```

- [ ] **Step 2: Run RED**

Run the focused tests and expect missing-function failures.

- [ ] **Step 3: Implement normalized row builders and source collectors**

Implement:

```python
def build_seed_row(
    dataset: str,
    method: str,
    seed: int,
    cold: dict[str, float],
    hot: dict[str, float],
    cold_count: int,
    hot_count: int,
    cold_source: str,
    hot_source: str,
    direct: dict[str, float] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "status": "ready",
        "cold_count": cold_count,
        "hot_count": hot_count,
        "cold_source": cold_source,
        "hot_source": hot_source,
    }
    for metric in METRICS:
        row[metric] = weighted_overall(cold[metric], cold_count, hot[metric], hot_count)
        if direct and metric in direct:
            validate_direct_overall(row[metric], direct[metric])
    return row


def unavailable_row(dataset: str, method: str, seed: int, reason: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "status": "unavailable_missing_warm_targets",
        "reason": reason,
        **{metric: math.nan for metric in METRICS},
    }
```

Collectors must:

- import `audit_significance_inputs.py` and `export_warm_target_table.py` by adding `paper_aaai27/scripts` to `sys.path`;
- pair each core cold spec with the warm spec of the same dataset, method, and seed;
- prefer per-item CSV means and counts, falling back to result JSON metrics;
- add SEMCo using `semco_specs()`;
- read KGRec report paths from the locked `main_table_summary.json`, using `test.full_all_item_macro` and verifying the cold/hot reconstruction;
- emit PCGNN unavailable rows when hot count is zero or the hot metric block is empty.

- [ ] **Step 4: Run GREEN**

Run the focused tests and expect all normalization tests to pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add paper_aaai27/scripts/export_overall_baseline_comparison.py tests/test_export_overall_baseline_comparison.py
git commit -m "feat: collect overall baseline artifacts"
```

### Task 3: Three-seed summary, ranks, improvements, and wide table

**Files:**
- Modify: `tests/test_export_overall_baseline_comparison.py`
- Modify: `paper_aaai27/scripts/export_overall_baseline_comparison.py`

- [ ] **Step 1: Write failing summary tests**

```python
def test_summarize_adds_sample_std_rank_and_ckg_improvement() -> None:
    rows = []
    for seed, ours_value in zip((2025, 2026, 2027), (0.21, 0.22, 0.23)):
        rows.append(
            {
                "dataset": "Toy",
                "method": "Baseline",
                "seed": seed,
                "status": "ready",
                **{metric: 0.2 for metric in MODULE.METRICS},
            }
        )
        rows.append(
            {
                "dataset": "Toy",
                "method": "CKG-RL",
                "seed": seed,
                "status": "ready",
                **{metric: ours_value for metric in MODULE.METRICS},
            }
        )
    detail = pd.DataFrame(rows)
    summary = MODULE.summarize_ready(detail)
    ours = summary[(summary.dataset == "Toy") & (summary.method == "CKG-RL")].iloc[0]
    assert ours["R@10_std"] == pytest.approx(0.01)
    assert ours["R@10_rank"] == 1
    assert ours["R@10_relative_improvement"] == pytest.approx(0.1)


def test_summarize_rejects_missing_seed() -> None:
    detail = pd.DataFrame(
        [
            {
                "dataset": "Toy",
                "method": "Baseline",
                "seed": seed,
                "status": "ready",
                **{metric: 0.2 for metric in MODULE.METRICS},
            }
            for seed in (2025, 2026)
        ]
    )
    with pytest.raises(ValueError, match="requires seeds 2025, 2026, 2027"):
        MODULE.summarize_ready(detail)
```

- [ ] **Step 2: Run RED**

Run the focused tests and expect missing summary-function failures.

- [ ] **Step 3: Implement summary and wide reshaping**

Implement `summarize_ready(detail)`, using sample standard deviation (`ddof=1`), descending ranks with `method="min"`, and strongest-baseline comparisons that exclude CKG-RL. Implement `build_wide(summary)` with method rows and dataset-metric columns for R@5, R@10, N@5, and N@10, preserving the main-table method order.

- [ ] **Step 4: Run GREEN**

Run the focused tests and expect summary tests to pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add paper_aaai27/scripts/export_overall_baseline_comparison.py tests/test_export_overall_baseline_comparison.py
git commit -m "feat: summarize overall baseline comparison"
```

### Task 4: CSV and workbook export

**Files:**
- Modify: `tests/test_export_overall_baseline_comparison.py`
- Modify: `paper_aaai27/scripts/export_overall_baseline_comparison.py`

- [ ] **Step 1: Write failing output-schema test**

```python
def test_write_outputs_creates_required_workbook_and_csvs(tmp_path: Path) -> None:
    paths = MODULE.write_outputs(detail, summary, coverage, tmp_path)
    assert {path.name for path in paths} == {
        "overall_baseline_comparison.xlsx",
        "overall_baseline_summary.csv",
        "overall_baseline_seed_detail.csv",
        "overall_baseline_wide.csv",
        "overall_baseline_coverage.csv",
    }
    workbook = openpyxl.load_workbook(tmp_path / "overall_baseline_comparison.xlsx", data_only=False)
    assert workbook.sheetnames == ["Summary", "Seed_Detail", "Coverage"]
    assert workbook["Summary"].freeze_panes == "A2"
```

- [ ] **Step 2: Run RED**

Run the focused test and expect a missing `write_outputs` failure.

- [ ] **Step 3: Implement writers and formatting**

Use pandas `ExcelWriter(engine="openpyxl")`, then apply Arial 10-point font, bold dark-blue headers, frozen header rows, filters, sensible column widths, and `0.0000` formatting to metric cells. Write values only; do not create formulas.

- [ ] **Step 4: Run GREEN**

Run the focused tests and load all generated files back with pandas/openpyxl to ensure schemas are readable.

- [ ] **Step 5: Commit Task 4**

```powershell
git add paper_aaai27/scripts/export_overall_baseline_comparison.py tests/test_export_overall_baseline_comparison.py
git commit -m "feat: export overall comparison workbook"
```

### Task 5: End-to-end artifact generation and verification

**Files:**
- Generate: `paper_aaai27/figures/overall_baseline_comparison/*`
- Modify only if verification exposes a tested defect: exporter and test files above.

- [ ] **Step 1: Add the end-to-end coverage test**

Require 108 dataset-method-seed coverage rows: 99 ready rows for ten eligible baselines plus CKG-RL, and nine unavailable PCGNN rows. Assert that the ready summary contains 33 dataset-method rows, representing 11 numeric methods across three datasets, and that PCGNN is the sole unavailable method for every dataset and seed.

- [ ] **Step 2: Run the full focused test module**

```powershell
.\py.bat -m pytest tests\test_export_overall_baseline_comparison.py -q --basetemp .pytest_tmp\overall_export_final
```

Expected: all tests pass.

- [ ] **Step 3: Run the exporter**

```powershell
.\py.bat paper_aaai27\scripts\export_overall_baseline_comparison.py
```

Expected: five output paths printed and no coverage or consistency exception.

- [ ] **Step 4: Verify generated artifacts**

Run a read-back audit that checks:

- workbook sheets are exactly `Summary`, `Seed_Detail`, and `Coverage`;
- no Excel error strings occur;
- summary has 33 ready dataset-method rows for ten baselines plus CKG-RL;
- coverage contains PCGNN as the only unavailable method;
- every ready method has seeds 2025, 2026, and 2027;
- CSV and workbook Summary values agree within `1e-12`;
- KGRec direct values match reconstructed values within tolerance.

- [ ] **Step 5: Run relevant regression tests**

```powershell
.\py.bat -m pytest tests\test_export_overall_baseline_comparison.py tests\test_build_recppo_mooccube_significance.py -q --basetemp .pytest_tmp\overall_export_regression
```

Expected: all tests pass.

- [ ] **Step 6: Commit implementation and generated artifacts**

```powershell
git add paper_aaai27/scripts/export_overall_baseline_comparison.py tests/test_export_overall_baseline_comparison.py paper_aaai27/figures/overall_baseline_comparison
git commit -m "feat: export overall baseline comparison"
```

## Plan Self-Review

- Spec coverage: source locking, Overall Item-Macro, ten eligible baselines, CKG-RL comparison, PCGNN coverage, six metrics, CSV/XLSX outputs, ranks, relative improvements, and verification are assigned to explicit tasks.
- Placeholder scan: no placeholder token or undefined follow-up step remains.
- Type consistency: all calculations use metric dictionaries keyed by `METRICS`; detail, summary, coverage, and wide frames have distinct documented roles.
