# KGRec Quick Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add epoch-0 validation and named KGRec loss diagnostics, then run a three-learning-rate seed-2025 strict-cold screen without overwriting formal results.

**Architecture:** Keep GPU model construction and evaluation in the existing runner. Add two pure helpers for averaging named loss dictionaries and applying validation/best-checkpoint state to progress rows, which makes the new behavior unit-testable without a GPU. Run the three configurations serially into isolated directories and select only from validation trajectories.

**Tech Stack:** Python 3.12, PyTorch 2.8, NumPy, pytest, PowerShell, CUDA.

---

### Task 1: Add failing tests for diagnostic helpers

**Files:**
- Modify: `tests/test_kgrec_strict_runner.py`
- Test: `tests/test_kgrec_strict_runner.py`

- [ ] **Step 1: Write failing tests for component averaging and validation state**

Add imports for `average_loss_components` and `apply_validation_result`, then add tests equivalent to:

```python
def test_average_loss_components_averages_named_batch_losses() -> None:
    averaged = average_loss_components([
        {"rec_loss": 2.0, "mae_loss": 0.4, "cl_loss": 0.2},
        {"rec_loss": 4.0, "mae_loss": 0.2, "cl_loss": 0.6},
    ])
    assert averaged == {"rec_loss": 3.0, "mae_loss": 0.3, "cl_loss": 0.4}


def test_apply_validation_result_supports_epoch_zero_as_best() -> None:
    row = {"epoch": 0}
    validation = {"full_cold_item_macro": {"N@10": 0.25}}
    best_score, bad_epochs, is_best = apply_validation_result(
        row=row,
        validation=validation,
        best_score=float("-inf"),
        bad_epochs=0,
    )
    assert is_best
    assert best_score == 0.25
    assert bad_epochs == 0
    assert row == {
        "epoch": 0,
        "validation_full_cold_item_macro": {"N@10": 0.25},
        "validation_score": 0.25,
        "best": True,
        "bad_epochs": 0,
    }
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
D:\Anaconda3\envs\zw\python.exe -m pytest tests/test_kgrec_strict_runner.py -q --basetemp D:\DeskTop\MOOCCube\outputs\kgrec_diag_tdd_red
```

Expected: collection fails because the two helpers do not exist.

### Task 2: Implement pure diagnostic helpers

**Files:**
- Modify: `paper_aaai27/scripts/run_kgrec_strict_seed.py`
- Test: `tests/test_kgrec_strict_runner.py`

- [ ] **Step 1: Implement named loss averaging**

Add a pure helper near the existing evaluation helpers:

```python
def average_loss_components(loss_rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not loss_rows:
        return {}
    keys = sorted({key for row in loss_rows for key in row})
    return {
        key: float(np.mean([float(row[key]) for row in loss_rows if key in row]))
        for key in keys
    }
```

- [ ] **Step 2: Implement validation state application**

Add:

```python
def apply_validation_result(
    *,
    row: dict[str, object],
    validation: Mapping[str, object],
    best_score: float,
    bad_epochs: int,
) -> tuple[float, int, bool]:
    score = float(validation["full_cold_item_macro"]["N@10"])
    is_best = score > best_score
    row["validation_full_cold_item_macro"] = validation["full_cold_item_macro"]
    row["validation_score"] = score
    row["best"] = is_best
    if is_best:
        best_score = score
        bad_epochs = 0
    else:
        bad_epochs += 1
    row["bad_epochs"] = bad_epochs
    return best_score, bad_epochs, is_best
```

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests in `test_kgrec_strict_runner.py` pass.

### Task 3: Integrate epoch-0 validation and component logging

**Files:**
- Modify: `paper_aaai27/scripts/run_kgrec_strict_seed.py`
- Test: `tests/test_kgrec_strict_runner.py`

- [ ] **Step 1: Evaluate and checkpoint epoch 0**

Immediately after opening `training_progress.jsonl`, evaluate validation before the training loop, create `{"epoch": 0}`, call `apply_validation_result`, save `best_model.pt` when best, store `best_validation`, and write the row. Do not add training-only fields to this row.

- [ ] **Step 2: Record mean component losses for trained epochs**

Collect each model-returned loss dictionary in the batch loop:

```python
loss_component_rows: list[dict[str, float]] = []
loss, loss_dict = model(batch)
loss_component_rows.append({key: float(value) for key, value in loss_dict.items()})
```

Add `train_loss_components = average_loss_components(loss_component_rows)` to every trained epoch row.

- [ ] **Step 3: Reuse the validation helper for trained epochs**

Replace the duplicated validation score/best/bad-epoch update block with `apply_validation_result`. Preserve checkpoint saves, `best_epoch`, `best_validation`, progress flushing, and patience behavior.

- [ ] **Step 4: Run all KGRec unit tests**

Run:

```powershell
D:\Anaconda3\envs\zw\python.exe -m pytest tests/test_kgrec_native_scatter.py tests/test_kgrec_strict_adapter.py tests/test_kgrec_strict_runner.py -q --basetemp D:\DeskTop\MOOCCube\outputs\kgrec_diag_full_pytest
```

Expected: all tests pass.

### Task 4: Run a bounded CUDA integration smoke

**Files:**
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/diagnostic_lr_screen_seed2025/_smoke/`

- [ ] **Step 1: Run one epoch and one training batch**

Run:

```powershell
D:\Anaconda3\envs\zw\python.exe -u paper_aaai27/scripts/run_kgrec_strict_seed.py --seed 2025 --epochs 1 --patience 1 --dim 8 --context-hops 1 --max-train-batches 1 --device cuda --output-dir paper_aaai27/baseline_sources/_kgrec_strict/diagnostic_lr_screen_seed2025/_smoke
```

- [ ] **Step 2: Verify smoke artifacts**

Confirm the report is complete, progress contains epochs 0 and 1, epoch 1 contains finite `rec_loss`, `mae_loss`, and `cl_loss`, and the checkpoint exists.

### Task 5: Run the serial seed-2025 learning-rate screen

**Files:**
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/diagnostic_lr_screen_seed2025/lr_1e-4/`
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/diagnostic_lr_screen_seed2025/lr_5e-5/`
- Create: `paper_aaai27/baseline_sources/_kgrec_strict/diagnostic_lr_screen_seed2025/lr_1e-5/`

- [ ] **Step 1: Run baseline-diagnostic**

Run the strict runner with `--seed 2025 --epochs 10 --patience 4 --lr 1e-4 --device cuda` and the `lr_1e-4` output directory.

- [ ] **Step 2: Run LR-0.5**

Run the same command with `--lr 5e-5` and the `lr_5e-5` output directory.

- [ ] **Step 3: Run LR-0.1**

Run the same command with `--lr 1e-5` and the `lr_1e-5` output directory.

All three commands use the default seed-2025 atomic directory, batch size 4096, dimension 64, and two context hops. Runs are serial.

### Task 6: Verify and compare validation trajectories

**Files:**
- Read: all three `training_progress.jsonl` files
- Read: all three `kgrec_strict_adapter_report.json` files

- [ ] **Step 1: Verify artifacts and process cleanup**

For each run, require `status: complete`, `device: cuda`, a nonempty checkpoint, progress beginning at epoch 0, finite trained loss components, and no remaining KGRec process.

- [ ] **Step 2: Compare validation-only evidence**

For each learning rate report epoch-0 score, peak cold NDCG@10, peak epoch, final validation score, cold Recall@10 trajectory, and mean component-loss trajectory. Do not use test metrics to choose the preferred configuration.

- [ ] **Step 3: Select the preferred diagnostic configuration**

Prefer the configuration with the strongest validation cold NDCG@10 while using curve stability and a post-epoch-1 optimum as secondary evidence. State whether lower learning rate resolves, delays, or fails to change the epoch-1 pattern.
