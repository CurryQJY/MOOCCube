# Official USIM Static Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the NeurIPS 2024 official USIM implementation from `USIM-main` on the current strict item-cold MOOCCube split.

**Architecture:** Keep the official repository code unchanged and add a project-level adapter. The adapter imports `USIM-main/warm_model/bprmf.py` and `USIM-main/cold_model/USIM.py`, trains the official BPRMF warm backbone, trains the official content mapper, runs official RL buffer rollouts, and evaluates with the existing full-ranking item-macro evaluator.

**Tech Stack:** Python, PyTorch, pandas, pytest, existing `hin_data_common.py` and `hin_eval_common.py`.

---

### Task 1: Adapter Helper Tests

**Files:**
- Create: `tests/test_usim_official_static_hin.py`
- Create later: `usim_official_static_hin.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
import json
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usim_official_static_hin import (
    build_official_rl_records,
    cold_item_ids_from_splits,
    load_split_cold_threshold,
)


def test_load_split_cold_threshold_prefers_static_summary(tmp_path):
    (tmp_path / "static_split_summary.json").write_text(
        json.dumps({"cold_threshold": 1}),
        encoding="utf-8",
    )

    assert load_split_cold_threshold(str(tmp_path), fallback=5) == 1


def test_cold_item_ids_from_splits_uses_threshold_and_unique_items():
    train_df = pd.DataFrame({"u_idx": [0, 1], "i_idx": [10, 11], "popularity": [3, 2]})
    val_df = pd.DataFrame({"u_idx": [2, 3], "i_idx": [12, 13], "popularity": [0, 2]})
    test_df = pd.DataFrame({"u_idx": [4, 5, 6], "i_idx": [12, 14, 14], "popularity": [0, 0, 0]})

    cold = cold_item_ids_from_splits(train_df, val_df, test_df, cold_threshold=1)

    assert cold.tolist() == [12, 14]


def test_build_official_rl_records_groups_users_and_excludes_cold_items():
    train_df = pd.DataFrame(
        {
            "u_idx": [0, 1, 2, 3],
            "i_idx": [10, 10, 11, 12],
            "popularity": [2, 2, 2, 0],
        }
    )
    content = torch.arange(13 * 3, dtype=torch.float32).view(13, 3)

    records = build_official_rl_records(train_df, content, excluded_item_ids=torch.tensor([12]))

    assert [r["item"] for r in records] == [10, 11]
    assert records[0]["user"] == [0, 1]
    assert records[1]["user"] == [2]
    assert torch.equal(records[0]["item_content"], content[10])
```

- [ ] **Step 2: Run tests to verify RED**

Run: `.\py.bat -m pytest tests/test_usim_official_static_hin.py -q`

Expected: import fails because `usim_official_static_hin.py` does not exist yet.

### Task 2: Official Runner Implementation

**Files:**
- Create: `usim_official_static_hin.py`

- [ ] **Step 1: Implement helpers and runner**

Add:
- `load_split_cold_threshold(split_dir, fallback)`
- `cold_item_ids_from_splits(train_df, val_df, test_df, cold_threshold)`
- `build_official_rl_records(train_df, content_emb, excluded_item_ids)`
- official import path setup for `USIM-main`
- BPRMF backbone training using official `BPRMF.calculate_loss`
- official `model.content_mapper` MSE pretraining
- official RL loop using `model.update_buffer()` and `model.optimize()`
- full-ranking evaluation with `evaluate_embedding_ranker(..., average_mode="item_macro")`

- [ ] **Step 2: Run tests to verify GREEN**

Run: `.\py.bat -m pytest tests/test_usim_official_static_hin.py -q`

Expected: all tests pass.

### Task 3: Smoke Execution

**Files:**
- No new files required unless output metrics are produced under the configured output directory.

- [ ] **Step 1: Run small smoke**

Run:

```powershell
$env:USIM_DATA_DIR='processed_data_hin_clean_pop5'
$env:USIM_STATIC_SPLIT_DIR='outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025'
$env:USIM_BASELINE_OUTPUT_DIR='outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/main_table_balanced_itemmacro_usim_official_smoke'
$env:USIM_OFFICIAL_BACKBONE_EPOCHS='1'
$env:USIM_OFFICIAL_MLP_EPOCHS='1'
$env:USIM_OFFICIAL_RL_EPOCHS='1'
$env:USIM_OFFICIAL_RL_BATCH_SIZE='8'
$env:USIM_OFFICIAL_MAX_RL_BATCHES='1'
$env:USIM_OFFICIAL_EVAL_BATCH_SIZE='2048'
.\py.bat -B usim_official_static_hin.py
```

Expected: official imports load, one BPR epoch runs, one mapper epoch runs, one RL batch runs, and a cold full-ranking item-macro report is written.
