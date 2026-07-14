# Validation Inference-Policy Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate static, PPO, greedy-similarity, course-fit, and fixed-random inference on the strict-cold validation folds of the three frozen main-table checkpoints.

**Architecture:** Extend the existing read-only Actor A/B wrapper with a validation-target split adapter and policy installer aliases. Add a dedicated runner that replays the exact main-table config for each policy and a reporter that ranks policies only by validation cold item-macro NDCG@10 with Recall@10 tie-breaking.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell, existing recovered model and full-ranking evaluator.

---

### Task 1: Validation target routing

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write the failing split-routing test**

```python
import pandas as pd


def test_validation_target_routes_validation_rows_without_test_rows():
    train = pd.DataFrame({"split": ["train"]})
    val = pd.DataFrame({"split": ["val"]})
    test = pd.DataFrame({"split": ["test"]})

    def split_fn(_):
        return train, val, test, {"test_rows": 1, "val_rows": 1}

    wrapped = ab.make_validation_target_split(split_fn)
    got_train, got_val, got_eval, info = wrapped(object())

    assert got_train.equals(train)
    assert got_val.equals(val)
    assert got_eval.equals(val)
    assert not got_eval.equals(test)
    assert info == {"test_rows": 1, "val_rows": 1}
```

- [ ] **Step 2: Run RED**

Run: `.\py.bat -m pytest tests\test_main_checkpoint_actor_inference_ab.py::test_validation_target_routes_validation_rows_without_test_rows -q --basetemp .pytest_tmp\validation_policy_red`

Expected: FAIL because `make_validation_target_split` is missing.

- [ ] **Step 3: Implement the split adapter and install it only when requested**

```python
ORIGINAL_STATIC_SPLIT = legacy._static_split_df


def make_validation_target_split(split_fn):
    def validation_target(df):
        train, val, test, info = split_fn(df)
        return train, val, val.copy(), info
    return validation_target


def install_evaluation_target(target):
    target = str(target).strip().lower()
    if target == "test":
        legacy._static_split_df = ORIGINAL_STATIC_SPLIT
    elif target == "validation":
        legacy._static_split_df = make_validation_target_split(ORIGINAL_STATIC_SPLIT)
    else:
        raise ValueError("USIM_ACTOR_EVAL_TARGET must be test or validation")
```

Keep split metadata byte-for-byte unchanged so checkpoint provenance remains valid. Add `evaluation_target` to `actor_inference_audit.json` and post-annotate the output manifest after evaluation; do not put routing-only fields into the checkpoint split fingerprint.

- [ ] **Step 4: Run GREEN and the existing wrapper tests**

Run: `.\py.bat -m pytest tests\test_main_checkpoint_actor_inference_ab.py -q --basetemp .pytest_tmp\validation_policy_green`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add main_checkpoint_actor_inference_ab.py tests/test_main_checkpoint_actor_inference_ab.py
git commit -m "feat: route inference probe to validation folds"
```

### Task 2: Five policy modes

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write failing policy-installation tests**

```python
@pytest.mark.parametrize(
    "mode,rollout,uses_refiner,uses_argmax",
    [
        ("static", "ppo", False, False),
        ("ppo", "ppo", True, True),
        ("greedy_similarity", "greedy_similarity", True, False),
        ("course_fit", "course_fit", True, False),
        ("random", "random", True, False),
    ],
)
def test_install_policy_mode(monkeypatch, mode, rollout, uses_refiner, uses_argmax):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)
    ab.install_mode(mode, eval_seed=7001)
    assert ab.INFERENCE_ROLLOUT_POLICY == rollout
    assert hasattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors") is uses_refiner
    assert (legacy.FixedSimpleAC.get_action_value is ab.deterministic_get_action_value) is uses_argmax
```

- [ ] **Step 2: Run RED**

Run: `.\py.bat -m pytest tests\test_main_checkpoint_actor_inference_ab.py::test_install_policy_mode -q --basetemp .pytest_tmp\policy_modes_red`

Expected: FAIL for unsupported modes.

- [ ] **Step 3: Implement policy aliases**

```python
POLICY_MODES = {
    "actor": "ppo",
    "ppo": "ppo",
    "greedy_similarity": "greedy_similarity",
    "course_fit": "course_fit",
    "random": "random",
}
INFERENCE_ROLLOUT_POLICY = "ppo"


def install_mode(mode, eval_seed):
    global INFERENCE_ROLLOUT_POLICY
    reset_audit()
    set_eval_seed(eval_seed)
    mode = str(mode).strip().lower()
    legacy.evaluate_usim = evaluate_with_bank_targets
    eval_core.build_eval_pos_item_vecs = bank_aligned_pos_item_vecs
    legacy.FixedSimpleAC.get_action_value = ORIGINAL_GET_ACTION_VALUE
    if mode == "static":
        INFERENCE_ROLLOUT_POLICY = "ppo"
        current = getattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", None)
        if current is infer_actor_refined_item_vectors:
            delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors")
        return
    if mode not in POLICY_MODES:
        raise ValueError(f"Unsupported inference policy: {mode}")
    rollout = POLICY_MODES[mode]
    INFERENCE_ROLLOUT_POLICY = rollout
    legacy.FixedSimpleAC.get_action_value = (
        deterministic_get_action_value if rollout == "ppo" else ORIGINAL_GET_ACTION_VALUE
    )
    legacy.Fast3FeedbackUSIM.infer_refined_item_vectors = infer_actor_refined_item_vectors
```

Inside `infer_actor_refined_item_vectors`, temporarily assign `self.cfg.rollout_policy = INFERENCE_ROLLOUT_POLICY` only around `run_usim_episode`, then restore the checkpoint configuration in `finally`. This keeps the training fingerprint at `rollout_policy=ppo` for every arm.

- [ ] **Step 4: Run GREEN and full focused tests**

Run: `.\py.bat -m pytest tests\test_main_checkpoint_actor_inference_ab.py tests\test_legacy_ppo_eval_probe.py -q --basetemp .pytest_tmp\policy_modes_green`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add main_checkpoint_actor_inference_ab.py tests/test_main_checkpoint_actor_inference_ab.py
git commit -m "feat: add validation inference policy modes"
```

### Task 3: Cold item-macro policy reporter

**Files:**
- Create: `tests/test_validation_inference_policy_report.py`
- Create: `validation_inference_policy_report.py`

- [ ] **Step 1: Write the failing ranking test**

```python
import pandas as pd

import validation_inference_policy_report as report


def test_rank_policies_uses_n10_then_r10_only():
    rows = pd.DataFrame([
        {"policy": "ppo", "seed": 2025, "N@10": 0.22, "R@10": 0.30},
        {"policy": "ppo", "seed": 2026, "N@10": 0.20, "R@10": 0.28},
        {"policy": "greedy_similarity", "seed": 2025, "N@10": 0.21, "R@10": 0.40},
        {"policy": "greedy_similarity", "seed": 2026, "N@10": 0.20, "R@10": 0.41},
    ])
    ranking = report.rank_policies(rows)
    assert ranking.iloc[0]["policy"] == "ppo"
    assert ranking.iloc[0]["mean_N@10"] == 0.21
```

- [ ] **Step 2: Run RED**

Run: `.\py.bat -m pytest tests\test_validation_inference_policy_report.py -q --basetemp .pytest_tmp\policy_report_red`

Expected: FAIL because the report module is missing.

- [ ] **Step 3: Implement the report functions and CLI**

```python
import argparse
from pathlib import Path
import pandas as pd

POLICIES = ["static", "ppo", "greedy_similarity", "course_fit", "random"]
METRIC_COLUMNS = {
    "R@5": "full_cold_item_macro_r5",
    "R@10": "full_cold_item_macro_r10",
    "R@20": "full_cold_item_macro_r20",
    "N@5": "full_cold_item_macro_n5",
    "N@10": "full_cold_item_macro_n10",
    "N@20": "full_cold_item_macro_n20",
}


def rank_policies(rows):
    summary = rows.groupby("policy", as_index=False).agg(
        seeds=("seed", "nunique"),
        **{f"mean_{m}": (m, "mean") for m in METRIC_COLUMNS},
        **{f"std_{m}": (m, "std") for m in METRIC_COLUMNS},
    )
    return summary.sort_values(
        ["mean_N@10", "mean_R@10"], ascending=[False, False], ignore_index=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for policy in POLICIES:
        for seed in args.seeds:
            path = root / policy / f"strict_item_cold_balanced_thr1_seed_{seed}" / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
            if not path.exists():
                continue
            raw = pd.read_csv(path).iloc[0]
            row = {"policy": policy, "seed": seed}
            row.update({metric: float(raw[column]) for metric, column in METRIC_COLUMNS.items()})
            rows.append(row)
    details = pd.DataFrame(rows)
    if details.empty:
        raise SystemExit("No validation policy results found")
    ranking = rank_policies(details)
    details.to_csv(root / "validation_policy_by_seed.csv", index=False)
    ranking.to_csv(root / "validation_policy_ranking.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run GREEN**

Run: `.\py.bat -m pytest tests\test_validation_inference_policy_report.py -q --basetemp .pytest_tmp\policy_report_green`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add validation_inference_policy_report.py tests/test_validation_inference_policy_report.py
git commit -m "feat: rank validation inference policies"
```

### Task 4: Serial validation policy runner

**Files:**
- Create: `run_validation_inference_policy_screen.ps1`

- [ ] **Step 1: Create the runner**

```powershell
param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027)
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$checkpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_candidate"
$outputRoot = "outputs\recppo_research_repair\validation_inference_policy_screen"
$policies = @("static", "ppo", "greedy_similarity", "course_fit", "random")
$completed = @()
try {
    $env:USIM_ACTOR_EVAL_TARGET = "validation"
    foreach ($seed in $Seeds) {
        $tag = "strict_item_cold_balanced_thr1_seed_$seed"
        if (-not (Test-Path (Join-Path $checkpointRoot "$tag\finished.pt"))) {
            Write-Host "SKIP seed=${seed}: finished.pt is not available"
            continue
        }
        foreach ($policy in $policies) {
            $policyRoot = Join-Path $outputRoot $policy
            $final = Join-Path $policyRoot "$tag\final_fullrank_usim_feedback_fast3_content_delta_static.csv"
            $audit = Join-Path $policyRoot "$tag\actor_inference_audit.json"
            if ((Test-Path $final) -and (Test-Path $audit)) {
                Write-Host "SKIP completed seed=$seed policy=$policy"
                continue
            }
            $env:USIM_ACTOR_INFERENCE_MODE = $policy
            $env:USIM_ACTOR_INFERENCE_SEED = "7001"
            & $runner `
                -ScriptPath "main_checkpoint_actor_inference_ab.py" `
                -Protocol strict_item_cold_balanced -ColdThresholds @(1) -Seeds @($seed) `
                -Epochs 60 -Patience 60 -EarlyStopAverageMode item_macro -EarlyStopScoreMode cold_only `
                -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode batch_random `
                -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
                -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
                -CourseRedundantW 0.02 -CourseRedundantMode concept -CourseSampleBeta 0.20 `
                -TrainForceCold $true -UsimSteps 5 -PpoLossWeight 1.0 -RolloutPolicy ppo -RlResidualScale 1.0 `
                -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
                -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
                -UseUsimRefinedEval $true -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
                -OutputRoot $policyRoot -CheckpointRoot $checkpointRoot `
                -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate
            if ($LASTEXITCODE -ne 0) { throw "Validation policy failed: seed=$seed policy=$policy" }
        }
        $completed += $seed
    }
    if ($completed.Count) {
        & .\py.bat validation_inference_policy_report.py --root $outputRoot --seeds $completed
        if ($LASTEXITCODE -ne 0) { throw "Validation policy report failed" }
    }
}
finally {
    Remove-Item Env:USIM_ACTOR_EVAL_TARGET -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:USIM_ACTOR_INFERENCE_SEED -ErrorAction SilentlyContinue
}
```

- [ ] **Step 2: Parse-check the runner**

Run the PowerShell parser over `run_validation_inference_policy_screen.ps1`.

Expected: zero syntax errors.

- [ ] **Step 3: Dry-run an unavailable seed**

Run: `& .\run_validation_inference_policy_screen.ps1 -Seeds @(9999)`

Expected: the runner prints a finished-checkpoint skip and exits 0.

- [ ] **Step 4: Commit**

```powershell
git add run_validation_inference_policy_screen.ps1
git commit -m "feat: run validation inference policy screen"
```

### Task 5: Verification and experiment

**Files:**
- Verify all files above.
- Generate only: `outputs/recppo_research_repair/validation_inference_policy_screen/`.

- [ ] **Step 1: Run focused regression tests**

Run all wrapper, report, and legacy probe tests with a workspace-local pytest base directory.

- [ ] **Step 2: Run the three-seed serial screen**

Run: `& .\run_validation_inference_policy_screen.ps1`

- [ ] **Step 3: Verify validation-only provenance**

Confirm every arm has 34 cold target courses, audit `evaluation_target=validation`, static has zero episode calls, and rollout modes have non-zero episode calls.

- [ ] **Step 4: Read the ranking table**

Report policy mean/std for cold item-macro @5/@10/@20, select by mean N@10 then mean R@10, and state that this frozen-checkpoint result does not prove training-policy causality.
