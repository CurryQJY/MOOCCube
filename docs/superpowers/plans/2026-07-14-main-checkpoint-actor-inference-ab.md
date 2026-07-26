# Main-Checkpoint Actor Inference A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare current static full-ranking inference with deterministic Actor-refined cold-course inference using the exact frozen checkpoints behind the recovered main-table results.

**Architecture:** Add an evaluation-only Python wrapper around `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`; it installs deterministic Actor inference without changing the recovered source. A PowerShell runner replays the exact main-table configuration against finished checkpoints, and a separate reporter joins aggregate and per-course CSVs. Tests cover deterministic action choice, cold-only refinement, leakage guards, audit counts, provenance rejection, and report arithmetic.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell, existing `fast3_delta.eval` full-ranking evaluator.

---

## File structure

- Create `main_checkpoint_actor_inference_ab.py`: evaluation-only mode installer, deterministic Actor action, cold-course refinement, and audit export.
- Create `actor_inference_ab_report.py`: aggregate arm-level and per-course comparisons without selecting on test metrics.
- Create `run_main_checkpoint_actor_inference_ab.ps1`: exact-config checkpoint replay for available finished seeds.
- Create `tests/test_main_checkpoint_actor_inference_ab.py`: behavioral and leakage tests for the wrapper.
- Create `tests/test_actor_inference_ab_report.py`: report-join and delta tests.
- Do not modify the recovered training source, `fast3_delta/eval.py`, checkpoints, or existing main-table outputs.

### Task 1: Deterministic Actor inference primitive

**Files:**
- Create: `tests/test_main_checkpoint_actor_inference_ab.py`
- Create: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write the failing deterministic-action test**

```python
import torch

import main_checkpoint_actor_inference_ab as ab
import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy
import fast3_delta.eval as eval_core


def test_deterministic_action_uses_actor_argmax():
    torch.manual_seed(7)
    agent = legacy.FixedSimpleAC(item_dim=4, time_dim=5)
    state = torch.randn(2, 4)
    time_step = torch.zeros(2, 1)
    candidates = torch.randn(2, 3, 4)
    action, log_prob, value, entropy = ab.deterministic_get_action_value(
        agent, state, time_step, candidates
    )
    with torch.no_grad():
        t_emb = torch.nn.functional.one_hot(time_step.squeeze(1).long(), num_classes=5).float()
        feat = agent.common(torch.cat([state, t_emb], dim=1))
        query = agent.actor_head(feat).unsqueeze(1)
        keys = agent.user_proj(candidates)
        expected = torch.matmul(query, keys.transpose(1, 2)).squeeze(1).argmax(dim=-1)
    assert torch.equal(action, expected)
    assert log_prob.shape == (2,)
    assert value.shape == (2, 1)
    assert entropy.shape == (2,)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `./py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py::test_deterministic_action_uses_actor_argmax -q`

Expected: FAIL because `main_checkpoint_actor_inference_ab` does not exist.

- [ ] **Step 3: Implement the deterministic action function and audit state**

```python
from dataclasses import dataclass, asdict
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy


@dataclass
class InferenceAudit:
    actor_calls: int = 0
    episode_calls: int = 0
    refined_items: int = 0
    cosine_sum: float = 0.0
    l2_sum: float = 0.0


AUDIT = InferenceAudit()
EVAL_SEED = 7001
ACTIVE_ITEM_BANK = None
ORIGINAL_EVALUATE = legacy.evaluate_usim
ORIGINAL_BUILD_POS = eval_core.build_eval_pos_item_vecs


def deterministic_get_action_value(self, item_state, time_step, candidates_emb, action_idx=None):
    AUDIT.actor_calls += 1
    t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=self.time_dim).float()
    feat = self.common(torch.cat([item_state, t_emb], dim=1))
    value = self.critic_head(feat)
    query = self.actor_head(feat).unsqueeze(1)
    keys = self.user_proj(candidates_emb)
    logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
    dist = Categorical(logits=logits)
    if action_idx is None:
        action_idx = logits.argmax(dim=-1)
    return action_idx, dist.log_prob(action_idx), value, dist.entropy()
```

- [ ] **Step 4: Run the deterministic-action test and verify GREEN**

Run: `./py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py::test_deterministic_action_uses_actor_argmax -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit the primitive**

```powershell
git add main_checkpoint_actor_inference_ab.py tests/test_main_checkpoint_actor_inference_ab.py
git commit -m "feat: add deterministic Actor inference primitive"
```

### Task 2: Cold-only inference refinement and leakage guards

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write failing tests for no-target refinement and mode installation**

```python
class TinyInferenceModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.training = False
        self.cfg = type("Cfg", (), {"emb_dim": 2, "candidate_strategy": "retrieve_sample"})()
        self.item_id_emb = type("Emb", (), {"weight": torch.zeros(3, 2)})()
        self.item_popularity = torch.tensor([3.0, 0.0, 0.0])
        self.calls = []

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def _build_user_bank_raw(self):
        bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return bank, torch.nn.functional.normalize(bank, dim=1)

    def get_item_vector(self, idx, llm_s, force_cold, disable_id_dropout=True):
        base = torch.stack([idx.float() + 1.0, torch.ones_like(idx).float()], dim=1)
        return base, base, base

    def run_usim_episode(self, init, target_emb, **kwargs):
        self.calls.append({"target_emb": target_emb, **kwargs})
        return init + 0.25, {}, {}

    def _blend_rl_episode_output(self, base, final):
        return final


def test_refinement_never_supplies_behavior_target():
    model = TinyInferenceModel()
    out = ab.infer_actor_refined_item_vectors(model, torch.tensor([1, 2]))
    assert out.shape == (2, 2)
    assert len(model.calls) == 1
    assert model.calls[0]["target_emb"] is None
    assert torch.equal(model.calls[0]["item_idx"], torch.tensor([1, 2]))


def test_install_static_does_not_attach_refinement(monkeypatch):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)
    ab.install_mode("static", eval_seed=7001)
    assert not hasattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors")


def test_install_actor_attaches_refinement(monkeypatch):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)
    ab.install_mode("actor", eval_seed=7001)
    assert legacy.Fast3FeedbackUSIM.infer_refined_item_vectors is ab.infer_actor_refined_item_vectors


def test_positive_target_vector_comes_from_the_same_refined_bank():
    ab.ACTIVE_ITEM_BANK = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    out = ab.bank_aligned_pos_item_vecs(
        model=None,
        item_idx=torch.tensor([2, 0]),
        llm_s=None,
        pop_sel=None,
        eval_type="cold",
    )
    assert torch.equal(out, torch.tensor([[3.0, 4.0], [1.0, 0.0]]))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `./py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py -q`

Expected: FAIL because `infer_actor_refined_item_vectors` and `install_mode` are missing.

- [ ] **Step 3: Implement cold-course refinement and installation**

```python
def reset_audit():
    global AUDIT
    AUDIT = InferenceAudit()


def set_eval_seed(seed):
    global EVAL_SEED
    EVAL_SEED = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_actor_refined_item_vectors(
    self, item_idx, llm_s=None, item_batch=1024, force_cold=True,
    user_bank_raw=None, user_seen_items=None,
):
    set_eval_seed(EVAL_SEED)
    item_idx = torch.as_tensor(item_idx, dtype=torch.long, device=self.device).view(-1)
    if item_idx.numel() == 0:
        return torch.empty((0, self.cfg.emb_dim), device=self.device)
    if llm_s is None:
        llm_s = torch.full((item_idx.numel(),), -1.0, device=self.device)
    was_training = self.training
    self.eval()
    outputs = []
    bank = user_bank_raw if user_bank_raw is not None else self._build_user_bank_raw()
    history_context = user_seen_items
    if history_context is None and getattr(self, "user_seen_index", None) is not None:
        history_context = {}
    try:
        with torch.no_grad():
            for start in range(0, item_idx.numel(), max(1, int(item_batch))):
                idx = item_idx[start:start + item_batch]
                score = llm_s[start:start + item_batch]
                base, _, _ = self.get_item_vector(
                    idx, score, force_cold=force_cold, disable_id_dropout=True
                )
                pop = None
                if self.item_popularity is not None:
                    pop = self.item_popularity.to(self.device).index_select(0, idx).float()
                final, _, _ = self.run_usim_episode(
                    base, target_emb=None, user_bank_raw=bank, item_idx=idx,
                    target_pop=pop, user_seen_items=history_context,
                )
                refined = self._blend_rl_episode_output(base, final)
                cos = F.cosine_similarity(base, refined, dim=1)
                l2 = torch.linalg.vector_norm(refined - base, dim=1)
                AUDIT.episode_calls += 1
                AUDIT.refined_items += int(idx.numel())
                AUDIT.cosine_sum += float(cos.sum().item())
                AUDIT.l2_sum += float(l2.sum().item())
                outputs.append(refined.detach())
    finally:
        self.train(was_training)
    return torch.cat(outputs, dim=0)


def bank_aligned_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    if ACTIVE_ITEM_BANK is not None:
        return ACTIVE_ITEM_BANK.index_select(0, item_idx)
    return ORIGINAL_BUILD_POS(model, item_idx, llm_s, pop_sel, eval_type)


def evaluate_with_bank_targets(*args, **kwargs):
    global ACTIVE_ITEM_BANK
    previous = ACTIVE_ITEM_BANK
    banks = kwargs.get("all_item_vecs")
    eval_type = kwargs.get("eval_type", "cold")
    ACTIVE_ITEM_BANK = eval_core.select_eval_item_bank(banks, eval_type) if banks is not None else None
    try:
        return ORIGINAL_EVALUATE(*args, **kwargs)
    finally:
        ACTIVE_ITEM_BANK = previous


def install_mode(mode, eval_seed):
    reset_audit()
    set_eval_seed(eval_seed)
    legacy.evaluate_usim = evaluate_with_bank_targets
    eval_core.build_eval_pos_item_vecs = bank_aligned_pos_item_vecs
    mode = mode.strip().lower()
    if mode == "static":
        return
    if mode != "actor":
        raise ValueError("USIM_ACTOR_INFERENCE_MODE must be static or actor")
    legacy.FixedSimpleAC.get_action_value = deterministic_get_action_value
    legacy.Fast3FeedbackUSIM.infer_refined_item_vectors = infer_actor_refined_item_vectors
```

- [ ] **Step 4: Run the wrapper tests and verify GREEN**

Run: `./py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the refinement path**

```powershell
git add main_checkpoint_actor_inference_ab.py tests/test_main_checkpoint_actor_inference_ab.py
git commit -m "feat: add cold-course Actor refinement probe"
```

### Task 3: Entrypoint, audit export, provenance enforcement, and read-only checkpoint loading

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write failing tests for audit export and checkpoint-write blocking**

```python
import json
import pytest


def test_audit_export_reports_mean_displacement(tmp_path):
    ab.AUDIT = ab.InferenceAudit(
        actor_calls=10, episode_calls=2, refined_items=4,
        cosine_sum=3.2, l2_sum=0.8,
    )
    path = tmp_path / "audit.json"
    ab.write_audit(path, mode="actor", eval_seed=7001)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mean_cosine"] == pytest.approx(0.8)
    assert payload["mean_l2"] == pytest.approx(0.2)


def test_checkpoint_write_blocker_preserves_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    target = checkpoint_dir / "finished.pt"
    target.write_bytes(b"original")
    blocker = ab.make_read_only_torch_save(checkpoint_dir, real_save=torch.save)
    blocker({"changed": True}, target)
    assert target.read_bytes() == b"original"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `./py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py -q`

Expected: FAIL because audit export and strict guard are missing.

- [ ] **Step 3: Implement audit output and main entrypoint**

```python
def write_audit(path, mode, eval_seed):
    payload = asdict(AUDIT)
    count = max(1, AUDIT.refined_items)
    payload.update({
        "mode": mode,
        "eval_seed": int(eval_seed),
        "mean_cosine": AUDIT.cosine_sum / count,
        "mean_l2": AUDIT.l2_sum / count,
    })
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_read_only_torch_save(checkpoint_dir, real_save):
    root = os.path.normcase(os.path.abspath(os.fspath(checkpoint_dir)))
    def guarded_save(obj, path, *args, **kwargs):
        target = os.path.normcase(os.path.abspath(os.fspath(path)))
        try:
            inside = os.path.commonpath([root, target]) == root
        except ValueError:
            inside = False
        if inside:
            print(f">> EVAL READ-ONLY: blocked checkpoint write {target}")
            return None
        return real_save(obj, path, *args, **kwargs)
    return guarded_save


def main():
    mode = os.environ.get("USIM_ACTOR_INFERENCE_MODE", "static")
    eval_seed = int(os.environ.get("USIM_ACTOR_INFERENCE_SEED", "7001"))
    checkpoint_dir = os.environ.get("USIM_FB_CKPT_DIR", "")
    if not checkpoint_dir:
        raise RuntimeError("USIM_FB_CKPT_DIR is required for checkpoint replay")
    torch.save = make_read_only_torch_save(checkpoint_dir, real_save=torch.save)
    install_mode(mode, eval_seed)
    legacy.main()
    output = legacy._feedback_output_path("actor_inference_audit.json")
    write_audit(output, mode=mode, eval_seed=eval_seed)
    print(
        f">> ACTOR INFERENCE AUDIT: mode={mode} actor_calls={AUDIT.actor_calls} "
        f"episode_calls={AUDIT.episode_calls} refined_items={AUDIT.refined_items}"
    )


if __name__ == "__main__":
    main()
```

Use the existing provenance guard unchanged in real runs; never replace it with an unconditional acceptance override. `USIM_FB_SAVE_CKPT=1` remains enabled so the legacy runner loads the checkpoint, while the wrapper blocks every `torch.save` whose resolved path is inside the source checkpoint directory.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `./py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the entrypoint**

```powershell
git add main_checkpoint_actor_inference_ab.py tests/test_main_checkpoint_actor_inference_ab.py
git commit -m "feat: add audited Actor inference entrypoint"
```

### Task 4: Comparison reporter

**Files:**
- Create: `tests/test_actor_inference_ab_report.py`
- Create: `actor_inference_ab_report.py`

- [ ] **Step 1: Write the failing report test**

```python
import pandas as pd
import pytest

import actor_inference_ab_report as report


def test_compare_seed_joins_per_item_and_computes_deltas(tmp_path):
    static = tmp_path / "static.csv"
    actor = tmp_path / "actor.csv"
    pd.DataFrame([
        {"item_id": 1, "count": 2, "R@10": 0.25, "N@10": 0.10},
        {"item_id": 2, "count": 3, "R@10": 0.50, "N@10": 0.20},
    ]).to_csv(static, index=False)
    pd.DataFrame([
        {"item_id": 1, "count": 2, "R@10": 0.50, "N@10": 0.15},
        {"item_id": 2, "count": 3, "R@10": 0.25, "N@10": 0.30},
    ]).to_csv(actor, index=False)
    detail, summary = report.compare_per_item(static, actor, seed=2025)
    assert list(detail["item_id"]) == [1, 2]
    assert detail.loc[0, "delta_R@10"] == pytest.approx(0.25)
    assert summary["delta_N@10"] == pytest.approx(0.075)
```

- [ ] **Step 2: Run the report test and verify RED**

Run: `./py.bat -m pytest tests/test_actor_inference_ab_report.py -q`

Expected: FAIL because `actor_inference_ab_report` does not exist.

- [ ] **Step 3: Implement the report join and CLI**

```python
import argparse
from pathlib import Path

import pandas as pd


METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]


def compare_per_item(static_path, actor_path, seed):
    left = pd.read_csv(static_path)
    right = pd.read_csv(actor_path)
    merged = left.merge(right, on=["item_id", "count"], suffixes=("_static", "_actor"), validate="one_to_one")
    for metric in METRICS:
        if f"{metric}_static" in merged and f"{metric}_actor" in merged:
            merged[f"delta_{metric}"] = merged[f"{metric}_actor"] - merged[f"{metric}_static"]
    summary = {"seed": int(seed), "cold_items": int(len(merged))}
    for metric in METRICS:
        column = f"delta_{metric}"
        if column in merged:
            summary[column] = float(merged[column].mean())
            summary[f"positive_{metric}"] = float((merged[column] > 0).mean())
    return merged, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    summaries = []
    details = []
    for seed in args.seeds:
        name = f"strict_item_cold_balanced_thr1_seed_{seed}"
        static = root / "static" / name / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
        actor = root / "actor" / name / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
        if not static.exists() or not actor.exists():
            continue
        detail, summary = compare_per_item(static, actor, seed)
        details.append(detail.assign(seed=seed))
        summaries.append(summary)
    if not summaries:
        raise SystemExit("No complete static/actor seed pairs found")
    pd.DataFrame(summaries).to_csv(root / "actor_inference_ab_seed_summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(root / "actor_inference_ab_per_item.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the report test and verify GREEN**

Run: `./py.bat -m pytest tests/test_actor_inference_ab_report.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit the reporter**

```powershell
git add actor_inference_ab_report.py tests/test_actor_inference_ab_report.py
git commit -m "feat: compare Actor inference per-course metrics"
```

### Task 5: Exact-config PowerShell runner

**Files:**
- Create: `run_main_checkpoint_actor_inference_ab.ps1`

- [ ] **Step 1: Write the runner with finished-checkpoint gating**

```powershell
param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int[]]$Seeds = @(2025, 2026, 2027)
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$runner = Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1"
$checkpointRoot = "checkpoints\recovery_validation\main_table_51ea12fc_candidate"
$outputRoot = "outputs\recppo_research_repair\main_checkpoint_actor_inference_ab"
$completed = @()
foreach ($seed in $Seeds) {
    $checkpointDir = Join-Path $checkpointRoot "strict_item_cold_balanced_thr1_seed_$seed"
    if (-not (Test-Path (Join-Path $checkpointDir "finished.pt"))) {
        Write-Host "SKIP seed=$seed: finished.pt is not available"
        continue
    }
    foreach ($mode in @("static", "actor")) {
        $env:USIM_ACTOR_INFERENCE_MODE = $mode
        $env:USIM_ACTOR_INFERENCE_SEED = "7001"
        & $runner `
            -ScriptPath "main_checkpoint_actor_inference_ab.py" `
            -Protocol "strict_item_cold_balanced" -ColdThresholds @(1) -Seeds @($seed) `
            -Epochs 60 -Patience 60 -EarlyStopAverageMode "item_macro" -EarlyStopScoreMode "cold_only" `
            -UseContentDelta $false -UsePseudoColdTrain $false -PseudoColdMode "batch_random" `
            -PseudoColdRatio 0.3 -PseudoColdMinPop 5 -UsePaac $false `
            -CoursePrereqW 0.08 -CoursePrereqGate 0.20 -CourseConceptW 0.04 -CourseDiffW 0.03 `
            -CourseRedundantW 0.02 -CourseRedundantConceptGate 1.0 -CourseRedundantMode "concept" `
            -CourseTermNorm "none" -CourseSampleBeta 0.20 -TrainForceCold $true -UsimSteps 5 `
            -PpoLossWeight 1.00 -RolloutPolicy "ppo" -RlResidualScale 1.00 `
            -UseCourseFeedback $true -UseCourseReward $true -UseCourseSample $true -UsePrereqAux $true `
            -CourseFeedbackOnlyCold $false -CourseSampleOnlyCold $false -PrereqAuxOnlyCold $false `
            -MaskKnownPosNeg $true -MaskSameItemNeg $true -RunSampledEval $false `
            -OutputRoot (Join-Path $outputRoot $mode) -CheckpointRoot $checkpointRoot `
            -SaveCkpt $true -AutoResume $true -ForceFresh $false -SaveOptState $true -SkipAggregate
        if ($LASTEXITCODE -ne 0) { throw "A/B evaluation failed: seed=$seed mode=$mode" }
    }
    $completed += $seed
}
if ($completed.Count -gt 0) {
    & .\py.bat actor_inference_ab_report.py --root $outputRoot --seeds $completed
    if ($LASTEXITCODE -ne 0) { throw "A/B report generation failed" }
}
Remove-Item Env:USIM_ACTOR_INFERENCE_MODE -ErrorAction SilentlyContinue
Remove-Item Env:USIM_ACTOR_INFERENCE_SEED -ErrorAction SilentlyContinue
```

- [ ] **Step 2: Syntax-check the runner without executing training/evaluation**

Run:

```powershell
$errors=$null; [System.Management.Automation.Language.Parser]::ParseFile(
  (Resolve-Path .\run_main_checkpoint_actor_inference_ab.ps1), [ref]$null, [ref]$errors
) | Out-Null; if($errors){$errors | Format-List; exit 1}
```

Expected: exit code 0 with no parser errors.

- [ ] **Step 3: Commit the runner**

```powershell
git add run_main_checkpoint_actor_inference_ab.ps1
git commit -m "feat: run main-checkpoint Actor inference A/B"
```

### Task 6: Verification and available-seed evaluation

**Files:**
- Verify: all files above
- Generate only: `outputs/recppo_research_repair/main_checkpoint_actor_inference_ab/`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\py.bat -m pytest tests/test_main_checkpoint_actor_inference_ab.py tests/test_actor_inference_ab_report.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the existing evaluation-path regression test**

Run: `.\py.bat -m pytest tests/test_legacy_ppo_eval_probe.py -q`

Expected: all tests pass; static legacy behavior remains unchanged.

- [ ] **Step 3: Evaluate only finished main-table checkpoints**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_main_checkpoint_actor_inference_ab.ps1
```

Expected on the current workspace: finished seeds run and unfinished seeds are skipped. Each static arm reports zero Actor/episode calls; each actor arm reports non-zero calls. The refined-item count equals the full strict-cold item-bank count from the manifest (validation plus test cold courses), while the reported target-course count remains 68 on MOOCCube.

- [ ] **Step 4: Verify the static arm reproduces the stored main-table metrics exactly**

Run a CSV comparison that loads both stored and replayed `final_fullrank_usim_feedback_fast3_content_delta_static.csv` files and asserts equality for all `full_cold_item_macro_*` columns.

Expected: maximum absolute difference equals 0.

- [ ] **Step 5: Inspect A/B outputs and report partial results honestly**

Run:

```powershell
Import-Csv outputs\recppo_research_repair\main_checkpoint_actor_inference_ab\actor_inference_ab_seed_summary.csv | Format-Table -AutoSize
```

Expected: one row per completed seed with Actor-minus-static deltas and per-course positive fractions. Label the result partial until all three seeds are complete.

- [ ] **Step 6: Commit only source/tests, never generated outputs or checkpoints**

Run: `git status --short`

Expected: generated outputs remain untracked/ignored; no checkpoint or existing result file is staged.
