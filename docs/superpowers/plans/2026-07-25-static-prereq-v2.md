# Static Prerequisite v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate an isolated, reproducible static prerequisite baseline before changing dual-path.

**Architecture:** Keep `static_content_scorer_clean.py` and all historical outputs untouched. Create `static_prereq_v2.py` with the same model/evaluator, canonical prerequisite-index wiring, explicit loss components, CLI-controlled weights, and provenance manifests. Validate the wiring with unit tests and a two-epoch smoke run, then run fresh weight-0 and weight-1 panels on seeds 2025/2026/2027.

**Tech Stack:** Python 3.11, PyTorch/CUDA, pandas, pytest, PowerShell, existing MOOCCube strict item-cold evaluator.

---

## File map

- Create: `static_prereq_v2.py` — isolated scorer entry point and manifest writer.
- Create: `tests/test_static_prereq_v2.py` — loss/index/manifest unit tests.
- Create: `run_static_prereq_v2.ps1` — bounded, unbuffered sequential runner for the six full runs.
- Create: `outputs/static_prereq_v2/` — smoke, controls, and prerequisite outputs only.
- Do not modify: `static_content_scorer_clean.py`, `dual_path_residual_clean.py`, or any historical output directory.

### Task 1: Create the isolated scorer and make the prerequisite loss observable

**Files:**
- Create: `static_prereq_v2.py`
- Test later: `tests/test_static_prereq_v2.py`

- [ ] **Step 1: Copy the current clean scorer into the isolated entry point.**

Run from `D:\DeskTop\MOOCCube`:

```powershell
Copy-Item -LiteralPath .\static_content_scorer_clean.py -Destination .\static_prereq_v2.py -Force
```

Do not edit the original file.

- [ ] **Step 2: Normalize the prerequisite index field.**

In `static_prereq_v2.py`, use only `self.prereq_idx` for the loaded tensor and
`self.prereq_mask` for its validity mask. The loss guard and lookup must be:

```python
if cfg.prereq_aux_weight <= 0.0 or model.prereq_idx is None:
    return item_vec_norm.new_zeros(())
pre_idx = model.prereq_idx.to(device).index_select(0, tgt)
```

- [ ] **Step 3: Refactor the loss so every branch includes all components.**

Replace the current early-return structure with one main-loss calculation and a
single total return. Preserve the current sampled-negative selection exactly.
The required shape is:

```python
def infonce_loss_parts(model, u_idx, i_idx, device):
    cfg = model.cfg
    z_u = F.normalize(model.user_vector(u_idx), dim=1)
    item_out, id_e_true, content_e = model.item_vector(
        i_idx, force_cold=False, apply_id_dropout=True, return_towers=True
    )
    z_i = F.normalize(item_out, dim=1)
    prereq = _prereq_aux_loss(model, i_idx, z_i, device)
    aux = _aux_infonce(cfg, id_e_true, content_e, device)
    logits = torch.mm(z_u, z_i.t()) / cfg.temp
    labels = torch.arange(logits.size(0), device=device)
    logits_m = logits.clone()
    diag = torch.arange(logits_m.size(0), device=device)
    logits_m[diag, diag] -= cfg.margin / cfg.temp

    if logits.size(0) <= 1:
        main = F.cross_entropy(logits_m, labels)
    else:
        n_total = min(cfg.train_num_negs, logits.size(0) - 1)
        if n_total <= 0:
            main = F.cross_entropy(logits_m, labels)
        else:
            # Keep the existing hard/random negative construction unchanged.
            neg = logits_m.clone()
            neg[diag, diag] = -1e9
            n_hard = max(0, min(int(n_total * cfg.hard_neg_ratio), n_total))
            n_rand = n_total - n_hard
            hard_idx = torch.empty(logits_m.size(0), 0, dtype=torch.long, device=device)
            rand_idx = torch.empty(logits_m.size(0), 0, dtype=torch.long, device=device)
            if n_hard > 0:
                _, hard_idx = torch.topk(neg, k=n_hard, dim=1)
            if n_rand > 0:
                rs = torch.rand_like(neg)
                rs[diag, diag] = -1e9
                if n_hard > 0:
                    rs.scatter_(1, hard_idx, -1e9)
                _, rand_idx = torch.topk(rs, k=n_rand, dim=1)
            cand = torch.cat([labels.view(-1, 1), hard_idx, rand_idx], dim=1)
            cand_logits = logits_m.gather(1, cand)
            main = F.cross_entropy(
                cand_logits,
                torch.zeros(logits.size(0), dtype=torch.long, device=device),
            )
    total = main + aux + prereq
    return total, {"main": main, "aux": aux, "prereq": prereq}


def infonce_loss(model, u_idx, i_idx, device):
    return infonce_loss_parts(model, u_idx, i_idx, device)[0]
```

Add small `get_git_head()` and `get_git_dirty()` helpers for the manifest. They
must return `None`/`False` when Git is unavailable so training does not depend on
Git being installed.

- [ ] **Step 4: Add explicit CLI overrides.**

Add these arguments and assign them to the config before training:

```python
ap.add_argument("--prereq-weight", type=float, default=1.0)
ap.add_argument("--aux-weight", type=float, default=0.3)
ap.add_argument("--prereq-path", default="outputs/prereq_target/prereq_index_topk10.pt")
```

The startup log must print both weights and the resolved prerequisite path.

- [ ] **Step 5: Log loss components and write provenance.**

Accumulate `parts["main"]`, `parts["aux"]`, and `parts["prereq"]` per epoch and
print all three averages. Before training, write `run_manifest.json` containing:
Create `out_dir` before writing the manifest, and fail with a clear message if
the output path is not writable.

```python
{
    "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    "git_head": get_git_head(),
    "git_dirty": get_git_dirty(),
    "argv": sys.argv,
    "seed": args.seed,
    "data_dir": args.data_dir,
    "split_dir": args.split_dir,
    "prereq_weight": cfg.prereq_aux_weight,
    "aux_weight": cfg.aux_weight,
    "prereq_path": cfg.prereq_path,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
```

### Task 2: Add tests before running training

**Files:**
- Create: `tests/test_static_prereq_v2.py`

- [ ] **Step 1: Add a deterministic prerequisite fixture and model factory.**

The fixture must contain at least four items, valid prerequisite rows for two
items, and one invalid/padded row. Construct a small `ScorerConfig` and a fixed
batch of user/item indices with `torch.manual_seed(7)`.

- [ ] **Step 2: Test canonical index loading and nonzero loss.**

Assert that `model.prereq_idx` exists, has the expected shape, and that
`_prereq_aux_loss(...)` is finite and strictly positive for the valid fixture.

- [ ] **Step 3: Test weight activation in the regular negative-sampling branch.**

With identical model parameters and batch, compute `infonce_loss_parts` at
weights 0 and 1. Assert:

```python
assert torch.isfinite(total0)
assert torch.isfinite(total1)
assert parts0["prereq"].item() == 0.0
assert parts1["prereq"].item() > 0.0
assert abs(total1.item() - total0.item()) > 1e-6
```

- [ ] **Step 4: Test no-valid-prerequisite rows.**

Use an all-`-1` prerequisite fixture and assert a finite zero prerequisite
component without changing the main or auxiliary loss behavior.

- [ ] **Step 5: Run the focused tests.**

Run:

```powershell
& D:\Anaconda3\envs\zw\python.exe -m pytest -q tests/test_static_prereq_v2.py
```

Expected: all tests pass. Do not start GPU training if this fails.

### Task 3: Run static checks and activation smoke

**Files:**
- Create: `outputs/static_prereq_v2/_smoke_seed2025/`

- [ ] **Step 1: Compile and dry-run.**

```powershell
& D:\Anaconda3\envs\zw\python.exe -m py_compile .\static_prereq_v2.py
& D:\Anaconda3\envs\zw\python.exe .\static_prereq_v2.py `
  --split-dir outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025 `
  --output-dir outputs/static_prereq_v2/_dry_seed2025 `
  --seed 2025 --prereq-weight 1.0 --dry-run
```

Expected: the index loads, the reported valid count is 440/698, and the command exits 0.

- [ ] **Step 2: Run the two-epoch smoke.**

```powershell
& D:\Anaconda3\envs\zw\python.exe -u .\static_prereq_v2.py `
  --split-dir outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025 `
  --output-dir outputs/static_prereq_v2/_smoke_seed2025 `
  --seed 2025 --epochs 2 --prereq-weight 1.0 `
  *> outputs/static_prereq_v2/_smoke_seed2025.log
```

Expected: clean exit, `prereq_loss` is finite and nonzero in the log, and
`run_manifest.json` is present. Stop and fix the code if any condition fails.

### Task 4: Run the fresh control and prerequisite panels

**Files:**
- Create: `run_static_prereq_v2.ps1`
- Create: `outputs/static_prereq_v2/control_seed{2025,2026,2027}/`
- Create: `outputs/static_prereq_v2/seed{2025,2026,2027}/`

- [ ] **Step 1: Create a sequential, unbuffered runner.**

The runner must invoke the same script with the same split/epochs for each seed,
write one log per run, stop on a nonzero exit code, and skip a run only when its
`test_metrics.json` and `run_manifest.json` both exist. It must never delete an
existing v2 output directory.

- [ ] **Step 2: Run v2 control (`--prereq-weight 0.0`) for all three seeds.**

Use output directories `control_seed2025`, `control_seed2026`, and
`control_seed2027`. Verify each has the four required artifacts before moving on.

- [ ] **Step 3: Run v2 prerequisite (`--prereq-weight 1.0`) for all three seeds.**

Use output directories `seed2025`, `seed2026`, and `seed2027`. Verify each has the
four required artifacts and a nonzero epoch-level prerequisite loss.

### Task 5: Aggregate and report without test leakage

**Files:**
- Create: `outputs/static_prereq_v2/summary.json`
- Create: `outputs/static_prereq_v2/summary.csv`

- [ ] **Step 1: Read only each run's `test_metrics.json` after training is complete.**

For each weight and seed, record best epoch, cold/hot/overall R/N@5/10/20, and
the manifest hash. Compute mean and standard deviation across the three seeds.

- [ ] **Step 2: Compare the paired v2 control versus v2 prerequisite.**

Report deltas for cold R/N@10 first, then hot and overall. Keep historical
`seed*_prereq` values in a separate labeled section; never pool them with v2.

- [ ] **Step 3: Run final verification.**

Check that every summary row points to an existing manifest and metrics file,
that no test-derived parameter was used, and that no training process remains.

- [ ] **Step 4: Commit only the new v2 source/tests/runner and summary metadata.**

Do not stage unrelated pre-existing worktree changes.

## Self-review checklist

- Spec coverage: fixes both known prereq wiring failures, adds observability,
  preserves protocol, runs smoke and 3-seed control/prerequisite panels, and
  records provenance.
- Placeholder scan: all implementation steps are concrete and specified.
- Type consistency: `prereq_idx`, `infonce_loss_parts`, `prereq_weight`, and all
  output directory names are used consistently across tasks.
