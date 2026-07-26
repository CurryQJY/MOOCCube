# CKG-RL Hot-Expert Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an isolated seed-2025 validation-only graph warm-expert preflight that determines whether the proposed CKG-RL dual-route model has a viable Hot backbone.

**Architecture:** Create a new graph-warm-expert entrypoint and launcher without touching the CKG-RL main-table implementation. The entrypoint trains a fresh CGRC-style graph expert from the exact existing split, owns its manifest and split-hash audit, and exports full-catalog Cold/Hot/Overall validation metrics each epoch. It deliberately excludes every CBI and simulator component; the only selection gate is pre-registered Hot capacity.

**Tech Stack:** Python, PyTorch, SciPy sparse graphs, PowerShell, pytest, existing strict-split/data/evaluator helpers.

---

### Task 1: Write the failing isolation and routing tests

**Files:**

- Create: `tests/test_ckg_hot_graph_preflight.py`

- [ ] **Step 1: Write a test for the warm-only configuration contract.**

```python
def test_preflight_config_disables_cbi_and_test_evaluation():
    from ckg_hot_graph_preflight import PreflightConfig

    cfg = PreflightConfig.for_seed(2025)

    assert cfg.seed == 2025
    assert cfg.use_cbi is False
    assert cfg.use_simulator is False
    assert cfg.use_ppo is False
    assert cfg.use_course_rewards is False
    assert cfg.test_evaluation is False
    assert cfg.hot_r10_floor == 0.2219
    assert cfg.hot_n10_floor == 0.1442
```

- [ ] **Step 2: Write a test for strict pseudo-cold edge removal.**

```python
def test_masked_graph_removes_every_edge_for_selected_item():
    import scipy.sparse as sp
    from ckg_hot_graph_preflight import drop_item_edges

    graph = sp.csr_matrix(([1, 1, 1], ([0, 1, 1], [2, 2, 3])), shape=(2, 4))
    masked = drop_item_edges(graph, [2])

    assert masked[:, 2].nnz == 0
    assert masked[:, 3].nnz == 1
```

- [ ] **Step 3: Write a test for mixed-bank count-weighted Overall aggregation.**

```python
def test_overall_uses_item_counts_not_interaction_counts():
    from ckg_hot_graph_preflight import count_weighted_overall

    assert count_weighted_overall(0.4, 2, 0.1, 8) == 0.16
```

- [ ] **Step 4: Run the focused test and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_hot_graph_preflight.py -q --basetemp .pytest_tmp/ckg_hot_preflight_red`

Expected: import failure because `ckg_hot_graph_preflight` does not exist.

### Task 2: Implement the isolated warm-expert entrypoint

**Files:**

- Create: `ckg_hot_graph_preflight.py`
- Test: `tests/test_ckg_hot_graph_preflight.py`

- [ ] **Step 1: Add `PreflightConfig` with a deterministic seed factory.**

```python
@dataclass(frozen=True)
class PreflightConfig:
    seed: int
    hot_r10_floor: float = 0.2219
    hot_n10_floor: float = 0.1442
    use_cbi: bool = False
    use_simulator: bool = False
    use_ppo: bool = False
    use_course_rewards: bool = False
    test_evaluation: bool = False

    @classmethod
    def for_seed(cls, seed: int) -> "PreflightConfig":
        return cls(seed=int(seed))
```

- [ ] **Step 2: Add `drop_item_edges` and `count_weighted_overall` exactly as tested.**

```python
def drop_item_edges(graph: sp.csr_matrix, item_ids: Sequence[int]) -> sp.csr_matrix:
    coo = graph.tocoo()
    keep = ~np.isin(coo.col, np.asarray(item_ids, dtype=np.int64))
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape)

def count_weighted_overall(cold_value, cold_count, hot_value, hot_count):
    return (cold_count * cold_value + hot_count * hot_value) / (cold_count + hot_count)
```

- [ ] **Step 3: Implement the warm graph expert using only the existing strict split and `static_train.pkl`.**

Use `load_hin_processed`, `InteractionDataset`, `collate_interactions`, and the sparse graph/reconstruction helpers from `cgrc_paper_static_hin.py`. The entrypoint must construct a fresh `CGRCNet`, never load a checkpoint, and train only the two losses below:

```python
loss = ranking_loss_on_full_train_graph + lambda_e * masked_warm_item_edge_reconstruction_loss
```

For every epoch, build the unified item bank, evaluate the validation loader separately for Cold and Hot with `evaluate_embedding_ranker(..., full_ranking=True, average_mode="item_macro")`, and write the count-weighted Overall values to `validation_epochs.csv`. Keep all CBI/simulation/PPO/course-reward code out of this entrypoint.

- [ ] **Step 4: Run the focused test and verify GREEN.**

Run: `./py.bat -m pytest tests/test_ckg_hot_graph_preflight.py -q --basetemp .pytest_tmp/ckg_hot_preflight_green`

Expected: PASS.

### Task 3: Add the reproducible single-seed launcher

**Files:**

- Create: `run_ckg_hot_graph_preflight_seed2025.ps1`
- Modify: `tests/test_ckg_hot_graph_preflight.py`

- [ ] **Step 1: Add a launcher contract test.**

```python
def test_launcher_locks_validation_only_seed_2025_preflight():
    source = Path("run_ckg_hot_graph_preflight_seed2025.ps1").read_text(encoding="utf-8")
    assert "Seeds = @(2025)" in source
    assert 'OutputRoot = "outputs\\ckg_hot_graph_preflight_seed2025"' in source
    assert "TestEvaluation = $false" in source
    assert "UseCbi = $false" in source
```

- [ ] **Step 2: Run the launcher test and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_hot_graph_preflight.py -q --basetemp .pytest_tmp/ckg_hot_launcher_red`

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement the launcher.**

The launcher must create fresh output/checkpoint/log roots, record exact input split hashes and protected-file hashes, set `test_evaluation=false`, set the fixed Hot floors `0.2219` and `0.1442`, and refuse to overwrite a completed manifest. It must not write to any protected main-table file.

- [ ] **Step 4: Run the focused suite and verify GREEN.**

Run: `./py.bat -m pytest tests/test_ckg_hot_graph_preflight.py -q --basetemp .pytest_tmp/ckg_hot_launcher_green`

Expected: PASS.

### Task 4: Validate before launch

**Files:**

- Test: `tests/test_ckg_hot_graph_preflight.py`
- Run: `run_ckg_hot_graph_preflight_seed2025.ps1`

- [ ] **Step 1: Run the focused suite.**

Run: `./py.bat -m pytest tests/test_ckg_hot_graph_preflight.py -q --basetemp .pytest_tmp/ckg_hot_preflight_final`

Expected: PASS.

- [ ] **Step 2: Run PowerShell dry-run.**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_ckg_hot_graph_preflight_seed2025.ps1 -DryRun`

Expected: JSON configuration with seed 2025, validation-only evaluation, and every CBI/simulation flag false.

- [ ] **Step 3: Launch only after the dry-run configuration and protected-file inventory are valid.**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_ckg_hot_graph_preflight_seed2025.ps1`

Expected: a fresh manifest and training log under `outputs/ckg_hot_graph_preflight_seed2025` and `background_logs/ckg_hot_graph_preflight_seed2025`.

### Task 5: Post-run decision gate

- [ ] Verify the manifest completed successfully and protected hashes did not change.
- [ ] Recompute per-epoch validation Cold/Hot/Overall from exported per-item rows.
- [ ] Pass only if the selected validation checkpoint has Hot R@10 >= `0.2219` and Hot N@10 >= `0.1442`; record Overall without using it as this preflight's pass/fail criterion.
- [ ] Do not run test or implement the CBI adapter unless the Hot preflight gate passes.
