# CKG-RL V3.2 Clean Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. The repository is deliberately dirty. Create only the listed clean-route files and do not modify the historical V2/V3/V3.1 routes.

**Goal:** Build and validate an isolated clean teacher-to-generator-to-USIM-policy route for the current CKG-RL method, with strict train/inference support parity and no cold-test leakage.

**Architecture:** `ckg_rl_usim_v32_clean.py` is a standalone orchestrator. It reads the frozen shared strict-cold split, derives deterministic warm/pseudo partitions, trains a clean behavioral teacher, trains a content-only generator outside pseudo items, then trains a frozen-space legal-candidate RecPPO policy. It writes a manifest and evaluation artifacts without relying on the historical static runner's warm checkpoint or dropout behavior.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell.

---

### Task 1: Lock the partition and visibility contracts in failing tests

**Files:**
- Create: `tests/test_ckg_rl_usim_v32_clean.py`
- Create: `ckg_rl_usim_v32_clean.py`

- [ ] **Step 1: Write failing partition tests.**

```python
def test_clean_partition_is_deterministic_disjoint_and_train_only():
    train = _frame([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 2)])
    val = _frame([(0, 0), (1, 9)])
    test = _frame([(2, 1), (3, 10)])
    first = build_clean_partitions(train, val, test, n_items=11, seed=7, pseudo_ratio=0.50, min_popularity=1)
    second = build_clean_partitions(train, val, test, n_items=11, seed=7, pseudo_ratio=0.50, min_popularity=1)
    assert first == second
    assert first.g_item_ids.isdisjoint(first.p_train_item_ids)
    assert first.g_item_ids.isdisjoint(first.p_val_item_ids)
    assert first.p_train_item_ids.isdisjoint(first.p_val_item_ids)
    assert set(first.h_val["i_idx"]).issubset({0, 1, 2})
    assert set(first.c_val["i_idx"]) == {9}
    assert set(first.h_test["i_idx"]).issubset({0, 1, 2})
    assert set(first.c_test["i_idx"]) == {10}
```

- [ ] **Step 2: Write failing teacher/generator visibility tests.**

```python
def test_clean_stage_views_exclude_outer_and_pseudo_behavior_targets():
    parts = _example_partitions()
    views = build_stage_views(parts)
    assert set(views.teacher_train["_row_id"]) == set(parts.h_train["_row_id"])
    assert set(views.teacher_val["_row_id"]) == set(parts.h_val["_row_id"])
    assert not set(views.generator_item_ids) & set(parts.p_train_item_ids)
    assert not set(views.generator_item_ids) & set(parts.p_val_item_ids)
    assert not set(views.teacher_train["_row_id"]) & set(parts.c_val["_row_id"])
    assert not set(views.teacher_train["_row_id"]) & set(parts.c_test["_row_id"])
```

- [ ] **Step 3: Run the focused file and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_partitions_red`

Expected: collection or import failure because `build_clean_partitions` and `build_stage_views` do not exist.

- [ ] **Step 4: Implement only `CleanPartitions`, `build_clean_partitions`, and `build_stage_views`.**

```python
@dataclass(frozen=True)
class CleanPartitions:
    h_train: pd.DataFrame
    h_val: pd.DataFrame
    h_test: pd.DataFrame
    c_val: pd.DataFrame
    c_test: pd.DataFrame
    g_item_ids: frozenset[int]
    p_train_item_ids: frozenset[int]
    p_val_item_ids: frozenset[int]

def build_clean_partitions(train_df, val_df, test_df, *, n_items, seed, pseudo_ratio, min_popularity):
    warm_ids = frozenset(int(value) for value in train_df["i_idx"].unique())
    h_val, c_val = _split_rows_by_ids(val_df, warm_ids)
    h_test, c_test = _split_rows_by_ids(test_df, warm_ids)
    p_train_ids, p_val_ids = _select_pseudo_item_sets(train_df, seed, pseudo_ratio, min_popularity)
    return CleanPartitions(train_df.copy(), h_val, h_test, c_val, c_test,
        warm_ids - p_train_ids - p_val_ids, p_train_ids, p_val_ids)
```

- [ ] **Step 5: Run the focused file and verify GREEN.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_partitions_green`

Expected: the partition and visibility tests pass.

### Task 2: Add and implement clean teacher and content-generator contracts

**Files:**
- Modify: `tests/test_ckg_rl_usim_v32_clean.py`
- Modify: `ckg_rl_usim_v32_clean.py`

- [ ] **Step 1: Write failing model-isolation tests.**

```python
def test_generator_has_no_item_id_parameters_and_maps_content_to_teacher_space():
    generator = ContentGenerator(content_dim=3, emb_dim=2, hidden_dim=4)
    assert not any("item" in name.lower() for name, _ in generator.named_parameters())
    assert generator(torch.zeros(2, 3)).shape == (2, 2)

def test_clean_teacher_ranking_loss_uses_only_behavioral_tables():
    teacher = CleanTeacher(n_users=3, n_items=4, emb_dim=2)
    assert teacher.ranking_loss(torch.tensor([0, 1]), torch.tensor([1, 2])).isfinite()
    assert not hasattr(teacher, "agent")
    assert not hasattr(teacher, "content_proj")
```

- [ ] **Step 2: Run the test and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_models_red`

Expected: the new imports or attributes fail because the clean models are absent.

- [ ] **Step 3: Implement the minimal clean models and stage trainers.**

```python
class CleanTeacher(nn.Module):
    def ranking_loss(self, user_ids, item_ids):
        user = F.normalize(self.user_emb(user_ids), dim=1)
        item = F.normalize(self.item_emb(item_ids), dim=1)
        return F.cross_entropy(user @ item.T / self.temperature, torch.arange(user.size(0), device=user.device))

class ContentGenerator(nn.Module):
    def forward(self, content):
        return self.net(content)

def train_content_generator(
    generator, teacher_items, content, generator_item_ids, *, epochs, batch_size, lr, device
):
    # Draw labels only by `generator_item_ids`; use normalized MSE plus cosine loss.
```

`train_clean_teacher` must receive only `stage_views.teacher_train` and `stage_views.teacher_val`; `train_content_generator` must accept only `stage_views.generator_item_ids`.

- [ ] **Step 4: Run the test file and verify GREEN.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_models_green`

Expected: all partition/model tests pass.

### Task 3: Lock legal USIM rollout, target-free inference, and course-history exclusion

**Files:**
- Modify: `tests/test_ckg_rl_usim_v32_clean.py`
- Modify: `ckg_rl_usim_v32_clean.py`

- [ ] **Step 1: Write failing policy tests.**

```python
def test_main_candidate_pool_is_state_retrieval_plus_end_for_train_and_inference():
    engine = _engine()
    state = torch.tensor([[1.0, 0.0]])
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert torch.equal(engine.legal_candidate_ids(state, users, count=2), torch.tensor([[0, 1]]))
    assert engine.legal_candidate_ids(state, users, count=2).tolist() == engine.legal_candidate_ids(state, users, count=2).tolist()

def test_inference_episode_rejects_oracle_inputs_and_never_requests_reward():
    engine = _engine()
    with pytest.raises(ValueError, match="oracle"):
        engine.rollout(torch.zeros(1, 2), user_bank=torch.zeros(3, 2), training=False, target_emb=torch.zeros(1, 2))
    result = engine.rollout(torch.zeros(1, 2), user_bank=torch.zeros(3, 2), training=False)
    assert result.final_state.shape == (1, 2)

def test_course_history_excludes_the_target_before_all_terms():
    history = {0: {1, 2}}
    target_free = target_excluded_history(history, target_item_ids=torch.tensor([2]), selected_user_ids=torch.tensor([0]))
    assert target_free == [{1}]
```

- [ ] **Step 2: Run the new tests and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_policy_red`

Expected: the clean engine APIs are absent.

- [ ] **Step 3: Implement `LegalUSIMPolicy`, `CleanUSIMEngine`, detached replay, and trust projection.**

```python
def rollout(
    self, initial_state, *, user_bank, training, target_emb=None,
    positive_user_ids=None, item_ids=None, user_history=None,
):
    if not training and (target_emb is not None or positive_user_ids is not None):
        raise ValueError("inference rollout forbids oracle target and positive users")
    candidates = self.legal_candidate_ids(state, user_bank, self.candidate_count)
    # `END` is a policy logit, never a user ID. Training reward alone observes targets.
```

`target_excluded_history` must copy histories and remove every row's target item before candidate-bias or reward computation. `project_displacement` must leave a zero delta unchanged and cap nonzero deltas at `max_delta`.

- [ ] **Step 4: Run the test file and verify GREEN.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_policy_green`

Expected: the legal candidate, target-free inference, target-history, terminal, and trust-cap tests pass.

### Task 4: Implement orchestration, evaluation, manifests, and launcher

**Files:**
- Modify: `ckg_rl_usim_v32_clean.py`
- Create: `run_ckg_rl_usim_v32_clean_seed2025.ps1`
- Modify: `tests/test_ckg_rl_usim_v32_clean.py`

- [ ] **Step 1: Write failing runner/launcher tests.**

```python
def test_clean_manifest_records_no_legacy_warm_checkpoint_or_random_mask(tmp_path):
    write_clean_manifest(tmp_path, _minimal_run_config(), _example_partitions(), stage_hashes={})
    payload = json.loads((tmp_path / "clean_manifest.json").read_text())
    assert payload["legacy_warm_checkpoint"] is None
    assert payload["random_id_dropout"] is False
    assert payload["main_candidate_mode"] == "legal_state_retrieval"
    assert payload["inference_oracle_access"] is False

def test_v32_launcher_is_fresh_and_disables_historical_routes():
    text = Path("run_ckg_rl_usim_v32_clean_seed2025.ps1").read_text(encoding="utf-8")
    assert 'ScriptPath = "ckg_rl_usim_v32_clean.py"' in text
    assert 'outputs\\ckg_rl_usim_v32_clean' in text
    assert 'USIM_ORIGINAL_V2' not in text
    assert 'USIM_V3_CORE' not in text
    assert 'USIM_CLEAN_RANDOM_ID_DROPOUT' in text
```

- [ ] **Step 2: Run the test and verify RED.**

Run: `./py.bat -m pytest tests/test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp/v32_runner_red`

Expected: missing manifest writer and launcher assertion failure.

- [ ] **Step 3: Implement `run_clean_pipeline` and full-ranking evaluation.**

```python
def run_clean_pipeline(config):
    train_df, val_df, test_df, content = load_clean_inputs(config)
    partitions = build_clean_partitions(
        train_df, val_df, test_df, n_items=content.size(0), seed=config.seed,
        pseudo_ratio=config.pseudo_ratio, min_popularity=config.pseudo_min_popularity,
    )
    views = build_stage_views(partitions)
    teacher, teacher_state = train_clean_teacher(views, content.size(0), config)
    generator, generator_state = train_content_generator(
        ContentGenerator(config.content_dim, config.emb_dim, config.hidden_dim),
        teacher.item_vectors(), content, views.generator_item_ids,
        epochs=config.generator_epochs, batch_size=config.batch_size, lr=config.generator_lr,
        device=config.device,
    )
    policy, policy_state = train_clean_policy(teacher, generator, views, content, config)
    validation = evaluate_clean_route(teacher, generator, policy, partitions.h_val, partitions.c_val, content, config)
    test = evaluate_clean_route(teacher, generator, policy, partitions.h_test, partitions.c_test, content, config)
    write_clean_manifest(config.output_dir, config, partitions, stage_hashes={
        "teacher": teacher_state["sha256"], "generator": generator_state["sha256"], "policy": policy_state["sha256"],
    })
```

`evaluate_clean_route` must use the teacher item bank unchanged for hot IDs and replace only strict-cold IDs with content-generator-plus-target-free-policy vectors. `C_test` and `H_test` are read after policy checkpoint selection and are not passed into any selection function.

- [ ] **Step 4: Create the seed-2025 PowerShell launcher.**

The launcher must reject existing roots, expose `-DryRun` and `-Smoke`, set `USIM_CLEAN_RANDOM_ID_DROPOUT=0`, set `USIM_CLEAN_CANDIDATE_MODE=legal_state_retrieval`, set isolated output/checkpoint roots, and never set `USIM_ORIGINAL_V2`, `USIM_V3_CORE`, or `USIM_FB_INIT_CKPT_DIR`.

- [ ] **Step 5: Run tests, compile, and dry run.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v32_clean.py -q --basetemp .pytest_tmp\v32_runner_green
.\py.bat -m py_compile ckg_rl_usim_v32_clean.py
.\run_ckg_rl_usim_v32_clean_seed2025.ps1 -DryRun
```

Expected: all V3.2 tests pass, the module compiles, and DryRun creates no run output.

### Task 5: Preflight and acceptance experiment

**Files:**
- Runtime create: `outputs/ckg_rl_usim_v32_clean/seed2025/`
- Runtime create: `checkpoints/ckg_rl_usim_v32_clean/seed2025/`
- Runtime create: `background_logs/ckg_rl_usim_v32_clean_seed2025_<timestamp>/`

- [ ] **Step 1: Run CPU smoke preflight.**

Run: `.\run_ckg_rl_usim_v32_clean_seed2025.ps1 -Smoke -Epochs 1 -TeacherEpochs 1 -GeneratorEpochs 1 -PolicyEpochs 1`

Expected: `smoke_report.json` records a train pseudo episode and a target-free strict-cold inference episode; no outer test selection occurs.

- [ ] **Step 2: Run full regression gate.**

Run:

```powershell
.\py.bat -m pytest tests\test_ckg_rl_usim_v32_clean.py tests\test_ckg_rl_usim_v3_core.py tests\test_usim_v2_core_alignment.py -q --basetemp .pytest_tmp\v32_full_regression
```

Expected: zero failures.

- [ ] **Step 3: Launch the isolated seed-2025 run in a hidden background process.**

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = "background_logs\\ckg_rl_usim_v32_clean_seed2025_$stamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','.\\run_ckg_rl_usim_v32_clean_seed2025.ps1') -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log') -PassThru
```

- [ ] **Step 4: Verify the run's final metrics and compare only matched strict split results.**

Read `final_metrics.json`, `clean_manifest.json`, and `validation_epochs.csv`. Compare cold/hot/overall item-macro R@10/N@10 with old CKG-RL and V3.1 only after confirming split hash and test-read timing in the manifest.
