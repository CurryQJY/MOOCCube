import inspect
import json
import hashlib
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import pandas as pd

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def test_adapter_config_locks_validation_only_masked_pseudocold():
    from ckg_frozen_hot_pseudocold_adapter import AdapterConfig

    cfg = AdapterConfig.for_seed(2025)

    assert cfg.seed == 2025
    assert not hasattr(cfg, "mask_ratio")
    assert cfg.n_items == 698
    assert cfg.train_zero_item_count == 102
    assert cfg.warm_item_count == 596
    assert cfg.pseudo_cold_item_count == 102
    assert cfg.trust_tau == pytest.approx(0.24929234)
    assert cfg.epochs == 15
    assert cfg.emb_dim == 64
    assert cfg.hidden_dim == 64
    assert cfg.layers_full == 2
    assert cfg.batch_size == 4096
    assert cfg.negatives_per_positive == 32
    assert cfg.ranking_temperature == pytest.approx(0.5)
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.weight_decay == 0.0
    assert cfg.delta_reg_weight == 0.0
    assert cfg.parity_atol == pytest.approx(1e-5)
    assert cfg.retention_tolerance == pytest.approx(0.003)
    assert cfg.test_evaluation is False
    assert cfg.use_cbi is False
    assert cfg.use_simulator is False
    assert cfg.use_ppo is False
    assert cfg.use_course_rewards is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_dim", 32),
        ("epochs", 10),
        ("negatives_per_positive", 16),
        ("ranking_temperature", 0.7),
        ("lr", 2e-3),
        ("weight_decay", 1e-4),
        ("delta_reg_weight", 0.001),
        ("cold_threshold", 7),
        ("parity_atol", 1e-4),
        ("retention_tolerance", 0.01),
    ],
)
def test_config_validation_rejects_any_unlocked_stage_b_knob(field, value):
    from ckg_frozen_hot_pseudocold_adapter import AdapterConfig, _validate_config

    with pytest.raises(ValueError):
        _validate_config(replace(AdapterConfig.for_seed(2025), **{field: value}))


def test_shared_adapter_is_content_only_and_starts_at_normalized_content_base():
    from ckg_frozen_hot_pseudocold_adapter import SharedColdAdapter

    adapter = SharedColdAdapter(emb_dim=4, hidden_dim=8, trust_tau=0.25)
    base = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

    output, delta = adapter(base)

    assert torch.allclose(output, F.normalize(base, dim=1))
    assert torch.allclose(delta, torch.zeros_like(delta))
    assert list(inspect.signature(adapter.forward).parameters) == ["content_base"]
    assert not any(isinstance(module, torch.nn.Embedding) for module in adapter.modules())
    assert all("item" not in name.lower() for name, _ in adapter.named_parameters())


class _FixedRawDelta(torch.nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, content_base: torch.Tensor) -> torch.Tensor:
        return self.value.to(content_base).expand_as(content_base)


def test_shared_adapter_projects_final_unit_sphere_chordal_shift_to_the_trust_radius():
    from ckg_frozen_hot_pseudocold_adapter import SharedColdAdapter

    tau = 0.24929234
    adapter = SharedColdAdapter(emb_dim=3, hidden_dim=2, trust_tau=tau)
    adapter.net = _FixedRawDelta(torch.tensor([[0.0, 100.0, 0.0]]))
    base = torch.tensor([[1.0, 0.0, 0.0]])

    output, delta = adapter(base)

    final_distance = (output - F.normalize(base, dim=1)).norm(dim=1)
    assert torch.allclose(final_distance, torch.full((1,), tau), atol=1e-6)
    assert torch.allclose(delta, output - F.normalize(base, dim=1), atol=1e-7)
    assert torch.all(final_distance <= tau + 1e-7)


class _FixedRawMatrix(torch.nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, content_base: torch.Tensor) -> torch.Tensor:
        return self.values.to(content_base)


def test_shared_adapter_never_exceeds_the_final_float32_chordal_cap():
    from ckg_frozen_hot_pseudocold_adapter import SharedColdAdapter

    tau = 0.24929234
    generator = torch.Generator().manual_seed(2025)
    base = F.normalize(torch.randn((20_000, 8), generator=generator), dim=1)
    raw_delta = torch.randn((20_000, 8), generator=generator) * 30.0
    adapter = SharedColdAdapter(emb_dim=8, hidden_dim=8, trust_tau=tau)
    adapter.net = _FixedRawMatrix(raw_delta)

    output, _ = adapter(base)

    assert float((output - base).norm(dim=1).max()) <= tau


def test_zero_initialized_adapter_has_a_nonzero_gradient_path_to_its_final_layer():
    from ckg_frozen_hot_pseudocold_adapter import SharedColdAdapter

    adapter = SharedColdAdapter(emb_dim=3, hidden_dim=4, trust_tau=0.24929234)
    output, _ = adapter(torch.tensor([[1.0, 0.0, 0.0]]))
    loss = output[:, 1].sum()

    loss.backward()

    assert adapter.net[-1].weight.grad is not None
    assert float(adapter.net[-1].weight.grad.abs().sum()) > 0.0


def test_masked_pseudocold_graph_removes_all_selected_item_edges():
    from ckg_frozen_hot_pseudocold_adapter import mask_item_edges

    graph = sp.csr_matrix(([1, 1, 1], ([0, 0, 1], [1, 2, 1])), shape=(2, 4))

    masked = mask_item_edges(graph, np.array([1], dtype=np.int64))

    assert masked[:, 1].nnz == 0
    assert masked[:, 2].nnz == 1
    assert graph[:, 1].nnz == 2


def test_pseudocold_selection_locks_catalog_counts_and_is_epoch_deterministic():
    from ckg_frozen_hot_pseudocold_adapter import (
        AdapterConfig,
        derive_train_item_partitions,
        select_epoch_pseudocold_items,
    )

    cfg = AdapterConfig.for_seed(2025)
    train_zero_mask = np.array([False] * 596 + [True] * 102)
    warm_ids, train_zero_ids = derive_train_item_partitions(train_zero_mask, cfg)
    first = select_epoch_pseudocold_items(warm_ids, epoch=4, cfg=cfg)
    second = select_epoch_pseudocold_items(warm_ids, epoch=4, cfg=cfg)

    assert warm_ids.size == 596
    assert train_zero_ids.size == 102
    assert first.size == 102
    assert np.array_equal(first, second)
    assert set(first).issubset(set(warm_ids))


def test_pseudocold_selection_rejects_catalog_count_drift():
    from ckg_frozen_hot_pseudocold_adapter import AdapterConfig, derive_train_item_partitions

    cfg = AdapterConfig.for_seed(2025)
    with pytest.raises(ValueError, match="596 warm"):
        derive_train_item_partitions(np.array([False] * 595 + [True] * 103), cfg)


def test_pseudocold_selection_audit_records_exact_count_ratio_and_id_hash():
    from ckg_frozen_hot_pseudocold_adapter import (
        AdapterConfig,
        pseudo_cold_selection_audit,
        select_epoch_pseudocold_items,
    )

    cfg = AdapterConfig.for_seed(2025)
    selected = select_epoch_pseudocold_items(np.arange(596), epoch=1, cfg=cfg)

    audit = pseudo_cold_selection_audit(selected, cfg)

    assert audit["pseudo_cold_item_count"] == 102
    assert audit["warm_item_count"] == 596
    assert audit["train_zero_item_count"] == 102
    assert audit["pseudo_cold_warm_ratio"] == pytest.approx(102 / 596)
    assert audit["pseudo_cold_ids_sha256"] == hashlib.sha256(selected.tobytes()).hexdigest()


def test_original_positives_are_never_negative_candidates():
    from ckg_frozen_hot_pseudocold_adapter import negative_candidates

    candidates = negative_candidates(
        user_ids=[0],
        original_user_rated=[{1, 2}],
        item_pool=np.array([1, 2, 3, 4]),
        per_user=8,
        rng=np.random.default_rng(2025),
    )

    assert len(candidates) == 1
    assert candidates[0].dtype == np.int64
    assert set(candidates[0]).isdisjoint({1, 2})
    assert set(candidates[0]).issubset({3, 4})


def test_negative_candidates_returns_empty_array_when_every_item_is_original_positive():
    from ckg_frozen_hot_pseudocold_adapter import negative_candidates

    candidates = negative_candidates(
        user_ids=[0],
        original_user_rated=[{1, 2}],
        item_pool=np.array([1, 2]),
        per_user=3,
        rng=np.random.default_rng(2025),
    )

    assert len(candidates) == 1
    assert candidates[0].dtype == np.int64
    assert candidates[0].size == 0


def test_training_negative_candidates_use_only_warm_train_ids():
    from ckg_frozen_hot_pseudocold_adapter import training_negative_candidates

    candidates = training_negative_candidates(
        user_ids=[0],
        original_user_rated=[{0, 1}],
        train_zero_mask=np.array([False, False, False, False, True, True]),
        per_user=8,
        rng=np.random.default_rng(2025),
    )

    assert set(candidates[0]).issubset({2, 3})
    assert set(candidates[0]).isdisjoint({0, 1, 4, 5})


def test_incomplete_removed_edge_negative_batch_fails_closed():
    from ckg_frozen_hot_pseudocold_adapter import require_complete_negative_candidates

    with pytest.raises(RuntimeError, match="warm-only negatives"):
        require_complete_negative_candidates(
            [np.array([1, 2], dtype=np.int64), np.empty(0, dtype=np.int64)], expected_count=2
        )


def test_item_balanced_objective_gives_each_selected_course_equal_total_weight():
    from ckg_frozen_hot_pseudocold_adapter import item_balanced_edge_objective

    edge_loss = torch.tensor([1.0, 1.0, 10.0])
    item_ids = torch.tensor([0, 0, 1])
    degree_by_item = torch.tensor([2.0, 1.0])

    objective = item_balanced_edge_objective(edge_loss, item_ids, degree_by_item, selected_item_count=2)

    assert objective.item() == pytest.approx(5.5)


def test_adapter_optimizer_is_plain_adam_with_no_weight_decay():
    from ckg_frozen_hot_pseudocold_adapter import AdapterConfig, SharedColdAdapter, build_adapter_optimizer

    cfg = AdapterConfig.for_seed(2025)
    adapter = SharedColdAdapter(cfg.emb_dim, cfg.hidden_dim, cfg.trust_tau)
    optimizer = build_adapter_optimizer(adapter, cfg)

    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


class _FixedAdapter(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = F.normalize(output, dim=1)
        self.calls = []

    def forward(self, content_base: torch.Tensor):
        self.calls.append(content_base.detach().clone())
        return self.output.to(content_base), torch.zeros_like(content_base)


def test_mixed_bank_routes_every_train_zero_item_through_adapter():
    from ckg_frozen_hot_pseudocold_adapter import build_true_eval_item_bank

    hot = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1)
    content = F.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.5, 0.5]]), dim=1)
    adapter = _FixedAdapter(torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]))

    bank = build_true_eval_item_bank(hot, content, np.array([False, True, True]), adapter)

    assert len(adapter.calls) == 1
    assert torch.allclose(adapter.calls[0], content)
    assert torch.allclose(bank[0], hot[0])
    assert torch.allclose(bank[1], adapter.output[1])
    assert torch.allclose(bank[2], adapter.output[2])
    assert torch.allclose(bank.norm(dim=1), torch.ones(3))


def _write_completed_preflight(tmp_path, *, manifest_status="completed", gate_status="completed", passed=True, epoch=15):
    manifest_path = tmp_path / "run_manifest.json"
    result_path = tmp_path / "preflight_result.json"
    manifest_path.write_text(
        json.dumps({"status": manifest_status, "gate_status": gate_status}),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "passed_hot_preflight": passed,
                "gate_status": gate_status,
                "selected_validation_epoch": {"epoch": epoch},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, result_path


def test_completed_hot_preflight_is_required(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter import require_completed_hot_preflight

    manifest_path, result_path = _write_completed_preflight(tmp_path, passed=False)

    with pytest.raises(ValueError, match="passed Hot preflight"):
        require_completed_hot_preflight(manifest_path, result_path, expected_epoch=15)


@pytest.mark.parametrize(
    ("manifest_status", "gate_status", "expected_error"),
    [
        ("failed", "completed", "completed Hot preflight manifest"),
        ("completed", "completed_gate_failed", "gate status"),
    ],
)
def test_hot_preflight_loader_fails_closed_on_manifest_or_gate_mismatch(
    tmp_path, manifest_status, gate_status, expected_error
):
    from ckg_frozen_hot_pseudocold_adapter import require_completed_hot_preflight

    manifest_path, result_path = _write_completed_preflight(
        tmp_path,
        manifest_status=manifest_status,
        gate_status=gate_status,
    )

    with pytest.raises(ValueError, match=expected_error):
        require_completed_hot_preflight(manifest_path, result_path, expected_epoch=15)


def test_hot_preflight_loader_requires_strict_boolean_and_selected_epoch(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter import require_completed_hot_preflight

    manifest_path, result_path = _write_completed_preflight(tmp_path, passed=1, epoch=14)

    with pytest.raises(ValueError, match="passed Hot preflight"):
        require_completed_hot_preflight(manifest_path, result_path, expected_epoch=15)

    manifest_path, result_path = _write_completed_preflight(tmp_path, passed=True, epoch=14)
    with pytest.raises(ValueError, match="selected validation epoch"):
        require_completed_hot_preflight(manifest_path, result_path, expected_epoch=15)


def test_hot_preflight_loader_returns_completed_records(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter import require_completed_hot_preflight

    manifest_path, result_path = _write_completed_preflight(tmp_path)

    manifest, result = require_completed_hot_preflight(manifest_path, result_path, expected_epoch=15)

    assert manifest["status"] == "completed"
    assert result["selected_validation_epoch"]["epoch"] == 15


def test_hot_preflight_loader_accepts_launcher_utf8_bom_records(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter import require_completed_hot_preflight

    manifest_path, result_path = _write_completed_preflight(tmp_path)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8-sig")
    result_path.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8-sig")

    manifest, result = require_completed_hot_preflight(manifest_path, result_path, expected_epoch=15)

    assert manifest["status"] == "completed"
    assert result["passed_hot_preflight"] is True


def test_hot_checkpoint_verification_requires_fixed_hash_and_architecture(tmp_path, monkeypatch):
    from ckg_frozen_hot_pseudocold_adapter import require_hot_checkpoint
    import ckg_frozen_hot_pseudocold_adapter as stage_b

    checkpoint_path = tmp_path / "epoch_015.pt"
    torch.save(
        {
            "epoch": 15,
            "model_state": {"item_lin.weight": torch.zeros((64, 3))},
            "config": {"emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint_path,
    )
    expected_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
    monkeypatch.setattr(stage_b, "HOT_CHECKPOINT_SHA256", expected_hash)

    payload = require_hot_checkpoint(checkpoint_path, expected_epoch=15)

    assert payload["epoch"] == 15
    assert "model_state" in payload


def test_hot_checkpoint_verification_fails_closed_on_hash_or_architecture_drift(tmp_path, monkeypatch):
    from ckg_frozen_hot_pseudocold_adapter import require_hot_checkpoint
    import ckg_frozen_hot_pseudocold_adapter as stage_b

    checkpoint_path = tmp_path / "epoch_015.pt"
    torch.save(
        {
            "epoch": 15,
            "model_state": {"item_lin.weight": torch.zeros((64, 3))},
            "config": {"emb_dim": 64, "mlp_hidden": 64, "layers_full": 1},
        },
        checkpoint_path,
    )
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
    monkeypatch.setattr(stage_b, "HOT_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        require_hot_checkpoint(checkpoint_path, expected_epoch=15)

    monkeypatch.setattr(stage_b, "HOT_CHECKPOINT_SHA256", actual_hash)
    with pytest.raises(ValueError, match="layers_full"):
        require_hot_checkpoint(checkpoint_path, expected_epoch=15)


def test_preflight_input_hash_verification_requires_before_and_after_manifest_hashes(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter import require_preflight_input_hashes

    meta = tmp_path / "meta.json"
    content = tmp_path / "content_emb.pt"
    train = tmp_path / "static_train.pkl"
    val = tmp_path / "static_val.pkl"
    cgrc = tmp_path / "cgrc_paper_static_hin.py"
    evaluator = tmp_path / "hin_eval_common.py"
    for path, body in {
        meta: b"meta",
        content: b"content",
        train: b"train",
        val: b"val",
        cgrc: b"cgrc",
        evaluator: b"eval",
    }.items():
        path.write_bytes(body)

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "data_sha256": {"meta": digest(meta), "content": digest(content)},
        "data_sha256_after": {"meta": digest(meta), "content": digest(content)},
        "split_sha256": {"train": digest(train), "val": digest(val)},
        "split_sha256_after": {"train": digest(train), "val": digest(val)},
        "source_sha256": {"cgrc": digest(cgrc), "evaluator": digest(evaluator)},
        "source_sha256_after": {"cgrc": digest(cgrc), "evaluator": digest(evaluator)},
    }

    require_preflight_input_hashes(
        manifest,
        data_files={"meta": meta, "content": content},
        split_files={"train": train, "val": val},
        source_files={"cgrc": cgrc, "evaluator": evaluator},
    )

    val.write_bytes(b"changed")
    with pytest.raises(ValueError, match="split_sha256"):
        require_preflight_input_hashes(
            manifest,
            data_files={"meta": meta, "content": content},
            split_files={"train": train, "val": val},
            source_files={"cgrc": cgrc, "evaluator": evaluator},
        )


def test_epoch_zero_parity_initializes_validation_rows_and_uses_runtime_baseline():
    from ckg_frozen_hot_pseudocold_adapter import initialize_validation_rows

    reference = {
        "selected_validation_epoch": {
            "epoch": 15,
            "cold_r10": 0.29,
            "cold_n10": 0.20,
            "hot_r10": 0.23,
            "hot_n10": 0.15,
            "overall_r10": 0.235,
            "overall_n10": 0.155,
        }
    }
    epoch_zero = {
        "epoch": 0,
        "cold_r10": 0.29,
        "cold_n10": 0.20,
        "hot_r10": 0.23,
        "hot_n10": 0.15,
        "overall_r10": 0.235,
        "overall_n10": 0.155,
    }

    rows, baseline = initialize_validation_rows(epoch_zero, reference, parity_atol=1e-5)

    assert rows == [epoch_zero]
    assert baseline == epoch_zero


def test_epoch_zero_parity_fails_before_training_on_preflight_metric_drift():
    from ckg_frozen_hot_pseudocold_adapter import initialize_validation_rows

    reference = {"selected_validation_epoch": {"epoch": 15, "cold_r10": 0.29, "cold_n10": 0.20, "hot_r10": 0.23, "hot_n10": 0.15, "overall_r10": 0.235, "overall_n10": 0.155}}
    epoch_zero = {"epoch": 0, "cold_r10": 0.28, "cold_n10": 0.20, "hot_r10": 0.23, "hot_n10": 0.15, "overall_r10": 0.235, "overall_n10": 0.155}

    with pytest.raises(ValueError, match="epoch-0 parity"):
        initialize_validation_rows(epoch_zero, reference, parity_atol=1e-5)


def test_validation_only_input_loader_never_needs_stream_or_test_split(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter import load_validation_only_inputs

    data_dir = tmp_path / "data"
    split_dir = tmp_path / "split"
    data_dir.mkdir()
    split_dir.mkdir()
    (data_dir / "meta.json").write_text(json.dumps({"n_users": 2, "n_items": 3}), encoding="utf-8")
    torch.save(torch.ones((3, 4)), data_dir / "content_emb.pt")
    train = pd.DataFrame({"u_idx": [0, 1], "i_idx": [0, 1], "popularity": [99, 99]})
    val = pd.DataFrame({"u_idx": [0, 1], "i_idx": [2, 1], "popularity": [99, 99]})
    train.to_pickle(split_dir / "static_train.pkl")
    val.to_pickle(split_dir / "static_val.pkl")
    (data_dir / "stream_data.pkl").write_bytes(b"must not be read")
    (split_dir / "static_test.pkl").write_bytes(b"must not be read")

    meta, content, loaded_train, loaded_val = load_validation_only_inputs(data_dir, split_dir)

    assert meta == {"n_users": 2, "n_items": 3}
    assert tuple(content.shape) == (3, 4)
    assert loaded_train["popularity"].tolist() == [1, 1]
    assert loaded_val["popularity"].tolist() == [0, 1]


def test_runner_entrypoint_is_present_and_contains_no_test_split_loader_path():
    from ckg_frozen_hot_pseudocold_adapter import run_adapter_preflight

    source = inspect.getsource(run_adapter_preflight)

    assert "static_test.pkl" not in source
    assert "stream_data.pkl" not in source
    assert "test_loader" not in source


def test_runner_defers_output_creation_until_provenance_checks_pass():
    from ckg_frozen_hot_pseudocold_adapter import run_adapter_preflight

    source = inspect.getsource(run_adapter_preflight)

    assert source.index("require_completed_hot_preflight") < source.index("output_dir.mkdir")
    assert source.index("require_preflight_input_hashes") < source.index("output_dir.mkdir")
    assert source.index("require_hot_checkpoint") < source.index("output_dir.mkdir")


def test_input_roots_are_anchored_to_the_repository_not_process_cwd():
    from ckg_frozen_hot_pseudocold_adapter import AdapterConfig, resolve_run_input_roots

    data_root, split_root = resolve_run_input_roots(AdapterConfig.for_seed(2025))

    assert data_root.name == "processed_data_hin_clean_pop5"
    assert split_root.name == "strict_item_cold_balanced_thr1_seed_2025"
    assert data_root.is_absolute()
    assert split_root.is_absolute()


def test_selection_rejects_hot_or_overall_regression_and_uses_cold_metric_order():
    from ckg_frozen_hot_pseudocold_adapter import select_adapter_epoch

    baseline = {"hot_r10": 0.232, "hot_n10": 0.152, "overall_r10": 0.235, "overall_n10": 0.155}
    rows = [
        {
            "epoch": 1,
            "cold_r10": 0.30,
            "cold_n10": 0.22,
            "hot_r10": 0.228,
            "hot_n10": 0.152,
            "overall_r10": 0.235,
            "overall_n10": 0.155,
        },
        {
            "epoch": 2,
            "cold_r10": 0.29,
            "cold_n10": 0.21,
            "hot_r10": 0.231,
            "hot_n10": 0.151,
            "overall_r10": 0.234,
            "overall_n10": 0.154,
        },
        {
            "epoch": 3,
            "cold_r10": 0.99,
            "cold_n10": 0.99,
            "hot_r10": 0.230,
            "hot_n10": 0.151,
            "overall_r10": 0.231,
            "overall_n10": 0.154,
        },
        {
            "epoch": 4,
            "cold_r10": 0.29,
            "cold_n10": 0.21,
            "hot_r10": 0.231,
            "hot_n10": 0.151,
            "overall_r10": 0.234,
            "overall_n10": 0.154,
        },
    ]

    best = select_adapter_epoch(rows, baseline, tolerance=0.003)

    assert best["epoch"] == 4
    assert best["passes_retention_guards"] is True


def test_selection_fails_when_no_epoch_preserves_hot_and_overall(tmp_path):
    del tmp_path
    from ckg_frozen_hot_pseudocold_adapter import select_adapter_epoch

    baseline = {"hot_r10": 0.232, "hot_n10": 0.152, "overall_r10": 0.235, "overall_n10": 0.155}
    rows = [
        {
            "epoch": 1,
            "cold_r10": 0.40,
            "cold_n10": 0.30,
            "hot_r10": 0.20,
            "hot_n10": 0.10,
            "overall_r10": 0.20,
            "overall_n10": 0.10,
        }
    ]

    with pytest.raises(ValueError, match="Hot and Overall guards"):
        select_adapter_epoch(rows, baseline)


def test_stage_b_launcher_is_fresh_validation_only_and_binds_the_hot_expert():
    source = Path("run_ckg_frozen_hot_pseudocold_adapter_seed2025.ps1").read_text(encoding="utf-8")

    assert '$outputRoot = "outputs\\ckg_frozen_hot_pseudocold_adapter_seed2025"' in source
    assert '$checkpointRoot = "checkpoints\\ckg_frozen_hot_pseudocold_adapter_seed2025"' in source
    assert '$logRoot = "background_logs\\ckg_frozen_hot_pseudocold_adapter_seed2025"' in source
    assert "Seeds = @(2025)" in source
    assert "TestEvaluation = $false" in source
    assert "UseCbi = $false" in source
    assert "UseSimulator = $false" in source
    assert "UsePpo = $false" in source
    assert "UseCourseRewards = $false" in source
    assert "epoch_015.pt" in source
    assert "Invoke-NativeLogged" in source
    assert "$LASTEXITCODE" in source
    assert "adapter_preflight_result.json" in source
    assert "Formal Stage B run requires fresh roots" in source


def _make_stage_b_launcher_fixture(repo: Path, *, passed: bool) -> None:
    workspace = Path(__file__).resolve().parents[1]
    shutil.copy2(
        workspace / "run_ckg_frozen_hot_pseudocold_adapter_seed2025.ps1",
        repo / "run_ckg_frozen_hot_pseudocold_adapter_seed2025.ps1",
    )
    for relative in (
        "usim_feedback_fast3_content_delta.py",
        "fast3_delta/eval.py",
        "fast3_delta/config.py",
        "run_fast3_main_table_config.ps1",
        "paper_aaai27/main.tex",
        "ckg_frozen_hot_pseudocold_adapter.py",
        "ckg_hot_graph_preflight.py",
        "cgrc_paper_static_hin.py",
        "hin_data_common.py",
        "hin_eval_common.py",
        "lightgcn_static_hin.py",
        "processed_data_hin_clean_pop5/meta.json",
        "processed_data_hin_clean_pop5/content_emb.pt",
        "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/static_train.pkl",
        "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/static_val.pkl",
        "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json",
        "outputs/ckg_hot_graph_preflight_seed2025/preflight_result.json",
        "checkpoints/ckg_hot_graph_preflight_seed2025/epoch_015.pt",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    gate = "completed" if passed else "completed_gate_failed"
    batch_exit = 0 if passed else 2
    (repo / "py.bat").write_text(
        "@echo off\r\n"
        "if not exist \"outputs\\ckg_frozen_hot_pseudocold_adapter_seed2025\" mkdir \"outputs\\ckg_frozen_hot_pseudocold_adapter_seed2025\"\r\n"
        "> \"outputs\\ckg_frozen_hot_pseudocold_adapter_seed2025\\validation_epochs.csv\" echo epoch,cold_r10,cold_n10\r\n"
        ">> \"outputs\\ckg_frozen_hot_pseudocold_adapter_seed2025\\validation_epochs.csv\" echo 0,0.2,0.1\r\n"
        f"> \"outputs\\ckg_frozen_hot_pseudocold_adapter_seed2025\\adapter_preflight_result.json\" echo {{\"passed_stage_b_screen\":{str(passed).lower()},\"gate_status\":\"{gate}\"}}\r\n"
        f"exit /b {batch_exit}\r\n",
        encoding="utf-8",
    )
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "stageb-test@example.invalid"),
        ("git", "config", "user.name", "Stage B Test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)


@pytest.mark.parametrize(("passed", "expected_exit", "expected_status"), [(True, 0, "completed"), (False, 2, "completed_gate_failed")])
def test_stage_b_launcher_records_completed_and_gate_failed_runs(tmp_path, passed, expected_exit, expected_status):
    repo = tmp_path / "stage_b_launcher"
    repo.mkdir()
    _make_stage_b_launcher_fixture(repo, passed=passed)

    completed = subprocess.run(
        [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(repo / "run_ckg_frozen_hot_pseudocold_adapter_seed2025.ps1"), "-Repo", str(repo),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(
        (repo / "outputs/ckg_frozen_hot_pseudocold_adapter_seed2025/run_manifest.json").read_text(encoding="utf-8-sig")
    )
    assert completed.returncode == expected_exit
    assert manifest["status"] == expected_status
    assert manifest["gate_status"] == expected_status
    assert manifest["validation_epochs"] == [{"epoch": "0", "cold_r10": "0.2", "cold_n10": "0.1"}]
