import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_replication_hot_config_allows_only_registered_replication_seeds():
    from ckg_hot_graph_preflight_replication import ReplicationHotConfig, validate_replication_hot_config

    cfg = ReplicationHotConfig.for_seed(2026)

    assert cfg.seed == 2026
    assert cfg.epochs == 15
    assert cfg.test_evaluation is False
    assert cfg.use_cbi is False
    assert cfg.use_simulator is False
    assert cfg.use_ppo is False
    assert cfg.use_course_rewards is False
    with pytest.raises(ValueError):
        validate_replication_hot_config(replace(cfg, seed=2025))


def test_replication_hot_loader_uses_only_meta_content_train_and_validation(tmp_path):
    from ckg_hot_graph_preflight_replication import load_hot_replication_inputs

    data_dir = tmp_path / "data"
    split_dir = tmp_path / "split"
    data_dir.mkdir()
    split_dir.mkdir()
    (data_dir / "meta.json").write_text(json.dumps({"n_users": 2, "n_items": 3}), encoding="utf-8")
    torch.save(torch.ones((3, 4)), data_dir / "content_emb.pt")
    pd.DataFrame({"u_idx": [0, 1], "i_idx": [0, 1], "popularity": [9, 9]}).to_pickle(
        split_dir / "static_train.pkl"
    )
    pd.DataFrame({"u_idx": [0, 1], "i_idx": [2, 1], "popularity": [9, 9]}).to_pickle(
        split_dir / "static_val.pkl"
    )
    (data_dir / "stream_data.pkl").write_bytes(b"must not be read")
    (split_dir / "static_test.pkl").write_bytes(b"must not be read")

    meta, content, train, validation = load_hot_replication_inputs(data_dir, split_dir)

    assert meta == {"n_users": 2, "n_items": 3}
    assert tuple(content.shape) == (3, 4)
    assert train["popularity"].tolist() == [1, 1]
    assert validation["popularity"].tolist() == [0, 1]


def test_hot_contract_uses_actual_selected_checkpoint_and_fixed_tau(tmp_path):
    from ckg_hot_replication_contract import build_selected_checkpoint_contract

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "epoch_012.pt"
    torch.save(
        {
            "epoch": 12,
            "model_state": {"item_lin.weight": torch.zeros((64, 3))},
            "config": {"seed": 2026, "emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint_path,
    )
    result = {"selected_validation_epoch": {"epoch": 12}, "passed_hot_preflight": True, "gate_status": "completed"}

    contract = build_selected_checkpoint_contract(
        seed=2026,
        result=result,
        checkpoint_dir=checkpoint_dir,
        warm_q75_audit=0.201,
    )

    assert contract["seed"] == 2026
    assert contract["epoch"] == 12
    assert contract["relative_path"] == "epoch_012.pt"
    assert contract["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert contract["fixed_trust_tau"] == pytest.approx(0.24929234)
    assert contract["warm_q75_audit"] == pytest.approx(0.201)

    torch.save(
        {
            "epoch": 12,
            "model_state": {"item_lin.weight": torch.zeros((64, 3))},
            "config": {"seed": 2027, "emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="seed"):
        build_selected_checkpoint_contract(
            seed=2026,
            result=result,
            checkpoint_dir=checkpoint_dir,
            warm_q75_audit=0.201,
        )


def test_replication_adapter_config_keeps_tau_fixed_and_rejects_unregistered_seed():
    from ckg_frozen_hot_pseudocold_adapter_replication import ReplicationAdapterConfig, validate_replication_adapter_config

    cfg = ReplicationAdapterConfig.for_seed(2027)

    assert cfg.seed == 2027
    assert cfg.trust_tau == pytest.approx(0.24929234)
    assert cfg.pseudo_cold_item_count == 102
    assert cfg.test_evaluation is False
    with pytest.raises(ValueError):
        validate_replication_adapter_config(replace(cfg, seed=2025))


def test_replication_adapter_requires_same_seed_dynamic_hot_contract(tmp_path):
    from ckg_frozen_hot_pseudocold_adapter_replication import require_replication_hot_contract

    checkpoint = tmp_path / "epoch_012.pt"
    torch.save(
        {
            "epoch": 12,
            "model_state": {"item_lin.weight": torch.zeros((64, 3))},
            "config": {"seed": 2026, "emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "passed_hot_preflight": True,
        "gate_status": "completed",
        "selected_validation_epoch": {"epoch": 12},
        "selected_checkpoint_contract": {
            "schema_version": 1,
            "seed": 2026,
            "epoch": 12,
            "relative_path": "epoch_012.pt",
            "sha256": digest,
            "architecture": {"emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
            "fixed_trust_tau": 0.24929234,
            "warm_q75_audit": 0.2,
        },
    }

    contract = require_replication_hot_contract(result, seed=2026, checkpoint_dir=tmp_path)

    assert contract["epoch"] == 12
    with pytest.raises(ValueError, match="seed"):
        require_replication_hot_contract(result, seed=2027, checkpoint_dir=tmp_path)

    result["config"] = {"seed": 2027}
    with pytest.raises(ValueError, match="seed"):
        require_replication_hot_contract(result, seed=2026, checkpoint_dir=tmp_path)

    result.pop("config")
    torch.save(
        {
            "epoch": 12,
            "model_state": {"item_lin.weight": torch.zeros((64, 3))},
            "config": {"seed": 2027, "emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint,
    )
    result["selected_checkpoint_contract"]["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="seed"):
        require_replication_hot_contract(result, seed=2026, checkpoint_dir=tmp_path)


def test_adapter_replication_launcher_checks_hot_input_hashes_before_creating_its_roots():
    source = Path("run_ckg_frozen_hot_pseudocold_replication.ps1").read_text(encoding="utf-8")

    assert "Assert-HotManifestInputHashes" in source
    assert source.index("Assert-HotManifestInputHashes -Manifest $hotManifest") < source.index(
        "New-Item -ItemType Directory -Force -Path $outputRoot"
    )


def test_hot_replication_launcher_is_seed_scoped_dynamic_and_train_validation_only():
    source = Path("run_ckg_hot_graph_preflight_replication.ps1").read_text(encoding="utf-8")

    assert "ValidateSet(2026, 2027)" in source
    assert "Invoke-NativeLogged" in source
    assert "selected_checkpoint_sha256" in source
    assert "ckg_hot_replication_contract.py" in source
    assert "static_test.pkl" not in source
    assert "stream_data.pkl" not in source
    assert "epoch_015.pt" not in source


def test_adapter_replication_launcher_binds_dynamic_hot_contract_and_is_validation_only():
    source = Path("run_ckg_frozen_hot_pseudocold_replication.ps1").read_text(encoding="utf-8")

    assert "ValidateSet(2026, 2027)" in source
    assert "TestEvaluation = $false" in source
    assert "Invoke-NativeLogged" in source
    assert "selected_checkpoint_sha256" in source
    assert "selected_checkpoint_contract" in source
    assert "epoch_015.pt" not in source


@pytest.mark.parametrize(
    ("launcher", "required_key"),
    [
        ("run_ckg_hot_graph_preflight_replication.ps1", "python_training_knobs"),
        ("run_ckg_frozen_hot_pseudocold_replication.ps1", "locked_protocol"),
    ],
)
def test_replication_launcher_dry_run_is_parseable_and_does_not_start_a_run(launcher, required_key):
    workspace = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(workspace / launcher),
            "-Repo",
            str(workspace),
            "-Seed",
            "2026",
            "-DryRun",
        ],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload[required_key]
    assert payload["test_evaluation"] is False
