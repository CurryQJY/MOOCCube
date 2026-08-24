import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch


def test_test_replay_uses_the_registered_checkpoint_and_rejects_hash_drift(tmp_path):
    from ckg_hot_graph_test_replay import resolve_frozen_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "epoch_007.pt"
    torch.save(
        {
            "epoch": 7,
            "model_state": {"item_lin.weight": torch.zeros((2, 3))},
            "config": {"seed": 2025, "emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint_path,
    )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    result_path = tmp_path / "preflight_result.json"
    result_path.write_text(
        json.dumps(
            {
                "test_evaluation": False,
                "passed_hot_preflight": True,
                "selected_validation_epoch": {"epoch": 7},
                "selected_checkpoint_contract": {
                    "seed": 2025,
                    "epoch": 7,
                    "relative_path": "epoch_007.pt",
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )

    frozen = resolve_frozen_checkpoint(
        result_path=result_path,
        checkpoint_dir=checkpoint_dir,
        seed=2025,
    )

    assert frozen["epoch"] == 7
    assert frozen["path"] == checkpoint_path
    assert frozen["sha256"] == digest

    checkpoint_path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="sha256"):
        resolve_frozen_checkpoint(
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            seed=2025,
        )


def test_test_replay_derives_a_verified_contract_for_legacy_seed_2025_result(tmp_path):
    from ckg_hot_graph_test_replay import resolve_frozen_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "epoch_015.pt"
    torch.save(
        {
            "epoch": 15,
            "model_state": {"item_lin.weight": torch.zeros((2, 3))},
            "config": {"seed": 2025, "emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
        },
        checkpoint_path,
    )
    result_path = tmp_path / "preflight_result.json"
    result_path.write_text(
        json.dumps(
            {
                "config": {"seed": 2025},
                "test_evaluation": False,
                "passed_hot_preflight": True,
                "selected_validation_epoch": {"epoch": 15},
            }
        ),
        encoding="utf-8",
    )

    frozen = resolve_frozen_checkpoint(
        result_path=result_path,
        checkpoint_dir=checkpoint_dir,
        seed=2025,
        expected_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )

    assert frozen["epoch"] == 15
    assert frozen["path"] == checkpoint_path
    assert frozen["sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    checkpoint_path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="sha256"):
        resolve_frozen_checkpoint(
            result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            seed=2025,
            expected_sha256=frozen["sha256"],
        )


def test_test_replay_loads_only_train_and_test_inputs(tmp_path):
    from ckg_hot_graph_test_replay import load_test_replay_inputs

    data_dir = tmp_path / "data"
    split_dir = tmp_path / "split"
    data_dir.mkdir()
    split_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"n_users": 2, "n_items": 3}), encoding="utf-8"
    )
    torch.save(torch.zeros((3, 4)), data_dir / "content_emb.pt")
    pd.DataFrame({"u_idx": [0, 1], "i_idx": [1, 1]}).to_pickle(
        split_dir / "static_train.pkl"
    )
    pd.DataFrame({"u_idx": [0], "i_idx": [2]}).to_pickle(split_dir / "static_test.pkl")

    meta, content, train, test = load_test_replay_inputs(data_dir, split_dir)

    assert meta["n_items"] == 3
    assert tuple(content.shape) == (3, 4)
    assert train["popularity"].tolist() == [2, 2]
    assert test["popularity"].tolist() == [0]


def test_test_replay_summary_uses_item_counts_and_preserves_selected_epoch():
    from ckg_hot_graph_test_replay import build_test_result

    result = build_test_result(
        frozen={"epoch": 15, "sha256": "abc"},
        cold={"R@5": 0.4, "R@10": 0.5, "N@5": 0.3, "N@10": 0.35},
        cold_count=2,
        hot={"R@5": 0.1, "R@10": 0.2, "N@5": 0.05, "N@10": 0.1},
        hot_count=8,
    )

    assert result["test_evaluation"] is True
    assert result["selected_validation_epoch"] == 15
    assert result["selected_checkpoint_sha256"] == "abc"
    assert result["cold_item_count"] == 2
    assert result["hot_item_count"] == 8
    assert result["overall_r10"] == pytest.approx(0.26)
    assert result["overall_n10"] == pytest.approx(0.15)


def test_test_replay_runs_a_frozen_checkpoint_without_a_validation_split(tmp_path):
    import cgrc_paper_static_hin as cgrc
    from ckg_hot_graph_test_replay import HotTestReplayConfig, run_test_replay

    data_dir = tmp_path / "data"
    split_dir = tmp_path / "split"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "test_output"
    for path in (data_dir, split_dir, checkpoint_dir):
        path.mkdir()
    content = torch.randn((4, 5))
    (data_dir / "meta.json").write_text(
        json.dumps({"n_users": 2, "n_items": 4}), encoding="utf-8"
    )
    torch.save(content, data_dir / "content_emb.pt")
    pd.DataFrame({"u_idx": [0, 1], "i_idx": [0, 1]}).to_pickle(
        split_dir / "static_train.pkl"
    )
    pd.DataFrame({"u_idx": [0, 1], "i_idx": [2, 1]}).to_pickle(
        split_dir / "static_test.pkl"
    )
    model = cgrc.CGRCNet(2, 4, 5, 4, 4, content)
    checkpoint_path = checkpoint_dir / "epoch_001.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state": model.state_dict(),
            "config": {
                "seed": 2025,
                "emb_dim": 4,
                "mlp_hidden": 4,
                "layers_full": 1,
                "cold_threshold": 1,
            },
        },
        checkpoint_path,
    )
    result_path = tmp_path / "preflight_result.json"
    result_path.write_text(
        json.dumps(
            {
                "config": {"seed": 2025},
                "test_evaluation": False,
                "passed_hot_preflight": True,
                "selected_validation_epoch": {"epoch": 1},
            }
        ),
        encoding="utf-8",
    )

    result = run_test_replay(
        HotTestReplayConfig(
            seed=2025,
            data_dir=data_dir,
            split_dir=split_dir,
            source_result_path=result_path,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            device="cpu",
        )
    )

    assert result["test_evaluation"] is True
    assert result["selected_validation_epoch"] == 1
    assert result["cold_item_count"] == 1
    assert result["hot_item_count"] == 1
    assert (output_dir / "test_result.json").is_file()


def test_registered_replays_use_only_the_three_frozen_preflight_sources(tmp_path):
    from ckg_hot_graph_test_replay import registered_test_replay_configs

    configs = registered_test_replay_configs(output_root=tmp_path / "replay")

    assert [cfg.seed for cfg in configs] == [2025, 2026, 2027]
    assert all("static_val.pkl" not in str(cfg.split_dir) for cfg in configs)
    assert all("ckg_hot_graph_preflight" in str(cfg.source_result_path) for cfg in configs)
    assert all("ckg_hot_graph_preflight" in str(cfg.checkpoint_dir) for cfg in configs)
    assert all(len(cfg.expected_sha256) == 64 for cfg in configs)
    assert [Path(cfg.output_dir).name for cfg in configs] == [
        "seed2025",
        "seed2026",
        "seed2027",
    ]


def test_test_replay_dry_run_exposes_only_frozen_test_inputs(tmp_path):
    from ckg_hot_graph_test_replay import build_test_replay_dry_run

    payload = build_test_replay_dry_run(output_root=tmp_path / "replay")

    assert payload["test_evaluation"] is True
    assert payload["test_history"] == "train_only"
    assert payload["seeds"] == [2025, 2026, 2027]
    assert all(row["selection"] == "frozen_validation_checkpoint" for row in payload["runs"])
    assert "epochs" not in payload
    assert "optimizer" not in payload


def test_registered_replay_batch_requires_a_fresh_root_and_writes_audit_files(
    tmp_path, monkeypatch
):
    import ckg_hot_graph_test_replay as replay

    output_root = tmp_path / "hot_test_replay"
    configs = [
        replay.HotTestReplayConfig(
            seed=seed,
            data_dir=tmp_path / "data",
            split_dir=tmp_path / "split",
            source_result_path=tmp_path / f"source_{seed}.json",
            checkpoint_dir=tmp_path / f"checkpoint_{seed}",
            output_dir=output_root / f"seed{seed}",
            device="cpu",
        )
        for seed in (2025, 2026, 2027)
    ]
    monkeypatch.setattr(
        replay,
        "registered_test_replay_configs",
        lambda *, output_root, device="": configs,
    )

    def fake_run(cfg):
        return {
            "seed": cfg.seed,
            "cold_r10": cfg.seed / 10000.0,
            "hot_r10": cfg.seed / 20000.0,
            "overall_r10": cfg.seed / 15000.0,
            "selected_validation_epoch": 15,
            "selected_checkpoint_sha256": f"sha-{cfg.seed}",
        }

    monkeypatch.setattr(replay, "run_test_replay", fake_run)

    summary = replay.run_registered_test_replays(output_root=output_root, device="cpu")

    assert [run["seed"] for run in summary["runs"]] == [2025, 2026, 2027]
    assert (output_root / "test_replay_summary.json").is_file()
    assert (output_root / "test_replay_summary.csv").is_file()
    manifest = json.loads((output_root / "test_replay_manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_history"] == "train_only"
    assert [entry["seed"] for entry in manifest["sources"]] == [2025, 2026, 2027]
    rows = pd.read_csv(output_root / "test_replay_summary.csv")
    assert rows["seed"].tolist() == [2025, 2026, 2027]

    with pytest.raises(FileExistsError, match="fresh"):
        replay.run_registered_test_replays(output_root=output_root, device="cpu")


def test_test_replay_cli_dry_run_prints_contract_without_creating_output(tmp_path, monkeypatch, capsys):
    from ckg_hot_graph_test_replay import main

    output_root = tmp_path / "dry_run_output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ckg_hot_graph_test_replay.py",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--dry-run",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["test_history"] == "train_only"
    assert payload["seeds"] == [2025, 2026, 2027]
    assert not output_root.exists()
