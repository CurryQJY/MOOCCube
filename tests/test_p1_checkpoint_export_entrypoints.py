import json
from pathlib import Path

import torch
import pytest

from cgrc_paper_static_hin import select_final_evaluation_view
from export_p1_ckgrl_topk import (
    build_runtime_environment,
    make_read_only_torch_save,
)
from export_p1_cgrc_topk import (
    build_cgrc_runtime_environment,
    build_export_manifest,
)
from export_p1_pcgnn_topk import (
    build_pcgnn_export_manifest,
    compare_checkpoint_validation_to_report,
    compare_replay_to_report,
    parse_args as parse_pcgnn_args,
    replay_and_export_pcgnn,
    select_pcgnn_analysis_view,
)


def test_read_only_torch_save_blocks_checkpoint_writes_only(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    outside = tmp_path / "outputs" / "result.pt"
    outside.parent.mkdir()
    inside = checkpoint_dir / "finished.pt"
    guarded = make_read_only_torch_save(checkpoint_dir, torch.save)

    guarded({"value": 1}, inside)
    guarded({"value": 2}, outside)

    assert not inside.exists()
    assert torch.load(outside, weights_only=True) == {"value": 2}


def test_runtime_environment_uses_manifest_then_protected_overrides(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": {"seed": 2026},
                "env": {
                    "USIM_STATIC_SEED": "wrong",
                    "USIM_FB_CKPT_DIR": "old-checkpoint",
                    "USIM_FB_OUTPUT_DIR": "old-output",
                    "USIM_PPO_LOSS_WEIGHT": "1",
                },
            }
        ),
        encoding="utf-8",
    )

    env = build_runtime_environment(
        manifest,
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        topk_output=tmp_path / "top20.jsonl",
        top_k=20,
    )

    assert env["USIM_STATIC_SEED"] == "2026"
    assert env["USIM_SEED"] == "2026"
    assert env["USIM_FB_CKPT_DIR"] == str(tmp_path / "checkpoint")
    assert env["USIM_FB_OUTPUT_DIR"] == str(tmp_path / "output")
    assert env["P1_TOPK_EXPORT_PATH"] == str(tmp_path / "top20.jsonl")
    assert env["P1_TOPK_EXPORT_K"] == "20"
    assert env["P1_TOPK_EXPORT_MODEL"] == "ckg_rl"
    assert env["USIM_PPO_LOSS_WEIGHT"] == "1"
    assert env["USIM_FB_SAVE_CKPT"] == "1"
    assert env["USIM_FB_AUTO_RESUME"] == "1"
    assert env["USIM_FB_FORCE_FRESH"] == "0"


def test_runtime_environment_preserves_ablation_controls_and_labels_export(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": {"seed": 2027},
                "env": {
                    "USIM_STEPS": "0",
                    "USIM_USE_COURSE_REWARD": "1",
                },
            }
        ),
        encoding="utf-8",
    )

    env = build_runtime_environment(
        manifest,
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        topk_output=tmp_path / "top20.jsonl",
        top_k=20,
        model_label="ckg_rl_wo_simulator",
    )

    assert env["USIM_STEPS"] == "0"
    assert env["USIM_USE_COURSE_REWARD"] == "1"
    assert env["P1_TOPK_EXPORT_MODEL"] == "ckg_rl_wo_simulator"


def test_cgrc_runtime_environment_is_read_only_and_seed_bound(tmp_path):
    env = build_cgrc_runtime_environment(
        seed=2026,
        split_dir=tmp_path / "split",
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        topk_output=tmp_path / "top20.jsonl",
        top_k=20,
    )

    assert env["USIM_STATIC_SEED"] == "2026"
    assert env["USIM_STATIC_SPLIT_DIR"] == str(tmp_path / "split")
    assert env["USIM_BASELINE_OUTPUT_DIR"] == str(tmp_path / "output")
    assert env["CGRC_PAPER_STATIC_SEED"] == "2026"
    assert env["CGRC_PAPER_SEED"] == "2026"
    assert env["CGRC_PAPER_CKPT_DIR"] == str(tmp_path / "checkpoint")
    assert env["CGRC_PAPER_EXPORT_TOPK_PATH"] == str(tmp_path / "top20.jsonl")
    assert env["CGRC_PAPER_EXPORT_TOPK_K"] == "20"
    assert env["CGRC_PAPER_SAVE_CKPT"] == "0"
    assert env["CGRC_PAPER_AUTO_RESUME"] == "1"
    assert env["CGRC_PAPER_FORCE_FRESH"] == "0"


def test_cgrc_validation_runtime_is_explicit(tmp_path):
    env = build_cgrc_runtime_environment(
        seed=2025,
        split_dir=tmp_path / "split",
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        topk_output=tmp_path / "top20.jsonl",
        analysis_split="validation",
    )

    assert env["CGRC_PAPER_EVAL_SPLIT"] == "validation"


def test_cgrc_final_view_selects_validation_loader_and_seen_history():
    view = select_final_evaluation_view(
        "validation",
        val_loader="val-loader",
        test_loader="test-loader",
        train_seen="train-seen",
        test_seen="test-seen",
        val_cold_items="val-cold",
        test_cold_items="test-cold",
    )

    assert view.name == "validation"
    assert view.loader == "val-loader"
    assert view.seen_items == "train-seen"
    assert view.cold_items == "val-cold"


def test_cgrc_manifest_binds_checkpoint_split_scripts_and_outputs(tmp_path):
    checkpoint = tmp_path / "best.pt"
    split = tmp_path / "static_test.pkl"
    script = tmp_path / "evaluator.py"
    topk = tmp_path / "top20.jsonl"
    native = tmp_path / "result.json"
    checkpoint.write_bytes(b"checkpoint")
    split.write_bytes(b"split")
    script.write_text("print('ok')\n", encoding="utf-8")
    topk.write_text('{"seed": 2025}\n', encoding="utf-8")
    native.write_text('[{"count_full_cold": 1}]\n', encoding="utf-8")

    checkpoint_hash = "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    manifest = build_export_manifest(
        seed=2025,
        top_k=20,
        checkpoint_paths=[checkpoint],
        checkpoint_sha256_before={str(checkpoint.resolve()): checkpoint_hash},
        checkpoint_sha256_after={str(checkpoint.resolve()): checkpoint_hash},
        split_paths=[split],
        script_paths=[script],
        topk_output=topk,
        native_result=native,
        record_count=1,
    )

    assert manifest["model"] == "cgrc"
    assert manifest["seed"] == 2025
    assert manifest["top_k"] == 20
    assert manifest["record_count"] == 1
    assert manifest["checkpoints"][0]["sha256_before"] == manifest["checkpoints"][0]["sha256_after"]
    assert manifest["topk_output"]["sha256"]
    assert manifest["native_result"]["sha256"]
    assert len(manifest["split_files"]) == 1
    assert len(manifest["script_files"]) == 1


def test_cgrc_manifest_rejects_checkpoint_mutation(tmp_path):
    checkpoint = tmp_path / "best.pt"
    topk = tmp_path / "top20.jsonl"
    native = tmp_path / "result.json"
    checkpoint.write_bytes(b"checkpoint")
    topk.write_text("{}\n", encoding="utf-8")
    native.write_text("[]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed during export"):
        build_export_manifest(
            seed=2025,
            top_k=20,
            checkpoint_paths=[checkpoint],
            checkpoint_sha256_before={str(checkpoint.resolve()): "before"},
            checkpoint_sha256_after={str(checkpoint.resolve()): "after"},
            split_paths=[],
            script_paths=[],
            topk_output=topk,
            native_result=native,
            record_count=1,
        )


def test_pcgnn_replay_exports_raw_ids_after_native_seen_masking(tmp_path):
    class FakeInteraction(dict):
        pass

    class FakePCGNN(torch.nn.Module):
        ITEM_SEQ = "item_seq"
        ITEM_SEQ_LEN = "item_length"

        def full_sort_predict(self, interaction):
            batch_size = int(interaction[self.ITEM_SEQ].shape[0])
            scores = torch.tensor([[99.0, 0.8, 0.9, 0.7, 0.6]], dtype=torch.float32)
            return scores.repeat(batch_size, 1)

    output = tmp_path / "pcgnn_top20.jsonl"
    examples = [
        {
            "user": 7,
            "history": [2],
            "target": 3,
            "raw_item": 103,
            "popularity": 0,
        }
    ]

    result = replay_and_export_pcgnn(
        model=FakePCGNN(),
        interaction_cls=FakeInteraction,
        examples=examples,
        user_seen_items={7: {2}},
        internal_to_raw={1: 101, 2: 102, 3: 103, 4: 104},
        max_len=3,
        batch_size=1,
        top_k=3,
        output_path=output,
        metadata={"model": "pcgnn", "seed": 2025},
        device=torch.device("cpu"),
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["recommended_item_ids"] == [101, 103, 104]
    assert rows[0]["recommended_scores"] == pytest.approx([0.8, 0.7, 0.6])
    assert rows[0]["target_item_id"] == 103
    assert rows[0]["sample_index"] == 0
    assert result["record_count"] == 1
    assert result["metrics"]["full_cold_item_macro"]["R@3"] == pytest.approx(1.0)


def test_pcgnn_validation_view_uses_validation_examples_only():
    selected = select_pcgnn_analysis_view(
        "validation",
        validation_examples=[{"target": 11}],
        test_examples=[{"target": 22}],
    )

    assert selected == [{"target": 11}]


def test_pcgnn_analysis_split_parses_validation_target():
    args = parse_pcgnn_args(["--seed", "2025", "--analysis-split", "validation"])

    assert args.analysis_split == "validation"


def test_pcgnn_replay_comparison_gates_counts_and_metric_drift():
    native = {
        "test_sequence_examples": 10,
        "test": {
            "count_full_cold_item_macro": 2,
            "rows_full_cold": 10,
            "full_cold_item_macro": {"R@10": 0.25, "N@10": 0.125},
        },
    }
    replay = {
        "record_count": 10,
        "metrics": {
            "count_full_cold_item_macro": 2,
            "rows_full_cold": 10,
            "full_cold_item_macro": {"R@10": 0.25, "N@10": 0.125},
        },
    }

    comparison = compare_replay_to_report(replay, native, tolerance=1e-12)

    assert comparison["passed"] is True
    assert comparison["max_abs_metric_drift"] == pytest.approx(0.0)

    replay["metrics"]["full_cold_item_macro"]["N@10"] = 0.2
    with pytest.raises(RuntimeError, match="metric drift"):
        compare_replay_to_report(replay, native, tolerance=1e-12)

    recorded = compare_replay_to_report(
        replay,
        native,
        tolerance=1e-12,
        raise_on_metric_drift=False,
    )
    assert recorded["passed"] is False
    assert recorded["max_abs_metric_drift"] == pytest.approx(0.075)


def test_pcgnn_checkpoint_validation_proves_best_state_identity():
    report = {
        "best_epoch": 4,
        "best_validation_score": 0.25,
        "validation_metric": "full_cold_item_macro.N@10",
        "validation": {
            "count_full_cold_item_macro": 2,
            "rows_full_cold": 10,
            "full_cold_item_macro": {"R@10": 0.5, "N@10": 0.25},
        },
    }
    checkpoint = {
        "epoch": 4,
        "validation_metric": "full_cold_item_macro.N@10",
        "validation_score": 0.25,
    }
    replay_validation = {
        "count_full_cold_item_macro": 2,
        "rows_full_cold": 10,
        "full_cold_item_macro": {"R@10": 0.5, "N@10": 0.25},
    }

    comparison = compare_checkpoint_validation_to_report(
        replay_validation,
        report,
        checkpoint,
        tolerance=1e-12,
    )

    assert comparison["passed"] is True
    assert comparison["max_abs_metric_drift"] == pytest.approx(0.0)

    checkpoint["epoch"] = 5
    with pytest.raises(RuntimeError, match="checkpoint epoch"):
        compare_checkpoint_validation_to_report(
            replay_validation,
            report,
            checkpoint,
            tolerance=1e-12,
        )


def test_pcgnn_manifest_binds_replay_and_rejects_checkpoint_mutation(tmp_path):
    checkpoint = tmp_path / "best_model.pt"
    report = tmp_path / "report.json"
    config = tmp_path / "config.yaml"
    split = tmp_path / "static_test.pkl"
    script = tmp_path / "exporter.py"
    topk = tmp_path / "pcgnn_top20.jsonl"
    replay = tmp_path / "pcgnn_replay_result.json"
    checkpoint.write_bytes(b"checkpoint")
    report.write_text("{}\n", encoding="utf-8")
    config.write_text("dataset: demo\n", encoding="utf-8")
    split.write_bytes(b"split")
    script.write_text("print('ok')\n", encoding="utf-8")
    topk.write_text('{"sample_index": 0}\n', encoding="utf-8")
    replay.write_text('{"record_count": 1}\n', encoding="utf-8")
    digest = "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"

    manifest = build_pcgnn_export_manifest(
        seed=2025,
        top_k=20,
        checkpoint_path=checkpoint,
        checkpoint_sha256_before=digest,
        checkpoint_sha256_after=digest,
        report_path=report,
        config_path=config,
        split_paths=[split],
        script_paths=[script],
        topk_output=topk,
        replay_result=replay,
        record_count=1,
    )

    assert manifest["model"] == "pcgnn"
    assert manifest["restored_state"] == "best_model.pt:model_state_dict"
    assert manifest["checkpoint"]["sha256_before"] == manifest["checkpoint"]["sha256_after"]
    assert manifest["report"]["sha256"]
    assert manifest["config"]["sha256"]
    assert manifest["replay_result"]["sha256"]

    with pytest.raises(RuntimeError, match="changed during export"):
        build_pcgnn_export_manifest(
            seed=2025,
            top_k=20,
            checkpoint_path=checkpoint,
            checkpoint_sha256_before="before",
            checkpoint_sha256_after="after",
            report_path=report,
            config_path=config,
            split_paths=[split],
            script_paths=[script],
            topk_output=topk,
            replay_result=replay,
            record_count=1,
        )
