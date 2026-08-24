from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch


def test_soft_anchor_loss_is_zero_for_content_and_scales_by_fixed_tau():
    from ckg_stageb_component_screen import soft_anchor_loss

    adapted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    selected = torch.tensor([False, True])

    loss = soft_anchor_loss(adapted, anchors, selected, trust_tau=0.5)

    assert loss.item() == pytest.approx(8.0)


def test_component_config_only_allows_registered_strict_soft_anchor_screen():
    from ckg_stageb_component_screen import ComponentScreenConfig, validate_component_config

    cfg = ComponentScreenConfig.for_seed(2027, component_name="soft_anchor_l2")

    validate_component_config(cfg)
    assert cfg.soft_anchor_weight == pytest.approx(0.10)
    with pytest.raises(ValueError, match="test evaluation"):
        validate_component_config(replace(cfg, test_evaluation=True))
    with pytest.raises(ValueError, match="component"):
        validate_component_config(replace(cfg, component_name="legacy_gate"))
    with pytest.raises(ValueError, match="data directory"):
        validate_component_config(replace(cfg, data_dir="other_data"))
    with pytest.raises(ValueError, match="phase"):
        validate_component_config(replace(cfg, phase="replication"))


@pytest.mark.parametrize(
    "field",
    ("parent_result_path", "hot_output_dir", "hot_checkpoint_dir"),
)
def test_component_config_binds_seed_specific_parent_and_hot_artifacts(field):
    from ckg_stageb_component_screen import ComponentScreenConfig, validate_component_config

    cfg = ComponentScreenConfig.for_seed(2027, component_name="soft_anchor_l2")

    with pytest.raises(ValueError, match="canonical"):
        validate_component_config(replace(cfg, **{field: "outputs/not_the_registered_artifact"}))


def test_candidate_acceptance_requires_immutable_guard_and_cold_gain():
    from ckg_stageb_component_screen import decide_single_seed_screen

    immutable = {
        "hot_r10": 0.240,
        "hot_n10": 0.156,
        "overall_r10": 0.241,
        "overall_n10": 0.157,
    }
    incumbent = {"cold_r10": 0.241, "cold_n10": 0.172}
    candidate = {
        "cold_r10": 0.241,
        "cold_n10": 0.175,
        "hot_r10": 0.238,
        "hot_n10": 0.154,
        "overall_r10": 0.239,
        "overall_n10": 0.155,
    }

    assert decide_single_seed_screen(candidate, incumbent, immutable) == "provisionally_accepted"

    candidate["overall_n10"] = 0.153
    assert decide_single_seed_screen(candidate, incumbent, immutable) == "rejected_retention_guard"

    candidate["overall_n10"] = 0.155
    candidate["cold_n10"] = 0.174
    assert decide_single_seed_screen(candidate, incumbent, immutable) == "rejected_insufficient_cold_gain"


def test_replication_phase_records_completion_without_contradicting_final_mean_rule():
    from ckg_stageb_component_screen import (
        ComponentScreenConfig,
        component_selection_decision,
        validate_component_config,
    )

    cfg = replace(
        ComponentScreenConfig.for_seed(2026, component_name="soft_anchor_l2"),
        phase="replication",
    )
    validate_component_config(cfg)
    candidate = {
        "cold_r10": 0.233,
        "cold_n10": 0.160,
        "hot_r10": 0.224,
        "hot_n10": 0.146,
        "overall_r10": 0.225,
        "overall_n10": 0.147,
    }
    incumbent = {"cold_r10": 0.232, "cold_n10": 0.159}
    immutable = {"hot_r10": 0.225, "hot_n10": 0.147, "overall_r10": 0.225, "overall_n10": 0.147}

    assert component_selection_decision(candidate, incumbent, immutable, cfg) == "replication_completed"


def test_component_training_loss_adds_soft_anchor_only_for_selected_pseudocold_items():
    from ckg_stageb_component_screen import component_training_loss

    ranking = torch.tensor(2.0)
    adapted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    selected = torch.tensor([False, True])

    total, anchor = component_training_loss(
        ranking,
        adapted,
        anchors,
        selected,
        trust_tau=0.5,
        soft_anchor_weight=0.1,
    )

    assert anchor.item() == pytest.approx(8.0)
    assert total.item() == pytest.approx(2.8)


def test_component_runner_source_is_validation_only():
    source = Path("ckg_stageb_component_screen.py").read_text(encoding="utf-8")

    assert "static_test.pkl" not in source
    assert "stream_data.pkl" not in source


def test_component_source_inventory_binds_runner_and_runtime_dependencies():
    from ckg_stageb_component_screen import component_source_files

    files = component_source_files()

    assert set(files) == {
        "component_runner",
        "adapter_helper",
        "replication_helper",
        "cgrc_model",
        "data_common",
        "eval_common",
        "lightgcn",
    }
    assert all(path.is_file() for path in files.values())


def test_parent_incumbent_rejects_noncompleted_or_test_derived_records(tmp_path):
    from ckg_stageb_component_screen import _require_parent_incumbent

    source = Path(
        "outputs/ckg_frozen_hot_pseudocold_adapter_replication_seed2027/adapter_preflight_result.json"
    )
    parent = json.loads(source.read_text(encoding="utf-8"))
    contract = parent["hot_checkpoint_contract"]
    parent["status"] = "completed_gate_failed"
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")

    with pytest.raises(ValueError, match="completed"):
        _require_parent_incumbent(path, seed=2027, contract=contract)

    parent["status"] = "completed"
    parent["test_evaluation"] = True
    path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="validation-only"):
        _require_parent_incumbent(path, seed=2027, contract=contract)

    parent["test_evaluation"] = False
    parent["config"]["delta_reg_weight"] = 0.1
    path.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="config mismatch"):
        _require_parent_incumbent(path, seed=2027, contract=contract)


@pytest.mark.parametrize(
    ("after_field", "message"),
    (
        ("source_sha256_after", "source"),
        ("hot_artifacts_sha256_after", "Hot artifact"),
    ),
)
def test_parent_run_manifest_binds_unchanged_sources_and_hot_artifacts(
    tmp_path, after_field, message
):
    from ckg_stageb_component_screen import (
        ComponentScreenConfig,
        _require_parent_run_manifest,
    )

    cfg = ComponentScreenConfig.for_seed(2027, component_name="soft_anchor_l2")
    parent_path = Path(cfg.parent_result_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest_path = parent_path.parent / "run_manifest.json"

    manifest = _require_parent_run_manifest(
        manifest_path,
        cfg=cfg,
        parent_path=parent_path,
        contract=parent["hot_checkpoint_contract"],
    )
    assert manifest["status"] == "completed"

    tampered = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    key = next(iter(tampered[after_field]))
    tampered[after_field][key] = "0" * 64
    altered_path = tmp_path / "run_manifest.json"
    altered_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _require_parent_run_manifest(
            altered_path,
            cfg=cfg,
            parent_path=parent_path,
            contract=parent["hot_checkpoint_contract"],
        )


def test_campaign_queue_starts_on_2027_and_replication_requires_provisional_acceptance():
    from ckg_stageb_component_campaign import campaign_steps

    steps = campaign_steps()

    assert [(step.component_name, step.seed, step.requires_previous_acceptance) for step in steps] == [
        ("soft_anchor_l2", 2027, False),
        ("soft_anchor_l2", 2026, True),
    ]


@pytest.mark.parametrize(
    ("result_patch", "config_patch", "message"),
    (
        ({}, {"seed": 2026}, "seed"),
        ({"phase": "replication"}, {"phase": "replication"}, "phase"),
        ({"component_name": "other_component"}, {"component_name": "other_component"}, "component"),
        ({"test_evaluation": True}, {"test_evaluation": True}, "test evaluation"),
    ),
)
def test_campaign_rejects_result_that_does_not_attest_to_queued_step(
    result_patch, config_patch, message
):
    from ckg_stageb_component_campaign import CampaignStep, validate_campaign_step_result

    step = CampaignStep("soft_anchor_l2", 2027, False)
    result = {
        "experiment": "ckg_stageb_component_screen",
        "status": "completed",
        "component_name": "soft_anchor_l2",
        "phase": "screen",
        "test_evaluation": False,
        "config": {
            "seed": 2027,
            "component_name": "soft_anchor_l2",
            "phase": "screen",
            "test_evaluation": False,
        },
    }
    result.update(result_patch)
    result["config"].update(config_patch)

    with pytest.raises(ValueError, match=message):
        validate_campaign_step_result(result, step)


def test_final_component_decision_requires_guarded_mean_cold_gain():
    from ckg_stageb_component_campaign import finalize_component_decision

    incumbent = {2027: {"cold_r10": 0.241, "cold_n10": 0.172}, 2026: {"cold_r10": 0.232, "cold_n10": 0.159}}
    accepted = {
        2027: {"cold_r10": 0.241, "cold_n10": 0.176, "passes_retention_guards": True},
        2026: {"cold_r10": 0.233, "cold_n10": 0.162, "passes_retention_guards": True},
    }

    assert finalize_component_decision(accepted, incumbent) == "accepted"

    accepted[2026]["cold_n10"] = 0.160
    assert finalize_component_decision(accepted, incumbent) == "rejected_mean_cold_gain"


def test_final_incumbent_promotes_only_an_accepted_component():
    from ckg_stageb_component_campaign import build_final_incumbent

    promoted = build_final_incumbent(
        "accepted",
        component_name="soft_anchor_l2",
        campaign_id="unit",
        result_paths={2026: "seed2026.json", 2027: "seed2027.json"},
    )
    rejected = build_final_incumbent(
        "rejected_mean_cold_gain",
        component_name="soft_anchor_l2",
        campaign_id="unit",
        result_paths={2027: "seed2027.json"},
    )

    assert promoted["admitted_components"] == ["soft_anchor_l2"]
    assert promoted["per_seed_component_results"]["2026"] == "seed2026.json"
    assert rejected["admitted_components"] == []


def test_component_inventory_records_enabled_redundant_and_protocol_rejected_items():
    from ckg_stageb_component_campaign import component_inventory

    inventory = {row["component_name"]: row for row in component_inventory()}

    assert inventory["shared_content_projector"]["status"] == "already_enabled"
    assert inventory["pseudo_cold_edge_masking"]["status"] == "already_enabled"
    assert inventory["legacy_cbi_trust_cone"]["status"] == "redundant"
    assert inventory["legacy_hot_id_content_gate"]["status"] == "protocol_rejected"
    assert inventory["legacy_simulator_rollout"]["status"] == "protocol_rejected"
    assert inventory["legacy_ppo"]["status"] == "protocol_rejected"
    assert inventory["legacy_course_reward"]["status"] == "protocol_rejected"


def test_campaign_description_is_strict_validation_only_and_serial():
    from ckg_stageb_component_campaign import CampaignConfig, describe_campaign

    description = describe_campaign(CampaignConfig.for_campaign_id("unit_test"))

    assert description["scope"] == "validation_only"
    assert description["test_evaluation"] is False
    assert description["steps"][0]["seed"] == 2027
    assert description["steps"][1]["requires_previous_acceptance"] is True


def test_campaign_ledger_appends_jsonl_events_and_keeps_snapshot_separate(tmp_path):
    from ckg_stageb_component_campaign import _write_ledgers

    ledger = {"events": [{"event": "campaign_started", "campaign_id": "unit"}], "runs": []}
    _write_ledgers(tmp_path, ledger)
    ledger["events"].append({"event": "run_completed", "campaign_id": "unit", "seed": 2027})
    _write_ledgers(tmp_path, ledger)

    lines = (tmp_path / "component_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["event"] for line in lines] == ["campaign_started", "run_completed"]
    assert (tmp_path / "component_ledger_snapshot.json").is_file()


def test_campaign_source_inventory_binds_controller_launcher_and_stageb_runtime_closure():
    from ckg_stageb_component_campaign import campaign_source_files

    files = campaign_source_files()

    assert set(files) == {
        "campaign_controller",
        "campaign_launcher",
        "component_runner",
        "adapter_helper",
        "replication_helper",
        "cgrc_model",
        "data_common",
        "eval_common",
        "lightgcn",
    }
    assert all(path.is_file() for path in files.values())


def test_campaign_aborts_before_next_seed_when_the_runtime_closure_drifts(tmp_path, monkeypatch):
    import ckg_stageb_component_campaign as campaign

    calls = []
    stable_hashes = {"runtime": "before"}
    changed_hashes = {"runtime": "after"}

    def fake_runner(cfg):
        calls.append(cfg)
        selected = {
            "epoch": 1,
            "cold_r10": 0.25,
            "cold_n10": 0.18,
            "hot_r10": 0.24,
            "hot_n10": 0.16,
            "overall_r10": 0.24,
            "overall_n10": 0.16,
            "passes_retention_guards": True,
        }
        return {
            "experiment": "ckg_stageb_component_screen",
            "status": "completed",
            "component_name": cfg.component_name,
            "phase": cfg.phase,
            "component_decision": "provisionally_accepted",
            "test_evaluation": False,
            "config": {
                "seed": cfg.seed,
                "component_name": cfg.component_name,
                "phase": cfg.phase,
                "test_evaluation": False,
            },
            "selected_validation_epoch": selected,
            "parent_selected_validation_epoch": selected,
        }

    def fake_hashes():
        return stable_hashes if not calls else changed_hashes

    monkeypatch.setattr(campaign, "run_component_screen", fake_runner)
    monkeypatch.setattr(campaign, "_campaign_source_hashes", fake_hashes)
    cfg = campaign.CampaignConfig(
        campaign_id="source_drift",
        output_root=str(tmp_path / "output"),
        checkpoint_root=str(tmp_path / "checkpoint"),
        log_root=str(tmp_path / "logs"),
    )

    summary = campaign.run_component_campaign(cfg)

    assert summary["campaign_status"] == "failed"
    assert [run_cfg.seed for run_cfg in calls] == [2027]


def test_campaign_finalizer_is_defined_before_cli_invocation():
    source = Path("ckg_stageb_component_campaign.py").read_text(encoding="utf-8")

    assert source.index("def finalize_component_decision") < source.index('if __name__ == "__main__":')


def test_campaign_launcher_is_backgrounded_and_supports_a_no_write_dry_run():
    source = Path("run_ckg_stageb_component_campaign.ps1").read_text(encoding="utf-8")

    assert "[switch]$DryRun" in source
    assert "ckg_stageb_component_campaign.py" in source
    assert "Start-Process" in source
    assert "-WindowStyle Hidden" in source
    assert "--dry-run" in source
