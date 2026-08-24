import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_lira_configuration_is_frozen_before_test():
    manifest = json.loads(
        (ROOT / "lira_v2_final_frozen_config.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "frozen_before_test"
    assert manifest["selection"]["target"] == "validation"
    assert manifest["selection"]["metric"] == "full_cold_item_macro_n10"
    assert manifest["model"]["steps"] == 3
    assert manifest["model"]["min_fit"] == 0.15
    assert manifest["model"]["min_gain"] == 0.001
    assert manifest["training"]["refinement_loss_weight"] == 0.5
    assert manifest["training"]["stability_loss_weight"] == 0.01
    assert manifest["rules"]["test_results_must_not_change_config"] is True
    assert manifest["rules"]["checkpoints_read_only"] is True


def test_final_lira_manifest_locks_three_validation_selected_checkpoints():
    manifest = json.loads(
        (ROOT / "lira_v2_final_frozen_config.json").read_text(encoding="utf-8")
    )

    checkpoints = manifest["checkpoints"]
    assert [row["seed"] for row in checkpoints] == [2025, 2026, 2027]
    assert [row["best_epoch"] for row in checkpoints] == [12, 12, 8]
    assert all(len(row["sha256"]) == 64 for row in checkpoints)
    assert all(row["path"].endswith("validation_finished.pt") for row in checkpoints)


def test_final_lira_manifest_excludes_failed_legacy_components():
    manifest = json.loads(
        (ROOT / "lira_v2_final_frozen_config.json").read_text(encoding="utf-8")
    )

    disabled = set(manifest["disabled_components"])
    for component in (
        "actor",
        "critic",
        "ppo",
        "reward",
        "llm_score",
        "sage",
        "cgrc",
        "content_delta",
    ):
        assert component in disabled
