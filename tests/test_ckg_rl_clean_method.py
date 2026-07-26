"""Contracts for the publishable clean CKG-RL methodology facade."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ckg_rl_clean_method import (
    CleanMethodConfig,
    CleanMethodStage,
    build_clean_method_manifest,
    validate_clean_method_config,
)


def test_clean_method_manifest_declares_three_separated_stages():
    manifest = build_clean_method_manifest(CleanMethodConfig())

    assert [stage["name"] for stage in manifest["stages"]] == [
        "behavior_teacher",
        "cold_course_generator",
        "bounded_policy_refinement",
    ]
    assert manifest["stages"][0]["may_read"] == ["H_train", "H_val"]
    assert manifest["stages"][1]["may_read"] == ["H_G", "teacher_item_vectors", "course_content"]
    assert manifest["stages"][2]["may_read"] == [
        "P_train",
        "P_val",
        "teacher_user_vectors",
        "teacher_item_vectors",
        "course_graph",
    ]
    assert manifest["inference"]["hot_route"] == "frozen_teacher_item_vectors"
    assert manifest["inference"]["cold_route"] == "content_generator_then_target_free_policy"
    assert manifest["evaluation"]["ranking_bank"] == "single_unified_catalog_bank"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("legacy_dual_vector_eval", True, "legacy dual-vector"),
        ("positive_recompute_eval", True, "positive recompute"),
        ("random_id_dropout", True, "random ID dropout"),
        ("inference_uses_oracle_targets", True, "oracle targets"),
        ("hot_items_mutable_in_policy", True, "hot item"),
        ("candidate_mode", "positive_residual", "legal_state_retrieval"),
    ],
)
def test_clean_method_config_rejects_legacy_or_target_leaking_controls(field, value, match):
    config = CleanMethodConfig(**{field: value})

    with pytest.raises(ValueError, match=match):
        validate_clean_method_config(config)


def test_clean_method_config_requires_pretest_policy_selection():
    config = CleanMethodConfig(selection_reads_test=True)

    with pytest.raises(ValueError, match="test split"):
        validate_clean_method_config(config)


def test_clean_method_manifest_is_json_serializable_and_records_policy_bounds():
    config = CleanMethodConfig(max_policy_delta=0.35, stability_anchor_count=128)
    manifest = build_clean_method_manifest(config)

    encoded = json.dumps(manifest, sort_keys=True)

    assert "max_policy_delta" in encoded
    assert manifest["policy_bounds"] == {
        "max_policy_delta": 0.35,
        "stability_anchor_count": 128,
        "hot_items_mutable_in_policy": False,
    }


def test_clean_method_stage_names_are_stable_for_downstream_manifests():
    assert [stage.value for stage in CleanMethodStage] == [
        "behavior_teacher",
        "cold_course_generator",
        "bounded_policy_refinement",
    ]
