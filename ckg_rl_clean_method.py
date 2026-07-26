"""Clean CKG-RL method contract shared by launchers and reports.

This module is deliberately small: it records the publishable method boundary
and rejects legacy controls that caused the old CKG-RL route to mix training,
inference, and evaluation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CleanMethodStage(str, Enum):
    """Stable names for the three clean CKG-RL stages."""

    BEHAVIOR_TEACHER = "behavior_teacher"
    COLD_COURSE_GENERATOR = "cold_course_generator"
    BOUNDED_POLICY_REFINEMENT = "bounded_policy_refinement"


@dataclass(frozen=True)
class CleanMethodConfig:
    """Method-level switches that must stay clean for publishable runs."""

    legacy_dual_vector_eval: bool = False
    positive_recompute_eval: bool = False
    random_id_dropout: bool = False
    inference_uses_oracle_targets: bool = False
    hot_items_mutable_in_policy: bool = False
    selection_reads_test: bool = False
    candidate_mode: str = "legal_state_retrieval"
    max_policy_delta: float = 1.0
    stability_anchor_count: int = 0


def validate_clean_method_config(config: CleanMethodConfig) -> None:
    """Reject controls that would reintroduce the legacy CKG-RL flaws."""

    if bool(config.legacy_dual_vector_eval):
        raise ValueError("clean method forbids legacy dual-vector evaluation")
    if bool(config.positive_recompute_eval):
        raise ValueError("clean method forbids positive recompute evaluation")
    if bool(config.random_id_dropout):
        raise ValueError("clean method forbids random ID dropout")
    if bool(config.inference_uses_oracle_targets):
        raise ValueError("clean inference forbids oracle targets")
    if bool(config.hot_items_mutable_in_policy):
        raise ValueError("policy refinement must not mutate hot item vectors")
    if bool(config.selection_reads_test):
        raise ValueError("policy selection must not read the test split")
    if str(config.candidate_mode).strip().lower() != "legal_state_retrieval":
        raise ValueError("clean method requires legal_state_retrieval candidate mode")
    if float(config.max_policy_delta) < 0.0:
        raise ValueError("max_policy_delta must be non-negative")
    if int(config.stability_anchor_count) < 0:
        raise ValueError("stability_anchor_count must be non-negative")


def build_clean_method_manifest(config: CleanMethodConfig) -> dict[str, Any]:
    """Build a JSON-serializable method contract for a clean CKG-RL run."""

    validate_clean_method_config(config)
    return {
        "method": "clean_ckg_rl",
        "stages": [
            {
                "name": CleanMethodStage.BEHAVIOR_TEACHER.value,
                "may_read": ["H_train", "H_val"],
                "must_not_read": ["C_val", "C_test", "P_train", "P_val"],
                "output": ["teacher_user_vectors", "teacher_item_vectors"],
            },
            {
                "name": CleanMethodStage.COLD_COURSE_GENERATOR.value,
                "may_read": ["H_G", "teacher_item_vectors", "course_content"],
                "must_not_read": ["P_train", "P_val", "C_val", "C_test", "item_id_embedding"],
                "output": ["cold_initial_vectors"],
            },
            {
                "name": CleanMethodStage.BOUNDED_POLICY_REFINEMENT.value,
                "may_read": [
                    "P_train",
                    "P_val",
                    "teacher_user_vectors",
                    "teacher_item_vectors",
                    "course_graph",
                ],
                "must_not_read": ["C_val", "C_test", "serving_user_positive_target"],
                "output": ["target_free_policy"],
            },
        ],
        "inference": {
            "hot_route": "frozen_teacher_item_vectors",
            "cold_route": "content_generator_then_target_free_policy",
            "oracle_access": False,
            "candidate_mode": "legal_state_retrieval",
        },
        "evaluation": {
            "ranking_bank": "single_unified_catalog_bank",
            "positive_recompute_eval": False,
            "legacy_dual_vector_eval": False,
            "selection_reads_test": False,
        },
        "policy_bounds": {
            "max_policy_delta": float(config.max_policy_delta),
            "stability_anchor_count": int(config.stability_anchor_count),
            "hot_items_mutable_in_policy": False,
        },
    }
