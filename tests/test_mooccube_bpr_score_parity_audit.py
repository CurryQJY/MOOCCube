from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper_aaai27/scripts/audit_mooccube_bpr_score_parity.py"
)
SPEC = importlib.util.spec_from_file_location("mooccube_bpr_score_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_raw_dot_and_cosine_can_produce_different_rankings() -> None:
    users = np.array([[1.0, 0.0]], dtype=np.float32)
    items = np.array([[0.9, 0.0], [2.0, 2.0]], dtype=np.float32)

    raw = MODULE.score_matrix(users, items, "raw_dot")
    cosine = MODULE.score_matrix(users, items, "cosine")

    assert raw.argmax(axis=1).tolist() == [1]
    assert cosine.argmax(axis=1).tolist() == [0]


def test_item_macro_evaluation_masks_train_history_but_restores_target() -> None:
    users = np.array([[1.0, 0.0]], dtype=np.float32)
    items = np.array([[1.0, 0.0], [0.2, 0.0]], dtype=np.float32)
    test = pd.DataFrame(
        {
            "u_idx": [0],
            "i_idx": [0],
            "popularity": [0],
        }
    )

    result = MODULE.evaluate_item_macro(
        user_embeddings=users,
        item_embeddings=items,
        test_df=test,
        train_seen={0: {0}},
        cold_threshold=1,
        score_mode="raw_dot",
        eval_group="cold",
        k_list=(1,),
    )

    assert result["target_rows"] == 1
    assert result["item_count"] == 1
    assert result["metrics"]["R@1"] == pytest.approx(1.0)
    assert result["metrics"]["N@1"] == pytest.approx(1.0)


def test_item_macro_accepts_explicit_eval_device_cpu() -> None:
    users = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    items = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]], dtype=np.float32)
    test = pd.DataFrame(
        {
            "u_idx": [0, 1],
            "i_idx": [0, 1],
            "popularity": [0, 0],
        }
    )

    implicit = MODULE.evaluate_item_macro(
        user_embeddings=users,
        item_embeddings=items,
        test_df=test,
        train_seen={},
        cold_threshold=1,
        score_mode="raw_dot",
        eval_group="cold",
        k_list=(1,),
    )
    explicit = MODULE.evaluate_item_macro(
        user_embeddings=users,
        item_embeddings=items,
        test_df=test,
        train_seen={},
        cold_threshold=1,
        score_mode="raw_dot",
        eval_group="cold",
        k_list=(1,),
        eval_device="cpu",
    )

    assert explicit["metrics"] == implicit["metrics"]
    assert explicit["per_item"].equals(implicit["per_item"])


def test_item_macro_groups_by_target_item_not_interaction_count() -> None:
    users = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    items = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]], dtype=np.float32)
    test = pd.DataFrame(
        {
            "u_idx": [0, 0, 1],
            "i_idx": [0, 0, 1],
            "popularity": [0, 0, 1],
        }
    )

    cold = MODULE.evaluate_item_macro(
        user_embeddings=users,
        item_embeddings=items,
        test_df=test,
        train_seen={},
        cold_threshold=1,
        score_mode="raw_dot",
        eval_group="cold",
        k_list=(1,),
    )
    hot = MODULE.evaluate_item_macro(
        user_embeddings=users,
        item_embeddings=items,
        test_df=test,
        train_seen={},
        cold_threshold=1,
        score_mode="raw_dot",
        eval_group="hot",
        k_list=(1,),
    )

    assert cold["target_rows"] == 2
    assert cold["item_count"] == 1
    assert cold["metrics"]["R@1"] == pytest.approx(1.0)
    assert hot["target_rows"] == 1
    assert hot["item_count"] == 1
    assert hot["metrics"]["R@1"] == pytest.approx(1.0)


def test_teacher_triplet_sampler_uses_train_only_users_positives_and_items() -> None:
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 1],
            "i_idx": [1, 2, 3],
            "popularity": [1, 1, 1],
        }
    )
    tables = MODULE.build_teacher_sampling_tables(train)
    users, positives, negatives = MODULE.sample_teacher_triplets(
        tables=tables,
        n_pairs=100,
        rng=np.random.default_rng(2025),
    )

    train_pairs = set(zip(train.u_idx.tolist(), train.i_idx.tolist()))
    assert set(users).issubset({0, 1})
    assert all((int(user), int(item)) in train_pairs for user, item in zip(users, positives))
    assert set(negatives).issubset({1, 2, 3})


def test_cli_accepts_eval_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--official-embedding", "teacher.npy", "--eval-device", "cuda"],
    )

    args = MODULE.parse_args()

    assert args.eval_device == "cuda"
