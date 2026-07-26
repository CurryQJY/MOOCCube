import dataclasses
import json
from types import MappingProxyType

import pytest
import torch

from fast3_delta.pseudocold import (
    build_pseudocold_plan,
    effective_item_difficulty,
    effective_item_popularity,
    mask_user_item_history,
)


def test_item_level_plan_is_deterministic_stratified_and_auditable():
    popularity = torch.tensor(
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 20, 21, 22, 23],
        dtype=torch.long,
    )

    first = build_pseudocold_plan(
        popularity,
        target_count=4,
        min_popularity=2,
        cold_threshold=1,
        seed=2025,
        n_strata=4,
    )
    second = build_pseudocold_plan(
        popularity.tolist(),
        target_count=4,
        min_popularity=2,
        cold_threshold=1,
        seed=2025,
        n_strata=4,
    )

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert len(first.selected_item_ids) == 4
    assert all(popularity[item_id] >= 2 for item_id in first.selected_item_ids)
    assert first.selected_mask.dtype is torch.bool
    assert first.selected_mask.tolist() == second.selected_mask.tolist()
    assert first.audit["schema_version"] == "pseudocold-v1"
    assert first.audit["eligible_item_count"] == 12
    assert first.audit["selected_item_count"] == 4
    assert [stratum["selected_count"] for stratum in first.audit["strata"]] == [1, 1, 1, 1]
    assert json.loads(json.dumps(first.to_dict()))["plan_hash"] == first.plan_hash


def test_plan_and_audit_are_immutable():
    plan = build_pseudocold_plan(
        torch.tensor([0, 2, 3, 4], dtype=torch.long),
        target_count=1,
        min_popularity=2,
        seed=7,
        n_strata=2,
    )

    assert isinstance(plan.audit, MappingProxyType)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.seed = 8
    with pytest.raises(TypeError):
        plan.audit["selected_item_count"] = 0
    with pytest.raises(TypeError):
        plan.audit["strata"][0]["selected_count"] = 0


def test_mask_user_item_history_copies_dense_sparse_and_mapping_inputs():
    mask = torch.tensor([False, True, False, True])
    dense = torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    mapping = {3: {0, 1, 3}, 4: {1, 2}}
    sparse = dense.to_sparse_coo()

    masked_dense = mask_user_item_history(dense, mask)
    masked_sparse = mask_user_item_history(sparse, mask).to_dense()
    masked_mapping = mask_user_item_history(mapping, mask)

    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    assert torch.equal(masked_dense, expected)
    assert torch.equal(masked_sparse, expected)
    assert masked_mapping == {3: {0}, 4: {2}}
    assert torch.equal(dense, torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]))
    assert mapping == {3: {0, 1, 3}, 4: {1, 2}}


def test_effective_item_stats_hide_selected_items_without_mutating_source():
    popularity = torch.tensor([0.0, 2.0, 8.0, 32.0])
    selected_mask = torch.tensor([False, True, False, True])

    effective_popularity = effective_item_popularity(popularity, selected_mask)
    difficulty = effective_item_difficulty(popularity, selected_mask)

    assert torch.equal(effective_popularity, torch.tensor([0.0, 0.0, 8.0, 0.0]))
    assert torch.equal(popularity, torch.tensor([0.0, 2.0, 8.0, 32.0]))
    assert difficulty.tolist() == pytest.approx([1.0, 1.0, 0.0, 1.0])


def test_plan_rejects_invalid_selection_arguments():
    popularity = torch.tensor([0, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="exactly one"):
        build_pseudocold_plan(popularity, target_count=1, ratio=0.5)
    with pytest.raises(ValueError, match="target_count"):
        build_pseudocold_plan(popularity, target_count=-1)
    with pytest.raises(ValueError, match="ratio"):
        build_pseudocold_plan(popularity, ratio=1.1)


def test_plan_rejects_complex_popularity_values():
    with pytest.raises(ValueError, match="real-valued"):
        build_pseudocold_plan(
            torch.tensor([0.0 + 0.0j, 2.0 + 1.0j]),
            target_count=1,
        )
