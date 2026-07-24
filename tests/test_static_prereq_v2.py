"""Behavioral tests for the isolated static prerequisite scorer."""

import copy

import pytest
import torch

from static_prereq_v2 import (
    ScorerConfig,
    StaticContentScorer,
    _prereq_aux_loss,
    infonce_loss_parts,
)


def _make_config(tmp_path, prereq_weight=1.0):
    cfg = ScorerConfig(n_users=4, n_items=4, content_dim=6)
    cfg.emb_dim = 8
    cfg.hidden_dim = 12
    cfg.batch_size = 4
    cfg.train_num_negs = 2
    cfg.hard_neg_ratio = 0.5
    cfg.dropout_prob = 0.0
    cfg.prereq_aux_weight = float(prereq_weight)
    cfg.prereq_path = str(tmp_path / "prereq.pt")
    return cfg


def _write_prereq(path, all_invalid=False):
    if all_invalid:
        idx = torch.full((4, 2), -1, dtype=torch.long)
        mask = torch.zeros(4, dtype=torch.bool)
    else:
        idx = torch.tensor(
            [[1, 2], [2, -1], [-1, -1], [0, -1]], dtype=torch.long
        )
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)
    torch.save({"prereq_idx": idx, "has_prereq": mask}, path)


def _make_model(tmp_path, prereq_weight=1.0, all_invalid=False):
    torch.manual_seed(7)
    cfg = _make_config(tmp_path, prereq_weight)
    _write_prereq(cfg.prereq_path, all_invalid=all_invalid)
    content = torch.randn(cfg.n_items, cfg.content_dim)
    model = StaticContentScorer(cfg, content)
    return cfg, model


def test_prereq_index_uses_canonical_field_and_produces_nonzero_loss(tmp_path):
    cfg, model = _make_model(tmp_path)
    assert model.prereq_idx.shape == (4, 2)
    assert model.prereq_mask.tolist() == [True, True, False, True]

    model.eval()
    items = torch.tensor([0, 1, 3], dtype=torch.long)
    with torch.no_grad():
        fused = model.item_vector(items, force_cold=False)
        z_i = torch.nn.functional.normalize(fused, dim=1)
        loss = _prereq_aux_loss(model, items, z_i, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_regular_negative_sampling_branch_responds_to_prereq_weight(tmp_path):
    cfg, model = _make_model(tmp_path, prereq_weight=1.0)
    model_copy = copy.deepcopy(model)
    model.eval()
    model_copy.eval()
    users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    items = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    cfg.prereq_aux_weight = 0.0
    total0, parts0 = infonce_loss_parts(model, users, items, torch.device("cpu"))
    model_copy.cfg.prereq_aux_weight = 1.0
    total1, parts1 = infonce_loss_parts(
        model_copy, users, items, torch.device("cpu")
    )

    assert torch.isfinite(total0)
    assert torch.isfinite(total1)
    assert parts0["prereq"].item() == pytest.approx(0.0)
    assert parts1["prereq"].item() > 0.0
    assert abs(total1.item() - total0.item()) > 1e-6


def test_no_valid_prerequisite_rows_return_zero_component(tmp_path):
    _, model = _make_model(tmp_path, all_invalid=True)
    model.eval()
    users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    items = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    total, parts = infonce_loss_parts(model, users, items, torch.device("cpu"))

    assert torch.isfinite(total)
    assert parts["prereq"].item() == pytest.approx(0.0)
