"""Unit tests for static_semco_clean (SEMCo-style pure-content scorer)."""

import torch

from static_semco_clean import entmax_bisect, sampled_entmax_loss


def test_entmax_bisect_sparse_and_normalized():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.0, -0.1]])
    probs = entmax_bisect(logits, alpha=1.5, n_iter=40)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all(probs >= 0)
    assert probs[0, 2].item() == 0.0


def test_entmax_alpha1_matches_softmax():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
    p_ent = entmax_bisect(logits, alpha=1.0)
    p_sm = torch.softmax(logits, dim=-1)
    assert torch.allclose(p_ent, p_sm, atol=1e-6)


def test_sampled_entmax_loss_clear_margin_near_zero():
    logits = torch.tensor([[10.0, 0.0, -1.0]])
    target = torch.zeros(1, dtype=torch.long)
    loss = sampled_entmax_loss(logits, target, alpha=1.5, n_iter=40)
    assert loss.item() < 1e-6


def test_sampled_entmax_alpha1_finite():
    logits = torch.randn(8, 16)
    target = torch.zeros(8, dtype=torch.long)
    loss = sampled_entmax_loss(logits, target, alpha=1.0)
    assert torch.isfinite(loss)
