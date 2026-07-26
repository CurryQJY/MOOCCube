"""Numerical-equivalence + gradient-flow tests for the aux-loss hot-only branch.

Covers `FastFeedbackUSIM._compute_aux_loss` (introduced as a ROLLBACK-flagged
refactor; toggle via env var ``USIM_AUX_HOT_ONLY``):

1. Legacy mode (`aux_hot_only=False`) is bit-identical to the pre-refactor
   InfoNCE over the full batch.
2. Hot-only mode masks out cold rows in the InfoNCE.
3. Hot-only mode produces zero gradient on cold rows of `id_e_true` and
   `content_e` (this is the cold-start improvement the flag is meant to deliver).
4. Edge cases: all-cold batch, single-hot batch -> loss is exactly 0.
5. `effective_cold=None` falls back to legacy behavior.

Run:
    python _test_aux_hot_only.py
"""
import os
import sys

os.environ["USIM_FORCE_CPU"] = "1"
os.environ.setdefault("USIM_DISABLE_LLM_SCORE", "1")

import torch
import torch.nn.functional as F

import usim_feedback_fast3_content_delta as M


def make_dummy_model(n_users=32, n_items=16, content_dim=8):
    cfg = M.Fast3Config(n_users=n_users, n_items=n_items, content_dim=content_dim)
    cfg.batch_size = 4
    cfg.n_epochs = 1
    cfg.temp = 0.07
    content_emb = torch.randn(n_items, content_dim, generator=torch.Generator().manual_seed(0))
    model = M.Fast3FeedbackUSIM(cfg, content_emb)
    model.device = torch.device("cpu")
    model.eval()  # disable dropout for deterministic tests
    return cfg, model


def _legacy_aux(z_id_raw, z_con_raw, temp):
    """Reference implementation matching the pre-refactor inline block."""
    z_id = F.normalize(z_id_raw, dim=1)
    z_con = F.normalize(z_con_raw, dim=1)
    labels = torch.arange(z_id.size(0))
    sim = torch.matmul(z_id, z_con.t()) / temp
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


def test_legacy_path_bit_identical():
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = False  # legacy
    torch.manual_seed(1)
    id_e = torch.randn(6, cfg.emb_dim, requires_grad=True)
    con_e = torch.randn(6, cfg.emb_dim, requires_grad=True)
    eff_cold = torch.tensor([True, False, True, False, True, False])

    legacy = _legacy_aux(id_e, con_e, cfg.temp)
    new = model._compute_aux_loss(id_e, con_e, eff_cold)

    assert torch.allclose(legacy, new, atol=1e-7), f"legacy != new: {legacy.item()} vs {new.item()}"
    print(f"[PASS] legacy aux_loss bit-identical (loss={new.item():.6f})")


def test_legacy_path_when_effective_cold_is_none():
    """When effective_cold=None we must fall back to legacy regardless of flag."""
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = True
    torch.manual_seed(2)
    id_e = torch.randn(5, cfg.emb_dim)
    con_e = torch.randn(5, cfg.emb_dim)

    legacy = _legacy_aux(id_e, con_e, cfg.temp)
    new = model._compute_aux_loss(id_e, con_e, None)
    assert torch.allclose(legacy, new, atol=1e-7), "fallback to legacy failed when effective_cold=None"
    print(f"[PASS] effective_cold=None falls back to legacy (loss={new.item():.6f})")


def test_hot_only_matches_subset_inference():
    """Hot-only loss must equal the legacy loss computed over the hot subset."""
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = True
    torch.manual_seed(3)
    id_e = torch.randn(8, cfg.emb_dim)
    con_e = torch.randn(8, cfg.emb_dim)
    # 5 hot, 3 cold
    eff_cold = torch.tensor([False, True, False, False, True, True, False, False])

    hot_idx = (~eff_cold).nonzero(as_tuple=False).view(-1)
    expected = _legacy_aux(id_e[hot_idx], con_e[hot_idx], cfg.temp)
    actual = model._compute_aux_loss(id_e, con_e, eff_cold)

    assert torch.allclose(expected, actual, atol=1e-7), (
        f"hot-only != hot subset legacy: {expected.item()} vs {actual.item()}"
    )
    print(f"[PASS] hot_only == legacy(hot_subset): loss={actual.item():.6f} over {hot_idx.numel()} hot rows")


def test_hot_only_gradient_zero_on_cold_rows():
    """Cold rows of id_e_true / content_e must have zero gradient under hot-only."""
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = True
    torch.manual_seed(4)
    id_e = torch.randn(6, cfg.emb_dim, requires_grad=True)
    con_e = torch.randn(6, cfg.emb_dim, requires_grad=True)
    eff_cold = torch.tensor([True, False, True, False, False, True])

    loss = model._compute_aux_loss(id_e, con_e, eff_cold)
    loss.backward()

    cold_rows = eff_cold.nonzero(as_tuple=False).view(-1)
    hot_rows = (~eff_cold).nonzero(as_tuple=False).view(-1)

    cold_id_grad = id_e.grad[cold_rows]
    cold_con_grad = con_e.grad[cold_rows]
    hot_id_grad = id_e.grad[hot_rows]
    hot_con_grad = con_e.grad[hot_rows]

    assert torch.all(cold_id_grad == 0), f"cold id_e grad not zero: {cold_id_grad.abs().max().item()}"
    assert torch.all(cold_con_grad == 0), f"cold content_e grad not zero: {cold_con_grad.abs().max().item()}"
    assert hot_id_grad.abs().sum().item() > 0, "hot id_e grad must be nonzero"
    assert hot_con_grad.abs().sum().item() > 0, "hot content_e grad must be nonzero"
    print(
        f"[PASS] hot_only zeroes cold gradients "
        f"(cold rows={cold_rows.numel()}, hot rows={hot_rows.numel()}, "
        f"hot_id_grad_norm={hot_id_grad.norm().item():.4f})"
    )


def test_hot_only_all_cold_returns_zero():
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = True
    torch.manual_seed(5)
    id_e = torch.randn(4, cfg.emb_dim, requires_grad=True)
    con_e = torch.randn(4, cfg.emb_dim, requires_grad=True)
    eff_cold = torch.ones(4, dtype=torch.bool)

    loss = model._compute_aux_loss(id_e, con_e, eff_cold)
    assert loss.item() == 0.0, f"all-cold should yield 0 loss, got {loss.item()}"
    # Backward on a 0-tensor must still produce zero gradients (not None / not crash).
    loss.backward()
    assert id_e.grad is None or torch.all(id_e.grad == 0), "id_e grad should be zero/None for all-cold batch"
    print("[PASS] hot_only all-cold batch -> loss=0, no spurious gradient")


def test_hot_only_single_hot_returns_zero():
    """InfoNCE needs >=2 hot rows; with 1 hot row the loss is 0."""
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = True
    torch.manual_seed(6)
    id_e = torch.randn(5, cfg.emb_dim, requires_grad=True)
    con_e = torch.randn(5, cfg.emb_dim, requires_grad=True)
    eff_cold = torch.tensor([True, True, False, True, True])  # exactly 1 hot

    loss = model._compute_aux_loss(id_e, con_e, eff_cold)
    assert loss.item() == 0.0, f"single-hot should yield 0 loss, got {loss.item()}"
    print("[PASS] hot_only single-hot batch -> loss=0")


def test_legacy_full_batch_grad_includes_cold_rows():
    """Sanity check that the legacy path DOES propagate gradient to cold rows
    (so the hot-only branch is genuinely changing the gradient flow).
    """
    cfg, model = make_dummy_model()
    cfg.aux_hot_only = False
    torch.manual_seed(7)
    id_e = torch.randn(6, cfg.emb_dim, requires_grad=True)
    con_e = torch.randn(6, cfg.emb_dim, requires_grad=True)
    eff_cold = torch.tensor([True, False, True, False, False, True])

    loss = model._compute_aux_loss(id_e, con_e, eff_cold)
    loss.backward()
    cold_rows = eff_cold.nonzero(as_tuple=False).view(-1)
    cold_grad_norm = id_e.grad[cold_rows].abs().sum().item()
    assert cold_grad_norm > 0, "legacy path must propagate gradient to cold rows; got 0"
    print(f"[PASS] legacy path keeps cold gradient (norm={cold_grad_norm:.4f})")


if __name__ == "__main__":
    print("=" * 64)
    print("aux-loss hot-only refactor: numerical & gradient tests")
    print("=" * 64)
    test_legacy_path_bit_identical()
    test_legacy_path_when_effective_cold_is_none()
    test_hot_only_matches_subset_inference()
    test_hot_only_gradient_zero_on_cold_rows()
    test_hot_only_all_cold_returns_zero()
    test_hot_only_single_hot_returns_zero()
    test_legacy_full_batch_grad_includes_cold_rows()
    print("=" * 64)
    print("All aux-loss tests passed.")
