"""Unit tests for dual_path_residual_clean (AlphaFuse-style residual scorer)."""

import torch

from dual_path_residual_clean import (
    DualPathResidualScorer,
    ResidualConfig,
    decompose_language_space,
)


def _toy_content(n_items=40, content_dim=32, rank=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Low-rank + small noise so SVD has clear rich/null structure
    factors = torch.randn(n_items, rank, generator=g)
    basis = torch.randn(rank, content_dim, generator=g)
    noise = 0.01 * torch.randn(n_items, content_dim, generator=g)
    return factors @ basis + noise


def test_decompose_shapes_and_dims():
    content = _toy_content()
    decomp = decompose_language_space(content, emb_dim=16, null_dim=6)
    assert decomp["language_codes"].shape == (40, 16)
    assert decomp["d_s"] == 10
    assert decomp["d_n"] == 6
    assert decomp["U_take"].shape == (32, 16)
    assert torch.isfinite(decomp["language_codes"]).all()


def test_decompose_threshold_mode():
    content = _toy_content(rank=4)
    decomp = decompose_language_space(
        content, emb_dim=12, null_dim=4, null_threshold=0.1
    )
    assert decomp["d_s"] + decomp["d_n"] == 12
    assert decomp["language_codes"].shape[1] == 12


def test_cold_residual_exactly_zero():
    content = _toy_content(n_items=20, content_dim=24)
    decomp = decompose_language_space(content, emb_dim=12, null_dim=4)
    cfg = ResidualConfig(n_users=10, n_items=20, content_dim=24)
    cfg.emb_dim = 12
    cfg.null_dim = 4
    model = DualPathResidualScorer(cfg, decomp["language_codes"])
    # Make residual large so zeroing is detectable
    with torch.no_grad():
        model.item_null_residual.weight.fill_(3.0)
    idx = torch.arange(20)
    cold = torch.zeros(20, dtype=torch.bool)
    cold[0:5] = True
    bank = model.encode_all(cold_item_mask=cold)
    lang = model.language_codes
    assert torch.allclose(bank[cold], lang[cold], atol=1e-6)
    # Warm should differ from pure language
    warm = ~cold
    assert (bank[warm] - lang[warm]).abs().max().item() > 1.0


def test_item_vector_return_parts():
    content = _toy_content(n_items=10, content_dim=16)
    decomp = decompose_language_space(content, emb_dim=8, null_dim=3)
    cfg = ResidualConfig(n_users=5, n_items=10, content_dim=16)
    cfg.emb_dim = 8
    cfg.null_dim = 3
    model = DualPathResidualScorer(cfg, decomp["language_codes"])
    idx = torch.tensor([0, 1, 2])
    fused, lang, res = model.item_vector(idx, force_cold=False, return_parts=True)
    assert fused.shape == (3, 8)
    assert lang.shape == (3, 8)
    assert res.shape == (3, 3)
    # Reconstruct: fused == lang + [0_ds || res]
    pad = torch.zeros(3, cfg.row_dim)
    recon = lang + torch.cat([pad, res], dim=-1)
    assert torch.allclose(fused, recon, atol=1e-6)


def test_force_cold_tensor_and_bool():
    content = _toy_content(n_items=8, content_dim=16)
    decomp = decompose_language_space(content, emb_dim=8, null_dim=2)
    cfg = ResidualConfig(n_users=4, n_items=8, content_dim=16)
    cfg.emb_dim = 8
    cfg.null_dim = 2
    model = DualPathResidualScorer(cfg, decomp["language_codes"])
    with torch.no_grad():
        model.item_null_residual.weight.fill_(1.0)
    idx = torch.arange(8)
    all_cold = model.item_vector(idx, force_cold=True)
    assert torch.allclose(all_cold, model.language_codes, atol=1e-6)
    mask = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    partial = model.item_vector(idx, force_cold=mask)
    assert torch.allclose(partial[0], model.language_codes[0], atol=1e-6)
    assert not torch.allclose(partial[1], model.language_codes[1], atol=1e-5)


def test_user_and_item_forward_finite():
    content = _toy_content(n_items=15, content_dim=20)
    decomp = decompose_language_space(content, emb_dim=10, null_dim=4)
    cfg = ResidualConfig(n_users=6, n_items=15, content_dim=20)
    cfg.emb_dim = 10
    cfg.null_dim = 4
    model = DualPathResidualScorer(cfg, decomp["language_codes"])
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([3, 4, 5])
    z_u = model.user_vector(u)
    z_i = model.item_vector(i)
    assert torch.isfinite(z_u).all()
    assert torch.isfinite(z_i).all()
    logits = torch.mm(
        torch.nn.functional.normalize(z_u, dim=1),
        torch.nn.functional.normalize(z_i, dim=1).t(),
    )
    assert torch.isfinite(logits).all()
