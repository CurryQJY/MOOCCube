from __future__ import annotations

import sys
from pathlib import Path

import torch


KGREC_MODULES = (
    Path(__file__).resolve().parents[1]
    / "paper_aaai27"
    / "baseline_sources"
    / "KGRec"
    / "modules"
)
sys.path.insert(0, str(KGREC_MODULES.parent))
sys.path.insert(0, str(KGREC_MODULES))

from native_scatter import scatter_mean, scatter_softmax, scatter_sum


def test_native_scatter_matches_grouped_reductions_and_preserves_gradients() -> None:
    source = torch.tensor(
        [[1.0, 0.0], [3.0, 2.0], [2.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    groups = torch.tensor([0, 1, 0, 1])

    assert torch.allclose(
        scatter_sum(source, groups, dim_size=2),
        torch.tensor([[3.0, 4.0], [8.0, 8.0]]),
    )
    assert torch.allclose(
        scatter_mean(source, groups, dim_size=2),
        torch.tensor([[1.5, 2.0], [4.0, 4.0]]),
    )

    normalized = scatter_softmax(source, groups, dim_size=2)
    assert torch.allclose(
        scatter_sum(normalized, groups, dim_size=2),
        torch.ones((2, 2)),
        atol=1e-6,
    )

    (normalized.square().sum()).backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_native_scatter_reductions_accept_torch_scatter_src_keyword() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    groups = torch.tensor([0, 0])

    assert torch.equal(
        scatter_sum(src=source, index=groups, dim=0, dim_size=1),
        torch.tensor([[4.0, 6.0]]),
    )
    assert torch.equal(
        scatter_mean(src=source, index=groups, dim=0, dim_size=1),
        torch.tensor([[2.0, 3.0]]),
    )


def test_kgrec_modules_import_without_pyg_extensions() -> None:
    from modules.AttnHGCN import AttnHGCN
    from modules.KGRec import KGRec

    assert AttnHGCN is not None
    assert KGRec is not None


def test_relation_aware_sampling_includes_every_noninteraction_relation() -> None:
    from modules.KGRec import _relation_aware_edge_sampling

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    edge_type = torch.arange(1, 7, dtype=torch.long)

    sampled_index, sampled_type = _relation_aware_edge_sampling(
        edge_index,
        edge_type,
        n_relations=7,
        samp_rate=1.0,
    )

    assert sampled_index.shape == edge_index.shape
    assert torch.equal(torch.sort(sampled_type).values, edge_type)

