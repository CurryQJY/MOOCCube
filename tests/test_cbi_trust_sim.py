from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbi_trust_sim import project_to_content_cone


def test_projection_keeps_in_domain_vector():
    anchor = torch.tensor([[1.0, 0.0]])
    state = torch.tensor([[0.9, 0.4358899]])

    projected, stats = project_to_content_cone(state, anchor, cosine_floor=0.8)

    assert torch.allclose(projected, F.normalize(state, dim=1), atol=1e-6)
    assert stats["projected_count"] == 0


def test_projection_hits_cosine_boundary():
    anchor = torch.tensor([[1.0, 0.0]])
    state = torch.tensor([[0.0, 1.0]])
    cosine_floor = 0.8660254037844386

    projected, stats = project_to_content_cone(state, anchor, cosine_floor=cosine_floor)

    assert torch.allclose(projected.norm(dim=1), torch.ones(1), atol=1e-6)
    assert torch.all((projected * anchor).sum(dim=1) >= cosine_floor - 1e-6)
    assert stats["projected_count"] == 1


def test_projection_handles_antiparallel_input_without_nan():
    anchor = torch.tensor([[1.0, 0.0]])

    projected, stats = project_to_content_cone(-anchor, anchor, cosine_floor=0.8660254037844386)

    assert torch.isfinite(projected).all()
    assert torch.equal(projected, anchor)
    assert stats["projected_count"] == 1

