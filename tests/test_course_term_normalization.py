from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usim_feedback_fast3_content_delta import _normalize_course_term_tensor


def test_batch_course_term_normalization_uses_positive_mean_scale():
    term = torch.tensor([[0.0], [0.5], [1.0]])

    normalized = _normalize_course_term_tensor(term, mode="batch", clip=2.0, eps=1e-6)

    expected = torch.tensor([[0.0], [2.0 / 3.0], [4.0 / 3.0]])
    assert torch.allclose(normalized, expected, atol=1e-5)


def test_course_term_normalization_none_preserves_values():
    term = torch.tensor([[0.0], [0.5], [3.0]])

    normalized = _normalize_course_term_tensor(term, mode="none", clip=2.0, eps=1e-6)

    assert torch.equal(normalized, term)


if __name__ == "__main__":
    test_batch_course_term_normalization_uses_positive_mean_scale()
    test_course_term_normalization_none_preserves_values()
    print("test_course_term_normalization.py passed")
