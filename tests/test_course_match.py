import os
import unittest

import torch

from fast3_delta.config import FeedbackConfig
from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM


class EnvPatch:
    def __init__(self, **updates):
        self.updates = updates
        self.previous = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = str(value)

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_model(mode="mean", topk=2):
    cfg = FeedbackConfig(n_users=1, n_items=4, content_dim=3)
    content_emb = torch.zeros((4, 3), dtype=torch.float32)
    model = Fast3FeedbackUSIM(cfg, content_emb)
    model.device = torch.device("cpu")
    model.item_concept_overlap = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.2, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return model


class CourseConceptMatchTests(unittest.TestCase):
    def test_mean_preserves_legacy_average(self):
        with EnvPatch(USIM_FB_COURSE_MATCH_MODE="mean", USIM_FB_COURSE_MATCH_TOPK="2"):
            model = make_model()
            seen_mat = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
            item_idx = torch.tensor([3], dtype=torch.long)

            match = model._compute_course_concept_match(seen_mat, item_idx)

            self.assertTrue(torch.allclose(match, torch.tensor([[0.4]], dtype=torch.float32)))

    def test_topk_uses_most_relevant_history(self):
        with EnvPatch(USIM_FB_COURSE_MATCH_MODE="topk", USIM_FB_COURSE_MATCH_TOPK="2"):
            model = make_model()
            seen_mat = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
            item_idx = torch.tensor([3], dtype=torch.long)

            match = model._compute_course_concept_match(seen_mat, item_idx)

            self.assertTrue(torch.allclose(match, torch.tensor([[0.6]], dtype=torch.float32)))

    def test_max_uses_single_best_history_item(self):
        with EnvPatch(USIM_FB_COURSE_MATCH_MODE="max", USIM_FB_COURSE_MATCH_TOPK="2"):
            model = make_model()
            seen_mat = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
            item_idx = torch.tensor([3], dtype=torch.long)

            match = model._compute_course_concept_match(seen_mat, item_idx)

            self.assertTrue(torch.allclose(match, torch.tensor([[1.0]], dtype=torch.float32)))

    def test_topk_excludes_target_item_when_configured(self):
        with EnvPatch(
            USIM_FB_COURSE_MATCH_MODE="topk",
            USIM_FB_COURSE_MATCH_TOPK="2",
            USIM_FB_COURSE_MATCH_EXCLUDE_TARGET="1",
        ):
            model = make_model()
            seen_mat = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
            item_idx = torch.tensor([3], dtype=torch.long)

            match = model._compute_course_concept_match(seen_mat, item_idx)

            self.assertTrue(torch.allclose(match, torch.tensor([[0.6]], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
