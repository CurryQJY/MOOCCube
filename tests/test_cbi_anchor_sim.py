from pathlib import Path
from types import SimpleNamespace

import torch

from cbi_anchor_sim import CBIAnchorFast3FeedbackUSIM
from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM


def test_anchor_simulator_replaces_any_caller_target_with_initial_cbi(monkeypatch):
    captured = []

    def fake_parent_episode(self, init_item_emb, target_emb=None, **kwargs):
        del self, kwargs
        captured.append(target_emb)
        return init_item_emb, {"rewards": []}, {"steps": 0}

    monkeypatch.setattr(Fast3FeedbackUSIM, "run_usim_episode", fake_parent_episode)
    model = CBIAnchorFast3FeedbackUSIM.__new__(CBIAnchorFast3FeedbackUSIM)
    initial_cbi = torch.tensor([[0.6, 0.8]], requires_grad=True)

    first = model.run_usim_episode(initial_cbi, target_emb=torch.tensor([[1.0, 0.0]]))
    second = model.run_usim_episode(initial_cbi, target_emb=torch.tensor([[0.0, 1.0]]))

    assert first[0] is initial_cbi
    assert second[0] is initial_cbi
    assert len(captured) == 2
    for effective_target in captured:
        assert torch.equal(effective_target, initial_cbi)
        assert effective_target.requires_grad is False
        assert effective_target.data_ptr() == initial_cbi.data_ptr()
