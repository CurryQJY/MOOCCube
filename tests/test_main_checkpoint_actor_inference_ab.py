import torch
import torch.nn.functional as F

import main_checkpoint_actor_inference_ab as ab
import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy


def test_deterministic_action_uses_actor_argmax():
    torch.manual_seed(7)
    agent = legacy.FixedSimpleAC(item_dim=4, time_dim=5)
    state = torch.randn(2, 4)
    time_step = torch.zeros(2, 1)
    candidates = torch.randn(2, 3, 4)

    action, log_prob, value, entropy = ab.deterministic_get_action_value(
        agent,
        state,
        time_step,
        candidates,
    )

    with torch.no_grad():
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=5).float()
        feat = agent.common(torch.cat([state, t_emb], dim=1))
        query = agent.actor_head(feat).unsqueeze(1)
        keys = agent.user_proj(candidates)
        expected = torch.matmul(query, keys.transpose(1, 2)).squeeze(1).argmax(dim=-1)

    assert torch.equal(action, expected)
    assert log_prob.shape == (2,)
    assert value.shape == (2, 1)
    assert entropy.shape == (2,)
