import pytest
import torch

import usim_feedback_fast3_content_delta_ppo_v2 as ppo_v2


@pytest.mark.parametrize("agent_cls", [ppo_v2.SimpleAC, ppo_v2.FixedSimpleAC])
def test_policy_uses_argmax_during_evaluation(agent_cls):
    agent = agent_cls(item_dim=4, time_dim=4)
    agent.eval()
    item_state = torch.zeros(2, 4)
    time_step = torch.zeros(2, 1)
    candidates = torch.randn(2, 3, 4)

    first, _, _, _ = agent.get_action_value(item_state, time_step, candidates)
    second, _, _, _ = agent.get_action_value(item_state, time_step, candidates)

    assert torch.equal(first, second)


def test_terminal_reward_is_only_applied_on_last_step():
    distance = torch.tensor([[0.4], [0.2]])
    gain = torch.tensor([[0.1], [0.05]])

    middle = ppo_v2._compose_ppo_distance_reward(
        distance, gain, step_idx=1, num_steps=5, terminal_weight=2.0, gain_weight=1.0
    )
    final = ppo_v2._compose_ppo_distance_reward(
        distance, gain, step_idx=4, num_steps=5, terminal_weight=2.0, gain_weight=1.0
    )

    assert torch.allclose(middle, gain)
    assert torch.allclose(final, gain - 2.0 * distance)
