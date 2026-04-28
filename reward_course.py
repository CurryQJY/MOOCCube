import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenCourseRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer(
            "reward_weights",
            torch.tensor(
                [
                    float(getattr(config, "feedback_reward_sim", 0.50)),
                    float(getattr(config, "feedback_reward_accept", 0.25)),
                    float(getattr(config, "feedback_reward_prereq", -0.35)),
                    float(getattr(config, "feedback_reward_diff", -0.20)),
                    float(getattr(config, "feedback_reward_concept", 0.20)),
                ],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        user_vec,
        item_vec,
        accept_prob=None,
        prereq_gap=None,
        difficulty_gap=None,
        concept_match=None,
    ):
        sim = F.cosine_similarity(user_vec, item_vec, dim=1).unsqueeze(1)
        accept_prob = torch.zeros_like(sim) if accept_prob is None else accept_prob
        prereq_gap = torch.zeros_like(sim) if prereq_gap is None else prereq_gap
        difficulty_gap = torch.zeros_like(sim) if difficulty_gap is None else difficulty_gap
        concept_match = torch.zeros_like(sim) if concept_match is None else concept_match

        score = (
            self.reward_weights[0] * sim +
            self.reward_weights[1] * accept_prob +
            self.reward_weights[2] * prereq_gap +
            self.reward_weights[3] * difficulty_gap +
            self.reward_weights[4] * concept_match
        )
        return score
