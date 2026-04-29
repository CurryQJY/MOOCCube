import torch


FEEDBACK_TYPE_NAMES = (
    "good_fit",
    "prereq_unmet",
    "difficulty_too_high",
    "topic_drift",
    "redundant_recommendation",
)


def feedback_feature_dim(n_types):
    return 7 + 2 * int(n_types)


class FeedbackMemoryState:
    def __init__(self, batch_size, n_types, device):
        self.batch_size = int(batch_size)
        self.n_types = int(n_types)
        self.device = device
        self.reset()

    def reset(self):
        self.count = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.accept_sum = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.reject_sum = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.prereq_sum = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.diff_sum = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.concept_sum = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.last_item = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        self.type_sum = torch.zeros((self.batch_size, self.n_types), dtype=torch.float32, device=self.device)
        self.last_type = torch.zeros((self.batch_size, self.n_types), dtype=torch.float32, device=self.device)

    def append(
        self,
        item_idx,
        accept_prob,
        feedback_probs,
        prereq_gap,
        difficulty_gap,
        concept_match,
    ):
        accept_prob = accept_prob.detach().float()
        feedback_probs = feedback_probs.detach().float()
        prereq_gap = prereq_gap.detach().float()
        difficulty_gap = difficulty_gap.detach().float()
        concept_match = concept_match.detach().float()

        if item_idx is None:
            item_norm = torch.zeros_like(self.last_item)
        else:
            item_norm = item_idx.detach().float().view(-1, 1)
            item_norm = item_norm / max(1.0, float(item_norm.max().item()) + 1.0)

        self.count = self.count + 1.0
        self.accept_sum = self.accept_sum + accept_prob
        self.reject_sum = self.reject_sum + (1.0 - accept_prob)
        self.prereq_sum = self.prereq_sum + prereq_gap
        self.diff_sum = self.diff_sum + difficulty_gap
        self.concept_sum = self.concept_sum + concept_match
        self.type_sum = self.type_sum + feedback_probs
        self.last_type = feedback_probs
        self.last_item = item_norm

    def summary(self):
        denom = self.count.clamp_min(1.0)
        avg_type = self.type_sum / denom
        summary = torch.cat(
            [
                self.count / denom.max().clamp_min(1.0),
                self.accept_sum / denom,
                self.reject_sum / denom,
                self.prereq_sum / denom,
                self.diff_sum / denom,
                self.concept_sum / denom,
                self.last_item,
                avg_type,
                self.last_type,
            ],
            dim=1,
        )
        return summary
