import torch

from semco_static_hin import entmax_bisect, sampled_entmax_loss, training_profiles


class DummyCfg:
    detach_query = True
    exclude_train_target = True
    singleton_policy = "global"


def test_entmax_bisect_returns_sparse_probability_rows():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.0, -0.1]])
    probs = entmax_bisect(logits, alpha=1.5, n_iter=40)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all(probs >= 0)
    assert probs[0, 2].item() == 0.0


def test_sampled_entmax_loss_is_zero_for_clear_correct_margin():
    logits = torch.tensor([[10.0, 0.0, -1.0]])
    target = torch.zeros(1, dtype=torch.long)
    loss = sampled_entmax_loss(logits, target, alpha=1.5, n_iter=40)
    assert loss.item() < 1e-6


def test_training_profiles_use_global_profile_for_singleton_leave_one_out():
    item_vectors = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        dim=1,
    )
    profile_sum_bank = torch.stack([item_vectors[0], item_vectors[0] + item_vectors[1]])
    hist_counts = torch.tensor([1.0, 2.0])
    users = torch.tensor([0, 1])
    pos_items = torch.tensor([0, 1])
    pos_pair_counts = torch.tensor([1.0, 1.0])

    profiles = training_profiles(
        DummyCfg(),
        profile_sum_bank,
        hist_counts,
        item_vectors,
        users,
        pos_items,
        pos_pair_counts,
    )

    global_profile = torch.nn.functional.normalize(item_vectors.mean(dim=0, keepdim=True), dim=1)[0]
    assert torch.allclose(profiles[0], global_profile, atol=1e-6)
    assert torch.allclose(profiles[1], item_vectors[0], atol=1e-6)
