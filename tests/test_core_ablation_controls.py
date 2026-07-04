import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast3_delta.config import Fast3Config
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


class CoreAblationControlTests(unittest.TestCase):
    def test_rollout_policy_env_selects_non_learning_actions(self):
        with EnvPatch(USIM_ROLLOUT_POLICY="greedy_similarity"):
            cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            self.assertEqual(cfg.rollout_policy, "greedy_similarity")
            model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
            model.device = torch.device("cpu")
            current_h = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
            current_h[0, 0] = 1.0
            candidates = torch.zeros((1, 3, cfg.emb_dim), dtype=torch.float32)
            candidates[0, 0, 1] = 1.0
            candidates[0, 1, 0] = 1.0
            candidates[0, 2, 0] = -1.0
            action_idx, log_prob, value, entropy = model._select_rollout_action(
                current_h,
                torch.zeros((1, 1), dtype=torch.float32),
                candidates,
                fit_score=None,
            )

            self.assertEqual(action_idx.tolist(), [1])
            self.assertTrue(torch.allclose(log_prob, torch.zeros_like(log_prob)))
            self.assertTrue(torch.allclose(value, torch.zeros_like(value)))
            self.assertTrue(torch.allclose(entropy, torch.zeros_like(entropy)))

        with EnvPatch(USIM_ROLLOUT_POLICY="course_fit"):
            cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            self.assertEqual(cfg.rollout_policy, "course_fit")
            model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
            model.device = torch.device("cpu")
            current_h = torch.zeros((1, cfg.emb_dim), dtype=torch.float32)
            candidates = torch.zeros((1, 3, cfg.emb_dim), dtype=torch.float32)
            fit_score = torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32)
            action_idx, _, _, _ = model._select_rollout_action(
                current_h,
                torch.zeros((1, 1), dtype=torch.float32),
                candidates,
                fit_score=fit_score,
            )

            self.assertEqual(action_idx.tolist(), [1])

    def test_ppo_loss_weight_env_controls_forward_loss_contribution(self):
        with EnvPatch(
            USIM_PPO_LOSS_WEIGHT="0.0",
            USIM_USE_CONTENT_DELTA="0",
            USIM_USE_PAAC="0",
            USIM_USE_PREREQ_AUX_LOSS="0",
            USIM_AUX_WEIGHT="0",
            USIM_DISABLE_LLM_SCORE="1",
            USIM_STEPS="0",
        ):
            cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            self.assertEqual(cfg.ppo_loss_weight, 0.0)
            cfg.dropout_prob = 0.0
            model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
            model.device = torch.device("cpu")
            model.eval()

            def fake_run_usim_episode(z_i_base, target_emb, **kwargs):
                return z_i_base, {"rewards": ["sentinel"]}, {"steps": 0}

            def fake_compute_ppo_loss(trajectory):
                self.assertEqual(trajectory["rewards"], ["sentinel"])
                return torch.tensor(7.0, dtype=torch.float32)

            model.run_usim_episode = fake_run_usim_episode
            model.compute_ppo_loss = fake_compute_ppo_loss

            batch = {
                "u": torch.tensor([0, 1], dtype=torch.long),
                "i": torch.tensor([0, 1], dtype=torch.long),
            }
            pop = torch.tensor([0, 0], dtype=torch.long)
            llm_s = torch.zeros(2, dtype=torch.float32)

            loss_without_ppo, stats_without_ppo = model.forward(batch, pop, llm_s)
            cfg.ppo_loss_weight = 1.0
            loss_with_ppo, stats_with_ppo = model.forward(batch, pop, llm_s)

            self.assertAlmostEqual(
                float((loss_with_ppo - loss_without_ppo).detach().cpu().item()),
                7.0,
                places=5,
            )
            self.assertEqual(stats_without_ppo["ppo_loss"], 0.0)
            self.assertEqual(stats_without_ppo["ppo_loss_raw"], 7.0)
            self.assertEqual(stats_with_ppo["ppo_loss"], 7.0)

    def test_zero_usim_steps_skips_simulator_and_has_zero_ppo_loss(self):
        with EnvPatch(USIM_STEPS="0"):
            cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
            model.device = torch.device("cpu")
            init = torch.randn((2, cfg.emb_dim), dtype=torch.float32)
            user_bank = torch.randn((3, cfg.emb_dim), dtype=torch.float32)

            final_h, trajectory, stats = model.run_usim_episode(
                init,
                target_emb=init.clone(),
                user_bank_raw=user_bank,
            )
            loss = model.compute_ppo_loss(trajectory)

            self.assertTrue(torch.allclose(final_h, init))
            self.assertEqual(stats["steps"], 0)
            self.assertEqual(len(trajectory["rewards"]), 0)
            self.assertEqual(float(loss.detach().cpu().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
