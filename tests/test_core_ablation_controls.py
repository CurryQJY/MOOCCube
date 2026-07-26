import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast3_delta.config import Fast3Config
from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM, _compute_early_stop_score


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

    def test_rl_residual_scale_blends_episode_output(self):
        with EnvPatch(USIM_RL_RESIDUAL_SCALE="0.1"):
            cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            self.assertAlmostEqual(cfg.rl_residual_scale, 0.1)
            model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
            z_i_base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            episode_h = z_i_base + 10.0

            blended = model._blend_rl_episode_output(z_i_base, episode_h)

            self.assertTrue(torch.allclose(blended, z_i_base + 1.0))

    def test_init_checkpoint_prefers_finished_then_latest(self):
        from usim_feedback_fast3_content_delta import _load_init_model_state_from_checkpoint_dir

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            latest_state = {"weight": torch.tensor([1.0])}
            finished_state = {"weight": torch.tensor([2.0])}
            torch.save({"model_state": latest_state}, ckpt_dir / "latest.pt")
            torch.save({"model_state": finished_state}, ckpt_dir / "finished.pt")

            path, model_state = _load_init_model_state_from_checkpoint_dir(str(ckpt_dir))

            self.assertEqual(Path(path).name, "finished.pt")
            self.assertTrue(torch.equal(model_state["weight"], finished_state["weight"]))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            latest_state = {"weight": torch.tensor([3.0])}
            torch.save({"model_state": latest_state}, ckpt_dir / "latest.pt")

            path, model_state = _load_init_model_state_from_checkpoint_dir(str(ckpt_dir))

            self.assertEqual(Path(path).name, "latest.pt")
            self.assertTrue(torch.equal(model_state["weight"], latest_state["weight"]))

    def test_init_checkpoint_filter_skips_shape_mismatches(self):
        from usim_feedback_fast3_content_delta import _filter_compatible_model_state

        with EnvPatch(USIM_STEPS="5"):
            target_cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            target = Fast3FeedbackUSIM(target_cfg, torch.zeros((4, 5), dtype=torch.float32))
        with EnvPatch(USIM_STEPS="0"):
            source_cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
            source = Fast3FeedbackUSIM(source_cfg, torch.zeros((4, 5), dtype=torch.float32))

        filtered, skipped = _filter_compatible_model_state(target, source.state_dict())

        self.assertIn("user_emb.weight", filtered)
        self.assertIn("agent.common.0.weight", skipped)
        self.assertNotIn("agent.common.0.weight", filtered)

    def test_early_stop_cold_rn_uses_cold_recall_and_ndcg(self):
        score = _compute_early_stop_score(
            {"R@10": 0.4, "N@10": 0.2},
            None,
            10,
            "cold_rn",
        )

        self.assertAlmostEqual(score, 2.0 * 0.4 * 0.2 / (0.4 + 0.2))

    def test_early_stop_balanced_rn_penalizes_hot_drop(self):
        score = _compute_early_stop_score(
            {"R@10": 0.4, "N@10": 0.2},
            {"R@10": 0.2, "N@10": 0.1},
            10,
            "balanced_rn",
        )

        self.assertAlmostEqual(
            score,
            4.0 / ((1.0 / 0.4) + (1.0 / 0.2) + (1.0 / 0.2) + (1.0 / 0.1)),
        )


if __name__ == "__main__":
    unittest.main()
