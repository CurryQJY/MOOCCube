from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast3_delta.config import FeedbackConfig
from fast3_delta.sg_urinit import apply_sg_urinit_, build_sg_urinit_weights


class SGURInitTest(unittest.TestCase):
    def _set_env(self, key, value):
        old_value = os.environ.get(key)
        os.environ[key] = value

        def restore():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

        self.addCleanup(restore)

    def test_config_reads_sg_urinit_switches(self):
        self._set_env("USIM_USE_SG_URINIT", "1")
        self._set_env("USIM_SG_URINIT_CLUSTER_K", "7")
        self._set_env("USIM_SG_URINIT_LOCAL_W", "0.8")
        self._set_env("USIM_SG_URINIT_GLOBAL_W", "0.2")
        self._set_env("USIM_SG_URINIT_TARGET_NORM", "0.05")
        self._set_env("USIM_SG_URINIT_MAX_ITER", "3")
        self._set_env("USIM_SG_URINIT_SEED", "99")

        cfg = FeedbackConfig(5, 6, content_dim=2)

        self.assertTrue(cfg.use_sg_urinit)
        self.assertEqual(cfg.sg_urinit_cluster_k, 7)
        self.assertAlmostEqual(cfg.sg_urinit_local_weight, 0.8)
        self.assertAlmostEqual(cfg.sg_urinit_global_weight, 0.2)
        self.assertAlmostEqual(cfg.sg_urinit_target_norm, 0.05)
        self.assertEqual(cfg.sg_urinit_max_iter, 3)
        self.assertEqual(cfg.sg_urinit_seed, 99)

    def test_build_sg_urinit_weights_uses_user_history_content_mean(self):
        train_df = pd.DataFrame(
            {
                "u_idx": [0, 0, 1],
                "i_idx": [0, 1, 2],
            }
        )
        content_emb = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )

        weights, mask, stats = build_sg_urinit_weights(
            train_df,
            content_emb,
            n_users=3,
            emb_dim=2,
            cluster_k=2,
            local_weight=1.0,
            global_weight=0.0,
            target_norm=0.2,
            seed=2025,
        )

        expected_user_0 = torch.tensor([1.0, 1.0])
        expected_user_0 = expected_user_0 / expected_user_0.norm() * 0.2
        expected_user_1 = torch.tensor([-0.2, 0.0])

        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False])))
        self.assertTrue(torch.allclose(weights[0], expected_user_0, atol=1e-6))
        self.assertTrue(torch.allclose(weights[1], expected_user_1, atol=1e-6))
        self.assertTrue(torch.allclose(weights[2], torch.zeros(2), atol=1e-6))
        self.assertEqual(stats["initialized_users"], 2)
        self.assertEqual(stats["cold_users"], 1)

    def test_apply_sg_urinit_only_overwrites_users_with_training_history(self):
        train_df = pd.DataFrame(
            {
                "u_idx": [0, 1],
                "i_idx": [0, 1],
            }
        )
        content_emb = torch.eye(2)
        model = SimpleNamespace(user_emb=torch.nn.Embedding(3, 2))
        with torch.no_grad():
            model.user_emb.weight.copy_(
                torch.tensor(
                    [
                        [0.01, 0.02],
                        [0.03, 0.04],
                        [0.05, 0.06],
                    ]
                )
            )
        original_no_history = model.user_emb.weight[2].detach().clone()
        cfg = SimpleNamespace(
            n_users=3,
            emb_dim=2,
            use_sg_urinit=True,
            sg_urinit_cluster_k=2,
            sg_urinit_local_weight=1.0,
            sg_urinit_global_weight=0.0,
            sg_urinit_target_norm=0.1,
            sg_urinit_seed=7,
        )

        stats = apply_sg_urinit_(model, train_df, content_emb, cfg)

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["initialized_users"], 2)
        self.assertTrue(torch.allclose(model.user_emb.weight[0], torch.tensor([0.1, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(model.user_emb.weight[1], torch.tensor([0.0, 0.1]), atol=1e-6))
        self.assertTrue(torch.allclose(model.user_emb.weight[2], original_no_history, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
