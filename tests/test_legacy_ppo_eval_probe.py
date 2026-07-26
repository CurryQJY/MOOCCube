import os

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy
import usim_feedback_fast3_content_delta_eval_probe as probe


def test_probe_keeps_split_seed_but_sets_independent_eval_seed(monkeypatch):
    monkeypatch.setenv("USIM_STATIC_SEED", "2025")
    probe.install_probe(eval_seed=9101, action_mode="sample")

    assert os.environ["USIM_STATIC_SEED"] == "2025"
    assert os.environ["USIM_SEED"] == "9101"
    assert legacy._checkpoint_config_matches({}, object())[0] is True


def test_probe_argmax_installs_deterministic_eval_policy():
    probe.install_probe(eval_seed=9102, action_mode="argmax")
    assert legacy.FixedSimpleAC.get_action_value is probe.deterministic_get_action_value


def test_probe_installs_episode_call_audit():
    probe.install_probe(eval_seed=9103, action_mode="sample")
    assert legacy.Fast3FeedbackUSIM.run_usim_episode is probe.audited_run_usim_episode
