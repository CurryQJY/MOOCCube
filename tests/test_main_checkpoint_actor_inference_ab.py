import json
from pathlib import Path

import pytest
import pandas as pd
import torch
import torch.nn.functional as F

import main_checkpoint_actor_inference_ab as ab
import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy


def test_wrapper_declares_static_runner_delegation_tokens():
    source = Path(ab.__file__).read_text(encoding="utf-8")
    assert "def run_static_experiment" in source
    assert "_static_split_df" in source


def test_validation_target_routes_validation_rows_without_test_rows():
    train = pd.DataFrame({"split": ["train"]})
    val = pd.DataFrame({"split": ["val"]})
    test = pd.DataFrame({"split": ["test"]})

    def split_fn(_):
        return train, val, test, {"test_rows": 1, "val_rows": 1}

    wrapped = ab.make_validation_target_split(split_fn)
    got_train, got_val, got_eval, info = wrapped(object())

    assert got_train.equals(train)
    assert got_val.equals(val)
    assert got_eval.equals(val)
    assert not got_eval.equals(test)
    assert info == {"test_rows": 1, "val_rows": 1}


def test_history_fingerprint_is_order_independent_and_counts_pairs():
    left = {1: {4, 2}, 0: {3}}
    right = {0: {3}, 1: {2, 4}}

    left_summary = ab.history_fingerprint(left)
    right_summary = ab.history_fingerprint(right)

    assert left_summary == right_summary
    assert left_summary["users"] == 2
    assert left_summary["pairs"] == 3


def test_audited_split_records_history_fingerprints_and_item_sets():
    train = pd.DataFrame({"u_idx": [0, 0, 1], "i_idx": [1, 2, 2]})
    val = pd.DataFrame({"u_idx": [0], "i_idx": [3]})
    test = pd.DataFrame({"u_idx": [1], "i_idx": [4]})

    def split_fn(_):
        return train, val, test, {"mode": "strict"}

    ab.reset_audit()
    wrapped = ab.make_audited_split(split_fn)
    got_train, got_val, got_test, info = wrapped(object())

    assert got_train.equals(train)
    assert got_val.equals(val)
    assert got_test.equals(test)
    assert info == {"mode": "strict"}
    assert ab.AUDIT.split_train_rows == 3
    assert ab.AUDIT.split_validation_rows == 1
    assert ab.AUDIT.split_test_rows == 1
    assert ab.AUDIT_CONTEXT.train_items == {1, 2}
    assert ab.AUDIT_CONTEXT.validation_items == {3}
    assert ab.AUDIT_CONTEXT.test_items == {4}
    assert ab.AUDIT_CONTEXT.train_history["pairs"] == 3
    assert ab.AUDIT_CONTEXT.train_plus_validation_history["pairs"] == 4


def test_audited_course_fit_counts_candidate_histories_containing_target(monkeypatch):
    class TinyCourseFitModel:
        user_seen_index = torch.tensor(
            [
                [False, False, True],
                [True, False, False],
                [False, False, False],
            ]
        )

    monkeypatch.setattr(
        ab,
        "ORIGINAL_COMPUTE_COURSE_FIT",
        lambda self, candidate_user_idx, item_idx, target_pop=None, user_seen_items=None: torch.zeros_like(
            candidate_user_idx, dtype=torch.float32
        ),
    )
    ab.reset_audit()

    result = ab.audited_compute_candidate_course_fit(
        TinyCourseFitModel(),
        candidate_user_idx=torch.tensor([[0, 2], [1, 2]]),
        item_idx=torch.tensor([2, 0]),
        user_seen_items={},
    )

    assert torch.equal(result, torch.zeros((2, 2)))
    assert ab.AUDIT.course_fit_calls == 1
    assert ab.AUDIT.course_fit_candidate_pairs == 4
    assert ab.AUDIT.target_seen_candidate_pairs == 2
    assert ab.AUDIT.course_fit_target_rows == 2
    assert ab.AUDIT.target_rows_with_seen_candidate == 2


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


class TinyInferenceModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.training = False
        self.cfg = type(
            "Cfg",
            (),
            {
                "emb_dim": 2,
                "candidate_strategy": "retrieve_sample",
                "rollout_policy": "ppo",
                "feedback_course_match_exclude_target": False,
            },
        )()
        self.item_id_emb = type("Emb", (), {"weight": torch.zeros(3, 2)})()
        self.item_popularity = torch.tensor([3.0, 0.0, 0.0])
        self.user_seen_index = torch.zeros((2, 3), dtype=torch.bool)
        self.calls = []

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def _build_user_bank_raw(self):
        bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return bank, F.normalize(bank, dim=1)

    def get_item_vector(self, idx, llm_s, force_cold, disable_id_dropout=True):
        base = torch.stack([idx.float() + 1.0, torch.ones_like(idx).float()], dim=1)
        return base, base, base

    def run_usim_episode(self, init, target_emb, **kwargs):
        self.calls.append(
            {
                "target_emb": target_emb,
                "exclude_target": self.cfg.feedback_course_match_exclude_target,
                **kwargs,
            }
        )
        return init + 0.25, {}, {}

    def _blend_rl_episode_output(self, base, final):
        return final


def test_refinement_never_supplies_behavior_target():
    model = TinyInferenceModel()

    out = ab.infer_actor_refined_item_vectors(model, torch.tensor([1, 2]))

    assert out.shape == (2, 2)
    assert len(model.calls) == 1
    assert model.calls[0]["target_emb"] is None
    assert torch.equal(model.calls[0]["item_idx"], torch.tensor([1, 2]))
    assert model.calls[0]["user_seen_items"] == {}


def test_audited_history_install_classifies_exact_train_only_mapping(monkeypatch):
    class TinyHistoryModel:
        installed = None

    train_seen = {0: {1, 2}, 1: {2}}
    train_val_seen = {0: {1, 2, 3}, 1: {2}}
    ab.reset_audit()
    ab.AUDIT_CONTEXT.train_history = ab.history_fingerprint(train_seen)
    ab.AUDIT_CONTEXT.train_plus_validation_history = ab.history_fingerprint(train_val_seen)
    monkeypatch.setattr(
        ab,
        "ORIGINAL_SET_USER_SEEN_INDEX",
        lambda self, value: setattr(self, "installed", value),
    )
    model = TinyHistoryModel()

    ab.audited_set_user_seen_index(model, train_seen)

    assert model.installed == train_seen
    assert ab.AUDIT.history_set_calls == 1
    assert ab.AUDIT.history_sources == ["train_only"]


def test_refinement_temporarily_overrides_target_exclusion_and_records_inputs():
    model = TinyInferenceModel()
    previous = ab.COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE
    ab.COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE = True
    ab.reset_audit()
    try:
        ab.infer_actor_refined_item_vectors(model, torch.tensor([1, 2]))
    finally:
        ab.COURSE_MATCH_EXCLUDE_TARGET_OVERRIDE = previous

    assert model.calls[0]["exclude_target"] is True
    assert model.cfg.feedback_course_match_exclude_target is False
    assert ab.AUDIT.behavior_target_none_calls == 1
    assert ab.AUDIT.behavior_target_non_null_calls == 0
    assert ab.AUDIT.refined_item_ids == [1, 2]
    assert ab.AUDIT.course_match_exclude_target_values == [True]


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("", None), ("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_parse_optional_bool(raw, expected):
    assert ab.parse_optional_bool(raw) is expected


def test_install_static_does_not_attach_refinement(monkeypatch):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)

    ab.install_mode("static", eval_seed=7001)

    assert not hasattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors")


def test_install_actor_attaches_refinement(monkeypatch):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)

    ab.install_mode("actor", eval_seed=7001)

    assert legacy.Fast3FeedbackUSIM.infer_refined_item_vectors is ab.infer_actor_refined_item_vectors


@pytest.mark.parametrize(
    "mode,rollout,uses_refiner,uses_argmax",
    [
        ("static", "ppo", False, False),
        ("ppo", "ppo", True, True),
        ("greedy_similarity", "greedy_similarity", True, False),
        ("course_fit", "course_fit", True, False),
        ("random", "random", True, False),
    ],
)
def test_install_policy_mode(monkeypatch, mode, rollout, uses_refiner, uses_argmax):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)

    ab.install_mode(mode, eval_seed=7001)

    assert ab.INFERENCE_ROLLOUT_POLICY == rollout
    assert hasattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors") is uses_refiner
    assert (legacy.FixedSimpleAC.get_action_value is ab.deterministic_get_action_value) is uses_argmax


def test_positive_target_vector_comes_from_the_same_refined_bank():
    previous = getattr(ab, "ACTIVE_ITEM_BANK", None)
    ab.ACTIVE_ITEM_BANK = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    try:
        out = ab.bank_aligned_pos_item_vecs(
            model=None,
            item_idx=torch.tensor([2, 0]),
            llm_s=None,
            pop_sel=None,
            eval_type="cold",
        )
    finally:
        ab.ACTIVE_ITEM_BANK = previous

    assert torch.equal(out, torch.tensor([[3.0, 4.0], [1.0, 0.0]]))


def test_audit_export_reports_mean_displacement_and_target(tmp_path, monkeypatch):
    monkeypatch.setenv("USIM_ACTOR_EVAL_TARGET", "validation")
    ab.AUDIT = ab.InferenceAudit(
        actor_calls=10,
        episode_calls=2,
        refined_items=4,
        cosine_sum=3.2,
        l2_sum=0.8,
    )
    path = tmp_path / "audit.json"

    ab.write_audit(path, mode="actor", eval_seed=7001)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mean_cosine"] == pytest.approx(0.8)
    assert payload["mean_l2"] == pytest.approx(0.2)
    assert payload["evaluation_target"] == "validation"


def test_audit_export_reports_history_target_seen_and_refined_composition(tmp_path):
    ab.AUDIT = ab.InferenceAudit(
        history_set_calls=2,
        history_sources=["train_only", "train_only"],
        course_fit_candidate_pairs=8,
        target_seen_candidate_pairs=0,
        course_fit_target_rows=4,
        target_rows_with_seen_candidate=0,
        refined_item_ids=[2, 3, 4],
        behavior_target_none_calls=1,
        behavior_target_non_null_calls=0,
        course_match_exclude_target_values=[True],
    )
    ab.AUDIT_CONTEXT = ab.InferenceAuditContext(
        train_items={1},
        validation_items={2},
        test_items={3},
    )
    path = tmp_path / "audit.json"

    ab.write_audit(path, mode="course_fit", eval_seed=7001)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["history_source_counts"] == {"train_only": 2}
    assert payload["history_all_train_only"] is True
    assert payload["target_seen_candidate_rate"] == 0.0
    assert payload["target_rows_with_seen_candidate_rate"] == 0.0
    assert payload["refined_item_composition"] == {
        "total_unique": 3,
        "train_present": 0,
        "validation_only": 1,
        "test_only": 1,
        "validation_and_test": 0,
        "neither_validation_nor_test": 1,
    }
    assert payload["effective_course_match_exclude_target"] == [True]


def test_checkpoint_write_blocker_preserves_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    target = checkpoint_dir / "finished.pt"
    target.write_bytes(b"original")
    blocker = ab.make_read_only_torch_save(checkpoint_dir, real_save=torch.save)

    blocker({"changed": True}, target)

    assert target.read_bytes() == b"original"


def test_checkpoint_write_blocker_allows_non_checkpoint_output(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    output = tmp_path / "output.pt"
    blocker = ab.make_read_only_torch_save(checkpoint_dir, real_save=torch.save)

    blocker({"value": 3}, output)

    assert torch.load(output, weights_only=True) == {"value": 3}
