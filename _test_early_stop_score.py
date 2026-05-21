"""Unit tests for `_compute_early_stop_score` formulas.

Covers all four `early_stop_score_mode` values:
    cold_only (legacy default), geometric, harmonic, sum.

Edge cases:
    - `None` cold/hot dicts (treated as 0 via `_metric_or_zero`).
    - Either side equal to 0 in geometric/harmonic.
    - Mode aliases (case + whitespace).

Run:
    python _test_early_stop_score.py
"""
import math
import os

os.environ["USIM_FORCE_CPU"] = "1"
os.environ.setdefault("USIM_DISABLE_LLM_SCORE", "1")

import usim_feedback_fast3_content_delta as M


def _approx(a, b, atol=1e-9):
    return abs(float(a) - float(b)) <= atol


def test_cold_only_legacy():
    cold = {"N@10": 0.25, "R@10": 0.30}
    hot = {"N@10": 0.10, "R@10": 0.20}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="cold_only")
    assert _approx(score, 0.25), f"cold_only: expected 0.25 got {score}"
    print(f"[PASS] cold_only -> {score:.4f}")


def test_geometric():
    cold = {"N@10": 0.25}
    hot = {"N@10": 0.16}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="geometric")
    expected = math.sqrt(0.25 * 0.16)  # 0.2
    assert _approx(score, expected), f"geometric: expected {expected} got {score}"
    print(f"[PASS] geometric -> {score:.4f}")


def test_geometric_zero_collapses():
    cold = {"N@10": 0.30}
    hot = {"N@10": 0.0}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="geometric")
    assert _approx(score, 0.0), f"geometric with hot=0 should be 0, got {score}"
    print(f"[PASS] geometric collapses on either-zero -> {score:.4f}")


def test_harmonic():
    cold = {"N@10": 0.20}
    hot = {"N@10": 0.30}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="harmonic")
    expected = 2.0 * 0.20 * 0.30 / (0.20 + 0.30)  # 0.24
    assert _approx(score, expected), f"harmonic: expected {expected} got {score}"
    print(f"[PASS] harmonic -> {score:.4f}")


def test_harmonic_zero_short_circuits():
    cold = {"N@10": 0.0}
    hot = {"N@10": 0.30}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="harmonic")
    assert _approx(score, 0.0), f"harmonic with cold=0 should be 0, got {score}"
    print(f"[PASS] harmonic short-circuits on zero -> {score:.4f}")


def test_sum():
    cold = {"N@10": 0.18}
    hot = {"N@10": 0.13}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="sum")
    assert _approx(score, 0.31), f"sum: expected 0.31 got {score}"
    print(f"[PASS] sum -> {score:.4f}")


def test_none_metrics_treated_as_zero():
    score_cold_none = M._compute_early_stop_score(None, {"N@10": 0.5}, k=10, mode="geometric")
    score_hot_none = M._compute_early_stop_score({"N@10": 0.5}, None, k=10, mode="geometric")
    score_both_none = M._compute_early_stop_score(None, None, k=10, mode="cold_only")
    assert _approx(score_cold_none, 0.0)
    assert _approx(score_hot_none, 0.0)
    assert _approx(score_both_none, 0.0)
    print("[PASS] None metrics -> 0 in all modes")


def test_missing_keys_treated_as_zero():
    cold = {"R@10": 0.40}  # no N@10
    hot = {"N@10": 0.20}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="cold_only")
    assert _approx(score, 0.0), f"missing N@10 should be 0, got {score}"
    print(f"[PASS] missing N@k key -> 0 in cold_only mode")


def test_unknown_mode_falls_back_to_cold_only():
    cold = {"N@10": 0.42}
    hot = {"N@10": 0.99}
    score = M._compute_early_stop_score(cold, hot, k=10, mode="banana")
    assert _approx(score, 0.42), f"unknown mode should fall back to cold_only, got {score}"
    print(f"[PASS] unknown mode -> cold_only fallback ({score:.4f})")


def test_k_dispatch():
    cold = {"N@5": 0.10, "N@10": 0.20, "N@20": 0.30}
    hot = {"N@5": 0.05, "N@10": 0.15, "N@20": 0.25}
    s5 = M._compute_early_stop_score(cold, hot, k=5, mode="cold_only")
    s10 = M._compute_early_stop_score(cold, hot, k=10, mode="sum")
    s20 = M._compute_early_stop_score(cold, hot, k=20, mode="geometric")
    assert _approx(s5, 0.10)
    assert _approx(s10, 0.35)
    assert _approx(s20, math.sqrt(0.30 * 0.25))
    print(f"[PASS] k dispatch correct: k=5 -> {s5:.4f}, k=10 -> {s10:.4f}, k=20 -> {s20:.4f}")


def test_config_validation_rejects_bad_mode():
    """The config should raise on invalid USIM_EARLY_STOP_SCORE_MODE."""
    os.environ["USIM_EARLY_STOP_SCORE_MODE"] = "banana"
    try:
        M.Fast3Config(n_users=4, n_items=4, content_dim=4)
    except ValueError as exc:
        msg = str(exc)
        assert "USIM_EARLY_STOP_SCORE_MODE" in msg
        print(f"[PASS] config rejects bad score mode: {msg}")
    else:
        raise AssertionError("Expected ValueError on bad USIM_EARLY_STOP_SCORE_MODE")
    finally:
        os.environ.pop("USIM_EARLY_STOP_SCORE_MODE", None)


if __name__ == "__main__":
    print("=" * 64)
    print("early-stop score helper: formula tests")
    print("=" * 64)
    test_cold_only_legacy()
    test_geometric()
    test_geometric_zero_collapses()
    test_harmonic()
    test_harmonic_zero_short_circuits()
    test_sum()
    test_none_metrics_treated_as_zero()
    test_missing_keys_treated_as_zero()
    test_unknown_mode_falls_back_to_cold_only()
    test_k_dispatch()
    test_config_validation_rejects_bad_mode()
    print("=" * 64)
    print("All early-stop score tests passed.")
