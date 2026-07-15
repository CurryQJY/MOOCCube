from __future__ import annotations

import math


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")


def weighted_overall(
    cold: float,
    cold_count: int,
    hot: float,
    hot_count: int,
) -> float:
    if not math.isfinite(float(cold)) or not math.isfinite(float(hot)):
        raise ValueError("cold and hot metrics must be finite")
    if cold_count <= 0 or hot_count <= 0:
        raise ValueError("cold and hot course counts must be positive")
    return (
        float(cold) * int(cold_count) + float(hot) * int(hot_count)
    ) / (int(cold_count) + int(hot_count))


def validate_direct_overall(
    reconstructed: float,
    direct: float,
    tolerance: float = 5e-5,
) -> None:
    if not math.isfinite(float(direct)) or abs(
        float(reconstructed) - float(direct)
    ) > float(tolerance):
        raise ValueError(
            "direct overall mismatch: "
            f"reconstructed={reconstructed:.12g}, direct={direct:.12g}"
        )


def unavailable_row(
    dataset: str,
    method: str,
    seed: int,
    reason: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "status": "unavailable_missing_warm_targets",
        "reason": reason,
        **{metric: math.nan for metric in METRICS},
    }
