#!/usr/bin/env python3
"""Compare ContentDelta ON/OFF single-variable ablation under research-v2 recipe."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "outputs" / "recppo_research_repair" / "content_delta_ablation_v2"
# Same code hash as research_v2; reuse as ON if fresh delta_on not present.
REF_ON = (
    ROOT
    / "outputs"
    / "recppo_compare"
    / "recppo_single_research_v2"
    / "strict_item_cold_balanced_thr1_seed_2025"
)

REPORT = "final_report_usim_feedback_fast3_content_delta_static.csv"
MANIFEST = "static_protocol_manifest.json"
METRICS = "mooc_metrics_usim_feedback_fast3_content_delta_static.csv"


def _find_seed_dir(base: Path, seed: int = 2025) -> Path | None:
    if not base.exists():
        return None
    # direct seed folder
    direct = base / f"strict_item_cold_balanced_thr1_seed_{seed}"
    if direct.exists():
        return direct
    # arm root may nest seed under protocol
    for p in base.rglob(f"strict_item_cold_balanced_thr1_seed_{seed}"):
        if p.is_dir() and (p / REPORT).exists():
            return p
    return None


def _read_report(seed_dir: Path) -> dict[str, dict[str, float]]:
    path = seed_dir / REPORT
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metric = row["metric"]
            out[metric] = {
                k: float(row[k])
                for k in (
                    "full_cold",
                    "full_hot",
                    "full_cold_item_macro",
                    "full_hot_item_macro",
                )
                if row.get(k) not in (None, "")
            }
    return out


def _read_manifest_flags(seed_dir: Path) -> dict:
    path = seed_dir / MANIFEST
    m = json.loads(path.read_text(encoding="utf-8"))
    env = m.get("env", {})
    mc = m.get("model_config", {})
    return {
        "script": Path(m.get("script", {}).get("path", "")).name,
        "script_sha256": m.get("script", {}).get("sha256"),
        "use_content_delta": env.get("USIM_USE_CONTENT_DELTA", mc.get("use_content_delta")),
        "n_epochs": env.get("USIM_N_EPOCHS", mc.get("n_epochs")),
        "patience": env.get("USIM_EARLY_STOP_PATIENCE", mc.get("early_stop_patience")),
        "rl_residual_scale": env.get("USIM_RL_RESIDUAL_SCALE", mc.get("rl_residual_scale")),
        "pseudo_mode": env.get("USIM_PSEUDO_COLD_MODE", mc.get("pseudo_cold_mode")),
        "pseudo_ratio": env.get("USIM_PSEUDO_COLD_RATIO", mc.get("pseudo_cold_ratio")),
        "pseudo_min_pop": env.get("USIM_PSEUDO_COLD_MIN_POP", mc.get("pseudo_cold_min_pop")),
        "use_pseudo": env.get("USIM_USE_PSEUDO_COLD_TRAIN", mc.get("use_pseudo_cold_train")),
        "ppo_w": env.get("USIM_PPO_LOSS_WEIGHT"),
        "rollout": env.get("USIM_ROLLOUT_POLICY"),
        "steps": env.get("USIM_STEPS"),
        "warmup": (mc.get("recppo") or {}).get("warmup_epochs"),
    }


def _best_val(seed_dir: Path) -> dict | None:
    path = seed_dir / METRICS
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["Val_full_cold_R@10"]))
    return {
        "best_epoch": int(best["Epoch"]),
        "val_cold_R10": float(best["Val_full_cold_R@10"]),
        "val_hot_R10": float(best["Val_full_hot_R@10"]),
        "val_cold_N10": float(best["Val_full_cold_N@10"]),
        "val_hot_N10": float(best["Val_full_hot_N@10"]),
        "n_epochs_logged": len(rows),
    }


def main() -> int:
    on_dir = _find_seed_dir(ABLATION / "delta_on") or REF_ON
    off_dir = _find_seed_dir(ABLATION / "delta_off")

    print("=== ContentDelta single-variable ablation (research-v2 recipe) ===")
    print(f"ON  dir: {on_dir}")
    print(f"OFF dir: {off_dir}")

    if on_dir is None or not on_dir.exists():
        print("ERROR: missing ON arm")
        return 1
    if off_dir is None or not off_dir.exists():
        print("ERROR: missing OFF arm (still running?)")
        return 2

    on_flags = _read_manifest_flags(on_dir)
    off_flags = _read_manifest_flags(off_dir)
    print("\n-- config check --")
    keys = sorted(set(on_flags) | set(off_flags))
    for k in keys:
        a, b = on_flags.get(k), off_flags.get(k)
        mark = "OK" if str(a) == str(b) or k == "use_content_delta" else "DIFF?"
        if k == "use_content_delta":
            mark = "FACTOR"
        print(f"  [{mark}] {k}: ON={a} | OFF={b}")

    if str(on_flags.get("use_content_delta")) not in ("1", "True", "true"):
        print("WARNING: ON arm does not have ContentDelta enabled")
    if str(off_flags.get("use_content_delta")) not in ("0", "False", "false"):
        print("WARNING: OFF arm still has ContentDelta enabled")

    on_rep = _read_report(on_dir)
    off_rep = _read_report(off_dir)

    print("\n-- test metrics (OFF - ON); positive => ContentDelta hurts --")
    header = f"{'metric':8} {'split':22} {'ON':10} {'OFF':10} {'d(OFF-ON)':12} {'rel%':8}"
    print(header)
    rows_out = []
    for metric in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20"):
        for split in (
            "full_cold",
            "full_cold_item_macro",
            "full_hot",
            "full_hot_item_macro",
        ):
            a = on_rep[metric][split]
            b = off_rep[metric][split]
            d = b - a
            rel = (d / a * 100.0) if a != 0 else float("nan")
            print(f"{metric:8} {split:22} {a:10.4f} {b:10.4f} {d:+12.4f} {rel:+7.2f}%")
            rows_out.append(
                {
                    "metric": metric,
                    "split": split,
                    "on": a,
                    "off": b,
                    "delta_off_minus_on": d,
                    "rel_pct": rel,
                }
            )

    on_val = _best_val(on_dir)
    off_val = _best_val(off_dir)
    print("\n-- validation best (by cold R@10) --")
    print("ON :", on_val)
    print("OFF:", off_val)

    # Headline judgment on main-table metric
    on_m = on_rep["R@10"]["full_cold_item_macro"]
    off_m = off_rep["R@10"]["full_cold_item_macro"]
    d_m = off_m - on_m
    print("\n-- headline (main-table style: cold item-macro R@10) --")
    print(f"ON (delta=1):  {on_m:.4f}")
    print(f"OFF(delta=0):  {off_m:.4f}")
    print(f"OFF - ON:      {d_m:+.4f}")
    if d_m > 0.003:
        print("Judgment: ContentDelta is a SIDE EFFECT under v2 recipe (OFF better).")
    elif d_m < -0.003:
        print("Judgment: ContentDelta is a POSITIVE effect under v2 recipe (ON better).")
    else:
        print("Judgment: ContentDelta is roughly NEUTRAL under v2 recipe (|d|<0.003).")

    out_csv = ABLATION / "content_delta_ablation_summary.csv"
    ABLATION.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "split", "on", "off", "delta_off_minus_on", "rel_pct"],
        )
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
