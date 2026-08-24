"""Verify that the ranking/pedagogy decoupling signs hold on the held-out TEST split.

The revised Figure 1 motivation was established on validation lists. The paper
text now states that the *sign* of each correlation is preserved on test. This
script discharges that claim honestly, reusing the exact same computation
primitives as the validation/Figure-1 pipeline:

  - build_real_risk_artifacts : identical structural artifacts (prereq, concept,
    difficulty complexity, redundancy) used for Figure 1.
  - _seed_inputs              : identical cold-target / history / popularity
    construction, but reading strict_item_cold_TEST rows.
  - analyze_export_record     : identical per-list NDCG@k and structural proxies.

Inputs are the frozen-checkpoint TEST exports produced by
export_p1_cgrc_topk.py / export_p1_pcgnn_topk.py with --analysis-split test.

Output: a per-course table plus Spearman correlations of NDCG@10 against the
four structural proxies, over cold courses that are actually exposed
(recall@10 > 0), pooled across three seeds -- mirroring the validation figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.analyze_p1_topk_motivation import (  # noqa: E402
    _seed_inputs,
    analyze_export_record,
    build_real_risk_artifacts,
)

SEEDS = (2025, 2026, 2027)
METRIC_K = 10
STRUCTURAL = [
    ("cold_structural_redundancy", "redundancy", "lower", "misaligned"),
    ("cold_prerequisite_gap", "prerequisite gap", "lower", "orthogonal"),
    ("cold_difficulty_gap", "difficulty gap", "lower", "orthogonal"),
    ("cold_concept_continuity", "concept continuity", "higher", "aligned"),
]

# validation-split reference signs, for a same-sign check
VALIDATION_RHO = {
    "cold_structural_redundancy": +0.528,
    "cold_prerequisite_gap": -0.151,
    "cold_difficulty_gap": -0.183,
    "cold_concept_continuity": +0.555,
}


def test_export_paths(root: Path, seed: int) -> dict[str, Path]:
    base = Path(root) / "outputs" / "test_motivation"
    split_id = f"strict_item_cold_balanced_thr1_seed_{seed}"
    return {
        "cgrc": base / "cgrc" / split_id / "top20_test.jsonl",
        "pcgnn": base / "pcgnn" / split_id / "pcgnn_top20.jsonl",
    }


def analyze_seed(root: Path, seed: int, artifacts, n_items: int) -> pd.DataFrame:
    split_root = root / "outputs" / "content_delta_pop5" / "static_item_cold_balanced"
    expected_pairs, histories, popularity = _seed_inputs(split_root, seed, n_items)

    # per-course accumulators, keyed (model, target_item_id)
    perf: dict[tuple, dict] = {}
    struct: dict[tuple, dict] = {}

    paths = test_export_paths(root, seed)
    for model, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing test export: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                # export records carry model/seed; normalize model label
                rec_model = model
                user_id = int(record["user_id"])
                target = int(record["target_item_id"])
                history = histories.get(user_id, np.empty(0, dtype=np.int64))
                # ensure sample_index present for the primitive
                record.setdefault("sample_index", 0)
                record["model"] = rec_model
                record["seed"] = int(seed)
                _, list_rows = analyze_export_record(
                    record,
                    history_item_ids=history,
                    train_popularity=popularity,
                    artifacts=artifacts,
                    cutoffs=(METRIC_K,),
                )
                row = list_rows[0]
                key = (rec_model, target)
                s = struct.setdefault(
                    key, {col: 0.0 for col, *_ in STRUCTURAL} | {"n": 0}
                )
                for col, *_ in STRUCTURAL:
                    s[col] += float(row[col])
                s["n"] += 1

                prefix = [int(i) for i in record["recommended_item_ids"][:METRIC_K]]
                rank = prefix.index(target) + 1 if target in prefix else None
                p = perf.setdefault(key, {"hit": 0.0, "ndcg": 0.0, "count": 0})
                p["count"] += 1
                p["hit"] += float(rank is not None)
                p["ndcg"] += 1.0 / math.log2(rank + 1.0) if rank is not None else 0.0

    rows = []
    for key, p in perf.items():
        model, target = key
        s = struct[key]
        entry = {
            "analysis_split": "test",
            "model": model,
            "seed": int(seed),
            "target_item_id": int(target),
            "cutoff": METRIC_K,
            "recall_at_10": p["hit"] / p["count"],
            "ndcg_at_10": p["ndcg"] / p["count"],
        }
        for col, *_ in STRUCTURAL:
            entry[col] = s[col] / s["n"] if s["n"] else float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify decoupling signs on the test split")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_aaai27" / "figures" / "test_motivation_analysis" / "course_macro.csv",
    )
    args = parser.parse_args()

    print("[test-verify] building structural artifacts (same as Figure 1) ...", flush=True)
    artifacts, stats = build_real_risk_artifacts(args.root)
    n_items = int(stats["n_items"])

    frames = []
    for seed in SEEDS:
        print(f"[test-verify] analyzing seed {seed} ...", flush=True)
        frames.append(analyze_seed(args.root, seed, artifacts, n_items))
    course = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    course.to_csv(args.output, index=False)
    print(f"[test-verify] wrote {args.output}  ({len(course)} course rows)", flush=True)

    exposed = course[course["recall_at_10"] > 0].copy()
    print(f"\n[test-verify] exposed cold courses (recall@10>0): n={len(exposed)}")
    print(f"  per model: {exposed['model'].value_counts().to_dict()}")
    print("\n  Spearman(NDCG@10, structural proxy) on TEST vs VALIDATION sign:")
    print("  " + "-" * 78)
    all_consistent = True
    for col, label, direction, tag in STRUCTURAL:
        rho, p = spearmanr(exposed["ndcg_at_10"], exposed[col])
        val = VALIDATION_RHO[col]
        same = (np.sign(rho) == np.sign(val))
        all_consistent = all_consistent and same
        flag = "OK " if same else "XX "
        print(
            f"  {flag} {label:20s} test rho={rho:+.3f} (p={p:.1e})  "
            f"val rho={val:+.3f}  [{tag}]"
        )
    print("  " + "-" * 78)
    print(f"  ALL SIGNS CONSISTENT WITH VALIDATION: {all_consistent}")

    # within-CGRC median split, mirroring the paper's robustness sentence
    cg = exposed[exposed["model"] == "cgrc"]
    if len(cg) >= 4:
        med = cg["ndcg_at_10"].median()
        hi = cg[cg["ndcg_at_10"] >= med]
        lo = cg[cg["ndcg_at_10"] < med]
        print(f"\n  CGRC median split (n_hi={len(hi)}, n_lo={len(lo)}):")
        for col, label, *_ in STRUCTURAL:
            print(f"    {label:20s} hi={hi[col].mean():+.3f}  lo={lo[col].mean():+.3f}")


if __name__ == "__main__":
    main()
