"""Write or validate the clean CKG-RL method contract.

This CLI is intentionally torch-free so launchers can run it before starting a
training job or allocating GPU resources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ckg_rl_clean_method import CleanMethodConfig, build_clean_method_manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the clean CKG-RL method contract")
    parser.add_argument("--output", required=True, help="Path to write the contract JSON")
    parser.add_argument("--max-policy-delta", type=float, default=1.0)
    parser.add_argument("--stability-anchor-count", type=int, default=0)
    parser.add_argument("--candidate-mode", default="legal_state_retrieval")
    parser.add_argument("--legacy-dual-vector-eval", action="store_true")
    parser.add_argument("--positive-recompute-eval", action="store_true")
    parser.add_argument("--random-id-dropout", action="store_true")
    parser.add_argument("--inference-uses-oracle-targets", action="store_true")
    parser.add_argument("--hot-items-mutable-in-policy", action="store_true")
    parser.add_argument("--selection-reads-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = build_clean_method_manifest(
            CleanMethodConfig(
                legacy_dual_vector_eval=bool(args.legacy_dual_vector_eval),
                positive_recompute_eval=bool(args.positive_recompute_eval),
                random_id_dropout=bool(args.random_id_dropout),
                inference_uses_oracle_targets=bool(args.inference_uses_oracle_targets),
                hot_items_mutable_in_policy=bool(args.hot_items_mutable_in_policy),
                selection_reads_test=bool(args.selection_reads_test),
                candidate_mode=str(args.candidate_mode),
                max_policy_delta=float(args.max_policy_delta),
                stability_anchor_count=int(args.stability_anchor_count),
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
