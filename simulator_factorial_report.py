"""Report the simulator-training by inference-policy factorial."""

import argparse
import json
from pathlib import Path

from ppo_loss_factorial_report import load_factorial, summarize_factorial


AUDIT_JSON = "actor_inference_audit.json"
SEED_DIR = "strict_item_cold_balanced_thr1_seed_{seed}"


def validate_simulator_step_audit(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_steps = payload.get("checkpoint_usim_steps_values")
    inference_steps = payload.get("effective_inference_usim_steps_values")
    if checkpoint_steps != [0] or inference_steps != [5]:
        raise ValueError(
            "Expected checkpoint T=0 and inference T=5, "
            f"got checkpoint={checkpoint_steps}, inference={inference_steps}"
        )
    return payload


def rename_factorial_columns(frame):
    exact = {
        "on_static": "t5_training_static",
        "on_course_fit": "t5_training_course_fit",
        "off_static": "t0_training_static",
        "off_course_fit": "t0_training_course_fit",
        "training_effect_static": "simulator_training_effect_static",
        "training_effect_course_fit": "simulator_training_effect_course_fit",
        "inference_effect_ppo_on": "course_fit_effect_t5_training",
        "inference_effect_ppo_off": "course_fit_effect_t0_training",
    }
    columns = {}
    for column in frame.columns:
        renamed = column
        for old, new in exact.items():
            if column == old or column.startswith(f"{old}_"):
                renamed = new + column[len(old):]
                break
        columns[column] = renamed
    return frame.rename(columns=columns)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-static-root", required=True)
    parser.add_argument("--t5-course-fit-root", required=True)
    parser.add_argument("--t0-static-root", required=True)
    parser.add_argument("--t0-course-fit-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    for seed in args.seeds:
        audit_path = (
            Path(args.t0_course_fit_root)
            / SEED_DIR.format(seed=seed)
            / AUDIT_JSON
        )
        validate_simulator_step_audit(audit_path)

    roots = {
        "on_static": args.t5_static_root,
        "on_course_fit": args.t5_course_fit_root,
        "off_static": args.t0_static_root,
        "off_course_fit": args.t0_course_fit_root,
    }
    rows = load_factorial(roots, args.seeds)
    by_seed, summary = summarize_factorial(rows)
    by_seed = rename_factorial_columns(by_seed)
    summary = rename_factorial_columns(summary)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(output_root / "simulator_factorial_by_seed.csv", index=False)
    summary.to_csv(output_root / "simulator_factorial_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
