import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PRESETS = {
    "usim_baseline": {
        "script": "usim.py",
        "env": {},
        "notes": "Plain USIM baseline for reference.",
    },
    "course_default": {
        "script": "usim_course.py",
        "env": {},
        "notes": "Current course-aware default settings.",
    },
    "course_epochs3": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
        },
        "notes": "Match the base USIM default epoch budget.",
    },
    "course_epochs3_no_score": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_TOPL": "0",
            "USIM_COURSE_SCORE_ALPHA": "0.0",
            "USIM_COURSE_SCORE_LAMBDA": "0.0",
        },
        "notes": "Remove course score adjustment to test train/eval mismatch.",
    },
    "course_epochs3_no_score_no_adapter": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_TOPL": "0",
            "USIM_COURSE_SCORE_ALPHA": "0.0",
            "USIM_COURSE_SCORE_LAMBDA": "0.0",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
        },
        "notes": "Disable both score adjustment and cold item adapter.",
    },
    "course_safe_rerank": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "15",
            "USIM_COURSE_SCORE_ALPHA": "0.04",
            "USIM_COURSE_SCORE_LAMBDA": "0.015",
            "USIM_COURSE_PENALTY_CAP": "0.08",
            "USIM_COURSE_SCORE_TRAIN": "0",
            "USIM_COURSE_CONCEPT_RECENT_K": "5",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
        },
        "notes": "Recommended first step: light rerank only for cold items, no adapter.",
    },
    "course_safe_rerank_adapter": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "15",
            "USIM_COURSE_SCORE_ALPHA": "0.04",
            "USIM_COURSE_SCORE_LAMBDA": "0.015",
            "USIM_COURSE_PENALTY_CAP": "0.08",
            "USIM_COURSE_SCORE_TRAIN": "0",
            "USIM_COURSE_CONCEPT_RECENT_K": "5",
            "USIM_USE_COLD_ITEM_ADAPTER": "1",
            "USIM_COLD_ITEM_ADAPTER_BETA": "0.20",
            "USIM_COLD_ITEM_VERY_COLD_THR": "2",
            "USIM_COLD_ITEM_BETA_VERY": "0.35",
            "USIM_COLD_ITEM_BETA_MILD": "0.15",
        },
        "notes": "Add a weak cold-item concept adapter on top of safe rerank.",
    },
    "course_safe_rerank_top10": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "10",
            "USIM_COURSE_SCORE_ALPHA": "0.03",
            "USIM_COURSE_SCORE_LAMBDA": "0.015",
            "USIM_COURSE_PENALTY_CAP": "0.06",
            "USIM_COURSE_SCORE_TRAIN": "0",
            "USIM_COURSE_CONCEPT_RECENT_K": "3",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
        },
        "notes": "More conservative cold-only rerank restricted to top-10.",
    },
    "course_safe_rerank_top20": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "20",
            "USIM_COURSE_SCORE_ALPHA": "0.05",
            "USIM_COURSE_SCORE_LAMBDA": "0.020",
            "USIM_COURSE_PENALTY_CAP": "0.10",
            "USIM_COURSE_SCORE_TRAIN": "0",
            "USIM_COURSE_CONCEPT_RECENT_K": "5",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
        },
        "notes": "Slightly stronger rerank to test whether top-L was too restrictive.",
    },
    "course_safe_rerank_train": {
        "script": "usim_course.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "15",
            "USIM_COURSE_SCORE_ALPHA": "0.03",
            "USIM_COURSE_SCORE_LAMBDA": "0.010",
            "USIM_COURSE_PENALTY_CAP": "0.06",
            "USIM_COURSE_SCORE_TRAIN": "1",
            "USIM_COURSE_CONCEPT_RECENT_K": "5",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
        },
        "notes": "Same safe rerank but enabled during training with smaller weights.",
    },
    "course_feedbacklite_safe": {
        "script": "usim_course_feedback_lite.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "15",
            "USIM_COURSE_SCORE_ALPHA": "0.03",
            "USIM_COURSE_SCORE_LAMBDA": "0.015",
            "USIM_COURSE_PENALTY_CAP": "0.08",
            "USIM_COURSE_CONCEPT_RECENT_K": "5",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
            "USIM_FEEDBACK_LITE_TOPL": "15",
            "USIM_FEEDBACK_LITE_ONLY_COLD": "1",
            "USIM_FEEDBACK_LITE_TRAIN": "0",
            "USIM_FEEDBACK_LITE_ACCEPT_ALPHA": "0.03",
            "USIM_FEEDBACK_LITE_GOOD_ALPHA": "0.04",
            "USIM_FEEDBACK_LITE_PREREQ_PENALTY": "0.04",
            "USIM_FEEDBACK_LITE_DIFF_PENALTY": "0.03",
            "USIM_FEEDBACK_LITE_TOPIC_PENALTY": "0.02",
            "USIM_FEEDBACK_LITE_REDUNDANT_PENALTY": "0.02",
            "USIM_FEEDBACK_LITE_ACCEPT_WEIGHT": "0.04",
            "USIM_FEEDBACK_LITE_TYPE_WEIGHT": "0.03",
            "USIM_FEEDBACK_LITE_HARD_NEG_LAMBDA": "0.0",
            "USIM_FEEDBACK_LITE_MARGIN_WEIGHT": "0.0",
        },
        "notes": "Use feedback-lite only as a weak cold reranker plus small auxiliary supervision.",
    },
    "course_feedbacklite_safer": {
        "script": "usim_course_feedback_lite.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "10",
            "USIM_COURSE_SCORE_ALPHA": "0.02",
            "USIM_COURSE_SCORE_LAMBDA": "0.010",
            "USIM_COURSE_PENALTY_CAP": "0.05",
            "USIM_COURSE_CONCEPT_RECENT_K": "3",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
            "USIM_FEEDBACK_LITE_TOPL": "10",
            "USIM_FEEDBACK_LITE_ONLY_COLD": "1",
            "USIM_FEEDBACK_LITE_TRAIN": "0",
            "USIM_FEEDBACK_LITE_ACCEPT_ALPHA": "0.02",
            "USIM_FEEDBACK_LITE_GOOD_ALPHA": "0.03",
            "USIM_FEEDBACK_LITE_PREREQ_PENALTY": "0.03",
            "USIM_FEEDBACK_LITE_DIFF_PENALTY": "0.02",
            "USIM_FEEDBACK_LITE_TOPIC_PENALTY": "0.02",
            "USIM_FEEDBACK_LITE_REDUNDANT_PENALTY": "0.01",
            "USIM_FEEDBACK_LITE_ACCEPT_WEIGHT": "0.03",
            "USIM_FEEDBACK_LITE_TYPE_WEIGHT": "0.02",
            "USIM_FEEDBACK_LITE_HARD_NEG_LAMBDA": "0.0",
            "USIM_FEEDBACK_LITE_MARGIN_WEIGHT": "0.0",
        },
        "notes": "Very conservative feedback-lite preset to test whether feedback weights were too strong.",
    },
    "course_feedbackquery_safe": {
        "script": "usim_course_feedback_query.py",
        "env": {
            "USIM_N_EPOCHS": "3",
            "USIM_COURSE_SCORE_ONLY_COLD": "1",
            "USIM_COURSE_SCORE_TOPL": "15",
            "USIM_COURSE_SCORE_ALPHA": "0.03",
            "USIM_COURSE_SCORE_LAMBDA": "0.015",
            "USIM_COURSE_PENALTY_CAP": "0.08",
            "USIM_COURSE_CONCEPT_RECENT_K": "5",
            "USIM_USE_COLD_ITEM_ADAPTER": "0",
            "USIM_FEEDBACK_LITE_TOPL": "10",
            "USIM_FEEDBACK_LITE_ONLY_COLD": "1",
            "USIM_FEEDBACK_LITE_TRAIN": "0",
            "USIM_FEEDBACK_LITE_ACCEPT_ALPHA": "0.02",
            "USIM_FEEDBACK_LITE_GOOD_ALPHA": "0.03",
            "USIM_FEEDBACK_LITE_PREREQ_PENALTY": "0.03",
            "USIM_FEEDBACK_LITE_DIFF_PENALTY": "0.02",
            "USIM_FEEDBACK_LITE_TOPIC_PENALTY": "0.02",
            "USIM_FEEDBACK_LITE_REDUNDANT_PENALTY": "0.01",
            "USIM_FEEDBACK_QUERY_TOPM": "20",
            "USIM_FEEDBACK_QUERY_SCALE": "0.08",
            "USIM_FEEDBACK_QUERY_TEMP": "0.20",
            "USIM_FEEDBACK_QUERY_ONLY_COLD": "1",
            "USIM_FEEDBACK_QUERY_AUX_WEIGHT": "0.02",
            "USIM_FEEDBACK_QUERY_MARGIN": "0.01",
        },
        "notes": "Weak query adaptation on top of conservative course and feedback signals.",
    },
}


METRIC_COLUMNS = (
    "sample_cold",
    "sample_hot",
    "full_cold",
    "full_hot",
)


def parse_final_metrics(output_text):
    metrics = {}
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 5:
            continue
        metric_name = parts[0]
        if not metric_name or metric_name[0] not in {"R", "N"} or "@" not in metric_name:
            continue
        try:
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        metrics[metric_name] = dict(zip(METRIC_COLUMNS, values))
    return metrics


def format_env_delta(env_delta):
    if not env_delta:
        return "(no overrides)"
    return " ".join(f"{key}={value}" for key, value in env_delta.items())


def run_preset(name, preset, root_dir, python_exe, log_dir, force_static=False, extra_env=None, dry_run=False):
    env = os.environ.copy()
    env.update(preset["env"])
    if force_static:
        env["USIM_STATIC"] = "1"
    if extra_env:
        env.update(extra_env)

    script_path = root_dir / preset["script"]
    command = [python_exe, str(script_path)]
    log_path = log_dir / f"{name}.log"

    if dry_run:
        return {
            "name": name,
            "script": preset["script"],
            "status": "dry_run",
            "command": command,
            "env": {k: env[k] for k in sorted(set(preset["env"].keys()) | ({"USIM_STATIC"} if force_static else set()) | (set(extra_env.keys()) if extra_env else set()))},
            "log_path": str(log_path),
            "notes": preset["notes"],
        }

    output_lines = []
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# preset: {name}\n")
        log_file.write(f"# notes: {preset['notes']}\n")
        log_file.write(f"# command: {' '.join(command)}\n")
        log_file.write(f"# env_overrides: {format_env_delta({k: env[k] for k in sorted(set(preset['env'].keys()) | (set(extra_env.keys()) if extra_env else set()) | ({'USIM_STATIC'} if force_static else set()))})}\n\n")
        proc = subprocess.Popen(
            command,
            cwd=str(root_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            output_lines.append(line)
        return_code = proc.wait()

    elapsed = time.time() - start
    output_text = "".join(output_lines)
    metrics = parse_final_metrics(output_text)
    status = "ok" if return_code == 0 else f"failed({return_code})"
    return {
        "name": name,
        "script": preset["script"],
        "status": status,
        "return_code": return_code,
        "seconds": round(elapsed, 2),
        "log_path": str(log_path),
        "notes": preset["notes"],
        "env": {k: env[k] for k in sorted(set(preset["env"].keys()) | ({"USIM_STATIC"} if force_static else set()) | (set(extra_env.keys()) if extra_env else set()))},
        "metrics": metrics,
    }


def write_summary(results, out_dir):
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "name",
        "script",
        "status",
        "seconds",
        "log_path",
        "notes",
        "R@10_sample_cold",
        "R@10_sample_hot",
        "R@10_full_cold",
        "R@10_full_hot",
        "N@10_sample_cold",
        "N@10_sample_hot",
        "N@10_full_cold",
        "N@10_full_hot",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "name": result.get("name"),
                "script": result.get("script"),
                "status": result.get("status"),
                "seconds": result.get("seconds"),
                "log_path": result.get("log_path"),
                "notes": result.get("notes"),
            }
            metrics = result.get("metrics", {})
            for metric_name in ("R@10", "N@10"):
                metric_values = metrics.get(metric_name, {})
                for metric_col in METRIC_COLUMNS:
                    row[f"{metric_name}_{metric_col}"] = metric_values.get(metric_col)
            writer.writerow(row)
    return json_path, csv_path


def parse_extra_env(pairs):
    env = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Bad --env override: {pair!r}, expected KEY=VALUE")
        key, value = pair.split("=", 1)
        env[key] = value
    return env


def main():
    parser = argparse.ArgumentParser(description="Run the most useful USIM course ablations.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit.")
    parser.add_argument("--only", nargs="+", help="Run only the specified preset names.")
    parser.add_argument("--static", action="store_true", help="Force USIM_STATIC=1 for all runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write no experiment output.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch experiments.")
    parser.add_argument("--log-dir", default=None, help="Directory for logs and summaries.")
    parser.add_argument("--env", action="append", default=[], help="Extra env override in KEY=VALUE form.")
    args = parser.parse_args()

    if args.list:
        for name, preset in PRESETS.items():
            print(f"{name:<32} -> {preset['script']}")
            print(f"  notes: {preset['notes']}")
            print(f"  env:   {format_env_delta(preset['env'])}")
        return

    selected_names = args.only if args.only else list(PRESETS.keys())
    unknown = [name for name in selected_names if name not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown preset(s): {', '.join(unknown)}")

    extra_env = parse_extra_env(args.env)
    root_dir = Path(__file__).resolve().parent
    if args.log_dir:
        out_dir = Path(args.log_dir).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = root_dir / "ablation_logs" / f"course_ablation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in selected_names:
        preset = PRESETS[name]
        print("=" * 90)
        print(f"[RUN] {name}")
        print(f"      script: {preset['script']}")
        print(f"      notes : {preset['notes']}")
        env_delta = dict(preset["env"])
        if args.static:
            env_delta["USIM_STATIC"] = "1"
        env_delta.update(extra_env)
        print(f"      env   : {format_env_delta(env_delta)}")
        result = run_preset(
            name=name,
            preset=preset,
            root_dir=root_dir,
            python_exe=args.python,
            log_dir=out_dir,
            force_static=args.static,
            extra_env=extra_env,
            dry_run=args.dry_run,
        )
        results.append(result)
        if result["status"] == "dry_run":
            print(f"[DRY-RUN] {name} | log={result['log_path']}")
            continue
        if result["status"].startswith("failed"):
            print(f"[FAIL] {name} -> {result['status']} | log={result['log_path']}")
        else:
            metric_summary = result.get("metrics", {}).get("R@10", {})
            if metric_summary:
                print(
                    "[DONE] "
                    f"{name} | R@10 sample cold={metric_summary.get('sample_cold', float('nan')):.4f} "
                    f"hot={metric_summary.get('sample_hot', float('nan')):.4f} | "
                    f"full cold={metric_summary.get('full_cold', float('nan')):.4f} "
                    f"hot={metric_summary.get('full_hot', float('nan')):.4f}"
                )
            else:
                print(f"[DONE] {name} | log={result['log_path']}")

    json_path, csv_path = write_summary(results, out_dir)
    print("=" * 90)
    print(f"Logs saved to: {out_dir}")
    print(f"Summary JSON : {json_path}")
    print(f"Summary CSV  : {csv_path}")


if __name__ == "__main__":
    main()
