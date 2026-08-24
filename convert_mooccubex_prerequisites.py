"""Convert MOOCCubeX prerequisite JSONL files to FAST3 relation format.

The original MOOCCubeX files store prerequisite pairs as raw concept names:
```
{"c1": "...", "c2": "...", "ground_truth": 1, ...}
```
Current course-aware artifacts expect concept ids already used by
``relations/course-concept.json`` and read tab-separated pairs as:
```
prerequisite_concept_id<TAB>target_concept_id
```

This converter keeps only human-positive pairs by default and maps each domain
file to the matching MOOCCubeX concept-id namespace:
cs -> Computer Science, math -> Mathematics, psy -> Psychology.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def _u(hex_codes: str) -> str:
    return "".join(chr(int(code, 16)) for code in hex_codes.split())


DOMAIN_BY_FILE = {
    "cs": _u("8ba1 7b97 673a 79d1 5b66 4e0e 6280 672f"),
    "math": _u("6570 5b66"),
    "psy": _u("5fc3 7406 5b66"),
}


def iter_json_records(path: Path) -> Iterable[dict]:
    """Read either JSONL or a JSON array without loading JSONL all at once."""
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is JSON but not a list")
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
            if isinstance(obj, dict):
                yield obj


def load_course_concepts(path: Path) -> set[str]:
    concept_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                concept_ids.add(parts[1])
    return concept_ids


def concept_id(raw_name: str, domain: str) -> str:
    return f"K_{raw_name}_{domain}"


def convert(base_dir: Path, output_path: Path, include_predicted: bool, pred_thr: float) -> dict:
    prereq_dir = base_dir / "prerequisites"
    relation_dir = base_dir / "relations"
    course_concept_path = relation_dir / "course-concept.json"
    course_concepts = load_course_concepts(course_concept_path)

    output_pairs: set[tuple[str, str]] = set()
    per_file: dict[str, dict] = {}

    for short, domain in DOMAIN_BY_FILE.items():
        path = prereq_dir / f"{short}.json"
        stats = Counter()
        missing_labels: Counter[str] = Counter()
        labels: set[str] = set()
        domain_labels: set[str] = set()

        for obj in iter_json_records(path):
            stats["total"] += 1
            gt = obj.get("ground_truth")
            stats[f"ground_truth:{gt}"] += 1

            use_pair = gt == 1
            if include_predicted and gt == -1:
                text_pred = obj.get("text_predict")
                graph_pred = obj.get("graph_predict")
                text_score = text_pred[1] if isinstance(text_pred, list) and len(text_pred) > 1 else 0.0
                graph_score = graph_pred[1] if isinstance(graph_pred, list) and len(graph_pred) > 1 else 0.0
                use_pair = max(float(text_score), float(graph_score)) >= pred_thr

            if not use_pair:
                continue

            c1 = str(obj.get("c1") or "").strip()
            c2 = str(obj.get("c2") or "").strip()
            if not c1 or not c2 or c1 == c2:
                stats["skipped_empty_or_self"] += 1
                continue

            stats["candidate_pairs"] += 1
            labels.update((c1, c2))
            prereq_id = concept_id(c1, domain)
            target_id = concept_id(c2, domain)
            if prereq_id not in course_concepts:
                missing_labels[c1] += 1
            else:
                domain_labels.add(c1)
            if target_id not in course_concepts:
                missing_labels[c2] += 1
            else:
                domain_labels.add(c2)
            if prereq_id in course_concepts and target_id in course_concepts:
                output_pairs.add((prereq_id, target_id))
                stats["converted_pairs"] += 1

        per_file[short] = {
            "domain": domain,
            "stats": dict(stats),
            "labels": len(labels),
            "domain_label_coverage": len(domain_labels),
            "missing_label_examples": [name for name, _ in missing_labels.most_common(10)],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for prereq_id, target_id in sorted(output_pairs):
            f.write(f"{prereq_id}\t{target_id}\n")

    return {
        "output_path": str(output_path),
        "output_pairs": len(output_pairs),
        "include_predicted": include_predicted,
        "pred_thr": pred_thr,
        "per_file": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="MOOCCubeX", type=Path)
    parser.add_argument(
        "--output",
        default=Path("MOOCCubeX") / "relations" / "prerequisite-dependency.json",
        type=Path,
    )
    parser.add_argument(
        "--include-predicted",
        action="store_true",
        help="Also include unlabeled pairs when text or graph confidence reaches --pred-thr.",
    )
    parser.add_argument("--pred-thr", default=0.95, type=float)
    args = parser.parse_args()

    summary = convert(args.base_dir, args.output, args.include_predicted, args.pred_thr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
