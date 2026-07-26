"""Build metadata-only pseudo prerequisites for MOOCCourse.

The generated graph intentionally uses only course metadata: course title,
coarse type label, and type_id. It does not read interaction rows, so the
result can be used with strict item-cold splits without test-behavior leakage.

The output is a separate relation bundle. Point experiments at it with:

  USIM_RELATION_DIR=processed_data_mooccourse/relations_metadata_prereq
  USIM_PREREQ_GRAPH_SOURCE=concept

Implementation detail: FAST3's concept-prerequisite path consumes concept-to-
concept prerequisite pairs. To express course-level pseudo edges without
changing the training code, this script adds one unique self concept per course
and writes prerequisite pairs between those self concepts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


FOUNDATION_KEYWORDS = (
    "intro",
    "introduction",
    "overview",
    "basic",
    "basics",
    "foundation",
    "fundamental",
    "fundamentals",
    "principle",
    "principles",
    "\u5bfc\u8bba",  # daolun / introduction
    "\u6982\u8bba",
    "\u57fa\u7840",
    "\u5165\u95e8",
    "\u539f\u7406",
    "\u521d\u7ea7",
    "\u666e\u901a",
    "\u901a\u8bc6",
    "\u7eea\u8bba",
)

ADVANCED_KEYWORDS = (
    "advanced",
    "practice",
    "practical",
    "project",
    "design",
    "system",
    "application",
    "case",
    "experiment",
    "seminar",
    "\u8fdb\u9636",
    "\u9ad8\u7ea7",
    "\u5b9e\u8df5",
    "\u5b9e\u9a8c",
    "\u5e94\u7528",
    "\u8bbe\u8ba1",
    "\u4e13\u9898",
    "\u7efc\u5408",
    "\u6848\u4f8b",
    "\u5de5\u7a0b",
    "\u7cfb\u7edf",
    "\u7814\u7a76",
    "\u5206\u6790",
    "\u9879\u76ee",
)

SEQUENCE_PATTERNS = (
    (re.compile(r"[\(\uff08]\s*(?:ii|2|two)\s*[\)\uff09]\s*$", re.I), 2),
    (re.compile(r"[\(\uff08]\s*(?:i|1|one)\s*[\)\uff09]\s*$", re.I), 1),
    (re.compile(r"[\(\uff08]\s*\u4e0b\s*[\)\uff09]\s*$"), 2),
    (re.compile(r"[\(\uff08]\s*\u4e0a\s*[\)\uff09]\s*$"), 1),
    (re.compile(r"(?:\s|^)(?:ii|2|two)\s*$", re.I), 2),
    (re.compile(r"(?:\s|^)(?:i|1|one)\s*$", re.I), 1),
)


@dataclass(frozen=True)
class CourseRecord:
    course_id: str
    name: str
    type_id: str
    type_label: str


@dataclass(frozen=True)
class PseudoPrereqEdge:
    source_id: str
    target_id: str
    rule: str
    confidence: float
    source_name: str
    target_name: str
    type_id: str
    type_label: str

    @property
    def source_concept(self) -> str:
        return course_self_concept(self.source_id)

    @property
    def target_concept(self) -> str:
        return course_self_concept(self.target_id)


def course_self_concept(course_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.:-]+", "_", str(course_id).strip())
    return f"MOOCCOURSE_COURSE_{safe}"


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normal_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"category\s+.*$", "", text)
    text = re.sub(r"type_id\s+\S+.*$", "", text)
    text = re.sub(r"[\s\t\r\n\-_:/\\,.;'\"`~!@#$%^&*+=|?<>\[\]{}]+", "", text)
    text = text.replace("(", "").replace(")", "")
    text = text.replace("\uff08", "").replace("\uff09", "")
    return text


def _word_tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = {w for w in re.findall(r"[a-z0-9]+", lowered) if len(w) >= 2}
    compact = _normal_for_match(text)
    if len(compact) >= 2:
        words.update(compact[i : i + 2] for i in range(len(compact) - 1))
    return words


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    return inter / float(len(a | b))


def _keyword_score(name: str, keywords: tuple[str, ...]) -> int:
    lowered = name.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _sequence_key(name: str) -> tuple[str, int] | None:
    for pattern, order in SEQUENCE_PATTERNS:
        if pattern.search(name):
            base = pattern.sub("", name).strip()
            base = _normal_for_match(base)
            if base:
                return base, order
    return None


def _same_type(a: CourseRecord, b: CourseRecord) -> bool:
    if a.type_id and b.type_id:
        return a.type_id == b.type_id
    return bool(a.type_label and a.type_label == b.type_label)


def _edge_sort_key(edge: PseudoPrereqEdge) -> tuple[float, str, str]:
    return (-edge.confidence, edge.source_id, edge.target_id)


def _make_edge(
    source: CourseRecord,
    target: CourseRecord,
    rule: str,
    confidence: float,
) -> PseudoPrereqEdge:
    return PseudoPrereqEdge(
        source_id=source.course_id,
        target_id=target.course_id,
        rule=rule,
        confidence=round(float(confidence), 6),
        source_name=source.name,
        target_name=target.name,
        type_id=target.type_id,
        type_label=target.type_label,
    )


def build_metadata_prereq_edges(
    records: list[CourseRecord],
    max_per_target: int = 5,
    min_confidence: float = 0.50,
    allow_type_foundation: bool = True,
) -> list[PseudoPrereqEdge]:
    """Infer conservative pseudo prerequisites from metadata only."""
    by_target: dict[str, list[PseudoPrereqEdge]] = defaultdict(list)
    record_by_id = {record.course_id: record for record in records}

    sequence_groups: dict[tuple[str, str], list[tuple[int, CourseRecord]]] = defaultdict(list)
    for record in records:
        seq = _sequence_key(record.name)
        if seq is None:
            continue
        base, order = seq
        sequence_groups[(record.type_id or record.type_label, base)].append((order, record))
    for group in sequence_groups.values():
        ordered = sorted(group, key=lambda x: (x[0], x[1].course_id))
        for idx in range(len(ordered) - 1):
            src_order, source = ordered[idx]
            dst_order, target = ordered[idx + 1]
            if dst_order > src_order and source.course_id != target.course_id:
                by_target[target.course_id].append(_make_edge(source, target, "sequence", 0.98))

    tokens = {record.course_id: _word_tokens(record.name) for record in records}
    foundation = {record.course_id: _keyword_score(record.name, FOUNDATION_KEYWORDS) for record in records}
    advanced = {record.course_id: _keyword_score(record.name, ADVANCED_KEYWORDS) for record in records}
    by_type: dict[str, list[CourseRecord]] = defaultdict(list)
    for record in records:
        by_type[record.type_id or record.type_label or "UNKNOWN"].append(record)

    for target in records:
        target_foundation = foundation[target.course_id]
        target_advanced = advanced[target.course_id]
        target_tokens = tokens[target.course_id]
        candidates = []
        for source in by_type[target.type_id or target.type_label or "UNKNOWN"]:
            if source.course_id == target.course_id or not _same_type(source, target):
                continue
            source_foundation = foundation[source.course_id]
            if source_foundation <= 0:
                continue
            similarity = _jaccard(tokens[source.course_id], target_tokens)
            source_is_more_basic = source_foundation > target_foundation
            lexical_signal = similarity >= 0.14
            target_is_specialized = target_advanced > 0
            if not (source_is_more_basic and (target_is_specialized or lexical_signal)):
                continue
            if not allow_type_foundation and not lexical_signal:
                continue
            confidence = 0.44
            confidence += min(source_foundation, 3) * 0.10
            confidence += min(target_advanced, 3) * 0.08
            confidence += min(similarity, 0.5) * 0.28
            if source_is_more_basic:
                confidence += 0.10
            if target.type_id and source.type_id == target.type_id:
                confidence += 0.04
            rule = "lexical_foundation" if lexical_signal else "type_foundation"
            if confidence >= min_confidence:
                candidates.append(_make_edge(source, target, rule, min(confidence, 0.90)))
        by_target[target.course_id].extend(candidates)

    deduped: list[PseudoPrereqEdge] = []
    for target_id, edges in by_target.items():
        best_by_pair: dict[tuple[str, str], PseudoPrereqEdge] = {}
        for edge in edges:
            if edge.source_id == edge.target_id:
                continue
            key = (edge.source_id, edge.target_id)
            prev = best_by_pair.get(key)
            if prev is None or edge.confidence > prev.confidence:
                best_by_pair[key] = edge
        capped = sorted(best_by_pair.values(), key=_edge_sort_key)[: max(1, int(max_per_target))]
        deduped.extend(capped)

    deduped.sort(key=lambda e: (e.target_id, -e.confidence, e.source_id))
    return [edge for edge in deduped if edge.source_id in record_by_id and edge.target_id in record_by_id]


def _parse_entity_name(raw_name: str) -> tuple[str, str]:
    title = raw_name
    type_label = ""
    if " category " in raw_name:
        title, rest = raw_name.split(" category ", 1)
        if " type_id " in rest:
            type_label = rest.split(" type_id ", 1)[0].strip()
        else:
            type_label = rest.strip()
    return title.strip(), type_label.strip()


def load_course_records(processed_dir: Path) -> list[CourseRecord]:
    entity_path = processed_dir / "entities" / "course.json"
    if not entity_path.exists():
        raise FileNotFoundError(f"Missing course entity file: {entity_path}")
    records = []
    with entity_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            course_id = _clean_text(obj.get("id"))
            raw_name = _clean_text(obj.get("name") or obj.get("about") or course_id)
            title, type_label = _parse_entity_name(raw_name)
            type_id = _clean_text(obj.get("core_id"))
            if not type_id:
                match = re.search(r"type_id\s+(\S+)", raw_name)
                type_id = match.group(1) if match else ""
            records.append(CourseRecord(course_id, title, type_id, type_label))
    records.sort(key=lambda r: r.course_id)
    return records


def write_relation_bundle(
    records: list[CourseRecord],
    edges: list[PseudoPrereqEdge],
    source_relation_dir: Path,
    output_relation_dir: Path,
) -> None:
    output_relation_dir.mkdir(parents=True, exist_ok=True)
    source_concept_path = source_relation_dir / "course-concept.json"
    output_concept_path = output_relation_dir / "course-concept.json"
    existing_lines = []
    if source_concept_path.exists():
        existing_lines = [
            line.rstrip("\n")
            for line in source_concept_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    concept_lines = list(dict.fromkeys(existing_lines + [f"{r.course_id}\t{course_self_concept(r.course_id)}" for r in records]))
    output_concept_path.write_text("\n".join(concept_lines) + "\n", encoding="utf-8")

    prereq_lines = [f"{edge.source_concept}\t{edge.target_concept}" for edge in edges]
    (output_relation_dir / "prerequisite-dependency.json").write_text(
        "\n".join(dict.fromkeys(prereq_lines)) + ("\n" if prereq_lines else ""),
        encoding="utf-8",
    )

    for extra_name in ("README_processed.txt",):
        src = source_relation_dir / extra_name
        if src.exists():
            shutil.copy2(src, output_relation_dir / extra_name)


def write_edge_table(edges: list[PseudoPrereqEdge], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "target_id",
                "source_name",
                "target_name",
                "type_id",
                "type_label",
                "rule",
                "confidence",
                "source_concept",
                "target_concept",
            ],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "source_name": edge.source_name,
                    "target_name": edge.target_name,
                    "type_id": edge.type_id,
                    "type_label": edge.type_label,
                    "rule": edge.rule,
                    "confidence": f"{edge.confidence:.6f}",
                    "source_concept": edge.source_concept,
                    "target_concept": edge.target_concept,
                }
            )


def _load_item_id_map(item_map_path: Path) -> dict[str, int]:
    mapping = {}
    if not item_map_path.exists():
        return mapping
    with item_map_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[str(row["course_id"])] = int(row["i_idx"])
    return mapping


def _split_coverage(edge_targets: set[str], item_id_map: dict[str, int], split_dir: Path) -> dict[str, object]:
    assignment_path = split_dir / "static_split_assignments.csv"
    if not assignment_path.exists():
        return {}
    train_items = set()
    test_items = set()
    with assignment_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item = int(row["i_idx"])
            split = row["split"]
            if split == "train":
                train_items.add(item)
            elif split == "test":
                test_items.add(item)
    target_iidx = {item_id_map[cid] for cid in edge_targets if cid in item_id_map}
    cold_test = test_items - train_items
    covered = cold_test & target_iidx
    return {
        "split": split_dir.name,
        "cold_test_items": len(cold_test),
        "cold_test_items_with_pseudo_prereq": len(covered),
        "cold_test_pseudo_prereq_coverage": round(len(covered) / max(1, len(cold_test)), 6),
    }


def build_audit(
    records: list[CourseRecord],
    edges: list[PseudoPrereqEdge],
    item_map_path: Path,
    split_root: Path | None,
    max_per_target: int,
    min_confidence: float,
    allow_type_foundation: bool,
) -> dict[str, object]:
    rule_counts = Counter(edge.rule for edge in edges)
    target_counts = Counter(edge.target_id for edge in edges)
    type_counts = Counter(edge.type_id for edge in edges)
    edge_targets = {edge.target_id for edge in edges}
    coverage = []
    if split_root and split_root.exists():
        item_id_map = _load_item_id_map(item_map_path)
        for split_dir in sorted(split_root.glob("strict_item_cold_balanced_thr1_seed_*")):
            cov = _split_coverage(edge_targets, item_id_map, split_dir)
            if cov:
                coverage.append(cov)
    return {
        "dataset": "MOOCCourse",
        "source": "metadata_only",
        "uses_interactions": False,
        "course_count": len(records),
        "edge_count": len(edges),
        "target_course_count": len(target_counts),
        "target_course_coverage": round(len(target_counts) / max(1, len(records)), 6),
        "max_per_target": int(max_per_target),
        "min_confidence": float(min_confidence),
        "allow_type_foundation": bool(allow_type_foundation),
        "rule_counts": dict(sorted(rule_counts.items())),
        "top_type_edge_counts": dict(type_counts.most_common(20)),
        "static_cold_coverage": coverage,
        "notes": [
            "Pseudo prerequisites are inferred from course title/type metadata only.",
            "Course-level edges are encoded as prerequisite pairs between unique course self concepts.",
            "This file is suitable for ablation as metadata-only pseudo structure, not as official prerequisites.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="processed_data_mooccourse")
    parser.add_argument("--source-relation-dir", default="")
    parser.add_argument("--output-relation-dir", default="")
    parser.add_argument("--max-per-target", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--split-root", default="outputs/mooccourse/course_ablation_e60_3seed/full")
    parser.add_argument(
        "--disable-type-foundation",
        action="store_true",
        help="Keep only sequence and lexical-foundation edges; drops broad same-type foundation edges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    source_relation_dir = Path(args.source_relation_dir) if args.source_relation_dir else processed_dir / "relations"
    output_relation_dir = Path(args.output_relation_dir) if args.output_relation_dir else processed_dir / "relations_metadata_prereq"

    records = load_course_records(processed_dir)
    edges = build_metadata_prereq_edges(
        records,
        max_per_target=args.max_per_target,
        min_confidence=args.min_confidence,
        allow_type_foundation=not args.disable_type_foundation,
    )
    write_relation_bundle(records, edges, source_relation_dir, output_relation_dir)
    write_edge_table(edges, output_relation_dir / "pseudo_prereq_edges.csv")
    audit = build_audit(
        records,
        edges,
        item_map_path=processed_dir / "_item_id_map.csv",
        split_root=Path(args.split_root) if args.split_root else None,
        max_per_target=args.max_per_target,
        min_confidence=args.min_confidence,
        allow_type_foundation=not args.disable_type_foundation,
    )
    (output_relation_dir / "pseudo_prereq_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Wrote metadata-only pseudo prerequisite relations to {output_relation_dir}")


if __name__ == "__main__":
    main()
