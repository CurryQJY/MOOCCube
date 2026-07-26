from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


Pair = tuple[str, str]
Triple = tuple[int, int, int]
RawTriple = tuple[str, str, str]

MOOCCUBE_RELATION_FILES = {
    "course-video": ("course_video", "course_external", "video"),
    "course-concept": ("course_concept", "course_external", "concept"),
    "teacher-course": ("course_teacher", "external_course", "teacher"),
    "school-course": ("course_school", "external_course", "school"),
}


@dataclass(frozen=True)
class AtomicKGRecData:
    user_to_id: dict[str, int]
    course_to_item: dict[str, int]
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    train_pairs: list[tuple[int, int]]
    validation_pairs: list[tuple[int, int]]
    test_pairs: list[tuple[int, int]]
    warm_item_ids: set[int]
    cold_item_ids: set[int]
    kg_triples: list[Triple]
    course_kg_degree: dict[int, int]

    @property
    def n_users(self) -> int:
        return len(self.user_to_id)

    @property
    def n_items(self) -> int:
        return len(self.course_to_item)

    @property
    def n_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def n_relations(self) -> int:
        return len(self.relation_to_id)


def _ordered_tokens(tokens: Iterable[str]) -> list[str]:
    return sorted({str(token) for token in tokens})


def normalize_mooccube_course_side_edges(
    relation_rows: Mapping[str, Sequence[Pair]],
    *,
    course_ids: Iterable[str],
) -> dict[str, list[Pair]]:
    course_set = {str(course) for course in course_ids}
    normalized: dict[str, list[Pair]] = {}
    for raw_relation, (relation_name, direction, entity_prefix) in MOOCCUBE_RELATION_FILES.items():
        rows = relation_rows.get(raw_relation, ())
        edges: list[Pair] = []
        for left, right in rows:
            if direction == "course_external":
                course, neighbor = str(left), f"{entity_prefix}:{right}"
            elif direction == "external_course":
                course, neighbor = str(right), f"{entity_prefix}:{left}"
            else:
                raise ValueError(f"Unsupported MOOCCube relation direction: {direction}")
            if course in course_set:
                edges.append((course, neighbor))
        if edges:
            normalized[relation_name] = edges
    return normalized


def _map_pairs(pairs: Sequence[Pair], user_to_id: Mapping[str, int], course_to_item: Mapping[str, int]) -> list[tuple[int, int]]:
    return [(int(user_to_id[user]), int(course_to_item[course])) for user, course in pairs]


def build_atomic_data(
    *,
    train_pairs: Sequence[Pair],
    validation_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    course_side_edges: Mapping[str, Sequence[Pair]],
) -> AtomicKGRecData:
    users = _ordered_tokens(user for pairs in (train_pairs, validation_pairs, test_pairs) for user, _ in pairs)
    courses = set(course for pairs in (train_pairs, validation_pairs, test_pairs) for _, course in pairs)
    for edges in course_side_edges.values():
        for course, _neighbor in edges:
            courses.add(str(course))

    user_to_id = {user: idx for idx, user in enumerate(users)}
    course_to_item = {course: idx for idx, course in enumerate(_ordered_tokens(courses))}
    n_items = len(course_to_item)

    external_nodes = _ordered_tokens(
        f"entity:{neighbor}" for edges in course_side_edges.values() for _course, neighbor in edges
    )
    entity_to_id = dict(course_to_item)
    entity_to_id.update({node: n_items + idx for idx, node in enumerate(external_nodes)})
    relation_to_id = {relation: idx for idx, relation in enumerate(_ordered_tokens(course_side_edges))}

    mapped_train = _map_pairs(train_pairs, user_to_id, course_to_item)
    mapped_validation = _map_pairs(validation_pairs, user_to_id, course_to_item)
    mapped_test = _map_pairs(test_pairs, user_to_id, course_to_item)

    warm_item_ids = {item for _user, item in mapped_train}
    eval_item_ids = {item for _user, item in mapped_validation + mapped_test}
    cold_item_ids = eval_item_ids - warm_item_ids

    triples: list[Triple] = []
    degree_counter: Counter[int] = Counter()
    for relation in sorted(course_side_edges):
        relation_id = relation_to_id[relation]
        for course, neighbor in course_side_edges[relation]:
            if course not in course_to_item:
                continue
            head = course_to_item[course]
            tail = entity_to_id[f"entity:{neighbor}"]
            triples.append((head, relation_id, tail))
            degree_counter[head] += 1

    return AtomicKGRecData(
        user_to_id=user_to_id,
        course_to_item=course_to_item,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        train_pairs=mapped_train,
        validation_pairs=mapped_validation,
        test_pairs=mapped_test,
        warm_item_ids=warm_item_ids,
        cold_item_ids=cold_item_ids,
        kg_triples=triples,
        course_kg_degree=dict(degree_counter),
    )


def build_atomic_data_from_kg_triples(
    *,
    train_pairs: Sequence[Pair],
    validation_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    kg_triples: Sequence[RawTriple],
) -> AtomicKGRecData:
    users = _ordered_tokens(user for pairs in (train_pairs, validation_pairs, test_pairs) for user, _ in pairs)
    courses = _ordered_tokens(course for pairs in (train_pairs, validation_pairs, test_pairs) for _, course in pairs)
    user_to_id = {user: idx for idx, user in enumerate(users)}
    course_to_item = {course: idx for idx, course in enumerate(courses)}
    n_items = len(course_to_item)

    kg_entities = _ordered_tokens(entity for head, _relation, tail in kg_triples for entity in (head, tail))
    external_entities = [entity for entity in kg_entities if entity not in course_to_item]
    entity_to_id = dict(course_to_item)
    entity_to_id.update({entity: n_items + idx for idx, entity in enumerate(external_entities)})
    relation_to_id = {
        relation: idx for idx, relation in enumerate(_ordered_tokens(relation for _head, relation, _tail in kg_triples))
    }

    mapped_train = _map_pairs(train_pairs, user_to_id, course_to_item)
    mapped_validation = _map_pairs(validation_pairs, user_to_id, course_to_item)
    mapped_test = _map_pairs(test_pairs, user_to_id, course_to_item)
    warm_item_ids = {item for _user, item in mapped_train}
    eval_item_ids = {item for _user, item in mapped_validation + mapped_test}
    cold_item_ids = eval_item_ids - warm_item_ids

    mapped_triples: list[Triple] = []
    degree_counter: Counter[int] = Counter()
    for head, relation, tail in kg_triples:
        head_id = entity_to_id[str(head)]
        tail_id = entity_to_id[str(tail)]
        mapped_triples.append((head_id, relation_to_id[str(relation)], tail_id))
        if head_id < n_items:
            degree_counter[head_id] += 1
        if tail_id < n_items:
            degree_counter[tail_id] += 1

    return AtomicKGRecData(
        user_to_id=user_to_id,
        course_to_item=course_to_item,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        train_pairs=mapped_train,
        validation_pairs=mapped_validation,
        test_pairs=mapped_test,
        warm_item_ids=warm_item_ids,
        cold_item_ids=cold_item_ids,
        kg_triples=mapped_triples,
        course_kg_degree=dict(degree_counter),
    )


def _group_items_by_user(pairs: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for user, item in pairs:
        grouped[int(user)].append(int(item))
    return {user: sorted(set(items)) for user, items in sorted(grouped.items())}


def _write_grouped_pairs(path: Path, pairs: Sequence[tuple[int, int]]) -> None:
    lines = []
    for user, items in _group_items_by_user(pairs).items():
        if items:
            lines.append(" ".join([str(user), *(str(item) for item in items)]))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_mapping(path: Path, mapping: Mapping[str, int]) -> None:
    rows = [f"{idx}\t{token}" for token, idx in sorted(mapping.items(), key=lambda item: item[1])]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _strict_manifest(atomic: AtomicKGRecData) -> dict[str, object]:
    train_items = {item for _user, item in atomic.train_pairs}
    entity_ids = sorted(atomic.entity_to_id.values())
    relation_ids = sorted(atomic.relation_to_id.values())
    cold_items_with_kg_edges = sorted(item for item in atomic.cold_item_ids if atomic.course_kg_degree.get(item, 0) > 0)
    return {
        "format": "KGRec atomic strict item-cold",
        "n_users": atomic.n_users,
        "n_items": atomic.n_items,
        "n_entities": atomic.n_entities,
        "n_relations": atomic.n_relations,
        "n_train_pairs": len(atomic.train_pairs),
        "n_validation_pairs": len(atomic.validation_pairs),
        "n_test_pairs": len(atomic.test_pairs),
        "n_kg_triples": len(atomic.kg_triples),
        "warm_item_ids": sorted(atomic.warm_item_ids),
        "cold_item_ids": sorted(atomic.cold_item_ids),
        "strict_checks": {
            "cold_items_absent_from_train": atomic.cold_item_ids.isdisjoint(train_items),
            "item_ids_contiguous": sorted(atomic.course_to_item.values()) == list(range(atomic.n_items)),
            "kg_entity_ids_contiguous": entity_ids == list(range(atomic.n_entities)),
            "relation_ids_contiguous": relation_ids == list(range(atomic.n_relations)),
            "cold_items_with_kg_edges": cold_items_with_kg_edges,
            "all_cold_items_have_kg_edges": len(cold_items_with_kg_edges) == len(atomic.cold_item_ids),
        },
        "course_kg_degree": {str(item): degree for item, degree in sorted(atomic.course_kg_degree.items())},
        "relation_to_id": dict(sorted(atomic.relation_to_id.items(), key=lambda item: item[1])),
    }


def write_kgrec_atomic_dataset(output_dir: str | Path, atomic: AtomicKGRecData) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_grouped_pairs(output_path / "train.txt", atomic.train_pairs)
    _write_grouped_pairs(output_path / "validation.txt", atomic.validation_pairs)
    _write_grouped_pairs(output_path / "test.txt", atomic.test_pairs)
    (output_path / "kg_final.txt").write_text(
        "\n".join(f"{head} {relation} {tail}" for head, relation, tail in atomic.kg_triples)
        + ("\n" if atomic.kg_triples else ""),
        encoding="utf-8",
    )
    _write_mapping(output_path / "user_list.txt", atomic.user_to_id)
    _write_mapping(output_path / "item_list.txt", atomic.course_to_item)
    _write_mapping(output_path / "entity_list.txt", atomic.entity_to_id)
    _write_mapping(output_path / "relation_list.txt", atomic.relation_to_id)

    manifest = _strict_manifest(atomic)
    (output_path / "strict_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _pairs_from_frame(frame) -> list[Pair]:
    required = {"user_id", "course_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split frame is missing columns: {sorted(missing)}")
    return [(str(row.user_id), str(row.course_id)) for row in frame[["user_id", "course_id"]].itertuples(index=False)]


def _read_split_pairs(split_root: Path) -> tuple[list[Pair], list[Pair], list[Pair]]:
    import pandas as pd

    train_path = split_root / "static_train.pkl"
    validation_path = split_root / "static_val.pkl"
    test_path = split_root / "static_test.pkl"
    for path in (train_path, validation_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return (
        _pairs_from_frame(pd.read_pickle(train_path)),
        _pairs_from_frame(pd.read_pickle(validation_path)),
        _pairs_from_frame(pd.read_pickle(test_path)),
    )


def _read_tsv_pairs(path: Path) -> list[Pair]:
    if not path.exists():
        return []
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


def _read_mooccube_relation_rows(relations_dir: Path) -> dict[str, list[Pair]]:
    return {
        raw_relation: _read_tsv_pairs(relations_dir / f"{raw_relation}.json")
        for raw_relation in MOOCCUBE_RELATION_FILES
    }


def _read_recbole_tsv(path: Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"RecBole TSV is empty: {path}") from exc
        header = [column.split(":", 1)[0] for column in raw_header]
        missing = set(required_columns) - set(header)
        if missing:
            raise ValueError(f"RecBole TSV {path} is missing columns: {sorted(missing)}")
        indices = {column: header.index(column) for column in required_columns}
        rows: list[dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if not values or not any(value.strip() for value in values):
                continue
            if len(values) != len(header):
                raise ValueError(f"RecBole TSV {path} line {line_number} has {len(values)} fields; expected {len(header)}")
            rows.append({column: values[index] for column, index in indices.items()})
    return rows


def export_recbole_kgrec_dataset(
    *,
    split_root: str | Path,
    link_path: str | Path,
    kg_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    split_path = Path(split_root)
    link_file = Path(link_path)
    kg_file = Path(kg_path)
    output_path = Path(output_dir)

    train_pairs, validation_pairs, test_pairs = _read_split_pairs(split_path)
    course_ids = {course for pairs in (train_pairs, validation_pairs, test_pairs) for _user, course in pairs}

    link_rows = _read_recbole_tsv(link_file, ("item_id", "entity_id"))
    item_to_entity: dict[str, str] = {}
    entity_to_item: dict[str, str] = {}
    for row in link_rows:
        item = str(row["item_id"])
        entity = str(row["entity_id"])
        if item in item_to_entity and item_to_entity[item] != entity:
            raise ValueError(f"RecBole link item {item!r} maps to multiple entities")
        if entity in entity_to_item and entity_to_item[entity] != item:
            raise ValueError(f"RecBole link entity {entity!r} maps to multiple items")
        item_to_entity[item] = entity
        entity_to_item[entity] = item

    missing_courses = sorted(course_ids - set(entity_to_item))
    if missing_courses:
        preview = ", ".join(missing_courses[:10])
        raise ValueError(f"Split courses missing from RecBole link entities: {preview}")

    kg_rows = _read_recbole_tsv(kg_file, ("head_id", "relation_id", "tail_id"))
    raw_triples: list[RawTriple] = [
        (str(row["head_id"]), str(row["relation_id"]), str(row["tail_id"]))
        for row in kg_rows
    ]
    atomic = build_atomic_data_from_kg_triples(
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        kg_triples=raw_triples,
    )
    missing_cold_items = sorted(item for item in atomic.cold_item_ids if atomic.course_kg_degree.get(item, 0) == 0)
    if missing_cold_items:
        missing_tokens = [
            course for course, item in sorted(atomic.course_to_item.items(), key=lambda entry: entry[1])
            if item in set(missing_cold_items)
        ]
        raise ValueError(f"Cold courses missing KG edges: {', '.join(missing_tokens[:10])}")

    manifest = write_kgrec_atomic_dataset(output_path, atomic)
    relation_counts = Counter(relation for _head, relation, _tail in raw_triples)
    manifest["source"] = {
        "split_root": str(split_path),
        "link_path": str(link_file),
        "kg_path": str(kg_file),
        "kg_scope": "full_arbitrary_entity_graph",
        "included_relations": sorted(relation_counts),
    }
    manifest["relation_edge_counts"] = {
        relation: int(relation_counts[relation]) for relation in sorted(relation_counts)
    }
    (output_path / "strict_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def export_mooccube_kgrec_dataset(
    *,
    split_root: str | Path,
    relations_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    split_path = Path(split_root)
    relations_path = Path(relations_dir)
    output_path = Path(output_dir)

    train_pairs, validation_pairs, test_pairs = _read_split_pairs(split_path)
    course_ids = {course for pairs in (train_pairs, validation_pairs, test_pairs) for _user, course in pairs}
    relation_rows = _read_mooccube_relation_rows(relations_path)
    course_side_edges = normalize_mooccube_course_side_edges(relation_rows, course_ids=course_ids)
    atomic = build_atomic_data(
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        course_side_edges=course_side_edges,
    )
    manifest = write_kgrec_atomic_dataset(output_path, atomic)
    manifest["source"] = {
        "split_root": str(split_path),
        "relations_dir": str(relations_path),
        "included_relations": sorted(course_side_edges),
    }
    manifest["relation_edge_counts"] = {relation: len(edges) for relation, edges in sorted(course_side_edges.items())}
    (output_path / "strict_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
