from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_ROOT = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "strict_item_cold_balanced_thr1_seed_2025"
)
DEFAULT_STREAM = ROOT / "processed_data_hin_clean_pop5" / "stream_data.pkl"
DEFAULT_REL_DIR = ROOT / "MOOCCube" / "relations"
DEFAULT_OUT = ROOT / "paper_aaai27" / "baseline_sources" / "_adaptability" / "mooccube_seed2025_smoke"
DEFAULT_PCGNN_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "PCGNN_recbole_drive" / "RecBole-master"
ALL_TARGETS = {"pcgnn", "upgpr", "msec"}


@dataclass
class TinyAdapterInput:
    n_users: int
    n_items: int
    train_pairs: list[tuple[int, int]]
    val_pairs: list[tuple[int, int]]
    test_pairs: list[tuple[int, int]]
    course_tokens: list[str]
    course_concepts: dict[int, list[str]] = field(default_factory=dict)
    course_teachers: dict[int, list[str]] = field(default_factory=dict)
    course_schools: dict[int, str] = field(default_factory=dict)
    course_videos: dict[int, list[str]] = field(default_factory=dict)
    user_videos: dict[int, list[str]] = field(default_factory=dict)
    video_concepts: dict[str, list[str]] = field(default_factory=dict)
    kg_triples: list[tuple[str, str, str]] = field(default_factory=list)


def safe_token(value: object) -> str:
    token = str(value) if value is not None and str(value) else "unknown"
    token = re.sub(r"\s+", "_", token.strip())
    token = token.replace("\t", "_")
    try:
        token.encode("ascii")
        return token
    except UnicodeEncodeError:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()[:12]
        return f"tok_{digest}"


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_targets(value: str) -> set[str]:
    raw = {part.strip().lower() for part in value.split(",") if part.strip()}
    targets = set(ALL_TARGETS) if raw == {"all"} else raw
    unknown = targets - ALL_TARGETS
    if unknown:
        raise ValueError(f"Unknown targets: {sorted(unknown)}")
    return targets


def write_pcgnn_atomic_dataset(root: Path, dataset_name: str, data: TinyAdapterInput) -> dict[str, object]:
    dataset_dir = root / "dataset" / dataset_name
    timestamp = 1

    def inter_lines(pairs: list[tuple[int, int]]) -> list[str]:
        nonlocal timestamp
        rows = ["user_id:token\titem_id:token\trating:float\ttimestamp:float"]
        for user, item in pairs:
            rows.append(f"{user}\t{item}\t1\t{timestamp}")
            timestamp += 1
        return rows

    write_text(dataset_dir / f"{dataset_name}.train.inter", inter_lines(data.train_pairs))
    write_text(dataset_dir / f"{dataset_name}.valid.inter", inter_lines(data.val_pairs))
    write_text(dataset_dir / f"{dataset_name}.test.inter", inter_lines(data.test_pairs))

    item_rows = ["item_id:token\tfirst_level_category:token\tsecond_level_category:token"]
    item_categories: dict[int, tuple[str, str]] = {}
    for item in range(data.n_items):
        concepts = [safe_token(x) for x in data.course_concepts.get(item, []) if safe_token(x)]
        first = concepts[0] if concepts else "unknown"
        second = concepts[1] if len(concepts) > 1 else first
        item_categories[item] = (first, second)
        item_rows.append(f"{item}\t{first}\t{second}")
    write_text(dataset_dir / f"{dataset_name}.item", item_rows)

    link_rows = ["item_id:token\tentity_id:token"]
    for item in range(data.n_items):
        entity = safe_token(data.course_tokens[item]) if item < len(data.course_tokens) else str(item)
        link_rows.append(f"{item}\t{entity}")
    write_text(dataset_dir / f"{dataset_name}.link", link_rows)

    kg_rows = ["head_id:token\trelation_id:token\ttail_id:token"]
    for head, relation, tail in data.kg_triples:
        kg_rows.append(f"{safe_token(head)}\t{safe_token(relation)}\t{safe_token(tail)}")
    for item in range(data.n_items):
        entity = safe_token(data.course_tokens[item]) if item < len(data.course_tokens) else str(item)
        for category in sorted(set(item_categories[item])):
            kg_rows.append(f"{entity}\titem_category\t{category}")
    if len(kg_rows) == 1:
        kg_rows.append("0\tself_loop\t0")
    write_text(dataset_dir / f"{dataset_name}.kg", kg_rows)

    config_text = f"""data_path: dataset/
dataset: {dataset_name}
benchmark_filename: ['train', 'valid', 'test']

field_separator: "\\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
LABEL_FIELD: label
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id

load_col:
    inter: [user_id, item_id, timestamp]
    item: [item_id, first_level_category, second_level_category]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]

filter_inter_by_user_or_item: True
rm_dup_inter: ~
normalize_all: True
eval_setting: TO_LS,full
topk: [5,10,20]
metrics: ["Recall", "MRR", "NDCG", "Hit", "Precision"]
training_neg_sample_num: 0
train_batch_size: 32
eval_batch_size: 64
epochs: 1
"""
    config_path = root / f"recbole_{dataset_name}.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    return {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "config_path": str(config_path),
        "train_rows": len(data.train_pairs),
        "validation_rows": len(data.val_pairs),
        "test_rows": len(data.test_pairs),
        "item_rows": data.n_items,
        "kg_rows": max(0, len(kg_rows) - 1),
        "protocol_notes": [
            "Uses benchmark_filename train/valid/test files to preserve the external strict split.",
            "RecBole still needs a custom item-macro evaluator for paper metrics.",
        ],
    }


def write_upgpr_processed_dataset(out_dir: Path, data: TinyAdapterInput) -> dict[str, object]:
    concept_to_id: dict[str, int] = {}
    teacher_to_id: dict[str, int] = {}
    school_to_id: dict[str, int] = {"unknown": 0}

    def get_id(mapping: dict[str, int], raw: str) -> int:
        token = safe_token(raw)
        if token not in mapping:
            mapping[token] = len(mapping)
        return mapping[token]

    for values in data.course_concepts.values():
        for concept in values:
            get_id(concept_to_id, concept)
    for values in data.course_teachers.values():
        for teacher in values:
            get_id(teacher_to_id, teacher)
    for school in data.course_schools.values():
        get_id(school_to_id, school)

    write_text(out_dir / "users.txt", [str(i) for i in range(data.n_users)])
    write_text(out_dir / "courses.txt", data.course_tokens[: data.n_items])
    write_text(out_dir / "concepts.txt", _ordered_tokens(concept_to_id))
    write_text(out_dir / "teachers.txt", _ordered_tokens(teacher_to_id))
    write_text(out_dir / "schools.txt", _ordered_tokens(school_to_id))

    def pair_rows(pairs: list[tuple[int, int]]) -> list[str]:
        return [f"{user} {item}" for user, item in pairs]

    # UPGPR's official preprocessor reads this file to build the training KG.
    # Keeping held-out interactions here would leak strict cold items.
    write_text(out_dir / "enrolments.txt", pair_rows(data.train_pairs))
    write_text(out_dir / "train.txt", pair_rows(data.train_pairs))
    write_text(out_dir / "validation.txt", pair_rows(data.val_pairs))
    write_text(out_dir / "test.txt", pair_rows(data.test_pairs))

    concept_rows: list[str] = []
    teacher_rows: list[str] = []
    school_rows: list[str] = []
    for item in range(data.n_items):
        concept_rows.append(" ".join(str(get_id(concept_to_id, c)) for c in data.course_concepts.get(item, [])))
        teacher_rows.append(" ".join(str(get_id(teacher_to_id, t)) for t in data.course_teachers.get(item, [])))
        school_rows.append(str(get_id(school_to_id, data.course_schools.get(item, "unknown"))))
    write_text(out_dir / "course_concepts.txt", concept_rows)
    write_text(out_dir / "course_teachers.txt", teacher_rows)
    write_text(out_dir / "course_school.txt", school_rows)

    return {
        "processed_dir": str(out_dir),
        "train_rows": len(data.train_pairs),
        "validation_rows": len(data.val_pairs),
        "test_rows": len(data.test_pairs),
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_concepts": len(concept_to_id),
        "n_teachers": len(teacher_to_id),
        "n_schools": len(school_to_id),
        "protocol_notes": [
            "Bypasses UPGPR's random splitter by writing train/validation/test files directly.",
            "Official path ranking must be converted to full-catalog scoring for paper metrics.",
        ],
    }


def write_msec_smoke_dataset(out_dir: Path, data: TinyAdapterInput) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted({v for values in data.course_videos.values() for v in values} | {v for values in data.user_videos.values() for v in values})
    concepts = sorted({safe_token(c) for values in data.course_concepts.values() for c in values} | {safe_token(c) for values in data.video_concepts.values() for c in values})
    video_to_id = {video: idx for idx, video in enumerate(videos)}
    concept_to_id = {concept: idx for idx, concept in enumerate(concepts)}

    train_uc = np.zeros((data.n_users, data.n_items), dtype=np.uint8)
    val_uc = np.zeros((data.n_users, data.n_items), dtype=np.uint8)
    for user, item in data.train_pairs:
        train_uc[user, item] = 1
    for user, item in data.val_pairs:
        val_uc[user, item] = 1

    train_uv = np.zeros((data.n_users, max(1, len(videos))), dtype=np.uint8)
    for user, values in data.user_videos.items():
        if user >= data.n_users:
            continue
        for video in values:
            if video in video_to_id:
                train_uv[user, video_to_id[video]] = 1

    ck = np.zeros((data.n_items, max(1, len(concepts))), dtype=np.uint8)
    for item, values in data.course_concepts.items():
        if item >= data.n_items:
            continue
        for concept in values:
            token = safe_token(concept)
            if token in concept_to_id:
                ck[item, concept_to_id[token]] = 1

    course_video = np.zeros((data.n_items, max(1, len(videos))), dtype=np.uint8)
    for item, values in data.course_videos.items():
        if item >= data.n_items:
            continue
        for video in values:
            if video in video_to_id:
                course_video[item, video_to_id[video]] = 1

    video_concept = np.zeros((max(1, len(videos)), max(1, len(concepts))), dtype=np.uint8)
    for video, values in data.video_concepts.items():
        if video not in video_to_id:
            continue
        for concept in values:
            token = safe_token(concept)
            if token in concept_to_id:
                video_concept[video_to_id[video], concept_to_id[token]] = 1

    np.save(out_dir / "train_uc.npy", train_uc)
    np.save(out_dir / "val_uc.npy", val_uc)
    np.save(out_dir / "train_uv.npy", train_uv)
    np.save(out_dir / "ck.npy", ck)
    np.save(out_dir / "course_video.npy", course_video)
    np.save(out_dir / "video_concept.npy", video_concept)
    np.save(out_dir / "user_course_features.npy", train_uc.astype(np.float32))

    return {
        "data_dir": str(out_dir),
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_videos": len(videos),
        "n_concepts": len(concepts),
        "train_uc_positive": int(train_uc.sum()),
        "val_uc_positive": int(val_uc.sum()),
        "protocol_notes": [
            "Exports MSEC matrix names, but DGL graph construction and sampled evaluator still need replacement.",
            "This smoke export keeps matrices small; full export would be much larger for user-course features.",
        ],
    }


def _ordered_tokens(mapping: dict[str, int]) -> list[str]:
    if not mapping:
        return []
    return [token for token, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not path.exists():
        return pairs
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def dataframe_pairs(df: pd.DataFrame, max_rows: int | None) -> list[tuple[int, int]]:
    rows = [(int(row.u_idx), int(row.i_idx)) for row in df[["u_idx", "i_idx"]].itertuples(index=False)]
    return rows[:max_rows] if max_rows is not None and max_rows >= 0 else rows


def build_smoke_input(
    split_root: Path,
    stream_path: Path,
    rel_dir: Path,
    max_train_pos: int,
    max_val_pos: int,
    max_test_pos: int,
    include_video_features: bool = True,
) -> TinyAdapterInput:
    train_df = pd.read_pickle(split_root / "static_train.pkl")
    val_df = pd.read_pickle(split_root / "static_val.pkl")
    test_df = pd.read_pickle(split_root / "static_test.pkl")
    stream_df = pd.read_pickle(stream_path)

    train_pairs_raw = dataframe_pairs(train_df, max_train_pos)
    val_cold = val_df[val_df["_split_source"].eq("strict_item_cold_val")]
    test_cold = test_df[test_df["_split_source"].eq("strict_item_cold_test")]
    val_pairs_raw = dataframe_pairs(val_cold, max_val_pos)
    test_pairs_raw = dataframe_pairs(test_cold, max_test_pos)

    users = sorted({u for u, _ in train_pairs_raw + val_pairs_raw + test_pairs_raw})
    user_to_local = {user: idx for idx, user in enumerate(users)}
    n_items = int(stream_df["i_idx"].max()) + 1

    def remap_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(user_to_local[u], i) for u, i in pairs if u in user_to_local]

    idx_to_course = (
        stream_df[["i_idx", "course_id"]]
        .drop_duplicates("i_idx")
        .sort_values("i_idx")
        .set_index("i_idx")["course_id"]
        .to_dict()
    )
    course_tokens = [safe_token(idx_to_course.get(i, f"course_{i}")) for i in range(n_items)]
    course_to_idx = {course: int(idx) for idx, course in idx_to_course.items()}

    course_concepts: dict[int, list[str]] = {i: [] for i in range(n_items)}
    for course, concept in read_pairs(rel_dir / "course-concept.json"):
        if course in course_to_idx:
            item = course_to_idx[course]
            if len(course_concepts[item]) < 8:
                course_concepts[item].append(safe_token(concept))

    course_teachers: dict[int, list[str]] = {i: [] for i in range(n_items)}
    for teacher, course in read_pairs(rel_dir / "teacher-course.json"):
        if course in course_to_idx:
            item = course_to_idx[course]
            if len(course_teachers[item]) < 4:
                course_teachers[item].append(safe_token(teacher))

    course_schools: dict[int, str] = {}
    for school, course in read_pairs(rel_dir / "school-course.json"):
        if course in course_to_idx:
            course_schools[course_to_idx[course]] = safe_token(school)

    selected_course_tokens = {course_tokens[i] for _, i in train_pairs_raw + val_pairs_raw + test_pairs_raw}
    kg_triples: list[tuple[str, str, str]] = []
    for item, concepts in course_concepts.items():
        course = course_tokens[item]
        if course in selected_course_tokens:
            for concept in concepts[:4]:
                kg_triples.append((course, "course_concept", concept))
    for pre, post in read_pairs(rel_dir / "prerequisite-dependency.json"):
        kg_triples.append((safe_token(pre), "prerequisite", safe_token(post)))
        if len(kg_triples) >= 5000:
            break

    course_videos: dict[int, list[str]] = {}
    user_videos: dict[int, list[str]] = {}
    video_concepts: dict[str, list[str]] = {}
    if include_video_features:
        course_videos = _limited_course_videos(rel_dir / "course-video.json", course_to_idx, limit_per_course=3)
        selected_raw_users = set()
        raw_user_map = stream_df[["u_idx", "user_id"]].drop_duplicates("u_idx").set_index("u_idx")["user_id"].to_dict()
        for user in users:
            if user in raw_user_map:
                selected_raw_users.add(raw_user_map[user])
        user_videos = _limited_user_videos(rel_dir / "user-video.json", raw_user_map, user_to_local, selected_raw_users, limit_per_user=5)
        selected_videos = {v for values in course_videos.values() for v in values} | {v for values in user_videos.values() for v in values}
        video_concepts = _limited_video_concepts(rel_dir / "video-concept.json", selected_videos, limit_per_video=5)

    return TinyAdapterInput(
        n_users=len(users),
        n_items=n_items,
        train_pairs=remap_pairs(train_pairs_raw),
        val_pairs=remap_pairs(val_pairs_raw),
        test_pairs=remap_pairs(test_pairs_raw),
        course_tokens=course_tokens,
        course_concepts=course_concepts,
        course_teachers=course_teachers,
        course_schools=course_schools,
        course_videos=course_videos,
        user_videos=user_videos,
        video_concepts=video_concepts,
        kg_triples=kg_triples,
    )


def _limited_course_videos(path: Path, course_to_idx: dict[str, int], limit_per_course: int) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for course, video in read_pairs(path):
        if course not in course_to_idx:
            continue
        item = course_to_idx[course]
        values = out.setdefault(item, [])
        if len(values) < limit_per_course:
            values.append(safe_token(video))
    return out


def _limited_user_videos(
    path: Path,
    raw_user_map: dict[int, str],
    user_to_local: dict[int, int],
    selected_raw_users: set[str],
    limit_per_user: int,
) -> dict[int, list[str]]:
    raw_to_local = {raw_user_map[user]: local for user, local in user_to_local.items() if user in raw_user_map}
    out: dict[int, list[str]] = {}
    if not selected_raw_users:
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] not in selected_raw_users:
                continue
            local = raw_to_local.get(parts[0])
            if local is None:
                continue
            values = out.setdefault(local, [])
            if len(values) < limit_per_user:
                values.append(safe_token(parts[1]))
    return out


def _limited_video_concepts(path: Path, selected_videos: set[str], limit_per_video: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not selected_videos:
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            video = safe_token(parts[0])
            if video not in selected_videos:
                continue
            values = out.setdefault(video, [])
            if len(values) < limit_per_video:
                values.append(safe_token(parts[1]))
    return out


def dependency_report() -> dict[str, str]:
    names = ["torch", "dgl", "recbole", "easydict", "wandb", "numpy", "pandas"]
    return {name: ("OK" if importlib.util.find_spec(name) else "MISSING") for name in names}


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    deps = report["dependencies"]
    rows = report["candidates"]
    lines = [
        "# Course Baseline Adaptability Experiment",
        "",
        f"Split: `{report['split_root']}`",
        f"Output: `{report['out_dir']}`",
        "",
        "## Dependency Gate",
        "",
        "| Package | Status |",
        "|---|---|",
    ]
    for name, status in deps.items():
        lines.append(f"| {name} | {status} |")
    lines.extend([
        "",
        "## Priority Assessment",
        "",
        "| Priority | Candidate | Smoke artifact | Loader/dependency status | Protocol fit | Recommendation |",
        "|---:|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['priority']} | {row['candidate']} | {row['artifact']} | "
            f"{row['loader_status']} | {row['protocol_fit']} | {row['recommendation']} |"
        )
    lines.extend([
        "",
        "## Key Takeaways",
        "",
        "- PCGNN remains the best next runnable course-specific baseline because its RecBole atomic format can preserve an external strict split.",
        "- PCGNN's files load in the modified RecBole tree, but the stock sequential build path still needs a protocol patch; otherwise validation/test sequence construction can collapse under external item-cold splits.",
        "- UPGPR is highly relevant and its Dataset/KnowledgeGraph reader accepts the exported files, but requires replacing path-only top-10 evaluation with full-catalog scoring.",
        "- MSEC-Rec is recent and course-specific, but current environment lacks DGL and its released evaluator is sampled-ranking.",
        "- KGAN remains a backup; the existing adapter smoke passes, but TF1/Keras and random official splitting make it less attractive.",
    ])
    write_text(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--rel-dir", type=Path, default=DEFAULT_REL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pcgnn-root", type=Path, default=DEFAULT_PCGNN_ROOT)
    parser.add_argument("--pcgnn-dataset-name", default="mooccube_strict_seed2025_smoke")
    parser.add_argument("--targets", default="all", help="Comma-separated subset of pcgnn,upgpr,msec, or all.")
    parser.add_argument("--max-train-pos", type=int, default=2000)
    parser.add_argument("--max-val-pos", type=int, default=500)
    parser.add_argument("--max-test-pos", type=int, default=500)
    args = parser.parse_args()

    targets = parse_targets(args.targets)

    data = build_smoke_input(
        args.split_root,
        args.stream,
        args.rel_dir,
        args.max_train_pos,
        args.max_val_pos,
        args.max_test_pos,
        include_video_features="msec" in targets,
    )

    artifacts: dict[str, object] = {}
    candidates: list[dict[str, object]] = []

    if "pcgnn" in targets:
        pcgnn_report = write_pcgnn_atomic_dataset(args.pcgnn_root, args.pcgnn_dataset_name, data)
        artifacts["pcgnn"] = pcgnn_report
        candidates.append(
            {
                "priority": 1,
                "candidate": "PCGNN",
                "artifact": pcgnn_report["dataset_dir"],
                "loader_status": "atomic files load; stock sequential dataloader needs patch for external strict valid/test histories",
                "protocol_fit": "medium-high: data split is preservable, official build/evaluator is not protocol-safe yet",
                "recommendation": "adapt first, but patch loader/build before training",
            }
        )
    if "upgpr" in targets:
        upgpr_report = write_upgpr_processed_dataset(args.out / "upgpr" / "processed_files", data)
        artifacts["upgpr"] = upgpr_report
        candidates.append(
            {
                "priority": 2,
                "candidate": "UPGPR",
                "artifact": upgpr_report["processed_dir"],
                "loader_status": "processed files exported; dependency isolation still recommended",
                "protocol_fit": "medium-high for data; path evaluator must be replaced",
                "recommendation": "adapt second in a clean dependency env",
            }
        )
    if "msec" in targets:
        msec_report = write_msec_smoke_dataset(args.out / "msec" / "data", data)
        artifacts["msec"] = msec_report
        candidates.append(
            {
                "priority": 3,
                "candidate": "MSEC-Rec",
                "artifact": msec_report["data_dir"],
                "loader_status": "matrix export ready; DGL dependency still required",
                "protocol_fit": "medium; sampled evaluator and DGL graph need changes",
                "recommendation": "adapt after PCGNN/UPGPR",
            }
        )
    deps = dependency_report()

    report = {
        "split_root": str(args.split_root),
        "out_dir": str(args.out),
        "smoke_input": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "train_pairs": len(data.train_pairs),
            "val_pairs": len(data.val_pairs),
            "test_pairs": len(data.test_pairs),
            "kg_triples": len(data.kg_triples),
        },
        "dependencies": deps,
        "artifacts": artifacts,
        "candidates": candidates,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adaptability_experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(args.out / "adaptability_experiment_report.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
