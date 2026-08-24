from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

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
DEFAULT_OUT = ROOT / "paper_aaai27" / "baseline_sources" / "_prepared" / "mooccube_seed2025"
REL_DIR = ROOT / "MOOCCube" / "relations"


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def write_rows(path: Path, rows: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")


def sample_negatives(
    positives: list[tuple[int, int]],
    all_items: list[int],
    user_all_pos: dict[int, set[int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    negatives: list[tuple[int, int]] = []
    for user, _ in positives:
        seen = user_all_pos.get(user, set())
        candidates = [item for item in all_items if item not in seen]
        if not candidates:
            continue
        negatives.append((user, rng.choice(candidates)))
    return negatives


def relation_entity_id(name: str, entity_map: dict[str, int]) -> int:
    if name not in entity_map:
        entity_map[name] = len(entity_map)
    return entity_map[name]


def build_kg(course_to_idx: dict[str, int], relation_width: int) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    entity_map: dict[str, int] = {}
    for course, idx in course_to_idx.items():
        entity_map[f"C::{course}"] = idx

    next_entity = max(course_to_idx.values(), default=-1) + 1
    for key in list(entity_map):
        if entity_map[key] >= next_entity:
            next_entity = entity_map[key] + 1

    def get_aux_entity(raw: str, prefix: str) -> int:
        nonlocal next_entity
        key = f"{prefix}::{raw}"
        if key not in entity_map:
            entity_map[key] = next_entity
            next_entity += 1
        return entity_map[key]

    triples: set[tuple[int, int, int]] = set()

    # 0/1: course-concept and reverse.
    for course, concept in read_pairs(REL_DIR / "course-concept.json"):
        if course not in course_to_idx:
            continue
        c = course_to_idx[course]
        k = get_aux_entity(concept, "K")
        triples.add((c, 0, k))
        triples.add((k, 1, c))

    # 2/3: course-teacher and reverse.
    for teacher, course in read_pairs(REL_DIR / "teacher-course.json"):
        if course not in course_to_idx:
            continue
        c = course_to_idx[course]
        t = get_aux_entity(teacher, "T")
        triples.add((c, 2, t))
        triples.add((t, 3, c))

    # 4/5: course-school and reverse.
    for school, course in read_pairs(REL_DIR / "school-course.json"):
        if course not in course_to_idx:
            continue
        c = course_to_idx[course]
        s = get_aux_entity(school, "S")
        triples.add((c, 4, s))
        triples.add((s, 5, c))

    # 6: concept prerequisite.
    for pre, post in read_pairs(REL_DIR / "prerequisite-dependency.json"):
        pre_id = get_aux_entity(pre, "K")
        post_id = get_aux_entity(post, "K")
        triples.add((pre_id, 6, post_id))

    # KGAN's original code has relation-shape hardcoding. Keep optional dummy
    # relation ids so the generated data can match that shape without changing
    # real course evidence.
    for rel in range(relation_width):
        triples.add((0, rel, 0))

    return sorted(triples), entity_map


def dataframe_pairs(df: pd.DataFrame, max_rows: int | None) -> list[tuple[int, int]]:
    pairs = [(int(row.u_idx), int(row.i_idx)) for row in df[["u_idx", "i_idx"]].itertuples(index=False)]
    if max_rows is not None and len(pairs) > max_rows:
        return pairs[:max_rows]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-pos", type=int, default=50000)
    parser.add_argument("--relation-width", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_df = pd.read_pickle(args.split_root / "static_train.pkl")
    val_df = pd.read_pickle(args.split_root / "static_val.pkl")
    test_df = pd.read_pickle(args.split_root / "static_test.pkl")
    stream_df = pd.read_pickle(args.stream)

    course_to_idx = {
        str(row.course_id): int(row.i_idx)
        for row in stream_df[["course_id", "i_idx"]].drop_duplicates("course_id").itertuples(index=False)
    }
    n_users = int(stream_df["u_idx"].max()) + 1
    n_items = int(stream_df["i_idx"].max()) + 1
    all_items = list(range(n_items))

    user_all_pos: dict[int, set[int]] = defaultdict(set)
    for row in stream_df[["u_idx", "i_idx"]].itertuples(index=False):
        user_all_pos[int(row.u_idx)].add(int(row.i_idx))

    train_pos = dataframe_pairs(train_df, args.max_train_pos)
    train_neg = sample_negatives(train_pos, all_items, user_all_pos, rng)
    val_cold = dataframe_pairs(val_df[val_df["_split_source"].eq("strict_item_cold_val")], None)
    test_cold = dataframe_pairs(test_df[test_df["_split_source"].eq("strict_item_cold_test")], None)

    train_rows = [(u, i, 1) for u, i in train_pos] + [(u, i, 0) for u, i in train_neg]
    rng.shuffle(train_rows)
    val_rows = [(u, i, 1) for u, i in val_cold]
    test_rows = [(u, i, 1) for u, i in test_cold]

    kg_rows, entity_map = build_kg(course_to_idx, relation_width=args.relation_width)

    kgan_dir = args.out / "kgan" / "data" / "course_strict_seed2025"
    write_rows(kgan_dir / "ratings_final.txt", train_rows)
    with (kgan_dir / "kg_final.txt").open("w", encoding="utf-8") as handle:
        for h, r, t in kg_rows:
            handle.write(f"{h}\t{r}\t{t}\n")
    write_rows(kgan_dir / "strict_train.txt", train_rows)
    write_rows(kgan_dir / "strict_val_cold_pos.txt", val_rows)
    write_rows(kgan_dir / "strict_test_cold_pos.txt", test_rows)

    idrmi_dir = args.out / "idrmi" / "Data" / "moocCube"
    write_rows(idrmi_dir / "train_set.txt", train_rows)
    write_rows(idrmi_dir / "eval_set.txt", val_rows)
    write_rows(idrmi_dir / "test_set.txt", test_rows)
    write_rows(idrmi_dir / "rating_index.tsv", train_rows + val_rows + test_rows)
    with (idrmi_dir / "kg_index.tsv").open("w", encoding="utf-8") as handle:
        for h, r, t in kg_rows:
            handle.write(f"{h}\t{r}\t{t}\n")

    user_items: dict[int, list[int]] = defaultdict(list)
    item_users: dict[int, list[int]] = defaultdict(list)
    for u, i in train_pos:
        user_items[u].append(i)
        item_users[i].append(u)
    with (idrmi_dir / "user_items.txt").open("w", encoding="utf-8") as handle:
        for user in sorted(user_items):
            items = " ".join(str(i) for i in sorted(set(user_items[user])))
            handle.write(f"{user} {items}\n")
    with (idrmi_dir / "item_users.txt").open("w", encoding="utf-8") as handle:
        for item in sorted(item_users):
            users = " ".join(str(u) for u in sorted(set(item_users[item])))
            handle.write(f"{item} {users}\n")

    report = {
        "split_root": str(args.split_root),
        "n_users": n_users,
        "n_items": n_items,
        "max_train_pos": args.max_train_pos,
        "train_positive_rows_exported": len(train_pos),
        "train_negative_rows_exported": len(train_neg),
        "val_cold_positive_rows": len(val_rows),
        "test_cold_positive_rows": len(test_rows),
        "kg_triples": len(kg_rows),
        "kg_entities": len(entity_map),
        "relation_width": args.relation_width,
        "kgan_dir": str(kgan_dir),
        "idrmi_dir": str(idrmi_dir),
        "notes": [
            "KGAN original loader randomly splits ratings_final.txt; strict protocol needs a custom loader/evaluator.",
            "IDRMI loader reads these files, but its NGCF adjacency construction does not populate R without source edits.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adapter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
