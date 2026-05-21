"""Copy legacy DropoutNet official-protocol results into epoch-tagged folders.

Older runs used a generic result subdirectory, so e120/e160 could overwrite one
another. This helper reads each result JSON, infers
``teacher{teacher_epochs}_student{student_epochs}``, and copies the JSON into
the corresponding epoch-tagged result directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_ROOT = Path("outputs/content_delta_pop5/static_item_cold_balanced")
DEFAULT_LEGACY_SUBDIR = "main_table_balanced_itemmacro_dropoutnet_official_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--legacy-subdir", default=DEFAULT_LEGACY_SUBDIR)
    parser.add_argument("--split-glob", default="strict_item_cold_balanced_thr1_seed_*")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) and data else data


def main() -> None:
    args = parse_args()
    copied = 0
    for split_dir in sorted(args.root.glob(args.split_glob)):
        if not split_dir.is_dir():
            continue
        src = split_dir / args.legacy_subdir / "dropoutnet_official_static_result.json"
        if not src.exists():
            continue
        obj = load_json(src)
        teacher = obj.get("teacher_epochs")
        student = obj.get("student_epochs")
        if teacher is None or student is None:
            print(f"Skip missing epochs: {src}")
            continue
        epoch_tag = obj.get("epoch_tag") or f"teacher{teacher}_student{student}"
        dst_dir = split_dir / f"main_table_balanced_itemmacro_dropoutnet_official_{epoch_tag}_v1"
        dst = dst_dir / src.name
        print(f"{src} -> {dst}")
        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} result file(s).")


if __name__ == "__main__":
    main()
