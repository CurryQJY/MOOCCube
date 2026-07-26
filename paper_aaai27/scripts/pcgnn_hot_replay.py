from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.pcgnn_strict_adapter import (  # noqa: E402
    _frame_rows,
    _limit_value,
    _read_split_rows,
    build_strict_eval_examples,
    build_user_seen_items,
    clean_argv_for_recbole,
    evaluate_pcgnn_full_item_macro,
    resolve_torch_device,
    resolve_workspace_path,
)
from paper_aaai27.scripts.priority_baseline_experiments import (  # noqa: E402
    PCGNN_ROOT,
    local_pcgnn_recbole,
    pcgnn_smoke_config_overrides,
)


BASE_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "_pcgnn_strict"
DATASET_PREFIX = {
    "mooccube": "mooccube",
    "junyi": "junyi",
    "coco": "coco",
}
TEST_SOURCES = {
    "cold": ("strict_item_cold_test",),
    "hot": ("strict_item_cold_warm_test",),
    "all": ("strict_item_cold_test", "strict_item_cold_warm_test"),
}
METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")


def select_test_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in TEST_SOURCES:
        raise ValueError(f"unsupported replay split: {split}")
    return frame[frame["_split_source"].isin(TEST_SOURCES[split])].copy()


def resolve_config_file(dataset_name: str) -> Path:
    return PCGNN_ROOT / f"recbole_{dataset_name}.yaml"


def combine_cold_hot_overall(report: dict[str, object]) -> dict[str, object]:
    cold = report.get("full_cold_item_macro", {})
    hot = report.get("full_hot_item_macro", {})
    cold_count = int(report.get("count_full_cold_item_macro", 0) or 0)
    hot_count = int(report.get("count_full_hot_item_macro", 0) or 0)
    if not isinstance(cold, dict) or not isinstance(hot, dict):
        raise ValueError("cold and hot item-macro blocks must be dictionaries")
    if cold_count <= 0 or hot_count <= 0:
        raise ValueError("cold and hot item counts must both be positive")
    all_metrics: dict[str, float] = {}
    for metric in METRICS:
        if metric not in cold or metric not in hot:
            continue
        cold_value = float(cold[metric])
        hot_value = float(hot[metric])
        if not math.isfinite(cold_value) or not math.isfinite(hot_value):
            continue
        all_metrics[metric] = (
            cold_value * cold_count + hot_value * hot_count
        ) / (cold_count + hot_count)
    return {
        **report,
        "full_all_item_macro": all_metrics,
        "count_full_all_item_macro": cold_count + hot_count,
    }


def pcgnn_run_dir(dataset: str, seed: int) -> Path:
    prefix = DATASET_PREFIX[dataset]
    return BASE_DIR / f"{prefix}_seed{int(seed)}_full_formal_kg_warm"


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def replay_one(
    dataset: str,
    seed: int,
    split: str,
    device_name: str,
    eval_batch_size: int,
    max_examples: int,
    output_filename: str,
) -> Path:
    import torch

    run_dir = pcgnn_run_dir(dataset, seed)
    source_report_path = run_dir / "pcgnn_strict_adapter_report.json"
    source_report = load_json(source_report_path)
    checkpoint_path = Path(
        str(source_report.get("best_checkpoint_path") or run_dir / "checkpoints" / "best_model.pt")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing PCGNN checkpoint: {checkpoint_path}")

    split_root = resolve_workspace_path(Path(str(source_report["split_root"])))
    dataset_name = str(source_report["dataset_name"])
    config_file = resolve_config_file(dataset_name)
    device = resolve_torch_device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_rows, _, test_df = _read_split_rows(split_root)
    selected_test_df = select_test_frame(test_df, split)
    selected_rows = _frame_rows(selected_test_df)

    with local_pcgnn_recbole(PCGNN_ROOT):
        from recbole.config import Config
        from recbole.data import Interaction, create_dataset, data_preparation
        from recbole.utils import get_model

        with clean_argv_for_recbole():
            config = Config(
                model="kg_model",
                dataset=dataset_name,
                config_file_list=[str(config_file)],
                config_dict=pcgnn_smoke_config_overrides(
                    train_batch_size=int(source_report.get("train_batch_size", 32)),
                    eval_batch_size=eval_batch_size,
                    device=device.type,
                ),
            )
        recbole_dataset = create_dataset(config)
        train_data, _, _ = data_preparation(config, recbole_dataset)
        token_map = {
            str(key): int(value)
            for key, value in recbole_dataset.field2token_id["item_id"].items()
        }
        max_len = int(config["MAX_ITEM_LIST_LENGTH"])
        examples = build_strict_eval_examples(
            train_rows,
            selected_rows,
            token_map,
            max_len=max_len,
            limit=_limit_value(max_examples),
        )
        user_seen_items = build_user_seen_items(train_rows, token_map)
        model = get_model(config["model"])(config, train_data).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        replay_test = evaluate_pcgnn_full_item_macro(
            model,
            Interaction,
            examples,
            user_seen_items,
            max_len=max_len,
            batch_size=eval_batch_size,
            k_list=source_report.get("k_list", [5, 10, 20]),
            cold_threshold=int(source_report.get("cold_threshold", 1)),
            device=device,
        )

    replay_test = combine_cold_hot_overall(replay_test) if split == "all" else replay_test
    output = {
        "model": "PCGNN",
        "protocol": "strict_item_cold_full_catalog_item_macro_hot_replay",
        "dataset": dataset,
        "seed": int(seed),
        "split": split,
        "source_report_path": str(source_report_path),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": source_report.get("best_epoch"),
        "best_validation_score": source_report.get("best_validation_score"),
        "dataset_name": dataset_name,
        "split_root": str(split_root),
        "config_file": str(config_file),
        "requested_device": device_name,
        "device": str(device),
        "eval_batch_size": int(eval_batch_size),
        "max_examples": int(max_examples),
        "selected_test_rows": int(len(selected_test_df)),
        "sequence_examples": int(len(examples)),
        "test": replay_test,
        "notes": [
            "Replay loads the retained PCGNN best checkpoint and evaluates strict test targets without retraining.",
            "The original formal PCGNN report selected only strict_item_cold_test rows; this replay includes strict_item_cold_warm_test rows when split=hot/all.",
            "Full-catalog scores are produced by PCGNN full_sort_predict and train-history items are masked by the same external evaluator as the original adapter.",
        ],
    }
    out_path = run_dir / output_filename
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_PREFIX), default=sorted(DATASET_PREFIX))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    parser.add_argument("--split", choices=sorted(TEST_SOURCES), default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--output-filename", default="pcgnn_hot_replay_report.json")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        for seed in args.seeds:
            path = replay_one(
                dataset=dataset,
                seed=seed,
                split=args.split,
                device_name=args.device,
                eval_batch_size=args.eval_batch_size,
                max_examples=args.max_examples,
                output_filename=args.output_filename,
            )
            print(path)


if __name__ == "__main__":
    main()
