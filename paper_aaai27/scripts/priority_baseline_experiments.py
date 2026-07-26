from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT / "outputs" / "content_delta_pop5" / "static_item_cold_balanced" / "strict_item_cold_balanced_thr1_seed_2025"
ADAPT_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "_adaptability" / "mooccube_seed2025_smoke"
PCGNN_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "PCGNN_recbole_drive" / "RecBole-master"
KGAN_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "KGAN" / "KGAN-master"
OUT_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "_priority_experiments" / "mooccube_seed2025"


def rows_from_frame(df: pd.DataFrame) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for row in df[["u_idx", "i_idx", "timestamp"]].itertuples(index=False):
        rows.append({"u_idx": int(row.u_idx), "i_idx": int(row.i_idx), "timestamp": int(row.timestamp)})
    return rows


def build_train_histories(train_rows: Iterable[dict[str, int]], token_map: dict[str, int], max_len: int) -> dict[int, list[int]]:
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in train_rows:
        token_id = token_map.get(str(row["i_idx"]))
        if token_id is not None:
            by_user[int(row["u_idx"])].append((int(row["timestamp"]), int(token_id)))
    histories: dict[int, list[int]] = {}
    for user, values in by_user.items():
        values.sort(key=lambda x: x[0])
        histories[user] = [item for _, item in values][-max_len:]
    return histories


def build_strict_sequence_examples(
    train_rows: Iterable[dict[str, int]],
    eval_rows: Iterable[dict[str, int]],
    token_map: dict[str, int],
    max_len: int,
    limit: int | None = None,
) -> list[dict[str, object]]:
    histories = build_train_histories(train_rows, token_map, max_len)
    examples: list[dict[str, object]] = []
    for row in sorted(eval_rows, key=lambda r: (int(r["u_idx"]), int(r["timestamp"]))):
        user = int(row["u_idx"])
        target = token_map.get(str(row["i_idx"]))
        history = histories.get(user, [])
        if target is None or not history:
            continue
        examples.append({"user": user, "history": history[-max_len:], "target": int(target), "raw_item": int(row["i_idx"])})
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_training_sequence_examples(
    train_rows: Iterable[dict[str, int]],
    token_map: dict[str, int],
    max_len: int,
    limit: int,
) -> list[dict[str, object]]:
    by_user: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in train_rows:
        if str(row["i_idx"]) in token_map:
            by_user[int(row["u_idx"])].append(row)
    examples: list[dict[str, object]] = []
    for user in sorted(by_user):
        history: list[int] = []
        for row in sorted(by_user[user], key=lambda r: int(r["timestamp"])):
            target = token_map.get(str(row["i_idx"]))
            if target is None:
                continue
            if history:
                examples.append({"user": user, "history": history[-max_len:], "target": int(target), "raw_item": int(row["i_idx"])})
                if len(examples) >= limit:
                    return examples
            history.append(int(target))
    return examples


def compute_relation_reachability(
    user_history: dict[int, set[int]],
    eval_pairs: list[tuple[int, int]],
    item_relations: dict[int, dict[str, set[str]]],
) -> dict[str, float | int]:
    reached = 0
    with_history = 0
    for user, target in eval_pairs:
        history = user_history.get(user, set())
        if not history:
            continue
        with_history += 1
        target_rel = item_relations.get(target, {})
        ok = False
        for item in history:
            hist_rel = item_relations.get(item, {})
            for name, values in target_rel.items():
                if values and values.intersection(hist_rel.get(name, set())):
                    ok = True
                    break
            if ok:
                break
        if ok:
            reached += 1
    return {
        "eval_pairs": len(eval_pairs),
        "with_train_history": with_history,
        "target_reachable": reached,
        "target_reachable_rate": reached / with_history if with_history else 0.0,
    }


def tensorize_examples(examples: list[dict[str, object]], max_len: int):
    import torch

    item_seq = torch.zeros((len(examples), max_len), dtype=torch.long)
    item_len = torch.zeros(len(examples), dtype=torch.long)
    target = torch.zeros(len(examples), dtype=torch.long)
    for row_idx, example in enumerate(examples):
        history = list(example["history"])[-max_len:]
        item_len[row_idx] = len(history)
        item_seq[row_idx, : len(history)] = torch.tensor(history, dtype=torch.long)
        target[row_idx] = int(example["target"])
    return item_seq, item_len, target


def pcgnn_smoke_config_overrides(
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, object]:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported PCGNN device [{device}]")
    return {
        "device": device,
        "use_gpu": device == "cuda",
        "gpu_id": 0 if device == "cuda" else -1,
        "show_progress": False,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
    }


@contextmanager
def local_pcgnn_recbole(pcgnn_root: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    for module in list(sys.modules):
        if module == "recbole" or module.startswith("recbole."):
            del sys.modules[module]
    os.chdir(pcgnn_root)
    sys.path.insert(0, str(pcgnn_root))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for module in list(sys.modules):
            if module == "recbole" or module.startswith("recbole."):
                del sys.modules[module]


def run_pcgnn_forward_experiment(max_train_examples: int, max_eval_examples: int) -> dict[str, object]:
    import torch

    train_df = pd.read_pickle(SPLIT_ROOT / "static_train.pkl")
    val_df = pd.read_pickle(SPLIT_ROOT / "static_val.pkl")
    val_cold = val_df[val_df["_split_source"].eq("strict_item_cold_val")]

    with local_pcgnn_recbole(PCGNN_ROOT):
        from recbole.config import Config
        from recbole.data import Interaction, create_dataset, data_preparation
        from recbole.utils import get_model

        config = Config(
            model="kg_model",
            dataset="mooccube_strict_seed2025_smoke",
            config_file_list=["recbole_mooccube_strict_seed2025_smoke.yaml"],
            config_dict=pcgnn_smoke_config_overrides(),
        )
        dataset = create_dataset(config)
        train_data, _, _ = data_preparation(config, dataset)
        token_map = {str(k): int(v) for k, v in dataset.field2token_id["item_id"].items()}
        train_rows = rows_from_frame(train_df)
        val_rows = rows_from_frame(val_cold)
        train_examples = build_training_sequence_examples(train_rows, token_map, config["MAX_ITEM_LIST_LENGTH"], max_train_examples)
        eval_examples = build_strict_sequence_examples(
            train_rows,
            val_rows,
            token_map,
            config["MAX_ITEM_LIST_LENGTH"],
            max_eval_examples,
        )
        model = get_model(config["model"])(config, train_data).to("cpu")
        model.train()
        train_seq, train_len, train_target = tensorize_examples(train_examples, config["MAX_ITEM_LIST_LENGTH"])
        interaction = Interaction(
            {
                model.ITEM_SEQ: train_seq,
                model.ITEM_SEQ_LEN: train_len,
                model.ITEM_ID: train_target,
            }
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss = model.calculate_rs_loss(interaction)
        loss.backward()
        optimizer.step()

        model.eval()
        eval_seq, eval_len, eval_target = tensorize_examples(eval_examples, config["MAX_ITEM_LIST_LENGTH"])
        eval_interaction = Interaction({model.ITEM_SEQ: eval_seq, model.ITEM_SEQ_LEN: eval_len})
        with torch.no_grad():
            scores = model.full_sort_predict(eval_interaction)
        for row_idx, example in enumerate(eval_examples):
            for seen in example["history"]:
                scores[row_idx, int(seen)] = -1e9
            scores[row_idx, 0] = -1e9
        top10 = torch.topk(scores, k=min(10, scores.shape[1]), dim=1).indices
        hits = []
        ndcgs = []
        ranks = []
        for row_idx, target in enumerate(eval_target.tolist()):
            order = torch.argsort(scores[row_idx], descending=True)
            rank = int((order == target).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            hit = 1.0 if target in top10[row_idx].tolist() else 0.0
            hits.append(hit)
            ndcgs.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

    return {
        "status": "ok",
        "train_sequence_examples": len(train_examples),
        "eval_sequence_examples": len(eval_examples),
        "loss_after_one_step": float(loss.detach().cpu().item()),
        "score_shape": list(scores.shape),
        "sample_recall_at_10": float(np.mean(hits)) if hits else 0.0,
        "sample_ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "median_target_rank": float(np.median(ranks)) if ranks else None,
        "interpretation": "Forward/loss/full-sort smoke only; metrics are not publishable because this is one mini-batch on a capped smoke dataset.",
    }


def load_upgpr_item_relations(processed_dir: Path) -> dict[int, dict[str, set[str]]]:
    concepts = (processed_dir / "course_concepts.txt").read_text(encoding="utf-8").splitlines()
    teachers = (processed_dir / "course_teachers.txt").read_text(encoding="utf-8").splitlines()
    schools = (processed_dir / "course_school.txt").read_text(encoding="utf-8").splitlines()
    out: dict[int, dict[str, set[str]]] = {}
    n_items = max(len(concepts), len(teachers), len(schools))
    for item in range(n_items):
        out[item] = {
            "concept": set(concepts[item].split()) if item < len(concepts) and concepts[item] else set(),
            "teacher": set(teachers[item].split()) if item < len(teachers) and teachers[item] else set(),
            "school": set(schools[item].split()) if item < len(schools) and schools[item] else set(),
        }
    return out


def run_upgpr_coverage_experiment() -> dict[str, object]:
    processed_dir = ADAPT_ROOT / "upgpr" / "processed_files"
    train_pairs = [tuple(map(int, line.split())) for line in (processed_dir / "train.txt").read_text().splitlines() if line.strip()]
    val_pairs = [tuple(map(int, line.split())) for line in (processed_dir / "validation.txt").read_text().splitlines() if line.strip()]
    test_pairs = [tuple(map(int, line.split())) for line in (processed_dir / "test.txt").read_text().splitlines() if line.strip()]
    history: dict[int, set[int]] = defaultdict(set)
    for user, item in train_pairs:
        history[user].add(item)
    item_rel = load_upgpr_item_relations(processed_dir)
    val = compute_relation_reachability(history, val_pairs, item_rel)
    test = compute_relation_reachability(history, test_pairs, item_rel)
    return {
        "status": "ok",
        "train_pairs": len(train_pairs),
        "validation": val,
        "test": test,
        "interpretation": "Coverage measures whether a target cold item shares concept/teacher/school with any train-history item; it is an upper-bound proxy for path-based candidate reachability, not trained UPGPR accuracy.",
    }


def run_msec_matrix_experiment() -> dict[str, object]:
    data_dir = ADAPT_ROOT / "msec" / "data"
    names = ["train_uc.npy", "val_uc.npy", "train_uv.npy", "ck.npy", "course_video.npy", "video_concept.npy"]
    matrices = {}
    for name in names:
        arr = np.load(data_dir / name, mmap_mode="r")
        nonzero = int(arr.sum())
        total = int(np.prod(arr.shape))
        matrices[name] = {
            "shape": list(arr.shape),
            "nonzero": nonzero,
            "density": nonzero / total if total else 0.0,
        }
    return {
        "status": "blocked_by_dependency",
        "dependency": "dgl missing in current py.bat environment",
        "matrices": matrices,
        "interpretation": "MSEC matrix inputs are structurally available; official graph/model execution requires DGL and evaluator replacement.",
    }


def run_kgan_loader_experiment() -> dict[str, object]:
    src_data = ROOT / "paper_aaai27" / "baseline_sources" / "_prepared_smoke" / "mooccube_seed2025" / "kgan" / "data" / "course_strict_seed2025"
    dst_data = KGAN_ROOT / "data" / "course_strict_seed2025"
    dst_data.mkdir(parents=True, exist_ok=True)
    for name in ["ratings_final.txt", "kg_final.txt"]:
        shutil.copy2(src_data / name, dst_data / name)
    for name in ["ratings_final.npy", "kg_final.npy"]:
        path = dst_data / name
        if path.exists():
            path.unlink()
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    os.chdir(KGAN_ROOT / "src")
    sys.path.insert(0, str(KGAN_ROOT / "src"))
    try:
        from data_loader import load_data

        args = SimpleNamespace(dataset="course_strict_seed2025", n_hop=2, n_memory=4)
        train_data, test_data, n_entity, n_relation, aggregate_set, relation_set = load_data(args)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        if "data_loader" in sys.modules:
            del sys.modules["data_loader"]
    return {
        "status": "ok_loader_only",
        "train_shape": list(train_data.shape),
        "test_shape": list(test_data.shape),
        "n_entity": int(n_entity),
        "n_relation": int(n_relation),
        "relation_set": len(relation_set),
        "aggregate_users": len(aggregate_set),
        "interpretation": "Official loader accepts adapted data but randomly splits ratings_final.txt, so it is not strict-protocol-safe without a custom loader/evaluator.",
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Priority Baseline Concrete Experiment Analysis",
        "",
        "## Summary",
        "",
        "| Priority | Candidate | Concrete experiment | Status | Main finding |",
        "|---:|---|---|---|---|",
    ]
    for row in report["summary"]:
        lines.append(f"| {row['priority']} | {row['candidate']} | {row['experiment']} | {row['status']} | {row['finding']} |")
    lines.extend(["", "## Details", ""])
    for key, value in report["details"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2))
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pcgnn-train-examples", type=int, default=32)
    parser.add_argument("--pcgnn-eval-examples", type=int, default=32)
    args = parser.parse_args()

    details: dict[str, object] = {}
    summary: list[dict[str, object]] = []

    try:
        details["PCGNN"] = run_pcgnn_forward_experiment(args.pcgnn_train_examples, args.pcgnn_eval_examples)
        summary.append({
            "priority": 1,
            "candidate": "PCGNN",
            "experiment": "one-step loss + strict-history full-sort smoke",
            "status": details["PCGNN"]["status"],
            "finding": f"loss={details['PCGNN']['loss_after_one_step']:.4f}, eval_cases={details['PCGNN']['eval_sequence_examples']}, median_rank={details['PCGNN']['median_target_rank']}",
        })
    except Exception as exc:
        details["PCGNN"] = {"status": "failed", "error": type(exc).__name__, "message": str(exc)}
        summary.append({"priority": 1, "candidate": "PCGNN", "experiment": "one-step loss + strict-history full-sort smoke", "status": "failed", "finding": str(exc)})

    details["UPGPR"] = run_upgpr_coverage_experiment()
    summary.append({
        "priority": 2,
        "candidate": "UPGPR",
        "experiment": "strict split path-reachability proxy",
        "status": details["UPGPR"]["status"],
        "finding": f"val_reach={details['UPGPR']['validation']['target_reachable_rate']:.3f}, test_reach={details['UPGPR']['test']['target_reachable_rate']:.3f}",
    })

    details["MSEC-Rec"] = run_msec_matrix_experiment()
    summary.append({
        "priority": 3,
        "candidate": "MSEC-Rec",
        "experiment": "matrix density and dependency gate",
        "status": details["MSEC-Rec"]["status"],
        "finding": "inputs exist but DGL is missing; sampled evaluator still needs replacement",
    })

    details["KGAN"] = run_kgan_loader_experiment()
    summary.append({
        "priority": 4,
        "candidate": "KGAN",
        "experiment": "official loader on adapted smoke data",
        "status": details["KGAN"]["status"],
        "finding": f"train={details['KGAN']['train_shape']}, test={details['KGAN']['test_shape']}, random split remains",
    })

    report = {"summary": summary, "details": details}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "priority_baseline_experiments.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.out_dir / "priority_baseline_experiments.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
