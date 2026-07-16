"""Validation-only inference-step causal probe from one frozen LIRA checkpoint."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as protocol
from lira.protocol_adapter import LIRAProtocolAdapter


def evaluate(checkpoint: Path, steps: int, output: Path, min_fit: float = 0.05) -> dict:
    os.environ.update(
        {
            "USIM_STATIC_SPLIT_MODE": "strict_item_cold_balanced",
            "USIM_STATIC_SEED": "2025",
            "USIM_STATIC_TRAIN_RATIO": "0.8",
            "USIM_COLD_THRESHOLD": "1",
            "USIM_DISABLE_LLM_SCORE": "1",
            "USIM_USE_REFINED_EVAL": "1",
            "USIM_FB_COURSE_MATCH_EXCLUDE_TARGET": "1",
            "LIRA_MIN_FIT": str(float(min_fit)),
        }
    )
    data_dir = Path("processed_data_hin_clean_pop5")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    frame = pd.read_pickle(data_dir / "stream_data.pkl")
    content = torch.load(data_dir / "content_emb.pt", map_location="cpu")
    cfg = protocol.Fast3Config(meta["n_users"], meta["n_items"], content.shape[1])
    cfg.usim_steps = int(steps)
    cfg.use_usim_refined_eval = True
    cfg.early_stop_average_mode = "item_macro"
    train, validation, _, _ = protocol._static_split_df(frame)
    train, validation, _, train_popularity = protocol._apply_train_popularity(
        train, validation, validation.copy(deep=True), cfg
    )
    train_seen = protocol._add_user_seen_from_df({}, train)
    artifacts, _ = protocol.build_course_artifacts(
        frame,
        cfg.n_items,
        relation_dir="MOOCCube/relations",
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    device = protocol._resolve_torch_device()
    model = LIRAProtocolAdapter(cfg, content).to(device)
    model.device = device
    model.set_course_artifacts(artifacts)
    model.set_feedback_item_stats(train_popularity)
    model.set_user_seen_index(train_seen)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("es_best_state") or payload["model_state"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    loader = DataLoader(
        protocol.StreamDataset(validation, {}),
        batch_size=2048,
        shuffle=False,
        collate_fn=protocol.collate_fn,
    )
    item_bank = protocol.build_eval_item_vecs(model, device, {}, item_batch=1024)
    cold, cold_count = protocol.evaluate_usim(
        model,
        loader,
        device,
        {},
        k_list=[5, 10, 20],
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=train_seen,
        all_item_vecs=item_bank,
        average_mode="item_macro",
    )
    hot, hot_count = protocol.evaluate_usim(
        model,
        loader,
        device,
        {},
        k_list=[5, 10, 20],
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=True,
        user_seen_items=train_seen,
        all_item_vecs=item_bank,
        average_mode="item_macro",
    )
    result = {
        "evaluation_target": "validation",
        "checkpoint": str(checkpoint),
        "checkpoint_best_epoch": payload.get("es_best", {}).get("epoch"),
        "inference_steps": int(steps),
        "min_fit": float(min_fit),
        "cold_count": int(cold_count),
        "hot_count": int(hot_count),
        "cold": cold,
        "hot": hot,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=[0, 1, 3], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-fit", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.steps, args.output, args.min_fit), indent=2))


if __name__ == "__main__":
    main()
