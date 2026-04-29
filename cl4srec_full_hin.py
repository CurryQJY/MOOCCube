import os
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cl4srec_static_hin import CL4SRecStaticModel
from hin_data_common import (
    add_user_seen_from_df,
    load_hin_processed,
    setup_seed,
    split_dataframe_by_periods,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report
from hhcor_static_hin import (
    HHCoRDataset,
    build_history_tensor,
    collate_hhcor,
    _update_histories_from_df,
)


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int = 768):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = int(os.environ.get("CL4SREC_EMB_DIM", "128"))
        self.hidden_dim = int(os.environ.get("CL4SREC_HIDDEN_DIM", "256"))
        self.n_heads = int(os.environ.get("CL4SREC_N_HEADS", "2"))
        self.n_layers = int(os.environ.get("CL4SREC_N_LAYERS", "2"))
        self.dropout = float(os.environ.get("CL4SREC_DROPOUT", "0.20"))
        self.user_hist_len = int(os.environ.get("CL4SREC_USER_HIST_LEN", "30"))
        self.content_weight = float(os.environ.get("CL4SREC_CONTENT_WEIGHT", "0.35"))

        self.batch_size = int(os.environ.get("CL4SREC_BATCH_SIZE", "2048"))
        self.n_epochs = int(os.environ.get("CL4SREC_FULL_EPOCHS", "3"))
        self.lr = float(os.environ.get("CL4SREC_LR", "5e-4"))
        self.temperature = float(os.environ.get("CL4SREC_TEMP", "0.10"))

        self.cl_weight = float(os.environ.get("CL4SREC_CL_WEIGHT", "0.20"))
        self.cl_temp = float(os.environ.get("CL4SREC_CL_TEMP", "0.20"))
        self.aug_mask_ratio = float(os.environ.get("CL4SREC_AUG_MASK_RATIO", "0.20"))
        self.aug_crop_ratio = float(os.environ.get("CL4SREC_AUG_CROP_RATIO", "0.20"))
        self.aug_reorder_ratio = float(os.environ.get("CL4SREC_AUG_REORDER_RATIO", "0.20"))

        self.cold_threshold = int(os.environ.get("CL4SREC_COLD_THRESHOLD", "5"))
        self.eval_n_neg = int(os.environ.get("CL4SREC_EVAL_N_NEG", "200"))
        self.warmup_periods = int(os.environ.get("CL4SREC_FULL_WARMUP", "2"))
        self.period_type = os.environ.get("CL4SREC_PERIOD_TYPE", "M")
        self.use_cumulative = os.environ.get("CL4SREC_USE_CUMULATIVE", "1") == "1"


def _weighted_add(dst: Dict[str, float], src: Dict[str, float], weight: int):
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v * weight


def _weighted_avg(src: Dict[str, float], count: int):
    if count < 1:
        return {}
    return {k: v / count for k, v in src.items()}


def main():
    setup_seed(2025)
    print("Loading data from processed_data_hin ...")
    meta, df, content_emb = load_hin_processed("processed_data_hin")
    cfg = Config(meta["n_users"], meta["n_items"], content_dim=content_emb.shape[1])
    periods = split_dataframe_by_periods(df, period_type=cfg.period_type)
    print(
        f"Streaming periods: {len(periods)} | warmup={cfg.warmup_periods} | "
        f"cumulative_train={cfg.use_cumulative} | epochs/period={cfg.n_epochs}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CL4SRecStaticModel(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"Model: CL4SRec stream | device={device}")

    k_list = [5, 10, 20]
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]

    accum_sample_cold = {k: 0.0 for k in metrics_keys}
    accum_sample_hot = {k: 0.0 for k in metrics_keys}
    accum_full_cold = {k: 0.0 for k in metrics_keys}
    accum_full_hot = {k: 0.0 for k in metrics_keys}
    count_sample_cold = 0
    count_sample_hot = 0
    count_full_cold = 0
    count_full_hot = 0

    user_seen_items = {}
    user_histories = {}
    accumulated_dfs = []

    for t, p_df in enumerate(periods):
        print(f"\n[Period {t + 1}/{len(periods)}] size={len(p_df)}")

        eval_hist, _ = build_history_tensor(
            p_df, base_histories=user_histories, max_len=cfg.user_hist_len, update_histories=False
        )
        eval_ds = HHCoRDataset(p_df, eval_hist)
        eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hhcor)

        if t >= cfg.warmup_periods:
            model.eval()
            with torch.no_grad():
                item_bank = model.get_item_bank().detach()
                get_user_fn = lambda b: model.encode_users(b["u"], b["hist"], item_bank)

                sample_cold, n_sc = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
                    user_seen_items=user_seen_items
                )
                sample_hot, n_sh = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
                    user_seen_items=user_seen_items
                )
                full_cold, n_fc = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
                    user_seen_items=user_seen_items
                )
                full_hot, n_fh = evaluate_embedding_ranker(
                    eval_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
                    k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
                    user_seen_items=user_seen_items
                )

            sample_cold = sample_cold or {}
            sample_hot = sample_hot or {}
            full_cold = full_cold or {}
            full_hot = full_hot or {}

            _weighted_add(accum_sample_cold, sample_cold, n_sc)
            _weighted_add(accum_sample_hot, sample_hot, n_sh)
            _weighted_add(accum_full_cold, full_cold, n_fc)
            _weighted_add(accum_full_hot, full_hot, n_fh)
            count_sample_cold += n_sc
            count_sample_hot += n_sh
            count_full_cold += n_fc
            count_full_hot += n_fh

            print(
                "  Eval: "
                f"sample_cold_R@10={sample_cold.get('R@10', 0.0):.4f}, "
                f"sample_hot_R@10={sample_hot.get('R@10', 0.0):.4f}, "
                f"full_cold_R@10={full_cold.get('R@10', 0.0):.4f}, "
                f"full_hot_R@10={full_hot.get('R@10', 0.0):.4f}"
            )
        else:
            print("  Warmup period: skip evaluation")

        if cfg.use_cumulative:
            accumulated_dfs.append(p_df)
            train_df = pd.concat(accumulated_dfs, ignore_index=True)
            train_hist, _ = build_history_tensor(
                train_df, base_histories={}, max_len=cfg.user_hist_len, update_histories=True
            )
        else:
            train_df = p_df
            train_hist, _ = build_history_tensor(
                train_df, base_histories=user_histories, max_len=cfg.user_hist_len, update_histories=True
            )

        if len(train_df) < 1:
            add_user_seen_from_df(user_seen_items, p_df)
            _update_histories_from_df(user_histories, p_df)
            continue

        train_ds = HHCoRDataset(train_df, train_hist)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_hhcor)

        model.train()
        for ep in range(cfg.n_epochs):
            total_loss = 0.0
            steps = 0
            for batch, _ in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1
            print(f"  Train epoch [{ep + 1}/{cfg.n_epochs}] loss={total_loss / max(1, steps):.4f}")

        add_user_seen_from_df(user_seen_items, p_df)
        _update_histories_from_df(user_histories, p_df)

    final_sample_cold = _weighted_avg(accum_sample_cold, count_sample_cold)
    final_sample_hot = _weighted_avg(accum_sample_hot, count_sample_hot)
    final_full_cold = _weighted_avg(accum_full_cold, count_full_cold)
    final_full_hot = _weighted_avg(accum_full_hot, count_full_hot)

    print_final_report(
        eval_n_neg=cfg.eval_n_neg,
        metrics_keys=metrics_keys,
        sample_cold=final_sample_cold,
        sample_hot=final_sample_hot,
        full_cold=final_full_cold,
        full_hot=final_full_hot,
        count_sample_cold=count_sample_cold,
        count_sample_hot=count_sample_hot,
        count_full_cold=count_full_cold,
        count_full_hot=count_full_hot,
        title="CL4SRec Streaming HIN"
    )

    out = {
        "sample_cold": final_sample_cold,
        "sample_hot": final_sample_hot,
        "full_cold": final_full_cold,
        "full_hot": final_full_hot,
        "count_sample_cold": count_sample_cold,
        "count_sample_hot": count_sample_hot,
        "count_full_cold": count_full_cold,
        "count_full_hot": count_full_hot,
        "periods": len(periods),
        "warmup_periods": cfg.warmup_periods,
        "use_cumulative": cfg.use_cumulative,
    }
    pd.DataFrame([out]).to_json("cl4srec_full_result.json", orient="records", force_ascii=False)
    print("Saved: cl4srec_full_result.json")


if __name__ == "__main__":
    main()
