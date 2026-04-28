import os
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hin_data_common import (
    add_user_seen_from_df,
    load_hin_processed,
    setup_seed,
    split_dataframe_by_periods,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report
from hhcor_static_hin import (
    HHCoRDataset,
    build_concept_adj,
    build_history_tensor,
    build_item_cluster_ids,
    build_item_course_mapping,
    build_prereq_adj,
    collate_hhcor,
    _update_histories_from_df,
)
from light_path_static_hin import LightPathStaticModel


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int = 768):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = 128
        self.hidden_dim = 256
        self.batch_size = 2048
        self.n_epochs = int(os.environ.get("LIGHT_FULL_EPOCHS", "3"))
        self.lr = 5e-4
        self.temperature = 0.10

        self.cold_threshold = int(os.environ.get("LIGHT_COLD_THRESHOLD", "5"))
        self.eval_n_neg = int(os.environ.get("LIGHT_EVAL_N_NEG", "200"))

        self.warmup_periods = int(os.environ.get("LIGHT_FULL_WARMUP", "2"))
        self.period_type = os.environ.get("LIGHT_PERIOD_TYPE", "M")
        self.use_cumulative = os.environ.get("LIGHT_USE_CUMULATIVE", "1") == "1"

        self.user_hist_len = int(os.environ.get("LIGHT_USER_HIST_LEN", "30"))
        self.n_heads = int(os.environ.get("LIGHT_N_HEADS", "2"))
        self.n_layers = int(os.environ.get("LIGHT_N_LAYERS", "2"))
        self.dropout = float(os.environ.get("LIGHT_DROPOUT", "0.20"))

        self.graph_topk = int(os.environ.get("LIGHT_GRAPH_TOPK", "20"))
        self.graph_mix_weight = float(os.environ.get("LIGHT_GRAPH_MIX_WEIGHT", "0.25"))

        self.prereq_min_support = int(os.environ.get("LIGHT_PREREQ_MIN_SUPPORT", "10"))
        self.prereq_max_per_item = int(os.environ.get("LIGHT_PREREQ_MAX_PER_ITEM", "8"))
        self.prereq_max_forward = int(os.environ.get("LIGHT_PREREQ_MAX_FORWARD", "20"))

        self.topo_aux_weight = float(os.environ.get("LIGHT_TOPO_AUX_WEIGHT", "0.15"))
        self.cluster_aux_weight = float(os.environ.get("LIGHT_CLUSTER_AUX_WEIGHT", "0.10"))


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

    idx_to_course = build_item_course_mapping(df, cfg.n_items)
    item_cluster_ids, subjects = build_item_cluster_ids(idx_to_course)
    relation_dir = os.path.join("MOOCCube", "relations")
    concept_adj = build_concept_adj(idx_to_course, relation_dir, topk=cfg.graph_topk)
    print(f"Cluster count(subject): {len(subjects)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightPathStaticModel(cfg, content_emb, n_clusters=len(subjects)).to(device)
    zero_adj = torch.zeros(cfg.n_items, cfg.n_items, dtype=torch.float32)
    model.set_graph_buffers(concept_adj, zero_adj, item_cluster_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"Model: LIGHT-path stream | device={device}")

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
                get_user_fn = lambda b: model.encode_users(b["u"], b["hist"], item_bank)[0]

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
        else:
            train_df = p_df

        prereq_adj = build_prereq_adj(
            train_df,
            n_items=cfg.n_items,
            min_support=cfg.prereq_min_support,
            max_per_item=cfg.prereq_max_per_item,
            max_forward=cfg.prereq_max_forward,
            topk=cfg.graph_topk
        )
        model.set_graph_buffers(concept_adj, prereq_adj, item_cluster_ids)

        if cfg.use_cumulative:
            train_hist, _ = build_history_tensor(
                train_df, base_histories={}, max_len=cfg.user_hist_len, update_histories=True
            )
        else:
            train_hist, _ = build_history_tensor(
                train_df, base_histories=user_histories, max_len=cfg.user_hist_len, update_histories=True
            )
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
        title="LIGHT-Style Topology Streaming HIN"
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
        "use_cumulative": cfg.use_cumulative
    }
    pd.DataFrame([out]).to_json("light_path_full_result.json", orient="records", force_ascii=False)
    print("Saved: light_path_full_result.json")


if __name__ == "__main__":
    main()

