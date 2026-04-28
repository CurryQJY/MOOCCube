import copy
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hin_data_common import (
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    load_hin_processed,
    setup_seed,
    static_split_df,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report
from hhcor_static_hin import (
    HHCoRDataset,
    build_history_tensor,
    collate_hhcor,
    _clone_user_histories,
    _update_histories_from_df,
)


class Config:
    def __init__(self, n_users: int, n_items: int, content_dim: int = 768):
        self.n_users = n_users
        self.n_items = n_items
        self.content_dim = content_dim

        self.emb_dim = int(os.environ.get("BERT4REC_EMB_DIM", "128"))
        self.hidden_dim = int(os.environ.get("BERT4REC_HIDDEN_DIM", "256"))
        self.n_heads = int(os.environ.get("BERT4REC_N_HEADS", "4"))
        self.n_layers = int(os.environ.get("BERT4REC_N_LAYERS", "2"))
        self.dropout = float(os.environ.get("BERT4REC_DROPOUT", "0.20"))
        self.user_hist_len = int(os.environ.get("BERT4REC_USER_HIST_LEN", "30"))
        self.content_weight = float(os.environ.get("BERT4REC_CONTENT_WEIGHT", "0.40"))

        self.batch_size = int(os.environ.get("BERT4REC_BATCH_SIZE", "2048"))
        self.n_epochs = int(os.environ.get("BERT4REC_STATIC_EPOCHS", "8"))
        self.lr = float(os.environ.get("BERT4REC_LR", "5e-4"))
        self.temperature = float(os.environ.get("BERT4REC_TEMP", "0.10"))

        self.cold_threshold = int(os.environ.get("BERT4REC_COLD_THRESHOLD", "5"))
        self.eval_n_neg = int(os.environ.get("BERT4REC_EVAL_N_NEG", "200"))
        self.static_seed = int(os.environ.get("BERT4REC_STATIC_SEED", "2025"))
        self.train_ratio = float(os.environ.get("BERT4REC_STATIC_TRAIN_RATIO", "0.8"))
        self.val_ratio = float(os.environ.get("BERT4REC_STATIC_VAL_RATIO", "0.1"))


class BERT4RecStaticModel(nn.Module):
    def __init__(self, cfg: Config, content_emb: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        # token ids: 0=PAD, 1..n_items=item, n_items+1=MASK
        self.user_emb = nn.Embedding(cfg.n_users, cfg.emb_dim)
        self.item_tok_emb = nn.Embedding(cfg.n_items + 2, cfg.emb_dim, padding_idx=0)
        self.item_con_emb = nn.Embedding.from_pretrained(content_emb, freeze=True)
        self.pos_emb = nn.Embedding(cfg.user_hist_len, cfg.emb_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_tok_emb.weight)

        self.content_proj = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
            nn.LayerNorm(cfg.emb_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.user_gate = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.Sigmoid(),
        )
        self.user_norm = nn.LayerNorm(cfg.emb_dim)

    def _hist_to_tokens(self, hist_idx: torch.Tensor) -> torch.Tensor:
        token = torch.zeros_like(hist_idx, dtype=torch.long)
        valid = (hist_idx >= 0) & (hist_idx < self.cfg.n_items)
        token[valid] = hist_idx[valid] + 1
        is_mask = hist_idx == self.cfg.n_items
        token[is_mask] = self.cfg.n_items + 1
        return token

    def get_item_bank(self) -> torch.Tensor:
        item_id = self.item_tok_emb.weight[1:self.cfg.n_items + 1]
        item_con = self.content_proj(self.item_con_emb.weight)
        return F.normalize(item_id + self.cfg.content_weight * item_con, dim=1)

    def encode_users(self, u_idx: torch.Tensor, hist_idx: torch.Tensor, item_bank: torch.Tensor) -> torch.Tensor:
        device = item_bank.device
        bsz, seq_len = hist_idx.shape
        emb_dim = item_bank.size(1)

        tokens = self._hist_to_tokens(hist_idx)
        non_pad = tokens != 0

        seq_emb = self.item_tok_emb(tokens)

        ext_bank = torch.zeros(self.cfg.n_items + 2, emb_dim, device=device)
        ext_bank[1:self.cfg.n_items + 1] = item_bank
        seq_emb = seq_emb + ext_bank[tokens]

        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos_idx))

        key_padding_mask = tokens == 0
        fully_padded = key_padding_mask.all(dim=1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded, 0] = False
            seq_emb = seq_emb.clone()
            seq_emb[fully_padded, 0, :] = 0.0

        seq_out = self.encoder(seq_emb, src_key_padding_mask=key_padding_mask)
        lengths = non_pad.sum(dim=1)
        last_pos = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, emb_dim)
        seq_vec = seq_out.gather(1, last_pos).squeeze(1)
        seq_vec = seq_vec.clone()
        seq_vec[lengths == 0] = 0.0

        u_id = self.user_emb(u_idx)
        gate = self.user_gate(torch.cat([u_id, seq_vec], dim=1))
        user_vec = self.user_norm(gate * u_id + (1.0 - gate) * seq_vec)
        return F.normalize(user_vec, dim=1)

    def forward(self, batch):
        item_bank = self.get_item_bank()
        z_u = self.encode_users(batch["u"], batch["hist"], item_bank)
        z_i = item_bank[batch["i"]]

        logits = torch.matmul(z_u, z_i.t()) / self.cfg.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def main():
    setup_seed(2025)
    print("Loading data from processed_data_hin ...")
    meta, df, content_emb = load_hin_processed("processed_data_hin")
    cfg = Config(meta["n_users"], meta["n_items"], content_dim=content_emb.shape[1])

    train_df, val_df, test_df = static_split_df(
        df,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio
    )
    print(
        f"Static split done: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | "
        f"cold_threshold={cfg.cold_threshold}, eval_n_neg={cfg.eval_n_neg}"
    )

    train_hist, train_histories = build_history_tensor(
        train_df, base_histories={}, max_len=cfg.user_hist_len, update_histories=True
    )
    val_hist, _ = build_history_tensor(
        val_df, base_histories=train_histories, max_len=cfg.user_hist_len, update_histories=False
    )
    train_val_histories = _clone_user_histories(train_histories)
    _update_histories_from_df(train_val_histories, val_df)
    test_hist, _ = build_history_tensor(
        test_df, base_histories=train_val_histories, max_len=cfg.user_hist_len, update_histories=False
    )

    train_ds = HHCoRDataset(train_df, train_hist)
    val_ds = HHCoRDataset(val_df, val_hist)
    test_ds = HHCoRDataset(test_df, test_hist)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_hhcor)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hhcor)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_hhcor)

    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)
    add_user_seen_from_df(test_seen, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT4RecStaticModel(cfg, content_emb).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"Model: BERT4Rec static | device={device} | epochs={cfg.n_epochs}")

    best_val = -1.0
    best_epoch = -1
    best_state = None
    k_list = [5, 10, 20]

    for epoch in range(cfg.n_epochs):
        model.train()
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

        avg_loss = total_loss / max(1, steps)

        model.eval()
        with torch.no_grad():
            item_bank = model.get_item_bank().detach()
            get_user_fn = lambda b: model.encode_users(b["u"], b["hist"], item_bank)
            val_full_cold, _ = evaluate_embedding_ranker(
                val_loader,
                device=device,
                n_items=cfg.n_items,
                cold_threshold=cfg.cold_threshold,
                get_user_vectors_fn=get_user_fn,
                all_item_vectors=item_bank,
                k_list=k_list,
                n_neg=cfg.eval_n_neg,
                eval_type="cold",
                full_ranking=True,
                user_seen_items=train_seen
            )
            val_key = val_full_cold.get("N@10", 0.0) if val_full_cold else 0.0
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch + 1}/{cfg.n_epochs}] loss={avg_loss:.4f} | val_full_cold_N@10={val_key:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restore best epoch={best_epoch}, val_full_cold_N@10={best_val:.4f}")

    model.eval()
    with torch.no_grad():
        item_bank = model.get_item_bank().detach()
        get_user_fn = lambda b: model.encode_users(b["u"], b["hist"], item_bank)

        sample_cold, n_sc = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=False,
            user_seen_items=test_seen
        )
        sample_hot, n_sh = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=False,
            user_seen_items=test_seen
        )
        full_cold, n_fc = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="cold", full_ranking=True,
            user_seen_items=test_seen
        )
        full_hot, n_fh = evaluate_embedding_ranker(
            test_loader, device, cfg.n_items, cfg.cold_threshold, get_user_fn, item_bank,
            k_list=k_list, n_neg=cfg.eval_n_neg, eval_type="hot", full_ranking=True,
            user_seen_items=test_seen
        )

    sample_cold = sample_cold or {}
    sample_hot = sample_hot or {}
    full_cold = full_cold or {}
    full_hot = full_hot or {}
    metrics_keys = [f"{m}@{k}" for m in ["R", "N"] for k in k_list]

    print_final_report(
        eval_n_neg=cfg.eval_n_neg,
        metrics_keys=metrics_keys,
        sample_cold=sample_cold,
        sample_hot=sample_hot,
        full_cold=full_cold,
        full_hot=full_hot,
        count_sample_cold=n_sc,
        count_sample_hot=n_sh,
        count_full_cold=n_fc,
        count_full_hot=n_fh,
        title="BERT4Rec Static HIN"
    )

    out = {
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
    }
    pd.DataFrame([out]).to_json("bert4rec_static_result.json", orient="records", force_ascii=False)
    print("Saved: bert4rec_static_result.json")


if __name__ == "__main__":
    main()
