from typing import Callable, Dict, Optional, Tuple

import torch


def compute_ranking_metrics(
    scores: torch.Tensor,
    target_indices: torch.Tensor,
    k_list=(5, 10, 20)
) -> Dict[str, float]:
    values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
    return {key: float(val.mean().item()) for key, val in values.items()}


def compute_ranking_metric_values(
    scores: torch.Tensor,
    target_indices: torch.Tensor,
    k_list=(5, 10, 20)
) -> Dict[str, torch.Tensor]:
    batch_size = scores.size(0)
    n_candidates = scores.size(1)
    max_k = min(max(k_list), n_candidates)
    _, topk_idx = torch.topk(scores, k=max_k, dim=1)

    targets = target_indices.view(-1, 1)
    results = {}
    for k in k_list:
        preds = topk_idx[:, :k]
        hits = (preds == targets).any(dim=1).float()
        results[f"R@{k}"] = hits

        rk = (preds == targets).nonzero(as_tuple=True)
        ndcg_vals = torch.zeros(batch_size, device=scores.device)
        if rk[0].numel() > 0:
            ndcg_vals[rk[0]] = 1.0 / torch.log2(rk[1].float() + 2.0)
        results[f"N@{k}"] = ndcg_vals
    return results


def precompute_item_vectors(
    n_items: int,
    device: torch.device,
    get_item_vectors_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = 2048
) -> torch.Tensor:
    all_vecs = []
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            item_idx = torch.arange(start, end, device=device, dtype=torch.long)
            item_vec = get_item_vectors_fn(item_idx)
            all_vecs.append(torch.nn.functional.normalize(item_vec, dim=1))
    return torch.cat(all_vecs, dim=0)


def evaluate_embedding_ranker(
    loader,
    device: torch.device,
    n_items: int,
    cold_threshold: int,
    get_user_vectors_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    all_item_vectors: torch.Tensor,
    k_list=(5, 10, 20),
    n_neg: int = 200,
    eval_type: str = "cold",
    full_ranking: bool = False,
    user_seen_items: Optional[Dict[int, set]] = None,
    score_adjust_fn: Optional[
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            torch.Tensor
        ]
    ] = None,
    normalize_user: bool = True,
    average_mode: str = "interaction",
) -> Tuple[Optional[Dict[str, float]], int]:
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    accum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in k_list}
    total_samples = 0
    item_accum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    item_counts: Dict[int, int] = {}
    seen_tensor_cache: Dict[int, Optional[torch.Tensor]] = {}

    with torch.no_grad():
        all_item_idx = torch.arange(n_items, device=device, dtype=torch.long)

        for batch, pop in loader:
            if eval_type == "cold":
                mask = pop < cold_threshold
            elif eval_type == "hot":
                mask = pop >= cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = int(mask.sum().item())
            if n_sel < 1:
                continue

            batch_sel = {k: v[mask].to(device) for k, v in batch.items()}
            pop_sel = pop[mask].to(device)
            u = batch_sel["u"]
            i = batch_sel["i"]
            user_ids = [int(x) for x in u.detach().cpu().tolist()]

            if user_seen_items is not None:
                for uid in user_ids:
                    if uid in seen_tensor_cache:
                        continue
                    seen_items = user_seen_items.get(uid)
                    if seen_items:
                        seen_idx = [x for x in seen_items if 0 <= x < n_items]
                        seen_tensor_cache[uid] = (
                            torch.tensor(seen_idx, dtype=torch.long, device=device)
                            if seen_idx else None
                        )
                    else:
                        seen_tensor_cache[uid] = None

            z_u = get_user_vectors_fn(batch_sel)
            if normalize_user:
                z_u = torch.nn.functional.normalize(z_u, dim=1)

            if full_ranking:
                scores = torch.mm(z_u, all_item_vectors.t())
                if user_seen_items is not None:
                    row_idx = torch.arange(n_sel, device=device)
                    target_scores = scores[row_idx, i].clone()
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache.get(uid)
                        if seen_idx is not None and seen_idx.numel() > 0:
                            scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                cand_idx = None
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                avail_counts = []
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache.get(uid) if user_seen_items is not None else None
                    if seen_idx is None:
                        avail = n_items - 1
                    else:
                        seen_ex_tgt = int((seen_idx != i[row]).sum().item())
                        avail = n_items - 1 - seen_ex_tgt
                    avail_counts.append(max(1, avail))

                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)
                for row, uid in enumerate(user_ids):
                    forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                    forbidden[i[row]] = True
                    seen_idx = seen_tensor_cache.get(uid) if user_seen_items is not None else None
                    if seen_idx is not None and seen_idx.numel() > 0:
                        forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]

                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                # Randomize candidate order to avoid positional tie-bias
                # when many scores are identical or nearly identical.
                perm = torch.argsort(
                    torch.rand(n_sel, cand_idx.size(1), device=device),
                    dim=1
                )
                cand_idx = cand_idx.gather(1, perm)
                cand_vecs = all_item_vectors[cand_idx]
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                target_indices = (cand_idx == i.unsqueeze(1)).nonzero(as_tuple=True)[1].view(-1)

            if score_adjust_fn is not None:
                uid_t = torch.tensor(user_ids, dtype=torch.long, device=device)
                scores = score_adjust_fn(scores, uid_t, i, pop_sel, cand_idx)

            batch_values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
            if average_mode == "item_macro":
                item_ids = [int(x) for x in i.detach().cpu().tolist()]
                for row, item_id in enumerate(item_ids):
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    for key, values in batch_values.items():
                        per_item = item_accum[key]
                        per_item[item_id] = per_item.get(item_id, 0.0) + float(values[row].detach().cpu().item())
            else:
                for k, values in batch_values.items():
                    accum[k] += float(values.sum().detach().cpu().item())
            total_samples += n_sel

    if total_samples < 1:
        return None, 0
    if average_mode == "item_macro":
        if not item_counts:
            return None, 0
        macro = {}
        for key, per_item in item_accum.items():
            item_values = [
                per_item.get(item_id, 0.0) / count
                for item_id, count in item_counts.items()
                if count > 0
            ]
            macro[key] = sum(item_values) / max(1, len(item_values))
        return macro, len(item_counts)
    return {k: v / total_samples for k, v in accum.items()}, total_samples


def print_final_report(
    eval_n_neg: int,
    metrics_keys,
    sample_cold: Dict[str, float],
    sample_hot: Dict[str, float],
    full_cold: Dict[str, float],
    full_hot: Dict[str, float],
    count_sample_cold: int,
    count_sample_hot: int,
    count_full_cold: int,
    count_full_hot: int,
    title: str
) -> None:
    print("\n" + "=" * 90)
    print(f"         FINAL REPORT: 采样评估 (1+{eval_n_neg}) vs 全库排名 ({title})")
    print("=" * 90)
    print(f"{'Metric':<10} | {'采样 Cold':<12} | {'采样 Hot':<12} | {'全库 Cold':<12} | {'全库 Hot':<12}")
    print("-" * 90)
    for m in metrics_keys:
        sc = sample_cold.get(m, 0.0)
        sh = sample_hot.get(m, 0.0)
        fc = full_cold.get(m, 0.0)
        fh = full_hot.get(m, 0.0)
        print(f"{m:<10} | {sc:<12.4f} | {sh:<12.4f} | {fc:<12.4f} | {fh:<12.4f}")
    print("-" * 90)
    print(f"采样 Samples: Cold={count_sample_cold}, Hot={count_sample_hot}")
    print(f"全库 Samples: Cold={count_full_cold}, Hot={count_full_hot}")
    print("=" * 90)
