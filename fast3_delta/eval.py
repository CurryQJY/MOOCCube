import csv
import os
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from ranking_topk_export import TopKJsonlExporter


def compute_ranking_metrics(scores, target_indices, k_list=(5, 10, 20)):
    values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
    return {key: float(val.mean().item()) for key, val in values.items()}


def compute_ranking_metric_values(scores, target_indices, k_list=(5, 10, 20), topk_indices=None):
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    targets = target_indices.view(-1, 1)
    actual_k = min(max(k_list), num_candidates)
    if topk_indices is None:
        _, topk_indices = torch.topk(scores, actual_k, dim=1)
    elif topk_indices.ndim != 2 or topk_indices.size(0) != batch_size or topk_indices.size(1) < actual_k:
        raise ValueError("topk_indices must cover max(k_list) for every score row")
    else:
        topk_indices = topk_indices[:, :actual_k]
    results = {}
    for k in k_list:
        preds = topk_indices[:, :k]
        hits = (preds == targets).any(dim=1).float()
        hit_ranks = torch.where(preds == targets)
        ndcg_vals = torch.zeros(batch_size, device=scores.device)
        if hit_ranks[1].numel() > 0:
            ranks = hit_ranks[1].float()
            ndcg_vals[hit_ranks[0]] = 1.0 / torch.log2(ranks + 2.0)
        results[f"R@{k}"] = hits
        results[f"N@{k}"] = ndcg_vals
    return results


def lookup_llm_score(llm_scores, item_idx, user_idx=None, allow_pair=True, allow_item=True):
    if llm_scores is None:
        return -1.0
    item_idx = int(item_idx)
    if allow_pair and user_idx is not None:
        pair_score = llm_scores.get((int(user_idx), item_idx))
        if pair_score is not None:
            return float(pair_score)
    if allow_item:
        item_score = llm_scores.get(item_idx)
        if item_score is not None:
            return float(item_score)
    return -1.0


def is_pair_llm_key(key):
    return isinstance(key, tuple) and len(key) == 2


def is_item_llm_key(key):
    return isinstance(key, (int, np.integer))


def count_llm_key_types(llm_scores):
    if not llm_scores:
        return 0, 0
    pair_count = 0
    item_count = 0
    for key in llm_scores:
        if is_pair_llm_key(key):
            pair_count += 1
        elif is_item_llm_key(key):
            item_count += 1
    return pair_count, item_count


def item_mean_llm_scores(llm_scores, keep_pair=False):
    item_scores = {}
    pair_sums = defaultdict(float)
    pair_counts = defaultdict(int)
    if not llm_scores:
        return item_scores

    for key, value in llm_scores.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue

        if is_pair_llm_key(key):
            item_idx = int(key[1])
            pair_sums[item_idx] += score
            pair_counts[item_idx] += 1
            if keep_pair:
                item_scores[(int(key[0]), item_idx)] = score
        elif is_item_llm_key(key):
            item_scores[int(key)] = score

    for item_idx, total in pair_sums.items():
        if item_idx not in item_scores and pair_counts[item_idx] > 0:
            item_scores[item_idx] = total / pair_counts[item_idx]
    return item_scores


def prepare_llm_scores(llm_scores, cfg):
    raw_pair, raw_item = count_llm_key_types(llm_scores)
    raw_total = len(llm_scores) if llm_scores else 0
    mode = str(getattr(cfg, "llm_bank_mode", "item") or "item").strip().lower()
    if mode in {"off", "disable", "disabled"}:
        mode = "none"
    elif mode in {"item", "item_avg", "item_mean", "mean_item"}:
        mode = "item_mean"
    elif mode in {"hybrid", "pair_item", "pair_item_mean"}:
        mode = "hybrid_mean"

    if getattr(cfg, "disable_llm_score", False) or mode == "none" or float(getattr(cfg, "llm_weight", 1.0)) <= 0.0:
        effective_scores = {}
        effective_mode = "none"
    elif mode == "item_mean":
        effective_scores = item_mean_llm_scores(llm_scores, keep_pair=False)
        effective_mode = "item_mean"
    elif mode == "hybrid_mean":
        effective_scores = item_mean_llm_scores(llm_scores, keep_pair=True)
        effective_mode = "hybrid_mean"
    elif mode == "item_only":
        effective_scores = {
            int(key): float(value)
            for key, value in (llm_scores or {}).items()
            if is_item_llm_key(key)
        }
        effective_mode = "item_only"
    else:
        effective_scores = llm_scores or {}
        effective_mode = "pair"

    eff_pair, eff_item = count_llm_key_types(effective_scores)
    summary = {
        "mode": effective_mode,
        "raw_total": raw_total,
        "raw_pair": raw_pair,
        "raw_item": raw_item,
        "effective_total": len(effective_scores),
        "effective_pair": eff_pair,
        "effective_item": eff_item,
    }
    return effective_scores, summary


def build_llm_score_tensor(llm_scores, user_ids, item_ids, device=None):
    values = [
        lookup_llm_score(llm_scores, item_idx, user_idx)
        for user_idx, item_idx in zip(user_ids, item_ids)
    ]
    return torch.tensor(values, dtype=torch.float, device=device)


def resolve_eval_force_cold(model, idx_batch, force_cold):
    if isinstance(force_cold, str):
        mode = force_cold.strip().lower()
        if mode in {"auto", "item", "item_pop"} and getattr(model, "item_popularity", None) is not None:
            return model.item_popularity[idx_batch].to(device=idx_batch.device) < float(model.cfg.cold_threshold)
        if mode in {"true", "cold", "all"}:
            return True
        if mode in {"false", "hot", "none", "off"}:
            return False
    return force_cold


def refined_eval_enabled(model):
    if getattr(model.cfg, "legacy_train_protocol", False):
        return False
    return bool(getattr(model.cfg, "use_usim_refined_eval", True)) and hasattr(
        model,
        "infer_refined_item_vectors",
    )


def strict_cold_item_mask(model, device):
    pop = getattr(model, "item_popularity", None)
    if pop is None:
        return None
    return pop.to(device=device).float().view(-1) < float(getattr(model.cfg, "cold_threshold", 1))


def item_llm_score_batch(llm_scores, item_idx, device):
    if llm_scores is None:
        return torch.full((item_idx.numel(),), -1.0, dtype=torch.float, device=device)
    values = [
        lookup_llm_score(llm_scores, int(idx), allow_pair=False, allow_item=True)
        for idx in item_idx.detach().cpu().tolist()
    ]
    return torch.tensor(values, dtype=torch.float, device=device)


def refined_item_vectors(model, item_idx, llm_s=None, item_batch=1024, force_cold=True):
    if llm_s is None:
        llm_s = torch.full((item_idx.numel(),), -1.0, dtype=torch.float, device=item_idx.device)
    return model.infer_refined_item_vectors(
        item_idx,
        llm_s=llm_s,
        item_batch=item_batch,
        force_cold=force_cold,
    )


def replace_strict_cold_with_refined(model, base_bank, device, llm_scores, item_batch=1024):
    if not refined_eval_enabled(model):
        return base_bank
    cold_mask = strict_cold_item_mask(model, device)
    if cold_mask is None or not bool(cold_mask.any().item()):
        return base_bank

    out = base_bank.clone()
    cold_idx = torch.nonzero(cold_mask, as_tuple=False).view(-1)
    with torch.no_grad():
        for start in range(0, cold_idx.numel(), item_batch):
            idx_batch = cold_idx[start : start + item_batch]
            llm_batch = item_llm_score_batch(llm_scores, idx_batch, device)
            refined = refined_item_vectors(
                model,
                idx_batch,
                llm_s=llm_batch,
                item_batch=item_batch,
                force_cold=True,
            )
            out[idx_batch] = F.normalize(refined, dim=1)
    return out


def build_all_item_vecs(model, device, llm_scores, item_batch=1024, force_cold=True):
    n_items = model.cfg.n_items
    all_item_idx = torch.arange(n_items, device=device)
    bank_mode = getattr(model.cfg, "llm_bank_mode", "none")
    if bank_mode in {"item", "item_mean", "hybrid_mean", "item_only"}:
        all_llm_s = torch.tensor(
            [
                lookup_llm_score(llm_scores, int(idx), allow_pair=False, allow_item=True)
                for idx in all_item_idx
            ],
            dtype=torch.float,
            device=device,
        )
    else:
        all_llm_s = torch.full((n_items,), -1.0, dtype=torch.float, device=device)
    all_item_vecs = []
    with torch.no_grad():
        for start in range(0, n_items, item_batch):
            end = min(start + item_batch, n_items)
            idx_batch = all_item_idx[start:end]
            llm_batch = all_llm_s[start:end]
            force_cold_batch = resolve_eval_force_cold(model, idx_batch, force_cold)
            z_i, _, _ = model.get_item_vector(idx_batch, llm_batch, force_cold=force_cold_batch)
            all_item_vecs.append(F.normalize(z_i, dim=1))
    return torch.cat(all_item_vecs, dim=0)


def build_eval_item_vecs(model, device, llm_scores, item_batch=1024):
    hot_force_cold = False
    cold_force_cold = True
    if getattr(model.cfg, "content_delta_cold_only", False):
        bank_mode = str(getattr(model.cfg, "content_delta_eval_bank_mode", "auto")).strip().lower()
        if bank_mode in {"auto", "item", "item_pop"}:
            hot_force_cold = "auto"
            cold_force_cold = "auto"
        elif bank_mode in {"hot", "none", "off"}:
            cold_force_cold = False
        else:
            cold_force_cold = True

    if getattr(model.cfg, "legacy_train_protocol", False):
        hot_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=hot_force_cold)
        cold_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=cold_force_cold)
        return {"cold": cold_bank, "hot": hot_bank, "all": hot_bank}

    was_training = model.training
    model.eval()
    try:
        hot_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=hot_force_cold)
        cold_bank = build_all_item_vecs(model, device, llm_scores, item_batch=item_batch, force_cold=cold_force_cold)
        if refined_eval_enabled(model):
            mixed_bank = replace_strict_cold_with_refined(
                model,
                hot_bank,
                device,
                llm_scores,
                item_batch=item_batch,
            )
            return {"cold": mixed_bank, "hot": mixed_bank, "all": mixed_bank}
        return {"cold": cold_bank, "hot": hot_bank, "all": hot_bank}
    finally:
        model.train(was_training)


def select_eval_item_bank(all_item_vecs, eval_type):
    if isinstance(all_item_vecs, dict):
        if eval_type in all_item_vecs:
            return all_item_vecs[eval_type]
        if eval_type == "all" and "hot" in all_item_vecs:
            return all_item_vecs["hot"]
        if "cold" in all_item_vecs:
            return all_item_vecs["cold"]
    return all_item_vecs


def build_eval_pos_item_vecs(model, item_idx, llm_s, pop_sel, eval_type):
    if item_idx.numel() < 1:
        return torch.empty((0, model.cfg.emb_dim), device=item_idx.device)
    if refined_eval_enabled(model):
        if eval_type == "cold":
            cold_mask = torch.ones_like(item_idx, dtype=torch.bool)
        elif eval_type == "hot":
            cold_mask = torch.zeros_like(item_idx, dtype=torch.bool)
        else:
            cold_mask = pop_sel.to(device=item_idx.device).float().view(-1) < float(model.cfg.cold_threshold)
        pos_vec = torch.empty((item_idx.size(0), model.cfg.emb_dim), device=item_idx.device)
        if cold_mask.any():
            cold_vec = refined_item_vectors(
                model,
                item_idx[cold_mask],
                llm_s=llm_s[cold_mask],
                item_batch=max(1, int(item_idx.numel())),
                force_cold=True,
            )
            pos_vec[cold_mask] = cold_vec
        hot_mask = ~cold_mask
        if hot_mask.any():
            hot_vec, _, _ = model.get_item_vector(item_idx[hot_mask], llm_s[hot_mask], force_cold=False)
            pos_vec[hot_mask] = hot_vec
        return F.normalize(pos_vec, dim=1)
    if eval_type == "cold":
        pos_vec, _, _ = model.get_item_vector(item_idx, llm_s, force_cold=True)
        return F.normalize(pos_vec, dim=1)
    if eval_type == "hot":
        pos_vec, _, _ = model.get_item_vector(item_idx, llm_s, force_cold=False)
        return F.normalize(pos_vec, dim=1)
    pos_vec = torch.empty((item_idx.size(0), model.cfg.emb_dim), device=item_idx.device)
    cold_mask = pop_sel < model.cfg.cold_threshold
    hot_mask = ~cold_mask
    if cold_mask.any():
        cold_vec, _, _ = model.get_item_vector(item_idx[cold_mask], llm_s[cold_mask], force_cold=True)
        pos_vec[cold_mask] = cold_vec
    if hot_mask.any():
        hot_vec, _, _ = model.get_item_vector(item_idx[hot_mask], llm_s[hot_mask], force_cold=False)
        pos_vec[hot_mask] = hot_vec
    return F.normalize(pos_vec, dim=1)


def evaluate_usim(
    model,
    loader,
    device,
    llm_scores,
    k_list=(5, 10, 20),
    n_neg=200,
    eval_type="cold",
    full_ranking=False,
    user_seen_items=None,
    all_item_vecs=None,
    average_mode="interaction",
    export_item_metrics_path=None,
    export_topk_path=None,
    export_topk_k=20,
    export_topk_metadata=None,
):
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    if export_topk_path and not full_ranking:
        raise ValueError("Top-K export requires full_ranking=True")
    model.eval()
    accum_metrics = {}
    total_samples = 0
    item_accum = {f"{m}@{k}": {} for m in ["R", "N"] for k in k_list}
    item_counts = {}
    seen_tensor_cache = {}
    seen_index = getattr(model, "user_seen_index", None)
    use_seen_index = seen_index is not None and user_seen_items is not None
    export_context = (
        TopKJsonlExporter(export_topk_path, top_k=export_topk_k, metadata=export_topk_metadata)
        if export_topk_path else nullcontext(None)
    )
    with export_context as topk_exporter, torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)
        if all_item_vecs is None:
            all_item_vecs = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
        item_bank = select_eval_item_bank(all_item_vecs, eval_type)
        for batch, pop, llm in loader:
            if eval_type == "cold":
                mask = pop < model.cfg.cold_threshold
            elif eval_type == "hot":
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)
            n_sel = mask.sum().item()
            if n_sel < 1:
                continue
            u = batch["u"][mask].to(device)
            i = batch["i"][mask].to(device)
            pop_sel = pop[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]
            item_ids = [int(x) for x in i.detach().cpu().tolist()]
            need_legacy_cache = (not use_seen_index) and (
                user_seen_items is not None
                or (not full_ranking)
                or model.cfg.use_course_rerank
            )
            if need_legacy_cache:
                for uid in user_ids:
                    if uid in seen_tensor_cache:
                        continue
                    seen_items = user_seen_items.get(uid) if user_seen_items else None
                    if seen_items:
                        seen_list = [it for it in seen_items if 0 <= it < n_items]
                        seen_tensor_cache[uid] = (
                            torch.tensor(seen_list, dtype=torch.long, device=device)
                            if seen_list else None
                        )
                    else:
                        seen_tensor_cache[uid] = None
            z_u = F.normalize(model.user_proj(model.user_emb(u)), dim=1)
            pos_llm = build_llm_score_tensor(llm_scores, user_ids, item_ids, device=device)
            pos_vec = build_eval_pos_item_vecs(model, i, pos_llm, pop_sel, eval_type)
            pos_scores = (z_u * pos_vec).sum(dim=1)
            if full_ranking:
                scores = torch.mm(z_u, item_bank.t())
                row_idx = torch.arange(n_sel, device=device)
                target_scores = pos_scores.clone()
                if use_seen_index:
                    seen_mask_full = seen_index.index_select(0, u)
                    scores = scores.masked_fill(seen_mask_full, -1e9)
                    scores[row_idx, i] = target_scores
                elif user_seen_items:
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache[uid]
                        if seen_idx is None:
                            continue
                        scores[row, seen_idx] = -1e9
                    scores[row_idx, i] = target_scores
                else:
                    scores[row_idx, i] = target_scores
                if hasattr(model, "apply_sage_two_expert_score_fusion"):
                    scores = model.apply_sage_two_expert_score_fusion(
                        scores,
                        user_ids=user_ids,
                        seen_tensor_cache=seen_tensor_cache,
                        cand_idx=None,
                    )
                scores = model.apply_course_rerank(scores, user_ids, seen_tensor_cache, cand_idx=None, target_pop=pop_sel)
                target_indices = i
            else:
                n_neg_eff = min(n_neg, max(1, n_items - 1))
                if use_seen_index:
                    forbidden_full = seen_index.index_select(0, u).clone()
                    row_idx_eval = torch.arange(n_sel, device=device)
                    forbidden_full[row_idx_eval, i] = True
                    avail_per_row = (~forbidden_full).sum(dim=1)
                    avail_counts = avail_per_row.clamp_min(1).tolist()
                else:
                    avail_counts = []
                    for row, uid in enumerate(user_ids):
                        seen_idx = seen_tensor_cache[uid]
                        if seen_idx is None:
                            avail = n_items - 1
                        else:
                            avail = n_items - 1 - int((seen_idx != i[row]).sum().item())
                        avail_counts.append(max(1, avail))
                n_neg_batch = min(n_neg_eff, min(avail_counts))
                neg_items = torch.empty((n_sel, n_neg_batch), dtype=torch.long, device=device)
                for row in range(n_sel):
                    if use_seen_index:
                        forbidden = forbidden_full[row]
                    else:
                        forbidden = torch.zeros(n_items, dtype=torch.bool, device=device)
                        forbidden[i[row]] = True
                        seen_idx = seen_tensor_cache[int(user_ids[row])]
                        if seen_idx is not None:
                            forbidden[seen_idx] = True
                    candidates = all_item_idx[~forbidden]
                    if candidates.numel() == 0:
                        candidates = all_item_idx[all_item_idx != i[row]]
                    pick = torch.randperm(candidates.numel(), device=device)[:n_neg_batch]
                    neg_items[row] = candidates[pick]
                cand_idx = torch.cat([i.unsqueeze(1), neg_items], dim=1)
                cand_vecs = item_bank[cand_idx].clone()
                cand_vecs[:, 0, :] = pos_vec
                scores = torch.bmm(cand_vecs, z_u.unsqueeze(2)).squeeze(2)
                if hasattr(model, "apply_sage_two_expert_score_fusion"):
                    scores = model.apply_sage_two_expert_score_fusion(
                        scores,
                        user_ids=user_ids,
                        seen_tensor_cache=seen_tensor_cache,
                        cand_idx=cand_idx,
                    )
                scores = model.apply_course_rerank(scores, user_ids, seen_tensor_cache, cand_idx=cand_idx, target_pop=pop_sel)
                target_indices = torch.zeros(n_sel, dtype=torch.long, device=device)
            metric_topk = None
            if topk_exporter is not None:
                shared_k = min(max(export_topk_k, max(k_list)), int(scores.size(1)))
                shared_scores, metric_topk = torch.topk(scores, k=shared_k, dim=1)
                topk_exporter.write_precomputed_batch(
                    metric_topk,
                    shared_scores,
                    user_ids=user_ids,
                    target_item_ids=item_ids,
                    target_popularity=pop_sel.detach().cpu().tolist(),
                )
            batch_values = compute_ranking_metric_values(
                scores,
                target_indices=target_indices,
                k_list=k_list,
                topk_indices=metric_topk,
            )
            if average_mode == "item_macro":
                for row, item_id in enumerate(item_ids):
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    for key, values in batch_values.items():
                        per_item = item_accum[key]
                        per_item[item_id] = per_item.get(item_id, 0.0) + float(values[row].detach().cpu().item())
            else:
                for key, values in batch_values.items():
                    accum_metrics[key] = accum_metrics.get(key, 0.0) + float(values.sum().detach().cpu().item())
            total_samples += n_sel
    if total_samples == 0:
        return None, 0
    if average_mode == "item_macro":
        if not item_counts:
            return None, 0
        macro = {}
        item_rows = []
        for key, per_item in item_accum.items():
            item_values = [
                per_item.get(item_id, 0.0) / count
                for item_id, count in item_counts.items()
                if count > 0
            ]
            macro[key] = sum(item_values) / max(1, len(item_values))
        if export_item_metrics_path:
            os.makedirs(os.path.dirname(export_item_metrics_path) or ".", exist_ok=True)
            for item_id in sorted(item_counts):
                row = {"item_id": int(item_id), "count": int(item_counts[item_id])}
                for key, per_item in item_accum.items():
                    row[key] = float(per_item.get(item_id, 0.0) / max(1, item_counts[item_id]))
                item_rows.append(row)
            fieldnames = ["item_id", "count"] + [f"{m}@{k}" for m in ["R", "N"] for k in k_list]
            with open(export_item_metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(item_rows)
        return macro, len(item_counts)
    return {key: value / total_samples for key, value in accum_metrics.items()}, total_samples


# Backward-compatible private names for the legacy monolithic entrypoint.
_lookup_llm_score = lookup_llm_score
_is_pair_llm_key = is_pair_llm_key
_is_item_llm_key = is_item_llm_key
_count_llm_key_types = count_llm_key_types
_item_mean_llm_scores = item_mean_llm_scores
_build_llm_score_tensor = build_llm_score_tensor
_resolve_eval_force_cold = resolve_eval_force_cold
_refined_eval_enabled = refined_eval_enabled
_strict_cold_item_mask = strict_cold_item_mask
_replace_strict_cold_with_refined = replace_strict_cold_with_refined
_select_eval_item_bank = select_eval_item_bank
_build_eval_pos_item_vecs = build_eval_pos_item_vecs
