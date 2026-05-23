"""Export per-user cold-item metric arrays for significance testing.

This script re-runs evaluation on saved checkpoints and stores per-user
Recall/NDCG values as .npz files (one per seed).

Usage:
    python export_peruser_metrics.py --variant full --seeds 2025 2026 2027

It will look for checkpoints in:
    outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/{variant}/
        seed{seed}/best_model.pt

And save per-user metrics to:
    outputs/content_delta_pop5/course_ablation_e60_3seed_corrected/{variant}/
        peruser_metrics_seed{seed}.npz

Each .npz file contains arrays keyed by "R@5", "R@10", "R@20", "N@5", "N@10", "N@20",
each of shape (n_cold_test_users,).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


K_LIST = [5, 10, 20]
ABLATION_ROOT = Path(
    os.environ.get(
        "USIM_COURSE_ABLATION_ROOT",
        "outputs/content_delta_pop5/course_ablation_e60_3seed_corrected",
    )
)


def evaluate_peruser(model, loader, device, llm_scores, k_list=K_LIST,
                     eval_type='cold', user_seen_items=None, all_item_vecs=None):
    """Evaluate and return per-user metric arrays instead of averaged values.
    
    Returns:
        dict[str, np.ndarray]: metric_name -> array of shape (n_users,)
    """
    from hin_eval_common import compute_ranking_metric_values

    model.eval()
    per_user_results = {f"R@{k}": [] for k in k_list}
    per_user_results.update({f"N@{k}": [] for k in k_list})
    seen_tensor_cache = {}

    with torch.no_grad():
        n_items = model.cfg.n_items
        all_item_idx = torch.arange(n_items, device=device)

        if all_item_vecs is None:
            # Import the appropriate build function
            try:
                from usim_feedback_fast3_content_delta_v2 import build_eval_item_vecs
            except ImportError:
                from usim import build_eval_item_vecs
            all_item_vecs = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)

        # Select item bank based on eval_type
        if eval_type == 'cold':
            item_bank = all_item_vecs[0] if isinstance(all_item_vecs, tuple) else all_item_vecs
        else:
            item_bank = all_item_vecs[1] if isinstance(all_item_vecs, tuple) else all_item_vecs

        # If it's a namedtuple/dict-like
        if hasattr(all_item_vecs, 'cold'):
            item_bank = all_item_vecs.cold if eval_type == 'cold' else all_item_vecs.hot
        elif isinstance(all_item_vecs, torch.Tensor):
            item_bank = all_item_vecs

        for batch, pop, llm in loader:
            if eval_type == 'cold':
                mask = pop < model.cfg.cold_threshold
            elif eval_type == 'hot':
                mask = pop >= model.cfg.cold_threshold
            else:
                mask = torch.ones_like(pop, dtype=torch.bool)

            n_sel = mask.sum().item()
            if n_sel < 1:
                continue

            u = batch['u'][mask].to(device)
            i = batch['i'][mask].to(device)
            pop_sel = pop[mask].to(device)
            user_ids = [int(x) for x in u.detach().cpu().tolist()]

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

            # Full ranking
            scores = torch.mm(z_u, item_bank.t())
            row_idx = torch.arange(n_sel, device=device)
            target_scores = scores[row_idx, i].clone()

            if user_seen_items:
                for row, uid in enumerate(user_ids):
                    seen_idx = seen_tensor_cache[uid]
                    if seen_idx is None:
                        continue
                    scores[row, seen_idx] = -1e9
                scores[row_idx, i] = target_scores
            else:
                scores[row_idx, i] = target_scores

            # Apply course rerank if available
            if hasattr(model, 'apply_course_rerank'):
                scores = model.apply_course_rerank(
                    scores, user_ids, seen_tensor_cache,
                    cand_idx=None, target_pop=pop_sel
                )

            target_indices = i

            # Get per-user metric values (not averaged)
            batch_values = compute_ranking_metric_values(scores, target_indices, k_list=k_list)
            for key, tensor in batch_values.items():
                per_user_results[key].append(tensor.cpu().numpy())

    # Concatenate all batches
    return {k: np.concatenate(v) for k, v in per_user_results.items() if v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="full",
                        choices=["full", "wo_course_reward", "wo_course_candidate",
                                 "wo_prereq_aux", "wo_all_course_signals"],
                        help="Which ablation variant to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--root", type=str, default=str(ABLATION_ROOT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.root)
    variant_dir = root / args.variant

    print(f"Variant: {args.variant}")
    print(f"Root: {root}")
    print(f"Seeds: {args.seeds}")
    print()

    # Check for checkpoint files
    for seed in args.seeds:
        ckpt_candidates = [
            variant_dir / f"seed{seed}" / "best_model.pt",
            variant_dir / f"seed_{seed}" / "best_model.pt",
            variant_dir / f"run_seed{seed}" / "best_model.pt",
        ]
        ckpt = next((p for p in ckpt_candidates if p.exists()), None)

        if ckpt is None:
            print(f"[seed={seed}] No checkpoint found. Searched:")
            for c in ckpt_candidates:
                print(f"    {c}")
            print()
            print("  --> You need to adapt the checkpoint path pattern above,")
            print("      or modify this script to match your directory structure.")
            print()
            continue

        print(f"[seed={seed}] Found checkpoint: {ckpt}")
        print(f"  TODO: Load model, dataset, and run evaluate_peruser()")
        print(f"  Save to: {variant_dir / f'peruser_metrics_seed{seed}.npz'}")
        print()

    print("=" * 70)
    print("INSTRUCTIONS:")
    print()
    print("This script provides the framework. To complete it, you need to:")
    print()
    print("1. Copy your model initialization and data loading code from your")
    print("   training script (e.g., usim_feedback_fast3_content_delta_v2.py)")
    print()
    print("2. For each seed:")
    print("   a. Load the checkpoint")
    print("   b. Build the test DataLoader")
    print("   c. Call evaluate_peruser() to get per-user arrays")
    print("   d. Save with: np.savez(output_path, **per_user_metrics)")
    print()
    print("3. Then run: python run_significance_test.py")
    print("=" * 70)
    print()
    print("QUICK ALTERNATIVE (if you want to skip re-evaluation):")
    print()
    print("Add the following to your existing evaluation loop in")
    print("usim_feedback_fast3_content_delta_v2.py, right after the final")
    print("evaluate_usim() call:")
    print()
    print('    from hin_eval_common import compute_ranking_metric_values')
    print('    # ... inside the eval loop, replace compute_ranking_metrics with:')
    print('    batch_values = compute_ranking_metric_values(scores, target_indices, k_list)')
    print('    # Accumulate per-user values instead of batch means')
    print()


if __name__ == "__main__":
    main()
