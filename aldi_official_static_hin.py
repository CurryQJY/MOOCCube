"""
Run an ALDI official-source static item-cold adaptation without modifying source.

This wrapper does three things:
1. Converts the shared static split into the data files expected by the
   official ALDI repository.
2. Copies the official source snapshot from third_party/ALDI into a runtime
   directory and applies TensorFlow compatibility patches only to that copy.
3. Runs official BPRMF + official ALDI training with local full-ranking
   evaluation and writes aldi_official_static_result.json.

The original files under third_party/ALDI are never edited.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

from hin_data_common import load_hin_processed, setup_seed, static_result_path, static_split_df


DEFAULT_SPLIT = Path("outputs/content_delta_pop5/static_item_cold/strict_item_cold_thr1_seed_2025")
DEFAULT_WORK = Path(".runtime_tmp/aldi_official_static")
DEFAULT_DATASET = "MOOCCubeStatic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Only write official-format data and runtime source.")
    parser.add_argument("--official-src", default=os.environ.get("ALDI_OFFICIAL_SRC", "third_party/ALDI"))
    parser.add_argument("--work-dir", default=os.environ.get("ALDI_OFFICIAL_WORK_DIR", str(DEFAULT_WORK)))
    parser.add_argument("--dataset", default=os.environ.get("ALDI_OFFICIAL_DATASET", DEFAULT_DATASET))
    parser.add_argument("--python", default=os.environ.get("ALDI_OFFICIAL_PYTHON", sys.executable))
    parser.add_argument("--conda-prefix", default=os.environ.get("ALDI_OFFICIAL_CONDA_PREFIX", ""))
    parser.add_argument("--force-teacher", action="store_true", default=os.environ.get("ALDI_OFFICIAL_FORCE_TEACHER", "0") == "1")
    parser.add_argument("--rebuild-source", action="store_true", default=os.environ.get("ALDI_OFFICIAL_REBUILD_SOURCE", "1") == "1")
    parser.add_argument(
        "--tf2-compat-patch",
        action="store_true",
        default=os.environ.get("ALDI_OFFICIAL_TF2_COMPAT_PATCH", "1") == "1",
        help="Patch official TF1 source imports to tensorflow.compat.v1 for TensorFlow 2.x runtimes.",
    )
    return parser.parse_args()


def _pairs(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": df["u_idx"].astype(np.int64).to_numpy(),
            "item": df["i_idx"].astype(np.int64).to_numpy(),
        }
    )


def _local_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": df["u_idx"].astype(np.int64).to_numpy(),
            "item": df["i_idx"].astype(np.int64).to_numpy(),
            "pop": df["popularity"].astype(np.int64).to_numpy(),
        }
    )


def _df_get_neighbors(input_df: pd.DataFrame, obj: str, max_num: int) -> np.ndarray:
    opp_obj = "item" if obj == "user" else "user"
    nei_array = np.zeros((max_num,), dtype=object)
    for key, values in input_df.groupby(obj):
        idx = int(key)
        if 0 <= idx < max_num:
            nei_array[idx] = values[opp_obj].astype(np.int64).to_numpy()
    return nei_array


def _unique_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if df.empty:
        return np.empty(0, dtype=np.int32)
    return np.array(sorted(set(df[col].astype(int).tolist())), dtype=np.int32)


def build_convert_dict(
    train_pairs: pd.DataFrame,
    warm_val: pd.DataFrame,
    warm_test: pd.DataFrame,
    cold_val: pd.DataFrame,
    cold_test: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> Dict:
    all_pairs = train_pairs.copy()
    overall_val_users = np.array(sorted(set(cold_val["user"]) & set(warm_val["user"])), dtype=np.int32)
    overall_test_users = np.array(sorted(set(cold_test["user"]) & set(warm_test["user"])), dtype=np.int32)
    overall_val = pd.concat([cold_val, warm_val], ignore_index=True)
    overall_val = overall_val[overall_val["user"].isin(overall_val_users)]
    overall_test = pd.concat([cold_test, warm_test], ignore_index=True)
    overall_test = overall_test[overall_test["user"].isin(overall_test_users)]

    user_array = np.arange(n_users, dtype=np.int32)
    item_array = np.arange(n_items, dtype=np.int32)
    warm_user = _unique_array(train_pairs, "user")
    warm_item = _unique_array(train_pairs, "item")
    cold_user = np.array(sorted(set(user_array.tolist()) - set(warm_user.tolist())), dtype=np.int32)
    cold_item = np.array(sorted(set(item_array.tolist()) - set(warm_item.tolist())), dtype=np.int32)

    return {
        "user_num": n_users,
        "item_num": n_items,
        "user_array": user_array,
        "item_array": item_array,
        "warm_user": warm_user,
        "warm_item": warm_item,
        "cold_user": cold_user,
        "cold_item": cold_item,
        "emb_user": warm_user,
        "warm_val_user": _unique_array(warm_val, "user"),
        "warm_test_user": _unique_array(warm_test, "user"),
        "cold_val_user": _unique_array(cold_val, "user"),
        "cold_test_user": _unique_array(cold_test, "user"),
        "hybrid_val_user": overall_val_users,
        "hybrid_test_user": overall_test_users,
        "overall_val_user": overall_val_users,
        "overall_test_user": overall_test_users,
        "emb_item": warm_item,
        "warm_val_item": _unique_array(warm_val, "item"),
        "warm_test_item": _unique_array(warm_test, "item"),
        "cold_val_item": _unique_array(cold_val, "item"),
        "cold_test_item": _unique_array(cold_test, "item"),
        "hybrid_val_item": _unique_array(overall_val, "item"),
        "hybrid_test_item": _unique_array(overall_test, "item"),
        "overall_val_item": _unique_array(overall_val, "item"),
        "overall_test_item": _unique_array(overall_test, "item"),
        "pos_user_nb": _df_get_neighbors(all_pairs, "user", n_users),
        "emb_user_nb": _df_get_neighbors(train_pairs, "user", n_users),
        "warm_val_user_nb": _df_get_neighbors(warm_val, "user", n_users),
        "warm_test_user_nb": _df_get_neighbors(warm_test, "user", n_users),
        "cold_val_user_nb": _df_get_neighbors(cold_val, "user", n_users),
        "cold_test_user_nb": _df_get_neighbors(cold_test, "user", n_users),
        "hybrid_val_user_nb": _df_get_neighbors(overall_val, "user", n_users),
        "hybrid_test_user_nb": _df_get_neighbors(overall_test, "user", n_users),
        "overall_val_user_nb": _df_get_neighbors(overall_val, "user", n_users),
        "overall_test_user_nb": _df_get_neighbors(overall_test, "user", n_users),
        "emb_item_nb": _df_get_neighbors(train_pairs, "item", n_items),
    }


def prepare_official_data(args) -> Path:
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin_clean_pop5")
    cold_threshold = int(os.environ.get("ALDI_OFFICIAL_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "1")))
    static_seed = int(os.environ.get("ALDI_OFFICIAL_STATIC_SEED", os.environ.get("USIM_STATIC_SEED", "2025")))
    setup_seed(static_seed)
    meta, df, content_emb = load_hin_processed(data_dir)
    train_df, val_df, test_df = static_split_df(df, seed=static_seed)

    work = Path(args.work_dir)
    dataset_dir = work / "data" / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = _pairs(train_df)
    val_local = _local_pairs(val_df)
    test_local = _local_pairs(test_df)
    warm_val = _pairs(val_df[val_df["popularity"] >= cold_threshold])
    warm_test = _pairs(test_df[test_df["popularity"] >= cold_threshold])
    cold_val = _pairs(val_df[val_df["popularity"] < cold_threshold])
    cold_test = _pairs(test_df[test_df["popularity"] < cold_threshold])

    train_pairs.to_csv(dataset_dir / "warm_emb.csv", index=False)
    warm_val.to_csv(dataset_dir / "warm_val.csv", index=False)
    warm_test.to_csv(dataset_dir / "warm_test.csv", index=False)
    cold_val.to_csv(dataset_dir / "cold_item_val.csv", index=False)
    cold_test.to_csv(dataset_dir / "cold_item_test.csv", index=False)
    train_pairs.to_csv(dataset_dir / "all.csv", index=False)
    val_local.to_csv(dataset_dir / "local_val.csv", index=False)
    test_local.to_csv(dataset_dir / "local_test.csv", index=False)
    np.save(dataset_dir / f"{args.dataset}_item_content.npy", content_emb.float().numpy())

    with (dataset_dir / "n_user_item.pkl").open("wb") as f:
        pickle.dump({"user": int(meta["n_users"]), "item": int(meta["n_items"])}, f, protocol=4)
    convert_dict = build_convert_dict(
        train_pairs=train_pairs,
        warm_val=warm_val,
        warm_test=warm_test,
        cold_val=cold_val,
        cold_test=cold_test,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
    )
    with (dataset_dir / "convert_dict.pkl").open("wb") as f:
        pickle.dump(convert_dict, f, protocol=4)

    manifest = {
        "dataset": args.dataset,
        "source_data_dir": data_dir,
        "split_dir": os.environ.get("USIM_STATIC_SPLIT_DIR", str(DEFAULT_SPLIT)),
        "cold_threshold": cold_threshold,
        "history_mask": os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only"),
        "train_rows": int(len(train_df)),
        "val_cold_rows": int(len(cold_val)),
        "val_hot_rows": int(len(warm_val)),
        "test_cold_rows": int(len(cold_test)),
        "test_hot_rows": int(len(warm_test)),
        "note": "all.csv intentionally contains train interactions only to match train_only masking.",
    }
    with (dataset_dir / "static_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote ALDI official-format data to {dataset_dir}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return dataset_dir


def _copytree_clean(src: Path, dst: Path, rebuild: bool) -> None:
    if rebuild and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["cold_start", "metric", "warm_model"]:
        src_dir = src / name
        if src_dir.exists():
            shutil.copytree(src_dir, dst / name, dirs_exist_ok=True)
    for name in ["main.py", "utils.py", "README.md", "SOURCE.md"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def _patch_tf_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("import tensorflow as tf", "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()")
    text = text.replace("tf.contrib.layers.l2_regularizer(reg)", "tf.keras.regularizers.l2(reg)")
    path.write_text(text, encoding="utf-8")


def _write_static_runner(src_dir: Path) -> None:
    runner = r'''
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cold_start
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embed_meth", type=str, default="bprmf")
    parser.add_argument("--embed_loss", type=str, default="BPR")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--model", type=str, default="ALDI")
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tws", type=int, default=0, choices=[0, 1])
    parser.add_argument("--freq_coef_M", type=float, default=4)
    parser.add_argument("--cold_threshold", type=int, default=1)
    parser.add_argument("--eval_n_neg", type=int, default=200)
    parser.add_argument("--result_json", type=str, required=True)
    parser.add_argument("--history_policy", type=str, default="train_only")
    parser.add_argument("--eval_batch_rows", type=int, default=4096)
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--auto_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--force_fresh", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    args.Ks = [5, 10, 20]
    return args


def build_seen(dataset_path, history_policy):
    train = pd.read_csv(os.path.join(dataset_path, "warm_emb.csv"), dtype=np.int64)
    seen = {}
    for user, item in zip(train["user"].values, train["item"].values):
        seen.setdefault(int(user), set()).add(int(item))
    if history_policy == "train_val":
        for filename in ["warm_val.csv", "cold_item_val.csv"]:
            df = pd.read_csv(os.path.join(dataset_path, filename), dtype=np.int64)
            for user, item in zip(df["user"].values, df["item"].values):
                seen.setdefault(int(user), set()).add(int(item))
    return seen


def item_freq_from_official(para_dict, train_data):
    item_freq = np.ones(shape=(para_dict["item_num"],), dtype=np.float32)
    item_to_user_neighbors = para_dict["emb_item_nb"][para_dict["warm_item"]]
    for item_index, user_neighbor_list in zip(para_dict["warm_item"], item_to_user_neighbors):
        neighborhoods = para_dict["emb_user_nb"][user_neighbor_list]
        item_freq[item_index] = sum([1.0 / len(neighborhood) for neighborhood in neighborhoods])
    return item_freq


def metric_from_scores(score_vec, target, ks):
    max_k = min(max(ks), score_vec.shape[0])
    top = np.argpartition(-score_vec, max_k - 1)[:max_k]
    top = top[np.argsort(-score_vec[top])]
    out = {}
    for k in ks:
        k_eff = min(k, len(top))
        preds = top[:k_eff]
        hit = np.where(preds == target)[0]
        out[f"R@{k}"] = 1.0 if hit.size > 0 else 0.0
        out[f"N@{k}"] = float(1.0 / np.log2(hit[0] + 2.0)) if hit.size > 0 else 0.0
    return out


def evaluate_local(
    model,
    para_dict,
    gen_user_emb,
    gen_item_emb,
    pairs,
    seen,
    args,
    eval_type,
    full_ranking,
    average_mode="interaction",
    export_item_metrics_path=None,
):
    average_mode = average_mode.strip().lower()
    if average_mode not in {"interaction", "item_macro"}:
        raise ValueError("average_mode must be 'interaction' or 'item_macro'")
    if eval_type == "cold":
        pairs = pairs[pairs["pop"] < args.cold_threshold]
    elif eval_type == "hot":
        pairs = pairs[pairs["pop"] >= args.cold_threshold]
    if pairs.empty:
        return {}, 0
    users = pairs["user"].astype(np.int64).to_numpy()
    items = pairs["item"].astype(np.int64).to_numpy()
    item_array = para_dict["item_array"]
    n_items = int(para_dict["item_num"])
    rng = np.random.default_rng(args.seed + (17 if full_ranking else 29) + (3 if eval_type == "hot" else 0))
    ks = args.Ks
    accum = {f"{m}@{k}": 0.0 for m in ["R", "N"] for k in ks}
    item_accum = {f"{m}@{k}": {} for m in ["R", "N"] for k in ks}
    item_counts = {}
    total = 0

    for beg in range(0, len(users), args.eval_batch_rows):
        end = min(beg + args.eval_batch_rows, len(users))
        u_batch = users[beg:end]
        i_batch = items[beg:end]
        ratings = model.get_user_rating(u_batch, item_array, gen_user_emb, gen_item_emb)
        for row, (uid, target) in enumerate(zip(u_batch.tolist(), i_batch.tolist())):
            score_vec = ratings[row].copy()
            target_score = score_vec[target]
            seen_items = seen.get(int(uid), set())
            if seen_items:
                seen_idx = [x for x in seen_items if 0 <= x < n_items]
                if seen_idx:
                    score_vec[np.asarray(seen_idx, dtype=np.int64)] = -1e10
            score_vec[target] = target_score
            if full_ranking:
                row_metrics = metric_from_scores(score_vec, int(target), ks)
            else:
                forbidden = set(seen_items)
                forbidden.add(int(target))
                pool = np.array([x for x in range(n_items) if x not in forbidden], dtype=np.int64)
                if pool.size == 0:
                    pool = np.array([x for x in range(n_items) if x != int(target)], dtype=np.int64)
                n_neg = min(args.eval_n_neg, pool.size)
                neg = rng.choice(pool, size=n_neg, replace=False)
                cand = np.concatenate([[int(target)], neg])
                rng.shuffle(cand)
                target_pos = int(np.where(cand == int(target))[0][0])
                row_metrics = metric_from_scores(score_vec[cand], target_pos, ks)
            if average_mode == "item_macro":
                item_counts[int(target)] = item_counts.get(int(target), 0) + 1
                for key, val in row_metrics.items():
                    per_item = item_accum[key]
                    per_item[int(target)] = per_item.get(int(target), 0.0) + float(val)
            else:
                for key, val in row_metrics.items():
                    accum[key] += val
            total += 1
    if average_mode == "item_macro":
        if not item_counts:
            return {}, 0
        macro = {}
        for key, per_item in item_accum.items():
            item_values = [
                per_item.get(item_id, 0.0) / count
                for item_id, count in item_counts.items()
                if count > 0
            ]
            macro[key] = sum(item_values) / max(1, len(item_values))
        if export_item_metrics_path:
            rows = []
            for item_id in sorted(item_counts):
                count = max(1, int(item_counts[item_id]))
                row = {"item_id": int(item_id), "count": int(item_counts[item_id])}
                for key, per_item in item_accum.items():
                    row[key] = float(per_item.get(item_id, 0.0) / count)
                rows.append(row)
            os.makedirs(os.path.dirname(export_item_metrics_path) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(export_item_metrics_path, index=False)
        return macro, len(item_counts)
    return {k: v / total for k, v in accum.items()}, total


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    utils.set_seed_tf(args.seed)
    dataset_path = os.path.join(args.datadir, args.dataset)
    content_data = np.load(os.path.join(dataset_path, f"{args.dataset}_item_content.npy"))
    para_dict = pickle.load(open(os.path.join(dataset_path, "convert_dict.pkl"), "rb"))
    train_data = pd.read_csv(os.path.join(dataset_path, "warm_emb.csv"), dtype=np.int64).values
    val_pairs = pd.read_csv(os.path.join(dataset_path, "local_val.csv"), dtype=np.int64)
    test_pairs = pd.read_csv(os.path.join(dataset_path, "local_test.csv"), dtype=np.int64)
    emb = np.load(os.path.join(dataset_path, f"{args.embed_meth}-{args.embed_loss}.npy"))
    user_num = int(para_dict["user_num"])
    user_emb = emb[:user_num]
    item_emb = emb[user_num:]
    item_freq = item_freq_from_official(para_dict, train_data)
    x_expect = (len(train_data) / para_dict["item_num"]) * (1 / (len(train_data) / para_dict["user_num"]))
    args.freq_coef_a = args.freq_coef_M / x_expect
    seen = build_seen(dataset_path, args.history_policy)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = getattr(cold_start, args.model)(sess, args, emb.shape[-1], content_data.shape[-1])
    save_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(".", "cold_start", "model_save")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{args.dataset}-{args.model}-official-static-best")
    latest_file = os.path.join(save_dir, f"{args.dataset}-{args.model}-official-static-latest")
    state_file = os.path.join(save_dir, f"{args.dataset}-{args.model}-official-static-state.json")
    saver = tf.train.Saver()

    best_val = -1.0
    best_epoch = -1
    patient = 0
    best_loss = None
    start_epoch = 0
    if args.ckpt_dir:
        print(f"Checkpoint: save=True resume={bool(args.auto_resume)} force_fresh={bool(args.force_fresh)} dir={args.ckpt_dir}")
    if args.ckpt_dir and args.auto_resume and not args.force_fresh and os.path.exists(latest_file + ".index"):
        saver.restore(sess, latest_file)
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
            start_epoch = int(saved_state.get("epoch", 0))
            best_val = float(saved_state.get("best_val", best_val))
            best_epoch = int(saved_state.get("best_epoch", best_epoch))
            best_loss = saved_state.get("best_loss", best_loss)
            patient = int(saved_state.get("patient", patient))
        print(f"Resume checkpoint: latest_epoch={start_epoch} | best_epoch={best_epoch} | best_score={best_val:.6f}")
    print(f"Official-source ALDI training: epochs={args.max_epoch}, batch={args.batch_size}")
    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        train_input = utils.bpr_neg_samp(para_dict["warm_user"], len(train_data), para_dict["emb_user_nb"], para_dict["warm_item"])
        epoch_loss = 0.0
        n_batch = 0
        for beg in range(0, len(train_input) - args.batch_size, args.batch_size):
            end = beg + args.batch_size
            batch = train_input[beg:end]
            loss = model.train(
                content_data[batch[:, 1]],
                item_emb[batch[:, 1]],
                content_data[batch[:, 2]],
                item_emb[batch[:, 2]],
                user_emb[batch[:, 0]],
                item_freq[batch[:, 1]],
                item_freq[batch[:, 2]],
            )
            epoch_loss += float(loss)
            n_batch += 1
        avg_loss = epoch_loss / max(1, n_batch)
        do_eval = (epoch % args.val_interval == 0) or (epoch == args.max_epoch)
        if do_eval:
            gen_user_emb = model.get_user_emb(user_emb)
            gen_item_emb = model.get_item_emb(content_data, item_emb, para_dict["warm_item"], para_dict["cold_item"])
            val_cold, n_val_cold = evaluate_local(
                model, para_dict, gen_user_emb, gen_item_emb, val_pairs, seen, args, "cold", True
            )
            val_key = val_cold.get("N@10", 0.0)
            if val_key > best_val:
                best_val = val_key
                best_epoch = epoch
                best_loss = avg_loss
                saver.save(sess, save_file)
                patient = 0
            else:
                patient += 1
            print(f"Epoch [{epoch}/{args.max_epoch}] loss={avg_loss:.4f} | val_full_cold_N@10={val_key:.4f} | best={best_val:.4f}")
            if args.patience > 0 and patient > args.patience:
                break
        else:
            print(f"Epoch [{epoch}/{args.max_epoch}] loss={avg_loss:.4f}")
        if args.ckpt_dir:
            saver.save(sess, latest_file)
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "best_loss": best_loss,
                        "patient": patient,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    saver.restore(sess, save_file)
    gen_user_emb = model.get_user_emb(user_emb)
    gen_item_emb = model.get_item_emb(content_data, item_emb, para_dict["warm_item"], para_dict["cold_item"])

    sample_cold, n_sc = evaluate_local(model, para_dict, gen_user_emb, gen_item_emb, test_pairs, seen, args, "cold", False)
    sample_hot, n_sh = evaluate_local(model, para_dict, gen_user_emb, gen_item_emb, test_pairs, seen, args, "hot", False)
    full_cold, n_fc = evaluate_local(model, para_dict, gen_user_emb, gen_item_emb, test_pairs, seen, args, "cold", True)
    full_hot, n_fh = evaluate_local(model, para_dict, gen_user_emb, gen_item_emb, test_pairs, seen, args, "hot", True)
    full_cold_item_macro, n_fc_item_macro = evaluate_local(
        model, para_dict, gen_user_emb, gen_item_emb, test_pairs, seen, args, "cold", True,
        average_mode="item_macro",
        export_item_metrics_path=os.path.join(os.path.dirname(args.result_json), "per_item_full_cold_aldi_official_static.csv"),
    )
    full_hot_item_macro, n_fh_item_macro = evaluate_local(
        model, para_dict, gen_user_emb, gen_item_emb, test_pairs, seen, args, "hot", True,
        average_mode="item_macro",
        export_item_metrics_path=os.path.join(os.path.dirname(args.result_json), "per_item_full_hot_aldi_official_static.csv"),
    )
    sess.close()

    result = {
        "model": "ALDI-official-source",
        "model_display": "ALDI (official-source)",
        "source": "Official ALDI TensorFlow source copied from third_party/ALDI into runtime and evaluated with the local static protocol.",
        "protocol": "static_item_cold",
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item_macro,
        "full_hot_item_macro": full_hot_item_macro,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item_macro,
        "count_full_hot_item_macro": n_fh_item_macro,
        "best_epoch": best_epoch,
        "best_val_full_cold_n10": best_val,
        "best_metric": "cold",
        "best_loss": best_loss,
        "eval_n_neg": args.eval_n_neg,
        "static_seed": args.seed,
        "checkpoint_dir": args.ckpt_dir or None,
        "resumed_from_epoch": start_epoch,
        "per_item_full_cold_path": os.path.join(os.path.dirname(args.result_json), "per_item_full_cold_aldi_official_static.csv"),
        "per_item_full_hot_path": os.path.join(os.path.dirname(args.result_json), "per_item_full_hot_aldi_official_static.csv"),
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "tws": int(args.tws),
        "note": "Original ALDI model/loss code is used from a runtime copy; source tree is not modified.",
    }
    os.makedirs(os.path.dirname(args.result_json), exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.result_json}")


if __name__ == "__main__":
    main()
'''
    (src_dir / "main_static_full_eval.py").write_text(textwrap.dedent(runner), encoding="utf-8")


def _write_convert_builder(src_dir: Path) -> None:
    builder = r'''
import argparse
import pickle

import numpy as np
import pandas as pd


def df_get_neighbors(input_df, obj, max_num):
    opp_obj = "item" if obj == "user" else "user"
    nei_array = np.zeros((max_num,), dtype=object)
    for key, values in input_df.groupby(obj):
        idx = int(key)
        if 0 <= idx < max_num:
            nei_array[idx] = values[opp_obj].astype(np.int64).to_numpy()
    return nei_array


def unique_array(df, col):
    if df.empty:
        return np.empty(0, dtype=np.int32)
    return np.array(sorted(set(df[col].astype(int).tolist())), dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    train = pd.read_csv(dataset_dir + "/warm_emb.csv", dtype=np.int64)
    warm_val = pd.read_csv(dataset_dir + "/warm_val.csv", dtype=np.int64)
    warm_test = pd.read_csv(dataset_dir + "/warm_test.csv", dtype=np.int64)
    cold_val = pd.read_csv(dataset_dir + "/cold_item_val.csv", dtype=np.int64)
    cold_test = pd.read_csv(dataset_dir + "/cold_item_test.csv", dtype=np.int64)
    with open(dataset_dir + "/n_user_item.pkl", "rb") as f:
        n_user_item = pickle.load(f)
    n_users = int(n_user_item["user"])
    n_items = int(n_user_item["item"])

    overall_val_users = np.array(sorted(set(cold_val["user"]) & set(warm_val["user"])), dtype=np.int32)
    overall_test_users = np.array(sorted(set(cold_test["user"]) & set(warm_test["user"])), dtype=np.int32)
    overall_val = pd.concat([cold_val, warm_val], ignore_index=True)
    overall_val = overall_val[overall_val["user"].isin(overall_val_users)]
    overall_test = pd.concat([cold_test, warm_test], ignore_index=True)
    overall_test = overall_test[overall_test["user"].isin(overall_test_users)]

    user_array = np.arange(n_users, dtype=np.int32)
    item_array = np.arange(n_items, dtype=np.int32)
    warm_user = unique_array(train, "user")
    warm_item = unique_array(train, "item")
    cold_user = np.array(sorted(set(user_array.tolist()) - set(warm_user.tolist())), dtype=np.int32)
    cold_item = np.array(sorted(set(item_array.tolist()) - set(warm_item.tolist())), dtype=np.int32)

    para_dict = {
        "user_num": n_users,
        "item_num": n_items,
        "user_array": user_array,
        "item_array": item_array,
        "warm_user": warm_user,
        "warm_item": warm_item,
        "cold_user": cold_user,
        "cold_item": cold_item,
        "emb_user": warm_user,
        "warm_val_user": unique_array(warm_val, "user"),
        "warm_test_user": unique_array(warm_test, "user"),
        "cold_val_user": unique_array(cold_val, "user"),
        "cold_test_user": unique_array(cold_test, "user"),
        "hybrid_val_user": overall_val_users,
        "hybrid_test_user": overall_test_users,
        "overall_val_user": overall_val_users,
        "overall_test_user": overall_test_users,
        "emb_item": warm_item,
        "warm_val_item": unique_array(warm_val, "item"),
        "warm_test_item": unique_array(warm_test, "item"),
        "cold_val_item": unique_array(cold_val, "item"),
        "cold_test_item": unique_array(cold_test, "item"),
        "hybrid_val_item": unique_array(overall_val, "item"),
        "hybrid_test_item": unique_array(overall_test, "item"),
        "overall_val_item": unique_array(overall_val, "item"),
        "overall_test_item": unique_array(overall_test, "item"),
        "pos_user_nb": df_get_neighbors(train, "user", n_users),
        "emb_user_nb": df_get_neighbors(train, "user", n_users),
        "warm_val_user_nb": df_get_neighbors(warm_val, "user", n_users),
        "warm_test_user_nb": df_get_neighbors(warm_test, "user", n_users),
        "cold_val_user_nb": df_get_neighbors(cold_val, "user", n_users),
        "cold_test_user_nb": df_get_neighbors(cold_test, "user", n_users),
        "hybrid_val_user_nb": df_get_neighbors(overall_val, "user", n_users),
        "hybrid_test_user_nb": df_get_neighbors(overall_test, "user", n_users),
        "overall_val_user_nb": df_get_neighbors(overall_val, "user", n_users),
        "overall_test_user_nb": df_get_neighbors(overall_test, "user", n_users),
        "emb_item_nb": df_get_neighbors(train, "item", n_items),
    }
    with open(dataset_dir + "/convert_dict.pkl", "wb") as f:
        pickle.dump(para_dict, f, protocol=4)
    print("Rebuilt convert_dict.pkl with runtime NumPy:", np.__version__)


if __name__ == "__main__":
    main()
'''
    (src_dir / "build_convert_dict_static.py").write_text(textwrap.dedent(builder), encoding="utf-8")


def prepare_runtime_source(args) -> Path:
    official_src = Path(args.official_src)
    if not official_src.exists():
        raise FileNotFoundError(f"Missing ALDI official source snapshot: {official_src}")
    src_dir = Path(args.work_dir) / "src"
    _copytree_clean(official_src, src_dir, rebuild=args.rebuild_source)
    if args.tf2_compat_patch:
        for rel in ["utils.py", "cold_start/ALDI.py", "warm_model/bprmf.py", "main.py"]:
            path = src_dir / rel
            if path.exists():
                _patch_tf_file(path)
    _write_static_runner(src_dir)
    _write_convert_builder(src_dir)
    print(f"Wrote runtime ALDI source copy to {src_dir}")
    return src_dir


def _env_int(*names: str, default: int) -> int:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return int(val)
    return int(default)


def _env_float(*names: str, default: float) -> float:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return float(val)
    return float(default)


def _env_bool_arg(*names: str, default: str = "0") -> str:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return "1" if str(val).strip().lower() in {"1", "true", "yes", "y", "on"} else "0"
    return "1" if str(default).strip().lower() in {"1", "true", "yes", "y", "on"} else "0"


def run_checked(cmd: Iterable[str], cwd: Path) -> None:
    print("Running:", " ".join([str(x) for x in cmd]))
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def python_cmd(args) -> list:
    if args.conda_prefix:
        conda_exe = os.environ.get("CONDA_EXE", r"D:\anaconda3\Scripts\conda.exe")
        return [conda_exe, "run", "-p", str(Path(args.conda_prefix).resolve()), "python"]
    return [args.python]


def check_tf_python(cmd_prefix: list) -> None:
    code = (
        "import sys; import tensorflow as tf; "
        "print(sys.executable); print(tf.__version__); "
        "import pandas, numpy, tqdm; print('tf-ready')"
    )
    subprocess.run(cmd_prefix + ["-c", code], check=True)


def run_official(args, src_dir: Path, dataset_dir: Path) -> None:
    py_cmd = python_cmd(args)
    try:
        check_tf_python(py_cmd)
    except Exception as exc:
        raise RuntimeError(
            "ALDI official-source run requires a Python environment with TensorFlow. "
            "Set ALDI_OFFICIAL_PYTHON to that python.exe or ALDI_OFFICIAL_CONDA_PREFIX to a conda prefix, "
            "or run with --prepare-only. "
            f"Current candidate failed: {' '.join(py_cmd)}"
        ) from exc

    run_checked(
        py_cmd
        + [
            "build_convert_dict_static.py",
            "--dataset-dir",
            str(dataset_dir.resolve()),
        ],
        cwd=src_dir,
    )

    data_root = dataset_dir.parent
    dataset = args.dataset
    gpu_id = os.environ.get("ALDI_OFFICIAL_GPU_ID", "0")
    seed = os.environ.get("ALDI_OFFICIAL_SEED", os.environ.get("USIM_STATIC_SEED", "2025"))
    emb_dim = os.environ.get("ALDI_OFFICIAL_EMB_DIM", os.environ.get("ALDI_EMB_DIM", "200"))
    teacher_epochs = _env_int("ALDI_OFFICIAL_TEACHER_EPOCHS", "ALDI_TEACHER_EPOCHS", default=200)
    teacher_interval = _env_int("ALDI_OFFICIAL_TEACHER_EVAL_INTERVAL", "ALDI_TEACHER_EVAL_INTERVAL", default=20)
    teacher_batch = _env_int("ALDI_OFFICIAL_TEACHER_BATCH_SIZE", "ALDI_BATCH_SIZE", default=4096)
    eval_users = os.environ.get("ALDI_OFFICIAL_N_TEST_USER", "2000")
    eval_user_batch = os.environ.get("ALDI_OFFICIAL_TEST_BATCH_US", "512")
    bpr_emb = dataset_dir / "bprmf-BPR.npy"
    if args.force_teacher or not bpr_emb.exists():
        run_checked(
            py_cmd
            + [
                "bprmf.py",
                "--dataset",
                dataset,
                "--datadir",
                str(data_root.resolve()),
                "--gpu_id",
                gpu_id,
                "--seed",
                seed,
                "--batch_size",
                str(teacher_batch),
                "--factor_num",
                emb_dim,
                "--max_epoch",
                str(teacher_epochs),
                "--interval",
                str(teacher_interval),
                "--patience",
                os.environ.get("ALDI_OFFICIAL_TEACHER_PATIENCE", "0"),
                "--Ks",
                "[10]",
                "--test_batch_us",
                eval_user_batch,
                "--n_test_user",
                eval_users,
            ],
            cwd=src_dir / "warm_model",
        )
    else:
        print(f"Reuse existing teacher embedding: {bpr_emb}")

    result_path = static_result_path("aldi_official_static_result.json")
    run_checked(
        py_cmd
        + [
            "main_static_full_eval.py",
            "--dataset",
            dataset,
            "--datadir",
            str(data_root.resolve()),
            "--gpu_id",
            gpu_id,
            "--seed",
            seed,
            "--batch_size",
            str(_env_int("ALDI_OFFICIAL_BATCH_SIZE", "ALDI_BATCH_SIZE", default=4096)),
            "--max_epoch",
            str(_env_int("ALDI_OFFICIAL_STATIC_EPOCHS", "ALDI_STATIC_EPOCHS", default=100)),
            "--val_interval",
            str(_env_int("ALDI_OFFICIAL_EVAL_INTERVAL", "ALDI_EVAL_INTERVAL", default=5)),
            "--patience",
            os.environ.get("ALDI_OFFICIAL_PATIENCE", "0"),
            "--lr",
            str(_env_float("ALDI_OFFICIAL_LR", "ALDI_LR", default=1e-3)),
            "--reg",
            str(_env_float("ALDI_OFFICIAL_REG", "ALDI_REG", default=1e-4)),
            "--alpha",
            str(_env_float("ALDI_OFFICIAL_ALPHA", "ALDI_ALPHA", default=0.9)),
            "--beta",
            str(_env_float("ALDI_OFFICIAL_BETA", "ALDI_BETA", default=0.05)),
            "--gamma",
            str(_env_float("ALDI_OFFICIAL_GAMMA", "ALDI_GAMMA", default=0.1)),
            "--tws",
            os.environ.get("ALDI_OFFICIAL_TWS", os.environ.get("ALDI_TWS", "0")),
            "--cold_threshold",
            os.environ.get("ALDI_OFFICIAL_COLD_THRESHOLD", os.environ.get("USIM_COLD_THRESHOLD", "1")),
            "--eval_n_neg",
            os.environ.get("ALDI_OFFICIAL_EVAL_N_NEG", os.environ.get("USIM_EVAL_N_NEG", "200")),
            "--history_policy",
            os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only"),
            "--eval_batch_rows",
            os.environ.get("ALDI_OFFICIAL_EVAL_BATCH_SIZE", os.environ.get("ALDI_EVAL_BATCH_SIZE", "4096")),
            "--ckpt_dir",
            os.environ.get("ALDI_OFFICIAL_CKPT_DIR", os.environ.get("BASELINE_CKPT_DIR", "")),
            "--auto_resume",
            _env_bool_arg("ALDI_OFFICIAL_AUTO_RESUME", "BASELINE_AUTO_RESUME", default="0"),
            "--force_fresh",
            _env_bool_arg("ALDI_OFFICIAL_FORCE_FRESH", "BASELINE_FORCE_FRESH", default="0"),
            "--result_json",
            str(Path(result_path).resolve()),
        ],
        cwd=src_dir,
    )


def main():
    args = parse_args()
    dataset_dir = prepare_official_data(args)
    src_dir = prepare_runtime_source(args)
    if args.prepare_only:
        print("Prepare-only complete. Set ALDI_OFFICIAL_PYTHON to a TensorFlow Python and rerun without --prepare-only.")
        return
    run_official(args, src_dir, dataset_dir)


if __name__ == "__main__":
    main()
