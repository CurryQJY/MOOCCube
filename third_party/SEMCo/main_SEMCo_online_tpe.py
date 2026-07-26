import argparse
import torch
import numpy as np
import pickle
from util.loader import DataLoader
from config.model_param import model_specific_param
from model_imports import *
import os
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray import tune
import copy
import ray
import optuna


def objective(config):
    from models.SEMCo_online import SEMCo_Online

    inp_args = copy.deepcopy(args)
    for k, v in config.items():
        if "_ind" not in k and "_vals" not in k:
            vars(inp_args)[k] = v

    vars(inp_args)["cl_weight"] = config["cl_weight_vals"][config["cl_weight_ind"]]
    vars(inp_args)["ema_momentum"] = config["ema_momentum_vals"][
        config["ema_momentum_ind"]
    ]
    vars(inp_args)["reg"] = config["reg_vals"][config["reg_ind"]]
    vars(inp_args)["student_warmup_epochs"] = int(config["student_warmup_epochs"])
    # vars(inp_args)['student_zero_epochs'] = int(config['student_zero_epochs'])

    out_dicts = []
    for rep in range(args.num_repeats):
        vars(inp_args)["repeater"] = rep
        model = SEMCo_Online(
            inp_args,
            training_data,
            warm_valid_data,
            cold_valid_data,
            all_valid_data,
            warm_test_data,
            cold_test_data,
            all_test_data,
            user_num,
            item_num,
            warm_user_idx,
            warm_item_idx,
            cold_user_idx,
            cold_item_idx,
            device,
            user_content=None,
            item_content=item_content_refs,
        )
        model.train()
        out_dicts.append(model.out_dict)
    avg_out_dict = {k: np.mean([d[k] for d in out_dicts]) for k in out_dicts[0]}
    ray.train.report(avg_out_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--topN", default="20")
    parser.add_argument("--bs", type=int, default=4096, help="training batch size")
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--reg", type=float, default=0.005)
    parser.add_argument("--runs", type=int, default=1, help="model runs")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--use_gpu", default=True, help="Whether to use CUDA")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA id")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--feat_dir", type=str, default="feats", help="Feat file location"
    )
    parser.add_argument(
        "--cold_object", default="item", type=str, choices=["user", "item"]
    )
    parser.add_argument("--fn", default="sparsemax", type=str)

    args, _ = parser.parse_known_args()
    parser = model_specific_param(args.model, parser)
    args = parser.parse_args()
    print(args)

    device = torch.device(
        "cuda:%d" % (args.gpu_id)
        if (torch.cuda.is_available() and args.use_gpu)
        else "cpu"
    )
    # data loader
    training_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv"
        )
    )
    # following the widely used setting in previous works, the 'all' set is used for validation.
    all_valid_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv"
        )
    )
    warm_valid_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv"
        )
    )
    cold_valid_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv"
        )
    )
    all_test_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv"
        )
    )
    warm_test_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv"
        )
    )
    cold_test_data = ray.put(
        DataLoader.load_data_set(
            f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv"
        )
    )

    # dataset information
    data_info_dict = pickle.load(
        open(f"./data/{args.dataset}/cold_{args.cold_object}/info_dict.pkl", "rb")
    )
    user_num = data_info_dict["user_num"]
    item_num = data_info_dict["item_num"]
    warm_user_idx = data_info_dict["warm_user"]
    warm_item_idx = data_info_dict["warm_item"]
    cold_user_idx = data_info_dict["cold_user"]
    cold_item_idx = data_info_dict["cold_item"]
    print(f"Dataset: {args.dataset}, User num: {user_num}, Item num: {item_num}.")

    # content obtaining
    feat_filenames = sorted(os.listdir(f"./data/{args.dataset}/{args.feat_dir}"))
    feat_files = [f"./data/{args.dataset}/{args.feat_dir}/{f}" for f in feat_filenames]
    item_content = [np.load(f).astype(np.float32) for f in feat_files]
    item_content_refs = [ray.put(x) for x in item_content]

    cl_weights = [0.1, 0.5, 1, 2, 5]
    regs = [0.0, 0.00001, 0.0001, 0.001]
    ema_momentums = [0.0, 0.5, 0.9, 0.99]
    search_space = {
        "decay_lr_epoch": [True, 15],
        "student_training_epochs": tune.qrandint(15, 25, 5),
        "student_zero_epochs": 0,
        "student_warmup_epochs": tune.quniform(
            0, 5.1, 1.7
        ),  # equivalent to [0,1,3,5] for ordinal modeling
        "reg_vals": regs,
        "reg_ind": tune.randint(0, len(regs) - 1),
        "teacher_sm_scale": (
            tune.qrandint(10, 14, 2)
            if args.fn == "sparsemax"
            else tune.quniform(1, 5, 0.5)
        ),
        "student_sm_scale": (
            tune.qrandint(6, 20, 2)
            if args.fn == "sparsemax"
            else tune.quniform(1, 5, 0.5)
        ),
        "distil_scale": (
            tune.qrandint(6, 12, 2)
            if args.fn == "sparsemax"
            else tune.quniform(1, 5, 0.5)
        ),
        "ema_momentum_vals": ema_momentums,
        "ema_momentum_ind": tune.randint(0, len(ema_momentums) - 1),
        "cl_weight_vals": cl_weights,
        "cl_weight_ind": tune.randint(0, len(cl_weights) - 1),
        "sub_batch_positives": tune.qrandint(5, 15, 5),
        "lr": 0.001,
        "bs": 2048,
        "student_emb_sizes": [192, 64],
        "teacher_emb_sizes": 384,
    }
    search_space["fn"] = args.fn
    algo = OptunaSearch(
        sampler=optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True)
    )
    algo = ConcurrencyLimiter(algo, max_concurrent=args.runs_per_gpu)
    objective_with_resources = tune.with_resources(
        tune.with_parameters(objective),
        resources={"cpu": 1.0, "gpu": 1 / args.runs_per_gpu},
    )
    tuner = tune.Tuner(
        objective_with_resources,
        tune_config=tune.TuneConfig(
            metric="val_cold_ndcg",
            mode="max",
            search_alg=algo,
            num_samples=80,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
