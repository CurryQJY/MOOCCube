import argparse
import torch
import numpy as np
import pickle
from util.loader import DataLoader
import os
import copy


def objective(config):
    from models.SEMCo import SEMCo

    inp_args = copy.deepcopy(args)
    for k, v in config.items():
        vars(inp_args)[k] = v

    model = SEMCo(
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
        item_content=item_content,
    )
    model.train()
    return model.out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clothing")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--topN", default="20")
    parser.add_argument("--bs", type=int, default=2048, help="training batch size")
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--reg", type=float, default=0.005)
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
    args = parser.parse_args()
    print(args)

    device = torch.device(
        "cuda:%d" % (args.gpu_id)
        if (torch.cuda.is_available() and args.use_gpu)
        else "cpu"
    )
    # data loader
    training_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/warm_train.csv"
    )
    # following the widely used setting in previous works, the 'all' set is used for validation.
    all_valid_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/overall_val.csv"
    )
    warm_valid_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/warm_val.csv"
    )
    cold_valid_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_val.csv"
    )
    all_test_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/overall_test.csv"
    )
    warm_test_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/warm_test.csv"
    )
    cold_test_data = DataLoader.load_data_set(
        f"./data/{args.dataset}/cold_{args.cold_object}/cold_{args.cold_object}_test.csv"
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

    search_space_final = {
        "clothing": {
            "sparsemax": {"reg": 0.001, "sm_scale": 12},
            "softmax": {"reg": 0.005, "sm_scale": 0.2},
            "entmax15": {"reg": 0.005, "sm_scale": 2.5},
        },
        "microlens": {
            "sparsemax": {"reg": 0.0, "sm_scale": 14},
            "softmax": {"reg": 0.0001, "sm_scale": 0.2},
            "entmax15": {"reg": 0.001, "sm_scale": 2.5},
        },
        "onion": {
            "sparsemax": {"reg": 0.0001, "sm_scale": 10},
            "softmax": {"reg": 0.01, "sm_scale": 0.1},
            "entmax15": {"reg": 0.0005, "sm_scale": 2},
        },
        "electronics": {
            "sparsemax": {"reg": 0.0001, "sm_scale": 12},
            "softmax": {"reg": 0.01, "sm_scale": 0.15},
            "entmax15": {"reg": 0.001, "sm_scale": 2},
        },
    }

    search_space = {
        "decay_lr_epoch": [True, 15],
        "save_emb": False,
        "lr": 0.001,
        "bs": 2048,
        "emb_sizes": (192, 64),
    }

    search_space["fn"] = args.fn
    search_space = {**search_space, **search_space_final[args.dataset][args.fn]}
    out_dict = objective(search_space)
    print(out_dict)
