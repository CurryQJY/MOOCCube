import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import (
    LearnedWeightedSum,
    next_batch_pairwise,
    sparse_mx_to_torch_sparse_tensor,
    omega_alpha,
)
import torch.nn.functional as F
from functools import partial
from entmax import entmax15, sparsemax


class SEMCo(BaseColdStartTrainer):
    def __init__(
        self,
        args,
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
        item_content=None,
    ):
        super(SEMCo, self).__init__(
            args,
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
            user_content=user_content,
            item_content=item_content,
        )
        self.patience = args.patience
        self.decay_lr, self.maxEpoch = args.decay_lr_epoch
        self.weight_decay = args.reg
        self.hidden_size, self.emb_size = args.emb_sizes
        self.lRate = args.lr
        self.eval_batch_size = args.eval_batch_size
        self.sm_scale = args.sm_scale

        self.model = SEMCo_Learner(self.data, self.emb_size, self.hidden_size)
        self.bestPerformance = []

        self.fn, self.alpha = (
            (sparsemax, 2) if args.fn == "sparsemax" else (entmax15, 1.5)
        )
        if args.fn == "softmax":
            self.loss_fn = self.InfoNCE
        else:
            self.loss_fn = self.entmax_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lRate,
            weight_decay=self.weight_decay,
        )
        if self.decay_lr:
            steps_per_epoch = len(self.data.training_data) // (self.batch_size)
            t_max = int(steps_per_epoch * self.maxEpoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=0.0
            )

        num_batches = 100
        batch_size = 1 + self.data.interaction_mat.shape[0] // num_batches
        eval_fns = {
            "val": partial(self.eval_valid_new, **{"batch_size": batch_size}),
            "test": partial(self.eval_test_new, **{"batch_size": batch_size}),
        }
        idx_map = {v: i for i, v in enumerate(self.data.mapped_warm_item_idx)}

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, item_idx, _ = batch
                item_idx = [idx_map[i] for i in item_idx]
                user_vecs, feats = model(True, user_idx)
                cl_loss = self.loss_fn(user_vecs, feats[item_idx])
                # Backward and optimize
                optimizer.zero_grad()
                cl_loss.backward()
                optimizer.step()
                if self.decay_lr:
                    scheduler.step()

        with torch.no_grad():
            self.user_emb, self.item_emb = self.model.eval()(perturbed=False)

        self.out_dict = {}
        for split in ["val", "test"]:
            eval_fns[split]()
            if split == "val":
                warm_recall, warm_ndcg = self.warm_valid_results
                cold_recall, cold_ndcg = self.cold_valid_results
            if split == "test":
                warm_recall, warm_ndcg = self.warm_test_results
                cold_recall, cold_ndcg = self.cold_test_results
            self.out_dict = {
                **self.out_dict,
                **{
                    "%s_cold_recall" % split: cold_recall,
                    "%s_cold_ndcg" % split: cold_ndcg,
                    "%s_warm_recall" % split: warm_recall,
                    "%s_warm_ndcg" % split: warm_ndcg,
                },
            }

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = self.scores[u]
            return score.numpy()

    def entmax_loss(self, view1, view2):
        z = (view1 @ view2.T) / self.sm_scale
        p_star = self.fn(z / self.sm_scale, dim=1)
        entropy = omega_alpha(p_star, self.alpha).mean()
        e_y = torch.eye(z.shape[0]).float().cuda()

        alignment_loss = torch.einsum("ij,ij->i", p_star - e_y, z).mean()
        return alignment_loss + entropy

    def InfoNCE(self, view1, view2):
        pos_score = (view1 @ view2.T) / self.sm_scale
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()


class SEMCo_Learner(nn.Module):
    def __init__(self, data, emb_size, hidden_size):
        super(SEMCo_Learner, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.feats_raw = [
            F.normalize(torch.from_numpy(x).cuda()) for x in data.mapped_item_content
        ]
        self.feats_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(arr.shape[1], self.hidden_size),
                    nn.BatchNorm1d(self.hidden_size, track_running_stats=False),
                    nn.ReLU(),
                )
                for arr in self.feats_raw
            ]
        )

        self.fuser = LearnedWeightedSum(self.hidden_size, len(self.feats_raw))
        self.final_layer = nn.Linear(self.hidden_size, self.emb_size)
        self.inter_mat_scaled = (
            data.interaction_mat / data.interaction_mat.sum(axis=1)
        ).tocsr()[:, data.mapped_warm_item_idx]
        self.ui_adj_tensor = sparse_mx_to_torch_sparse_tensor(
            self.inter_mat_scaled
        ).cuda()

    def forward(self, perturbed=False, user_idx=None):
        if perturbed:
            interaction_tensor_dropped = sparse_mx_to_torch_sparse_tensor(
                self.inter_mat_scaled[user_idx]
            ).cuda()
            feats_lst = [
                self.feats_layers[i](arr[self.data.mapped_warm_item_idx])
                for i, arr in enumerate(self.feats_raw)
            ]
            fused = self.fuser(feats_lst)
            feats_out = self.final_layer(fused)
            user_vecs = torch.sparse.mm(
                interaction_tensor_dropped, F.normalize(feats_out)
            )
        else:
            feats_lst = [
                self.feats_layers[i](arr) for i, arr in enumerate(self.feats_raw)
            ]
            fused = self.fuser(feats_lst)
            feats_out = self.final_layer(fused)
            user_vecs = torch.sparse.mm(
                self.ui_adj_tensor,
                F.normalize(feats_out)[self.data.mapped_warm_item_idx],
            )

        return F.normalize(user_vecs), F.normalize(feats_out)
