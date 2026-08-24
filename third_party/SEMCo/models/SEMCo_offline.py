import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import (
    GlobalPoolSampler,
    next_batch_pairwise,
    sparse_mx_to_torch_sparse_tensor,
    omega_alpha,
)
import torch.nn.functional as F
import random
import pickle
from functools import partial
from entmax import entmax15, sparsemax


class SEMCo_Offline(BaseColdStartTrainer):
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
        super(SEMCo_Offline, self).__init__(
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
        self.student_hidden_size, self.student_emb_size = args.student_emb_sizes
        self.lRate = args.lr
        self.eval_batch_size = args.eval_batch_size
        self.student_sm_scale = args.student_sm_scale
        self.student_distil_scale, self.teacher_distil_scale = (
            args.distil_scale,
            args.distil_scale,
        )
        self.cl_weight = args.cl_weight
        self.sub_batch_positives = args.sub_batch_positives
        self.fn_name = args.fn

        self.student_model = Student_Learner(
            args, self.data, self.student_emb_size, self.student_hidden_size
        )

        self.fn, self.alpha = (
            (sparsemax, 2) if args.fn == "sparsemax" else (entmax15, 1.5)
        )
        self.sampler = GlobalPoolSampler(self.data.interaction_mat)

        if args.fn == "softmax":
            self.loss_fn = self.InfoNCE
            self.distil_loss_fn = self.distil_loss_softmax
        else:
            self.loss_fn = self.entmax_loss
            self.distil_loss_fn = self.distil_loss

    def train(self):
        student_model = self.student_model.cuda()
        student_optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=self.lRate,
            weight_decay=0.0,
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

                teacher_input = self.student_model.embedding_dict["item_emb"][
                    self.data.mapped_warm_item_idx
                ]
                student_user_vecs, student_feats = student_model(
                    teacher_input, True, user_idx
                )
                student_cl_loss = self.loss_fn(
                    student_user_vecs, student_feats[item_idx]
                )

                user_unique = list(set(user_idx))
                sub_batch = set([])
                num_sample = self.batch_size // self.sub_batch_positives
                if num_sample > len(user_unique):
                    user_sample = user_unique
                    sub_batch = sub_batch.union(
                        set(
                            self.sampler.sample_unique(
                                user_sample, self.sub_batch_positives
                            )
                            .cpu()
                            .numpy()
                        )
                    )
                else:
                    while len(sub_batch) <= self.batch_size:
                        user_sample = random.sample(user_unique, num_sample)
                        sub_batch = sub_batch.union(
                            set(
                                self.sampler.sample_unique(
                                    user_sample, self.sub_batch_positives
                                )
                                .cpu()
                                .numpy()
                            )
                        )
                sub_batch = random.sample(
                    list(sub_batch), min(self.batch_size, len(sub_batch))
                )
                sub_batch = [idx_map[i] for i in sub_batch]
                distil_loss = self.distil_loss_fn(
                    student_feats[sub_batch], teacher_input[sub_batch]
                )

                total_student_loss = self.cl_weight * student_cl_loss + distil_loss
                # Backward and optimize
                student_optimizer.zero_grad()
                total_student_loss.backward()
                student_optimizer.step()

        with torch.no_grad():
            self.user_emb, self.item_emb = self.student_model.eval()(
                self.student_model.embedding_dict["item_emb"], perturbed=False
            )

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
        z = (view1 @ view2.T) / self.student_sm_scale
        p_star = self.fn(z / self.student_sm_scale, dim=1)
        entropy = omega_alpha(p_star, self.alpha).mean()
        e_y = torch.eye(z.shape[0]).float().cuda()

        alignment_loss = torch.einsum("ij,ij->i", p_star - e_y, z).mean()
        return alignment_loss + entropy

    def InfoNCE(self, view1, view2):
        pos_score = (view1 @ view2.T) / self.student_sm_scale
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def distil_loss(self, student_feats, teacher_feats):
        z = student_feats @ student_feats.T
        z.fill_diagonal_(-1)
        z /= self.student_distil_scale
        p_star = self.fn(z, dim=1)
        entropy = omega_alpha(p_star, self.alpha).mean()

        z_teacher = teacher_feats @ teacher_feats.T
        z_teacher.fill_diagonal_(-1)
        z_teacher /= self.teacher_distil_scale
        e_y = self.fn(z_teacher, dim=1)

        alignment_loss = torch.einsum("ij,ij->i", p_star - e_y, z).mean()
        return alignment_loss + entropy

    def distil_loss_softmax(self, student_feats, teacher_feats):
        z = student_feats @ student_feats.T
        z.fill_diagonal_(-1)
        p_star = torch.softmax(z / self.student_distil_scale, dim=1)

        z_teacher = teacher_feats @ teacher_feats.T
        z_teacher.fill_diagonal_(-1)
        e_y = torch.softmax(z_teacher / self.teacher_distil_scale, dim=1)

        log_pred = torch.log(p_star + 1e-9)
        loss = torch.mean(torch.sum(-e_y * log_pred, dim=-1))
        return loss


class Student_Learner(nn.Module):
    def __init__(self, args, data, emb_size, hidden_size):
        super(Student_Learner, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        with open(args.teacher_file, "rb") as f:
            item_teacher = pickle.load(f)[1]

        self.teacher_dim = item_teacher.shape[1]

        self.embedding_dict = nn.ParameterDict(
            {
                "item_emb": item_teacher.cuda(),
            }
        )
        self.embedding_dict["item_emb"].requires_grad = False

        self.final_layer = nn.Sequential(
            nn.Linear(self.teacher_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size),
        )
        self.inter_mat_scaled = (
            data.interaction_mat / data.interaction_mat.sum(axis=1)
        ).tocsr()[:, self.data.mapped_warm_item_idx]
        self.ui_adj_tensor = sparse_mx_to_torch_sparse_tensor(
            self.inter_mat_scaled
        ).cuda()

    def forward(self, teacher_input, perturbed=False, user_idx=None):
        if perturbed:
            interaction_tensor_dropped = sparse_mx_to_torch_sparse_tensor(
                self.inter_mat_scaled[user_idx]
            ).cuda()
        else:
            interaction_tensor_dropped = self.ui_adj_tensor

        feats_out = self.final_layer(teacher_input)
        user_vecs = torch.sparse.mm(interaction_tensor_dropped, F.normalize(feats_out))
        return F.normalize(user_vecs), F.normalize(feats_out)
