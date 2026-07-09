import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import (
    ExponentialMovingAverage,
    GlobalPoolSampler,
    LearnedWeightedSum,
    next_batch_pairwise,
    sparse_mx_to_torch_sparse_tensor,
    get_warmup_scheduler,
    omega_alpha,
)
import torch.nn.functional as F
import random
from functools import partial
from entmax import entmax15, sparsemax


class SEMCo_Online(BaseColdStartTrainer):
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
        super(SEMCo_Online, self).__init__(
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
        self.teacher_hidden_size, self.teacher_emb_size = 384, args.teacher_emb_sizes
        self.student_hidden_size, self.student_emb_size = args.student_emb_sizes
        self.student_zero_epochs = args.student_zero_epochs
        self.student_warmup_epochs = args.student_warmup_epochs
        self.student_training_epochs = args.student_training_epochs
        self.lRate = args.lr
        self.eval_batch_size = args.eval_batch_size

        self.teacher_sm_scale = args.teacher_sm_scale
        self.student_sm_scale = args.student_sm_scale
        self.student_distil_scale, self.teacher_distil_scale = (
            args.distil_scale,
            args.distil_scale,
        )
        self.cl_weight = args.cl_weight
        self.ema_momentum = args.ema_momentum
        self.sub_batch_positives = args.sub_batch_positives
        self.teacher_model = Teacher_Learner(
            args, self.data, self.teacher_emb_size, self.teacher_hidden_size
        )
        self.student_model = Student_Learner(
            args,
            self.data,
            self.student_emb_size,
            self.student_hidden_size,
            self.teacher_emb_size,
        )

        self.fn, self.alpha = (
            (sparsemax, 2) if args.fn == "sparsemax" else (entmax15, 1.5)
        )
        self.sampler = GlobalPoolSampler(self.data.interaction_mat)

    def train(self):
        teacher_model = self.teacher_model.cuda()
        teacher_optimizer = torch.optim.Adam(
            teacher_model.parameters(),
            lr=self.lRate,
            weight_decay=self.weight_decay,
        )
        steps_per_epoch = len(self.data.training_data) // (self.batch_size)
        t_max = int(steps_per_epoch * self.maxEpoch)
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            teacher_optimizer, T_max=t_max, eta_min=0.0
        )

        student_model = self.student_model.cuda()
        student_optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=self.lRate,
            weight_decay=0.0,
        )
        student_scheduler = get_warmup_scheduler(
            student_optimizer,
            zero_steps=self.student_zero_epochs * steps_per_epoch,
            warmup_steps=self.student_warmup_epochs * steps_per_epoch,
        )

        num_batches = 100
        batch_size = 1 + self.data.interaction_mat.shape[0] // num_batches
        eval_fns = {
            "val": partial(self.eval_valid_new, **{"batch_size": batch_size}),
            "test": partial(self.eval_test_new, **{"batch_size": batch_size}),
        }
        idx_map = {v: i for i, v in enumerate(self.data.mapped_warm_item_idx)}

        ema = ExponentialMovingAverage(
            (len(self.data.mapped_warm_item_idx), self.teacher_emb_size),
            self.ema_momentum,
            "cuda",
        )
        for epoch in range(self.student_training_epochs):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, item_idx, _ = batch
                item_idx = [idx_map[i] for i in item_idx]

                teacher_user_vecs, teacher_feats = teacher_model(True, user_idx)
                ema.update(teacher_feats)

                if epoch < self.maxEpoch:
                    teacher_cl_loss = self.entmax_loss(
                        teacher_user_vecs,
                        teacher_feats[item_idx],
                        self.teacher_sm_scale,
                    )
                    # Backward and optimize
                    teacher_optimizer.zero_grad()
                    teacher_cl_loss.backward()
                    teacher_optimizer.step()
                    teacher_scheduler.step()

                if epoch >= self.student_zero_epochs:
                    teacher_input = ema.get()
                    student_user_vecs, student_feats = student_model(
                        teacher_input, True, user_idx
                    )
                    student_cl_loss = self.entmax_loss(
                        student_user_vecs,
                        student_feats[item_idx],
                        self.student_sm_scale,
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
                    distil_loss = self.distil_loss(
                        student_feats[sub_batch], teacher_input[sub_batch]
                    )

                    total_student_loss = self.cl_weight * student_cl_loss + distil_loss
                    # Backward and optimize
                    student_optimizer.zero_grad()
                    total_student_loss.backward()
                    student_optimizer.step()
                    student_scheduler.step()

        with torch.no_grad():
            _, teacher_output = self.teacher_model.eval()(perturbed=False)
            self.user_emb, self.item_emb = self.student_model.eval()(
                teacher_output, perturbed=False
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

    def entmax_loss(self, view1, view2, sm_scale):
        z = (view1 @ view2.T) / sm_scale
        p_star = self.fn(z / sm_scale, dim=1)
        entropy = omega_alpha(p_star, self.alpha).mean()
        e_y = torch.eye(z.shape[0]).float().cuda()

        alignment_loss = torch.einsum("ij,ij->i", p_star - e_y, z).mean()
        return alignment_loss + entropy

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


class Student_Learner(nn.Module):
    def __init__(self, args, data, emb_size, hidden_size, teacher_dim):
        super(Student_Learner, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.final_layer = nn.Sequential(
            nn.Linear(teacher_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size),
        )
        self.inter_mat_scaled = (
            data.interaction_mat / data.interaction_mat.sum(axis=1)
        ).tocsr()
        self.ui_adj_tensor = sparse_mx_to_torch_sparse_tensor(
            self.inter_mat_scaled
        ).cuda()

    def forward(self, teacher_input, perturbed=False, user_idx=None):
        if perturbed:
            interaction_tensor_dropped = sparse_mx_to_torch_sparse_tensor(
                self.inter_mat_scaled[user_idx][:, self.data.mapped_warm_item_idx]
            ).cuda()
        else:
            interaction_tensor_dropped = self.ui_adj_tensor

        feats_out = self.final_layer(teacher_input)
        user_vecs = torch.sparse.mm(interaction_tensor_dropped, F.normalize(feats_out))
        return F.normalize(user_vecs), F.normalize(feats_out)


class Teacher_Learner(nn.Module):
    def __init__(self, args, data, emb_size, hidden_size):
        super(Teacher_Learner, self).__init__()
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
