from random import shuffle, choice
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import os
import scipy.sparse as sp


class LearnedWeightedSum(nn.Module):
    def __init__(self, input_dim, num_inputs=2):

        super().__init__()
        self.num_inputs = num_inputs
        self.input_dim = input_dim

        # Maps concatenated inputs to weights
        self.attention = nn.Linear(input_dim * num_inputs, num_inputs)

    def forward(self, inputs):
        """
        Args:
            inputs: List/tuple of 2D tensors [batch_size, input_dim]

        Returns:
            Weighted sum and attention weights (both [batch_size, input_dim] and [batch_size, num_inputs])
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        stacked = torch.stack(inputs, dim=1)
        batch_size = stacked.shape[0]

        concatenated = stacked.view(batch_size, -1)
        attention_scores = self.attention(concatenated)
        attention_weights = torch.softmax(attention_scores, dim=1)
        result = torch.einsum("bi,bid->bd", attention_weights, stacked)

        return result


class GlobalPoolSampler:
    def __init__(self, adj_matrix, device="cuda"):
        self.indptr = torch.from_numpy(adj_matrix.indptr).to(device).long()
        self.indices = torch.from_numpy(adj_matrix.indices).to(device).long()
        self.device = device

    def sample_unique(self, user_indices, num_samples=5):
        """
        Samples items for users and returns a flat tensor of unique item IDs.
        """
        if not isinstance(user_indices, torch.Tensor):
            user_indices = torch.tensor(user_indices, device=self.device)

        # 1. Get the slice boundaries for the users in the batch
        starts = self.indptr[user_indices]
        ends = self.indptr[user_indices + 1]
        counts = ends - starts

        # 2. Filter out users with zero interactions to avoid errors
        valid_mask = counts > 0
        valid_starts = starts[valid_mask]
        valid_counts = counts[valid_mask]

        if valid_starts.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        # 3. Vectorized random sampling across the valid batch
        num_valid = valid_starts.size(0)
        random_noise = torch.rand((num_valid, num_samples), device=self.device)
        offsets = (random_noise * valid_counts.unsqueeze(1)).long()

        # 4. Global lookup and flattening
        all_samples = self.indices[valid_starts.unsqueeze(1) + offsets].view(-1)

        # 5. Extract unique items
        unique_items = torch.unique(all_samples, sorted=False)

        return unique_items


class ExponentialMovingAverage:
    def __init__(self, shape, momentum=0.99, device="cpu"):
        self.momentum = momentum
        self.ema = torch.zeros(shape, device=device)
        self.initialized = False

    def update(self, X):
        if not self.initialized:
            self.ema = X.clone().detach()
            self.initialized = True
        else:
            self.ema = self.momentum * self.ema + (1 - self.momentum) * X.detach()
        self.ema = F.normalize(self.ema)

    def get(self):
        """Returns the current EMA value."""
        return self.ema

    def reset(self):
        """Reset the EMA to zeros."""
        self.ema.zero_()
        self.initialized = False


def get_warmup_scheduler(optimizer, zero_steps, warmup_steps):
    """
    Creates a scheduler that returns 0 for the first `zero_steps` steps,
    then linearly increases from 0 to optimizer's base LR over `warmup_steps` steps
    """

    def lr_lambda(step):
        if step < zero_steps:
            return 0.0
        elif step < zero_steps + warmup_steps:
            # Linear warmup from 0 to base LR
            progress = (step - zero_steps) / warmup_steps
            return progress
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def omega_alpha(P, alpha):
    return (1 - torch.sum(P**alpha, dim=-1)) / (alpha * (alpha - 1))


def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        # item_list = list(data.item.keys())
        item_list = list(data.source_warm_item_idx)
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def set_seed(seed, cuda):
    print("Set Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    zeros = torch.zeros(sparse_mx.shape, dtype=torch.float32)
    values = torch.from_numpy(sparse_mx.data)
    zeros[sparse_mx.row, sparse_mx.col] = values
    return zeros
