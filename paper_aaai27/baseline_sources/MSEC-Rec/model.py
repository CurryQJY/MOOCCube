import torch
import numpy as np
import dgl
from torch import nn
from torch.nn import functional as F
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.attention_weights = None

    def forward(self, z):
        # z: (num_nodes, num_meta_paths, in_size)
        w = self.project(z).mean(0)  # (num_meta_paths, 1)
        beta = torch.softmax(w, dim=0)  # (num_meta_paths, 1)


        self.attention_weights = beta.detach().cpu()

        beta = beta.expand((z.shape[0],) + beta.shape)  # (num_nodes, num_meta_paths, 1)
        return (beta * z).sum(1)  # (num_nodes, in_size)


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList([
            GATConv(in_size, out_size, layer_num_heads, dropout, dropout,
                    activation=F.elu, allow_zero_in_degree=True)
            for _ in meta_paths
        ])
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = [tuple(meta_path) for meta_path in meta_paths]
        self._cached_graph = None
        self._cached_coalesced_graph = {}


        self.edge_attn_weights = {}


        self.rw_start_ntype = 'user'
        self.rw_num_traces = 2000
        self.rw_restart_prob = 0.0


    def _sample_metapath_graph(self, g, meta_path):
        start_nodes = g.nodes(self.rw_start_ntype)
        expanded_start_nodes = start_nodes.repeat_interleave(self.rw_num_traces)

        traces, _ = dgl.sampling.random_walk(
            g, expanded_start_nodes,
            metapath=meta_path,
            restart_prob=self.rw_restart_prob
        )

        src_nodes, dst_nodes = [], []
        for trace in traces.tolist():
            if trace[0] != -1 and trace[-1] != -1:
                src_nodes.append(trace[0])
                dst_nodes.append(trace[-1])

        num_user_nodes = g.num_nodes('user')
        if len(src_nodes) == 0:
            sampled_g = dgl.metapath_reachable_graph(g, meta_path)
        else:
            sampled_g = dgl.graph(
                (src_nodes, dst_nodes),
                num_nodes=num_user_nodes
            )
        return sampled_g

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = self._sample_metapath_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path].to(h.device)


            h_mp, attn = self.gat_layers[i](new_g, h, get_attention=True)  
            semantic_embeddings.append(h_mp.flatten(1))


            self.edge_attn_weights[meta_path] = attn.detach().cpu()

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout))
        self.additional_layers = nn.ModuleList()
        for _ in range(1):
            self.additional_layers.append(
                HANLayer(meta_paths, hidden_size * num_heads[-1], hidden_size, num_heads[-1], dropout))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        for layer in self.additional_layers:
            h = layer(g, h)
        return h



class BprMF(nn.Module):
    def __init__(self, n_user, m_item, dim, reg):
        super(BprMF, self).__init__()
        self.n_user = n_user
        self.m_item = m_item
        self.dim = dim
        self.reg = reg

        self.Embedding_User = nn.Embedding(n_user, dim)
        self.Embedding_Item = nn.Embedding(m_item, dim)
        nn.init.xavier_uniform_(self.Embedding_User.weight)
        nn.init.xavier_uniform_(self.Embedding_Item.weight)

    def forward(self, users, items):
        users_emb = self.Embedding_User(users)
        items_emb = self.Embedding_Item(items)
        scores = torch.sum(users_emb * items_emb, dim=-1)
        return scores

    def bpr_loss(self, users, pos_items, neg_items):
        """
        users: [batch]
        pos_items: [batch]
        neg_items: [batch, num_neg]
        """
        # Embedding lookup
        user_emb = self.Embedding_User(users)  # [batch, d]
        pos_emb = self.Embedding_Item(pos_items)  # [batch, d]
        neg_emb = self.Embedding_Item(neg_items)  # [batch, num_neg, d]

        pos_scores = torch.sum(user_emb * pos_emb, dim=-1, keepdim=True)  # [batch, 1]

        user_emb_expanded = user_emb.unsqueeze(1)  # [batch, 1, d]

        neg_scores = torch.sum(user_emb_expanded * neg_emb, dim=-1)  # [batch, num_neg]

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss


class HybridModel(nn.Module):
    def __init__(self, han, bprmf, alpha=0.5):
        super(HybridModel, self).__init__()
        self.han = han
        self.bprmf = bprmf
        self.alpha = alpha

    def forward(self, g, h, users, pos_items, neg_items):
        node_embeddings = self.han(g, h)
        user_embeddings = node_embeddings[:self.bprmf.n_user]
        self.bprmf.Embedding_User.weight.data = user_embeddings
        item_embeddings = self.bprmf.Embedding_Item.weight.data

        anchor, positive, negative = user_embeddings[users], item_embeddings[pos_items], item_embeddings[neg_items]
        han_loss = self.contrastive_loss(anchor, positive)
        bprmf_loss=self.bprmf.bpr_loss(users, pos_items, neg_items)
        loss = self.alpha * han_loss + (1 - self.alpha) * bprmf_loss
        return loss

    def contrastive_loss(self, anchor, positive):
        loss=F.mse_loss(anchor,positive,reduction="mean")
        return loss

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
        

