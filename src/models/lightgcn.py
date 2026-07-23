# coding: utf-8
r"""
LightGCN
################################################
Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Plain collaborative-filtering LightGCN (no multimodal features). Serves as the
traditional CF baseline; it is unaffected by missing modalities by construction.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class LightGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']

        self.n_nodes = self.n_users + self.n_items

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        # scipy>=1.12 removed dok_matrix._update; build a COO matrix directly instead
        _rows, _cols = zip(*data_dict.keys())
        A = sp.coo_matrix((list(data_dict.values()), (list(_rows), list(_cols))), shape=A.shape, dtype=np.float32)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)

        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()

        u_e = u_embeddings[users]
        pos_e = i_embeddings[pos_items]
        neg_e = i_embeddings[neg_items]

        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)

        if self.reg_weight:
            mf_loss = self.mf_loss(pos_scores, neg_scores)
            reg_loss = self.reg_loss(
                self.user_embedding(users), self.item_embedding(pos_items), self.item_embedding(neg_items))

            return mf_loss + self.reg_weight * reg_loss

        # reg_weight=0 reproduces the historical 3-modality (tiktok) objective:
        # plain -mean(logsigmoid(pos - neg)) with no regularization term.
        # 3-modality historical behavior preserved
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return mf_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        u_e = u_embeddings[user]
        scores = torch.matmul(u_e, i_embeddings.transpose(0, 1))
        return scores
