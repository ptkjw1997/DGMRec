# coding: utf-8
r"""
MF-BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.

Plain matrix-factorization BPR (no multimodal features). Serves as the
traditional CF baseline; it is unaffected by missing modalities by
construction. (The previous file under this name was the DA-MRS framework
with an MF backbone and has been preserved as damrs_mf_backbone.py.bak.)
"""

import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class MFBPR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MFBPR, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.reg_weight = config['reg_weight']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_e = self.user_embedding(users)
        pos_e = self.item_embedding(pos_items)
        neg_e = self.item_embedding(neg_items)

        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_e, pos_e, neg_e)

        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_e = self.user_embedding(user)
        scores = torch.matmul(u_e, self.item_embedding.weight.transpose(0, 1))
        return scores
