# coding: utf-8

import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class BM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.new_items = config['new_items']
        if config['new_items'] :
            self.new_items_set = np.load(f"../data/{config['dataset']}/new_items.npy")
            self.old_items_set = np.setdiff1d(np.arange(self.n_items), self.new_items_set)

            with torch.no_grad() :
                self.item_id_embedding.weight[self.new_items_set] = 0.0
        else :
            self.new_items_set = self.old_items_set = np.arange(self.n_items)

        if config['missing_modal'] :
            self.preprocess_missing_modal(config)
        self.missing_modal = config['missing_modal']
        self.missing_imputation = config['missing_imputation']

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)
        if self.a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)  # NOTE: historical 3-modality behavior preserved (re-inits text_trs, not audio_trs)

    def preprocess_missing_modal(self, config) :

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        if 'a' in self.missing_items :
            # NOTE: historical 3-modality (tiktok) branch
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t'],
                                                    self.missing_items['tv'], self.missing_items['ta']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v'],
                                                    self.missing_items['tv'], self.missing_items['va']))
            self.missing_items_a = np.concatenate((self.missing_items['all'], self.missing_items['a'],
                                                    self.missing_items['ta'], self.missing_items['va']))
        else :
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))

        if config['missing_imputation'] == 0 :
            self.v_feat[self.missing_items_v] = 0.0
            self.t_feat[self.missing_items_t] = 0.0
            if self.a_feat is not None :
                self.a_feat[self.missing_items_a] = 0.0
        elif config['missing_imputation'] == 1 :
            non_missing_item_t = np.setdiff1d(self.old_items_set, self.missing_items_t)
            non_missing_item_v = np.setdiff1d(self.old_items_set, self.missing_items_v)
            if self.a_feat is not None :
                non_missing_item_a = np.setdiff1d(self.old_items_set, self.missing_items_a)

            image_mean = self.v_feat[non_missing_item_v].mean(dim = 0)
            text_mean = self.t_feat[non_missing_item_t].mean(dim = 0)
            if self.a_feat is not None :
                audio_mean = self.a_feat[non_missing_item_a].mean(dim = 0)

            self.v_feat[self.missing_items_v] = image_mean
            self.t_feat[self.missing_items_t] = text_mean
            if self.a_feat is not None :
                self.a_feat[self.missing_items_a] = audio_mean
        elif config['missing_imputation'] == 2 :
            pass
        else :
            assert False, f"Missing Imputation Must bo 0 or 1 or 2, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']


    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        # scipy>=1.12 removed dok_matrix._update; build a COO matrix directly instead
        _rows, _cols = zip(*data_dict.keys())
        A = sp.coo_matrix((list(data_dict.values()), (list(_rows), list(_cols))), shape=A.shape, dtype=np.float32)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online, a_feat_online = None, None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
            if self.missing_modal and self.a_feat is None :
                # NOTE: 2-modality (src) tree masks missing-modality items; 3-modality (tiktok) tree does not
                t_mask = torch.ones(self.n_items).to(self.device)
                t_mask[self.missing_items_t] = 0.0

                t_feat_online = torch.einsum("ij, i -> ij", t_feat_online, t_mask)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)
            if self.missing_modal and self.a_feat is None :
                # NOTE: 2-modality (src) tree masks missing-modality items; 3-modality (tiktok) tree does not
                v_mask = torch.ones(self.n_items).to(self.device)
                v_mask[self.missing_items_v] = 0.0

                v_feat_online = torch.einsum("ij, i -> ij", v_feat_online, v_mask)
        if self.a_feat is not None:
            a_feat_online = self.audio_trs(self.audio_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

            if self.a_feat is not None:
                a_feat_target = a_feat_online.clone()
                a_feat_target = F.dropout(a_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt, loss_a, loss_at = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()
        if self.a_feat is not None:
            a_feat_online = self.predictor(a_feat_online)
            a_feat_online = a_feat_online[items, :]
            a_feat_target = a_feat_target[items, :]
            loss_a = 1 - cosine_similarity(a_feat_online, i_target.detach(), dim=-1).mean()
            loss_at = 1 - cosine_similarity(a_feat_online, a_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        if self.a_feat is not None:
            # NOTE: historical 3-modality (tiktok) branch (adds audio loss terms)
            return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
                   self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt + loss_a + loss_at).mean()
        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)

        # if self.new_items :
        #     modal_online = (self.predictor(self.text_trs(self.text_embedding.weight)) + self.predictor(self.image_trs(self.image_embedding.weight))) / 2.0

        #     i_online[self.new_items_set] = modal_online[self.new_items_set]

        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

