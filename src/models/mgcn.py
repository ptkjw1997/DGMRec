# coding: utf-8
# @email: y463213402@gmail.com
r"""
MGCN
################################################
Reference:
    https://github.com/demonph10/MGCN
    ACM MM'2023: [Multi-View Graph Convolutional Network for Multimedia Recommendation]
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class MGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MGCN, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.new_items = config['new_items']
        if config['new_items'] :
            self.new_items_set = np.load(f"../data/{config['dataset']}/new_items.npy")
            self.old_items_set = np.setdiff1d(np.arange(self.n_items), self.new_items_set)
        else :
            self.new_items_set = self.old_items_set = np.arange(self.n_items)

        self.complete_items = np.arange(self.n_items)
        self.missing_modal = config['missing_modal']
        self.missing_imputation = config['missing_imputation']
        self.missing_generation = config['missing_generation']
        self.cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)
        if config['missing_modal'] :
            self.preprocess_missing_modal(config)


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)

            image_adj = build_sim(self.image_embedding.weight.detach())
            if self.missing_modal :
                image_adj[self.missing_items_v, :] = image_adj[:, self.missing_items_v] = 0.0
                image_adj[self.missing_items_v, self.missing_items_v] = 1.0
            image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            # image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,norm_type='sym')
            image_adj = compute_normalized_laplacian(image_adj)

            image_adj = image_adj.to_sparse_coo()
            self.image_original_adj = image_adj.cuda()

            if self.new_items :
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj[self.new_items_set, :] = 0.0
                image_adj[:, self.new_items_set] = 0.0
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian(image_adj).to_sparse_coo()
                self.image_original_adj_newitems = image_adj.cuda()
            # image_adj = build_sim(self.image_embedding.weight.detach())
            # image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
            #                                         norm_type='sym')
            # self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

            text_adj = build_sim(self.text_embedding.weight.detach())
            if self.missing_modal :
                text_adj[self.missing_items_t, :] = text_adj[:, self.missing_items_t] = 0.0
                text_adj[self.missing_items_t, self.missing_items_t] = 1.0
            # text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            text_adj = compute_normalized_laplacian(text_adj)

            text_adj = text_adj.to_sparse_coo()
            self.text_original_adj = text_adj.cuda()

            if self.new_items :
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj[self.new_items_set, :] = 0.0
                text_adj[:, self.new_items_set] = 0.0
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo()
                self.text_original_adj_newitems = text_adj.cuda()

            # text_adj = build_sim(self.text_embedding.weight.detach())
            # text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            # self.text_original_adj = text_adj.cuda()

        if self.a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)

            audio_adj = build_sim(self.audio_embedding.weight.detach())
            if self.missing_modal :
                audio_adj[self.missing_items_a, :] = audio_adj[:, self.missing_items_a] = 0.0
                audio_adj[self.missing_items_a, self.missing_items_a] = 1.0
            audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
            audio_adj = compute_normalized_laplacian(audio_adj)

            audio_adj = audio_adj.to_sparse_coo()
            self.audio_original_adj = audio_adj.cuda()

            if self.new_items :
                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj[self.new_items_set, :] = 0.0
                audio_adj[:, self.new_items_set] = 0.0
                audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
                audio_adj = compute_normalized_laplacian(audio_adj).to_sparse_coo()
                self.audio_original_adj_newitems = audio_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        if self.a_feat is not None:
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        if self.a_feat is not None:
            self.gate_a = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        if self.a_feat is not None:
            self.gate_audio_prefer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )

        self.tau = 0.5

    def preprocess_missing_modal(self, config) :

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        if 'a' in self.missing_items:
            # 3-modality (tiktok) masks: keys {all, t, v, a, tv, ta, va}
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t'],
                                                    self.missing_items['tv'], self.missing_items['ta']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v'],
                                                    self.missing_items['tv'], self.missing_items['va']))
            self.missing_items_a = np.concatenate((self.missing_items['all'], self.missing_items['a'],
                                                    self.missing_items['ta'], self.missing_items['va']))

            self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(np.union1d(self.missing_items_v, self.missing_items_t), self.missing_items_a))
        else:
            # 2-modality (Amazon) masks: keys {all, t, v}
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))
            self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t))

            self.items_tv = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_t, self.missing_items_v))

        if config['missing_imputation'] == 0 :
            self.v_feat[self.missing_items_v] = 0.0
            self.t_feat[self.missing_items_t] = 0.0
            if self.a_feat is not None:
                self.a_feat[self.missing_items_a] = 0.0
        elif config['missing_imputation'] == 1 :
            non_missing_item_t = np.setdiff1d(self.old_items_set, self.missing_items_t)
            non_missing_item_v = np.setdiff1d(self.old_items_set, self.missing_items_v)
            if self.a_feat is not None:
                non_missing_item_a = np.setdiff1d(self.old_items_set, self.missing_items_a)

            image_mean = self.v_feat[non_missing_item_v].mean(dim = 0)
            text_mean = self.t_feat[non_missing_item_t].mean(dim = 0)
            if self.a_feat is not None:
                audio_mean = self.a_feat[non_missing_item_a].mean(dim = 0)

            self.v_feat[self.missing_items_v] = image_mean
            self.t_feat[self.missing_items_t] = text_mean
            if self.a_feat is not None:
                self.a_feat[self.missing_items_a] = audio_mean
        else :
            assert False, f"Missing Imputation Must bo 0 or 1, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if self.a_feat is not None:
            audio_feats = self.audio_trs(self.audio_embedding.weight)

        # Behavior-Guided Purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))
        if self.a_feat is not None:
            audio_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_a(audio_feats))

        if self.new_items and self.training :
            image_org_adj = self.image_original_adj_newitems
            text_org_adj = self.text_original_adj_newitems
            if self.a_feat is not None:
                audio_org_adj = self.audio_original_adj_newitems

            mask = torch.ones(image_item_embeds.shape[0]).to(self.device)
            mask[self.new_items_set] = 0.0
            with torch.no_grad() :
                image_item_embeds = torch.einsum("ij, i -> ij", image_item_embeds, mask)
                text_item_embeds = torch.einsum("ij, i -> ij", text_item_embeds, mask)
                if self.a_feat is not None:
                    audio_item_embeds = torch.einsum("ij, i -> ij", audio_item_embeds, mask)
                # item_embeds = torch.einsum("ij, i -> ij", item_embeds, mask)
        else :
            image_org_adj = self.image_original_adj
            text_org_adj = self.text_original_adj
            if self.a_feat is not None:
                audio_org_adj = self.audio_original_adj

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(image_org_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(image_org_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(text_org_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(text_org_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        if self.a_feat is not None:
            if self.sparse:
                for i in range(self.n_layers):
                    audio_item_embeds = torch.sparse.mm(audio_org_adj, audio_item_embeds)
            else:
                for i in range(self.n_layers):
                    audio_item_embeds = torch.mm(audio_org_adj, audio_item_embeds)
            audio_user_embeds = torch.sparse.mm(self.R, audio_item_embeds)
            audio_embeds = torch.cat([audio_user_embeds, audio_item_embeds], dim=0)

        # Behavior-Aware Fuser
        if self.a_feat is not None:
            att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds), self.query_common(audio_embeds)], dim=-1)
        else:
            att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        if self.a_feat is not None:
            common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + \
                weight_common[:, 1].unsqueeze(dim=1) * text_embeds + \
                weight_common[:, 2].unsqueeze(dim=1) * audio_embeds
        else:
            common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
                dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds
        if self.a_feat is not None:
            sep_audio_embeds = audio_embeds - common_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        if self.a_feat is not None:
            audio_prefer = self.gate_audio_prefer(content_embeds)

        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (tiktok fuser: 4-way average)
            sep_audio_embeds = torch.multiply(audio_prefer, sep_audio_embeds)

            side_embeds = (sep_image_embeds + sep_text_embeds +sep_audio_embeds+ common_embeds) / 4
        else:
            side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores