# coding: utf-8

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class DAMRS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DAMRS, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']

        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']

        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers']

        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.kl_weight = config['kl_weight']
        self.neighbor_weight = config['neighbor_weight']
        # self.gen_weight = config['gen_weight']
        self.build_item_graph = True

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.new_items = config['new_items']
        if config['new_items'] :
            self.new_items_set = np.load(f"../data/{config['dataset']}/new_items.npy")
            self.old_items_set = np.setdiff1d(np.arange(self.n_items), self.new_items_set)
        else :
            self.new_items_set = self.old_items_set = np.arange(self.n_items)

        self.complete_items = np.arange(self.n_items)
        if config['missing_modal'] :
            self.preprocess_missing_modal(config)
        self.missing_modal = config['missing_modal']
        self.missing_imputation = config['missing_imputation']

        if self.a_feat is not None:
            # 3-modality historical behavior preserved ('normalize' unset in released tiktok runs -> no-op)
            if config['normalize'] :
                self.v_feat = F.normalize(self.v_feat)
                self.t_feat = F.normalize(self.t_feat)
                self.a_feat = F.normalize(self.a_feat)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        if self.a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=True)
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.embedding_dim)

        if self.a_feat is not None:
            # 3-modality historical behavior preserved (tiktok build order: delete_new_items=True graphs first)
            self.image_adj, self.text_adj, self.audio_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), self.audio_embedding.weight.detach(), delete_new_items = True)
            self.image_adj_infer, self.text_adj_infer, self.audio_adj_infer = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), self.audio_embedding.weight.detach(), delete_new_items = False)
        else :
            self.image_adj_infer, self.text_adj_infer = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), delete_new_items = False)
            self.image_adj, self.text_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), delete_new_items = True)
        # self.image_adj_newitems, self.text_adj_newitems = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach())
        # self.image_adj, self.text_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), delete_new_items = False)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()

        __, self.session_adj = self.get_session_adj()

        # # Loss Gen #
        # # Modality Generation
        # self.image_generator = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.v_feat.shape[1])
        # )
        # self.text_generator = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.t_feat.shape[1])
        # )
        # nn.init.xavier_uniform_(self.image_generator[0].weight); nn.init.xavier_uniform_(self.image_generator[2].weight)
        # nn.init.xavier_uniform_(self.text_generator[0].weight); nn.init.xavier_uniform_(self.text_generator[2].weight)

        # self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        # # Loss Gen #

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    # def pre_epoch_processing(self):
    #     if self.missing_modal :

    #         ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
    #         all_embeddings = [ego_embeddings]
    #         for i in range(self.n_ui_layers):
    #             side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
    #             ego_embeddings = side_embeddings
    #             all_embeddings += [ego_embeddings]
    #         all_embeddings = torch.stack(all_embeddings, dim=1)
    #         all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
    #         u_g_embeddings, _ = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

    #         del ego_embeddings, side_embeddings

    #         item_embs_agg = torch.sparse.mm(self.adj.t(), u_g_embeddings) * self.num_inters[self.n_users:]
    #         image_embs_gen = self.image_generator(item_embs_agg)
    #         text_embs_gen = self.text_generator(item_embs_agg)

    #         with torch.no_grad() :
    #             self.image_embedding.weight[self.missing_items_v] = image_embs_gen[self.missing_items_v]
    #             self.text_embedding.weight[self.missing_items_t] = text_embs_gen[self.missing_items_t]


    #         self.image_adj_newitems, self.text_adj_newitems = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach())
    #         self.image_adj, self.text_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), delete_new_items = False)


    def preprocess_missing_modal(self, config) :

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_imputation = config['missing_imputation']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        if 'a' in self.missing_items :
            # 3-modality missing-mask preprocessing (dict keys: all, t, v, a, tv, ta, va)
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t'],
                                                    self.missing_items['tv'], self.missing_items['ta']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v'],
                                                    self.missing_items['tv'], self.missing_items['va']))
            self.missing_items_a = np.concatenate((self.missing_items['all'], self.missing_items['a'],
                                                    self.missing_items['ta'], self.missing_items['va']))

            if config['missing_imputation'] == 0 :
                self.v_feat[self.missing_items_v] = 0.0
                self.t_feat[self.missing_items_t] = 0.0
                self.a_feat[self.missing_items_a] = 0.0
            elif config['missing_imputation'] == 1 :
                non_missing_item_t = np.setdiff1d(self.old_items_set, self.missing_items_t)
                non_missing_item_v = np.setdiff1d(self.old_items_set, self.missing_items_v)
                non_missing_item_a = np.setdiff1d(self.old_items_set, self.missing_items_a)

                image_mean = self.v_feat[non_missing_item_v].mean(dim = 0)
                text_mean = self.t_feat[non_missing_item_t].mean(dim = 0)
                audio_mean = self.a_feat[non_missing_item_a].mean(dim = 0)

                self.v_feat[self.missing_items_v] = image_mean
                self.t_feat[self.missing_items_t] = text_mean
                self.a_feat[self.missing_items_a] = audio_mean
            elif config['missing_imputation'] == 2 :
                # 3-modality historical behavior preserved (no feature imputation here, unlike the 2-modality tree)
                pass
            else :
                assert False, f"Missing Imputation Must bo 0 or 1 or 2, Not {config['missing_imputation']}"
            self.missing_imputation = config['missing_imputation']
            return

        self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
        self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))
        self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t))

        if config['missing_imputation'] == 0 :
            self.v_feat[self.missing_items_v] = 0.0
            self.t_feat[self.missing_items_t] = 0.0
        elif config['missing_imputation'] == 1 :
            non_missing_item_t = np.setdiff1d(self.old_items_set, self.missing_items_t)
            non_missing_item_v = np.setdiff1d(self.old_items_set, self.missing_items_v)

            image_mean = self.v_feat[non_missing_item_v].mean(dim = 0)
            text_mean = self.t_feat[non_missing_item_t].mean(dim = 0)

            self.v_feat[self.missing_items_v] = image_mean
            self.t_feat[self.missing_items_t] = text_mean
        elif config['missing_imputation'] == 2 :
            non_missing_item_t = np.setdiff1d(self.old_items_set, self.missing_items_t)
            non_missing_item_v = np.setdiff1d(self.old_items_set, self.missing_items_v)

            image_mean = self.v_feat[non_missing_item_v].mean(dim = 0)
            text_mean = self.t_feat[non_missing_item_t].mean(dim = 0)

            self.v_feat[self.missing_items_v] = image_mean
            self.t_feat[self.missing_items_t] = text_mean
            pass
        else :
            assert False, f"Missing Imputation Must bo 0 or 1 or 2, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']

    def get_knn_adj_mat(self, v_embeddings, t_embeddings, a_embeddings = None, delete_new_items = True):
        if a_embeddings is not None :
            # 3-modality historical behavior preserved (+1e-5 norm eps, imputation==1 similarity masking,
            # audio-aware mask zeroing order; the tiktok tree's unused _random() helper is not ported)
            v_context_norm = v_embeddings.div(torch.norm(v_embeddings, p=2, dim=-1, keepdim=True) + 1e-5)
            v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

            t_context_norm = t_embeddings.div(torch.norm(t_embeddings, p=2, dim=-1, keepdim=True) + 1e-5)
            t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

            a_context_norm = a_embeddings.div(torch.norm(a_embeddings, p=2, dim=-1, keepdim=True) + 1e-5)
            a_sim = torch.mm(a_context_norm, a_context_norm.transpose(1, 0))

            t_mean, v_mean, a_mean = t_sim.mean(), v_sim.mean(), a_sim.mean()


            mask_v = v_sim < v_mean
            mask_t = t_sim < t_mean
            mask_a = a_sim < a_mean

            t_sim[mask_v] = 0
            v_sim[mask_t] = 0
            a_sim[mask_v] = 0
            a_sim[mask_t] = 0
            a_sim[mask_a] = 0
            t_sim[mask_t] = 0
            v_sim[mask_v] = 0
            t_sim[mask_a] = 0
            v_sim[mask_a] = 0

            if self.missing_imputation == 1 :
                v_sim[:, self.missing_items_v] = v_sim[self.missing_items_v] = 0.0
                v_sim[self.missing_items_v, self.missing_items_v] = 1.0

                t_sim[:, self.missing_items_t] = t_sim[self.missing_items_t] = 0.0
                t_sim[self.missing_items_t, self.missing_items_t] = 1.0

                a_sim[:, self.missing_items_a] = a_sim[self.missing_items_a] = 0.0
                a_sim[self.missing_items_a, self.missing_items_a] = 1.0


            if self.new_items and delete_new_items:

                v_sim[:, self.new_items_set] = v_sim[self.new_items_set] = 0.0
                v_sim[self.new_items_set, self.new_items_set]= 1.0
                t_sim[:, self.new_items_set] = t_sim[self.new_items_set] = 0.0
                t_sim[self.new_items_set, self.new_items_set]= 1.0
                a_sim[:, self.new_items_set] = a_sim[self.new_items_set] = 0.0
                a_sim[self.new_items_set, self.new_items_set]= 1.0


            index_xt, index_xv, index_xa = [], [], []
            index_v = []
            index_t = []
            index_a = []

            for i in range(self.n_items):

                for _sim, index_x in zip([t_sim, v_sim, a_sim], [(index_xt, index_t), (index_xv, index_v), (index_xa, index_a)]) :
                    item_num_ = len(torch.nonzero(_sim[i]))
                    if item_num_ <= self.knn_k :
                        _, _knn_ind = torch.topk(_sim[i], item_num_)
                    else :
                        _, _knn_ind = torch.topk(_sim[i], self.knn_k)

                    index_x[0].append(torch.ones_like(_knn_ind) * i)
                    index_x[1].append(_knn_ind)

            index_xt = torch.cat(index_xt, dim=0).cuda()
            index_xv = torch.cat(index_xv, dim=0).cuda()
            index_xa = torch.cat(index_xa, dim=0).cuda()
            index_v = torch.cat(index_v, dim=0).cuda()
            index_t = torch.cat(index_t, dim=0).cuda()
            index_a = torch.cat(index_a, dim=0).cuda()

            adj_size = (self.n_items, self.n_items)
            del v_sim, t_sim, a_sim

            v_indices = torch.stack((torch.flatten(index_xv), torch.flatten(index_v)), 0)
            t_indices = torch.stack((torch.flatten(index_xt), torch.flatten(index_t)), 0)
            a_indices = torch.stack((torch.flatten(index_xa), torch.flatten(index_a)), 0)
            # norm
            return self.compute_normalized_laplacian(v_indices, adj_size), self.compute_normalized_laplacian(t_indices, adj_size), self.compute_normalized_laplacian(a_indices, adj_size)

        v_context_norm = v_embeddings.div(torch.norm(v_embeddings, p=2, dim=-1, keepdim=True))
        v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

        t_context_norm = t_embeddings.div(torch.norm(t_embeddings, p=2, dim=-1, keepdim=True))
        t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

        if self.missing_imputation == 2 :
            v_sim[self.missing_items['v']] = t_sim[self.missing_items['v']]
            t_sim[self.missing_items['t']] = v_sim[self.missing_items['t']]

        t_mean, v_mean = t_sim.mean(), v_sim.mean()

        mask_v = v_sim < v_mean
        mask_t = t_sim < t_mean

        t_sim[mask_v] = 0
        v_sim[mask_t] = 0
        t_sim[mask_t] = 0
        v_sim[mask_v] = 0

        if self.new_items and delete_new_items:

            v_sim[:, self.new_items_set] = v_sim[self.new_items_set] = 0.0
            v_sim[self.new_items_set, self.new_items_set]= 1.0
            t_sim[:, self.new_items_set] = t_sim[self.new_items_set] = 0.0
            t_sim[self.new_items_set, self.new_items_set]= 1.0

        index_xt, index_xv = [], []
        index_v = []
        index_t = []

        for i in range(self.n_items):
            for _sim, index_x in zip([t_sim, v_sim], [(index_xt, index_t), (index_xv, index_v)]) :
                item_num_ = len(torch.nonzero(_sim[i]))
                if item_num_ <= self.knn_k :
                    _, _knn_ind = torch.topk(_sim[i], item_num_)
                else :
                    _, _knn_ind = torch.topk(_sim[i], self.knn_k)

                index_x[0].append(torch.ones_like(_knn_ind) * i)
                index_x[1].append(_knn_ind)


        index_xt = torch.cat(index_xt, dim=0).cuda()
        index_xv = torch.cat(index_xv, dim=0).cuda()
        index_v = torch.cat(index_v, dim=0).cuda()
        index_t = torch.cat(index_t, dim=0).cuda()

        adj_size = (self.n_items, self.n_items)
        del v_sim, t_sim

        v_indices = torch.stack((torch.flatten(index_xv), torch.flatten(index_v)), 0)
        t_indices = torch.stack((torch.flatten(index_xt), torch.flatten(index_t)), 0)
        # norm
        return self.compute_normalized_laplacian(v_indices, adj_size), self.compute_normalized_laplacian(t_indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_session_adj(self):
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in self.item_graph_dict.keys():
                item_graph_sample = self.item_graph_dict[i][0]
                item_graph_weight = self.item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])
        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0).to(self.device)
        # norm
        return indices, self.compute_normalized_laplacian(indices, (self.n_items, self.n_items))

    def label_prediction(self, emb, aug_emb):
        if self.new_items :
            aug_emb = aug_emb[self.old_items_set]
        n_emb = F.normalize(emb, dim=1)
        n_aug_emb = F.normalize(aug_emb, dim=1)
        prob = torch.mm(n_emb, n_aug_emb.transpose(0, 1))
        prob = F.softmax(prob, dim=1)
        del n_emb, n_aug_emb
        return prob

    def generate_pesudo_labels(self, prob1, prob2, prob3, prob4 = None):
        if prob4 is not None :
            # 3-modality historical behavior preserved (4 label sources; topk size = self.knn_k, not hardcoded 10)
            positive = prob1 + prob2 + prob3 + prob4 + prob4
            _ , mm_pos_ind = torch.topk(positive, self.knn_k, dim=-1)
            prob = prob4.clone()
            prob.scatter_(1, mm_pos_ind, 0)
            _, single_pos_ind = torch.topk(prob, self.knn_k, dim=-1)
            return mm_pos_ind, single_pos_ind
        positive = prob1 + prob2 + prob3 + prob3
        _ , mm_pos_ind = torch.topk(positive, 10, dim=-1)
        prob = prob3.clone()
        prob.scatter_(1, mm_pos_ind, 0)
        _, single_pos_ind = torch.topk(prob, 10, dim=-1)
        return mm_pos_ind, single_pos_ind

    def neighbor_discrimination(self, mm_positive, s_positive, emb, aug_emb, temperature=0.2):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=2)

        if self.new_items :
            aug_emb = aug_emb[self.old_items_set]

        n_aug_emb = F.normalize(aug_emb, dim=1)
        n_emb = F.normalize(emb , dim=1)

        mm_pos_emb = n_aug_emb[mm_positive]
        s_pos_emb = n_aug_emb[s_positive]

        emb2 = torch.reshape(n_emb, [-1, 1, self.embedding_dim])
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (tile size = self.knn_k, not hardcoded 10)
            emb2 = torch.tile(emb2, [1, self.knn_k, 1])
        else :
            emb2 = torch.tile(emb2, [1, 10, 1])

        mm_pos_score = score(emb2, mm_pos_emb)
        s_pos_score = score(emb2, s_pos_emb)
        ttl_score = torch.matmul(n_emb, n_aug_emb.transpose(0, 1))

        mm_pos_score = torch.sum(torch.exp(mm_pos_score / temperature), dim=1)
        s_pos_score = torch.sum(torch.exp(s_pos_score / temperature), dim=1)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1) # 1

        cl_loss = - torch.log(mm_pos_score / (ttl_score) + 10e-10) - torch.log(s_pos_score / (ttl_score - mm_pos_score) + 10e-10)
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (returns per-item vector; caller masks by tva_index then averages)
            return cl_loss
        return torch.mean(cl_loss)

    def KL(self, p1, p2):
        return p1 * torch.log(p1) - p1 * torch.log(p2) + \
               (1 - p1) * torch.log(1 - p1) - (1 - p1) * torch.log(1 - p2)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        # scipy>=1.12 removed dok_matrix._update; build a COO matrix directly instead
        _rows, _cols = zip(*data_dict.keys())
        A = sp.coo_matrix((list(data_dict.values()), (list(_rows), list(_cols))), shape=A.shape, dtype=np.float32)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        self.num_inters = sumArr
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


    def forward(self, infer = True):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        del ego_embeddings, side_embeddings

        if infer :
            text_adj, image_adj = self.text_adj_infer, self.image_adj_infer
        else :
            text_adj, image_adj = self.text_adj, self.image_adj

        # text emb
        h_t = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_t = torch.sparse.mm(text_adj, h_t)

        # image emb
        h_v = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_v = torch.sparse.mm(image_adj, h_v)

        if self.a_feat is not None:
            # audio emb
            audio_adj = self.audio_adj_infer if infer else self.audio_adj
            h_a = self.item_id_embedding.weight.clone()
            for i in range(self.n_layers):
                h_a = torch.sparse.mm(audio_adj, h_a)

        # session emb
        h_s = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_s = torch.sparse.mm(self.session_adj, h_s)

        if self.a_feat is not None:
            return u_g_embeddings, i_g_embeddings, h_t, h_v, h_s, h_a
        return u_g_embeddings, i_g_embeddings, h_t, h_v, h_s

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        if self.a_feat is not None:
            # 3-modality historical behavior preserved: only tva_index is consumed below
            # (the tiktok tree's unused t/v/a/tv/ta/va index arrays are not ported)
            if self.missing_modal :
                tva_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(np.union1d(self.missing_items_t, self.missing_items_v), self.missing_items_a))
            else :
                tva_index = pos_items.detach().cpu().numpy()

        if self.new_items and np.isin(neg_items.detach().cpu().numpy(), self.new_items_set).sum() != 0 :
            assert False, "New Item Error !!"

        if self.a_feat is not None:
            user_embeddings, item_embeddings, h_t, h_v, h_s, h_a = self.forward(infer = False)
        else :
            user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward(infer = False)
        # Loss Gen #
        # item_embs_agg = torch.sparse.mm(self.adj.t(), user_embeddings) * self.num_inters[self.n_users:]
        # image_embs_gen = self.image_generator(item_embs_agg)
        # text_embs_gen = self.text_generator(item_embs_agg)
        # unique_item_id, inverse = torch.unique(torch.cat((pos_items, neg_items)), return_inverse = True, sorted=False)
        # unique_item_id= unique_item_id.detach().cpu().numpy()

        # if self.missing_modal :
        #     complete_idx = np.isin(unique_item_id, self.complete_items)
        # else :
        #     complete_idx = np.ones_like(unique_item_id, dtype = 'bool')

        # loss_gen = F.mse_loss(self.image_embedding.weight[unique_item_id][complete_idx], image_embs_gen[unique_item_id][complete_idx])
        # loss_gen += F.mse_loss(self.text_embedding.weight[unique_item_id][complete_idx], text_embs_gen[unique_item_id][complete_idx])
        # Loss Gen #

        u_idx = torch.unique(users, return_inverse=True, sorted=False)
        i_idx = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        u_id = u_idx[0]
        i_id = i_idx[0]

        # text
        label_prediction_t = self.label_prediction(h_t[i_id], h_t)
        # visual
        label_prediction_v = self.label_prediction(h_v[i_id], h_v)
        # session
        label_prediction_s = self.label_prediction(h_s[i_id], h_s)

        if self.a_feat is not None:
            # audio
            label_prediction_a = self.label_prediction(h_a[i_id], h_a)

            # 3-modality historical behavior preserved (4-way pseudo labels, tva_index masking, *0.5/4.0 scaling)
            mm_postive_s, s_postive_s = self.generate_pesudo_labels(label_prediction_t, label_prediction_v, label_prediction_a, label_prediction_s)
            neighbor_dis_loss_1 = self.neighbor_discrimination(mm_postive_s, s_postive_s, h_s[i_id], h_s)
            neighbor_dis_loss_1 = torch.mean(neighbor_dis_loss_1[np.isin(i_id.cpu().numpy(), tva_index)])

            mm_postive_v, s_postive_v = self.generate_pesudo_labels(label_prediction_t, label_prediction_a, label_prediction_s, label_prediction_v)
            neighbor_dis_loss_2 = self.neighbor_discrimination(mm_postive_v, s_postive_v, h_v[i_id], h_v)
            neighbor_dis_loss_2 = torch.mean(neighbor_dis_loss_2[np.isin(i_id.cpu().numpy(), tva_index)])

            mm_postive_t, s_postive_t = self.generate_pesudo_labels(label_prediction_v, label_prediction_a, label_prediction_s, label_prediction_t)
            neighbor_dis_loss_3 = self.neighbor_discrimination(mm_postive_t, s_postive_t, h_t[i_id], h_t)
            neighbor_dis_loss_3 = torch.mean(neighbor_dis_loss_3[np.isin(i_id.cpu().numpy(), tva_index)])

            mm_postive_a, s_postive_a = self.generate_pesudo_labels(label_prediction_v, label_prediction_t, label_prediction_s, label_prediction_a)
            neighbor_dis_loss_4 = self.neighbor_discrimination(mm_postive_a, s_postive_a, h_a[i_id], h_a)
            neighbor_dis_loss_4 = torch.mean(neighbor_dis_loss_4[np.isin(i_id.cpu().numpy(), tva_index)])

            neighbor_dis_loss = ( neighbor_dis_loss_1 + neighbor_dis_loss_2 + neighbor_dis_loss_3 + neighbor_dis_loss_4) *0.5 / 4.0
        else :
            mm_postive_s, s_postive_s = self.generate_pesudo_labels(label_prediction_t, label_prediction_v, label_prediction_s)
            neighbor_dis_loss_1 = self.neighbor_discrimination(mm_postive_s, s_postive_s, h_s[i_id], h_s)

            mm_postive_v, s_postive_v = self.generate_pesudo_labels(label_prediction_t, label_prediction_s, label_prediction_v)
            neighbor_dis_loss_2 = self.neighbor_discrimination(mm_postive_v, s_postive_v, h_v[i_id], h_v)

            mm_postive_t, s_postive_t = self.generate_pesudo_labels(label_prediction_v, label_prediction_s, label_prediction_t)
            neighbor_dis_loss_3 = self.neighbor_discrimination(mm_postive_t, s_postive_t, h_t[i_id], h_t)

            neighbor_dis_loss = (neighbor_dis_loss_1 + neighbor_dis_loss_2 + neighbor_dis_loss_3) / 3.0

        n_u_g_embeddings = user_embeddings[u_id]
        if self.a_feat is not None:
            it_embeddings = (h_t + h_v + h_a + h_s)/4.0 # (h_t + h_v+ h_a)/3.0
        else :
            it_embeddings = (h_t + h_s + h_v)/3.0

        p_g = F.sigmoid(torch.matmul(n_u_g_embeddings, F.normalize(item_embeddings[i_id], dim=-1).transpose(0, 1)))
        p_t = F.sigmoid(torch.matmul(n_u_g_embeddings, F.normalize(it_embeddings[i_id], dim=-1).transpose(0, 1)))

        KL_loss = torch.mean(self.KL(p_g, p_t) + self.KL(p_t, p_g))

        if self.a_feat is not None:
            p_weight, n_weight = self.get_weight_modal(users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s, h_a)
        else :
            p_weight, n_weight = self.get_weight_modal(users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s)

        u_g_embeddings = user_embeddings[users]
        if self.a_feat is not None:
            ia_embeddings = item_embeddings + (h_t + h_v+ h_a + h_s)/4.0 # (h_t + h_v+ h_a) /3.0 #
        else :
            ia_embeddings = item_embeddings + (h_t + h_v + h_s)/3.0
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, p_weight, n_weight)

        return batch_mf_loss + self.neighbor_weight * (neighbor_dis_loss) + KL_loss * self.kl_weight #+ loss_gen * self.gen_weight


    def full_sort_predict(self, interaction):
        user = interaction[0]
        if self.a_feat is not None:
            user_embeddings, item_embeddings, h_t, h_v, h_s, h_a = self.forward(infer = True) #
        else :
            user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward(infer = True) #

        user_e = user_embeddings[user, :]
        if self.a_feat is not None:
            i_embedding = (h_t + h_v+ h_a+ h_s)/4.0 # ) /3.0 #
        else :
            i_embedding = (h_v+ h_t+ h_s)/ 3.0
        all_item_e = item_embeddings + i_embedding
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))

        return score


    def get_weight_modal(self, users, pos_items, neg_items, user_embeddings, h_t, h_v, h_s, h_a = None):
        u_g_embeddings = user_embeddings[users]

        p_t = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_t[pos_items], dim=-1)), dim=1)
        p_v = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_s[pos_items], dim=-1)), dim=1)
        p_s = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_v[pos_items], dim=-1)), dim=1)
        if h_a is not None:
            p_a = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_a[pos_items], dim=-1)), dim=1)

        n_t = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_t[neg_items], dim=-1)), dim=1)
        n_v = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_s[neg_items], dim=-1)), dim=1)
        n_s = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_v[neg_items], dim=-1)), dim=1)
        if h_a is not None:
            n_a = torch.sum(torch.mul(u_g_embeddings, F.normalize(h_a[neg_items], dim=-1)), dim=1)

        if h_a is not None:
            p_tensor = F.sigmoid(torch.stack([p_t, p_v, p_s, p_a]))
        else :
            p_tensor = F.sigmoid(torch.stack([p_t, p_v, p_s]))
        p_variance = torch.var(p_tensor, dim=0).data
        p_mean_value = torch.mean(p_tensor, dim=0).data
        p_max_value, _ = torch.max(p_tensor, dim=0)

        if h_a is not None:
            n_tensor = F.sigmoid(torch.stack([n_t, n_v, n_s, n_a]))
        else :
            n_tensor = F.sigmoid(torch.stack([n_t, n_v, n_s]))
        n_mean_value = torch.mean(n_tensor).data

        p_mean_probability = torch.pow(p_mean_value, 1.0).data
        p_var_probability = torch.pow(torch.exp(-p_variance).data, 2.0) # 0 ~ 1
        pos_weight = p_mean_probability * p_var_probability
        pos_weight = torch.clamp(pos_weight, 0, 1).data

        mask = torch.zeros_like(p_mean_value)
        mask[p_mean_value < n_mean_value] = 1

        neg_weight_max = torch.pow((p_max_value - n_mean_value.data), 1.0) * mask
        neg_weight = torch.clamp(neg_weight_max, 0, 1).data
        # print(neg_weight)

        return pos_weight, neg_weight

    def bpr_loss(self, users, pos_items, neg_items, p_weight, n_weight):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        p_maxi = torch.log(F.sigmoid(pos_scores - neg_scores)) * p_weight
        n_maxi = torch.log(F.sigmoid(neg_scores - pos_scores)) * n_weight
        mf_loss = -torch.mean(p_maxi + n_maxi)
        # mf_loss = -torch.sum(maxi)
        return mf_loss
