# coding: utf-8

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class CI2MG(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CI2MG, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_nodes = self.n_users + self.n_items
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layers = config['n_hyper_layers']
        self.lamb_1 = config['lamb_1']
        self.lamb_2 = config['lamb_2']
        self.temp = config['temp']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

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

        if config['missing_modal'] :
            self.preprocess_missing_modal(config)
        self.missing_modal = config['missing_modal']
        self.missing_imputation = config['missing_imputation']


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            if self.a_feat is None:
                nn.init.xavier_uniform_(self.image_trs.weight)
                nn.init.xavier_uniform_(self.text_trs.weight)
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (trs xavier inits run after audio_trs creation)
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=True)
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
            nn.init.xavier_uniform_(self.text_trs.weight)
            nn.init.xavier_uniform_(self.audio_trs.weight)


        self.user_weight = nn.ModuleList([nn.Linear(64, 64, bias = False) for _ in range(self.n_ui_layers)]).to(self.device)
        self.item_weight = nn.ModuleList([nn.Linear(64, 64, bias = False) for _ in range(self.n_ui_layers)]).to(self.device)
        for i in range(self.n_ui_layers) :
            nn.init.xavier_uniform_(self.user_weight[i].weight)
            nn.init.xavier_uniform_(self.item_weight[i].weight)

        self.image_prot = nn.Embedding(64, 64).to(self.device)
        self.text_prot = nn.Embedding(64, 64).to(self.device)
        if self.a_feat is not None:
            self.audio_prot = nn.Embedding(64, 64).to(self.device)
        nn.init.xavier_uniform_(self.image_prot.weight)
        nn.init.xavier_uniform_(self.text_prot.weight)
        if self.a_feat is not None:
            nn.init.xavier_uniform_(self.audio_prot.weight)

        self.hgcn_weight_image = nn.Linear(64, 64, bias = False).to(self.device)
        self.hgcn_weight_text = nn.Linear(64, 64, bias = False).to(self.device)
        if self.a_feat is not None:
            self.hgcn_weight_audio = nn.Linear(64, 64, bias = False).to(self.device)
        # nn.init.xavier_uniform_(self.hgcn_weight_image.weight)
        # nn.init.xavier_uniform_(self.hgcn_weight_text.weight)

        self.skh_delta_image = torch.zeros(self.n_items, requires_grad = True).to(self.device)
        self.skh_delta_text = torch.zeros(self.n_items, requires_grad = True).to(self.device)
        if self.a_feat is not None:
            self.skh_delta_audio = torch.zeros(self.n_items, requires_grad = True).to(self.device)

        self.image_enhance = nn.Linear(64 * 2, 64)
        self.text_enhance = nn.Linear(64 * 2, 64)
        if self.a_feat is not None:
            self.audio_enhance = nn.Linear(64 * 2, 64)
        nn.init.xavier_uniform_(self.image_enhance.weight)
        nn.init.xavier_uniform_(self.text_enhance.weight)
        if self.a_feat is not None:
            nn.init.xavier_uniform_(self.audio_enhance.weight)
        self.bn = nn.BatchNorm1d(64)

        pass

    def preprocess_missing_modal(self, config) :

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
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
            else :
                # 3-modality historical behavior preserved (imputation 2 is not supported by the tiktok tree)
                assert False, f"Missing Imputation Must bo 0 or 1, Not {config['missing_imputation']}"
            self.missing_imputation = config['missing_imputation']
            return

        self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
        self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))

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
            pass
        else :
            assert False, f"Missing Imputation Must bo 0 or 1 or 2, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']

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
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        D = sp.diags(diag)
        L = D * A
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))


    def forward(self):
        u_g_embeddings, i_g_embeddings = self.user_embedding.weight, self.item_id_embedding.weight
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        del ego_embeddings
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        if self.missing_modal :
            t_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_t)
            v_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_v)
            tv_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_v))
        else :
            t_index = pos_items.detach().cpu().numpy()
            v_index = pos_items.detach().cpu().numpy()
            tv_index = pos_items.detach().cpu().numpy()

        if self.new_items and np.isin(neg_items.detach().cpu().numpy(), self.new_items_set).sum() != 0 :
            assert False, "New Item Error !!"
        user_embeddings, item_embeddings = self.forward()

        image_emb_raw, text_emb_raw = self.image_trs(self.image_embedding.weight), self.text_trs(self.text_embedding.weight)
        if self.a_feat is not None:
            audio_emb_raw = self.audio_trs(self.audio_embedding.weight)

        image_emb = torch.cat((torch.zeros((self.n_users, image_emb_raw.shape[1]), device = self.device), image_emb_raw), dim=0)
        image_emb = torch.sparse.mm(self.norm_adj, image_emb)
        user_image_emb, _ = torch.split(image_emb, [self.n_users, self.n_items], dim=0)

        image_emb = torch.cat((user_image_emb, torch.zeros((self.n_items, image_emb_raw.shape[1]), device = self.device)), dim=0)
        image_emb = torch.sparse.mm(self.norm_adj, image_emb)
        _, item_image_emb = torch.split(image_emb, [self.n_users, self.n_items], dim=0)

        text_emb = torch.cat((torch.zeros((self.n_users, text_emb_raw.shape[1]), device = self.device), text_emb_raw), dim=0)
        text_emb = torch.sparse.mm(self.norm_adj, text_emb)
        user_text_emb, _ = torch.split(text_emb, [self.n_users, self.n_items], dim=0)

        text_emb = torch.cat((user_text_emb, torch.zeros((self.n_items, text_emb_raw.shape[1]), device = self.device)), dim=0)
        text_emb = torch.sparse.mm(self.norm_adj, text_emb)
        _, item_text_emb = torch.split(text_emb, [self.n_users, self.n_items], dim=0)

        if self.a_feat is not None:
            audio_emb = torch.cat((torch.zeros((self.n_users, audio_emb_raw.shape[1]), device = self.device), audio_emb_raw), dim=0)
            audio_emb = torch.sparse.mm(self.norm_adj, audio_emb)
            user_audio_emb, _ = torch.split(audio_emb, [self.n_users, self.n_items], dim=0)

            audio_emb = torch.cat((user_audio_emb, torch.zeros((self.n_items, audio_emb_raw.shape[1]), device = self.device)), dim=0)
            audio_emb = torch.sparse.mm(self.norm_adj, audio_emb)
            _, item_audio_emb = torch.split(audio_emb, [self.n_users, self.n_items], dim=0)

        # Intra-Modality Generation

        # HyperGraph
        H_image = F.normalize(item_image_emb) @ F.normalize(self.image_prot.weight).T
        H_text = F.normalize(item_text_emb) @ F.normalize(self.text_prot.weight).T
        if self.a_feat is not None:
            H_audio = F.normalize(item_audio_emb) @ F.normalize(self.audio_prot.weight).T

        D = torch.sum(H_image, axis = 1)
        zero_idx = torch.where(D == 0)[0]
        if len(zero_idx) != 0 :
            D[zero_idx] = 1.0
        D = torch.diag(torch.pow(D, -1))

        B = torch.sum(H_image, axis = 0)
        zero_idx = torch.where(B == 0)[0]
        if len(zero_idx) != 0 :
            B[zero_idx] = 1.0
        B = torch.diag(torch.pow(B, -1))

        R = D @ H_image @ self.hgcn_weight_image.weight @ B @ H_image.T
        image_hyper_emb = item_image_emb.clone()
        for _ in range(self.n_hyper_layers) :
            image_hyper_emb = torch.sigmoid(R @ image_hyper_emb)


        D = torch.sum(H_text, axis = 1)
        zero_idx = torch.where(D == 0)[0]
        if len(zero_idx) != 0 :
            D[zero_idx] = 1.0
        D = torch.diag(torch.pow(D, -1))

        B = torch.sum(H_text, axis = 0)
        zero_idx = torch.where(B == 0)[0]
        if len(zero_idx) != 0 :
            B[zero_idx] = 1.0
        B = torch.diag(torch.pow(B, -1))

        R = D @ H_text @ self.hgcn_weight_text.weight @ B @ H_text.T
        text_hyper_emb = item_text_emb.clone()
        for _ in range(self.n_hyper_layers) :
            text_hyper_emb = torch.sigmoid(R @ text_hyper_emb)


        if self.a_feat is not None:
            D = torch.sum(H_audio, axis = 1)
            zero_idx = torch.where(D == 0)[0]
            if len(zero_idx) != 0 :
                D[zero_idx] = 1.0
            D = torch.diag(torch.pow(D, -1))

            B = torch.sum(H_audio, axis = 0)
            zero_idx = torch.where(B == 0)[0]
            if len(zero_idx) != 0 :
                B[zero_idx] = 1.0
            B = torch.diag(torch.pow(B, -1))

            R = D @ H_audio @ self.hgcn_weight_audio.weight @ B @ H_audio.T
            audio_hyper_emb = item_audio_emb.clone()
            for _ in range(self.n_hyper_layers) :
                audio_hyper_emb = torch.sigmoid(R @ audio_hyper_emb)

        # Inter-Modalit Generation

        if self.a_feat is not None:
            # 3-modality historical behavior preserved (pairwise-sum OT couplings)
            dist_it2a = torch.cdist(image_hyper_emb + text_hyper_emb, audio_hyper_emb) ** 2
            ot_it2a = sinkhorn_algorithm2(dist_it2a, 5.0, 10); del dist_it2a
            dist_ia2t = torch.cdist(image_hyper_emb + audio_hyper_emb, text_hyper_emb) ** 2
            ot_ia2t = sinkhorn_algorithm2(dist_ia2t, 5.0, 10); del dist_ia2t
            dist_ta2i = torch.cdist(text_hyper_emb + audio_hyper_emb, image_hyper_emb) ** 2
            ot_ta2i = sinkhorn_algorithm2(dist_ta2i, 5.0, 10); del dist_ta2i

            image_rep = (ot_ta2i  + self.skh_delta_image) @ image_hyper_emb
            text_rep = (ot_ia2t  + self.skh_delta_text) @ text_hyper_emb
            audio_rep = (ot_it2a  + self.skh_delta_audio) @ audio_hyper_emb
        else :
            dist = torch.cdist(image_hyper_emb, text_hyper_emb, p=2) ** 2
            ot_i2t = sinkhorn_algorithm2(dist, 5.0, 10)
            ot_t2i = sinkhorn_algorithm2(dist.T, 5.0, 10)

            image_rep = (ot_t2i  + self.skh_delta_image) @ image_hyper_emb
            text_rep = (ot_i2t  + self.skh_delta_text) @ text_hyper_emb

        image_pos_score = torch.exp(torch.cosine_similarity(image_hyper_emb[v_index], image_rep[v_index]) / self.temp)
        image_neg_score = torch.exp(torch.cosine_similarity(image_hyper_emb[v_index], image_rep[v_index]) / self.temp)
        text_pos_score = torch.exp(torch.cosine_similarity(text_hyper_emb[t_index], text_rep[t_index]) / self.temp)
        text_neg_score = torch.exp(torch.cosine_similarity(text_hyper_emb[t_index], text_rep[t_index]) / self.temp)
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (audio scores index with t_index, not a_index)
            audio_pos_score = torch.exp(torch.cosine_similarity(audio_hyper_emb[t_index], audio_rep[t_index]) / self.temp)
            audio_neg_score = torch.exp(torch.cosine_similarity(audio_hyper_emb[t_index], audio_rep[t_index]) / self.temp)

        loss_s = -torch.log(image_pos_score / (image_pos_score + image_neg_score)).mean()
        loss_s += -torch.log(text_pos_score / (text_pos_score + text_neg_score)).mean()
        if self.a_feat is not None:
            loss_s += -torch.log(audio_pos_score / (audio_pos_score + audio_neg_score)).mean()


        image_emb_final = torch.cat([image_hyper_emb, image_rep], dim = 1)
        text_emb_final = torch.cat([text_hyper_emb, text_rep], dim = 1)
        if self.a_feat is not None:
            audio_emb_final = torch.cat([audio_hyper_emb, audio_rep], dim = 1)

        image_enh_emb = (self.bn(self.image_enhance(image_emb_final)))
        text_enh_emb = (self.bn(self.text_enhance(text_emb_final)))
        if self.a_feat is not None:
            audio_enh_emb = (self.bn(self.audio_enhance(audio_emb_final)))

        loss_rec = F.mse_loss(image_enh_emb[v_index], image_emb_raw[v_index]) + F.mse_loss(text_enh_emb[t_index], text_emb_raw[t_index])
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (audio reconstruction indexed with t_index)
            loss_rec = loss_rec + F.mse_loss(audio_enh_emb[t_index], audio_emb_raw[t_index])

        if self.a_feat is not None:
            final_user_emb = torch.cat([user_embeddings, F.normalize(user_image_emb), F.normalize(user_text_emb), F.normalize(user_audio_emb)], dim = 1)
            final_item_emb = torch.cat([item_embeddings, F.normalize(image_enh_emb), F.normalize(text_enh_emb), F.normalize(audio_enh_emb)], dim = 1)
        else :
            final_user_emb = torch.cat([user_embeddings, F.normalize(user_image_emb), F.normalize(user_text_emb)], dim = 1)
            final_item_emb = torch.cat([item_embeddings, F.normalize(image_enh_emb), F.normalize(text_enh_emb)], dim = 1)

        loss_bpr = self.bpr_loss(final_user_emb[users], final_item_emb[pos_items], final_item_emb[neg_items])
        print(f"BPR Loss : {loss_bpr:.4f} | S Loss : {loss_s:.4f} | Rec Loss : {loss_rec:.4f}")
        return loss_rec + loss_s * self.lamb_1 + loss_bpr * self.lamb_2

    def InfoNCE_v2(self, view1, view2, temperature = 0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embeddings, item_embeddings = self.forward()

        image_emb_raw, text_emb_raw = self.image_trs(self.image_embedding.weight), self.text_trs(self.text_embedding.weight)
        if self.a_feat is not None:
            audio_emb_raw = self.audio_trs(self.audio_embedding.weight)

        image_emb = torch.cat((torch.zeros((self.n_users, image_emb_raw.shape[1]), device = self.device), image_emb_raw), dim=0)
        image_emb = torch.sparse.mm(self.norm_adj, image_emb)
        user_image_emb, _ = torch.split(image_emb, [self.n_users, self.n_items], dim=0)

        image_emb = torch.cat((user_image_emb, torch.zeros((self.n_items, image_emb_raw.shape[1]), device = self.device)), dim=0)
        image_emb = torch.sparse.mm(self.norm_adj, image_emb)
        _, item_image_emb = torch.split(image_emb, [self.n_users, self.n_items], dim=0)

        text_emb = torch.cat((torch.zeros((self.n_users, text_emb_raw.shape[1]), device = self.device), text_emb_raw), dim=0)
        text_emb = torch.sparse.mm(self.norm_adj, text_emb)
        user_text_emb, _ = torch.split(text_emb, [self.n_users, self.n_items], dim=0)

        text_emb = torch.cat((user_text_emb, torch.zeros((self.n_items, text_emb_raw.shape[1]), device = self.device)), dim=0)
        text_emb = torch.sparse.mm(self.norm_adj, text_emb)
        _, item_text_emb = torch.split(text_emb, [self.n_users, self.n_items], dim=0)

        if self.a_feat is not None:
            audio_emb = torch.cat((torch.zeros((self.n_users, audio_emb_raw.shape[1]), device = self.device), audio_emb_raw), dim=0)
            audio_emb = torch.sparse.mm(self.norm_adj, audio_emb)
            user_audio_emb, _ = torch.split(audio_emb, [self.n_users, self.n_items], dim=0)

            audio_emb = torch.cat((user_audio_emb, torch.zeros((self.n_items, audio_emb_raw.shape[1]), device = self.device)), dim=0)
            audio_emb = torch.sparse.mm(self.norm_adj, audio_emb)
            _, item_audio_emb = torch.split(audio_emb, [self.n_users, self.n_items], dim=0)

        # Intra-Modality Generation

        # HyperGraph
        H_image = F.normalize(item_image_emb) @ F.normalize(self.image_prot.weight).T
        H_text = F.normalize(item_text_emb) @ F.normalize(self.text_prot.weight).T
        if self.a_feat is not None:
            H_audio = F.normalize(item_audio_emb) @ F.normalize(self.audio_prot.weight).T

        D = torch.sum(H_image, axis = 1)
        zero_idx = torch.where(D == 0)[0]
        if len(zero_idx) != 0 :
            D[zero_idx] = 1.0
        D = torch.diag(torch.pow(D, -1))

        B = torch.sum(H_image, axis = 0)
        zero_idx = torch.where(B == 0)[0]
        if len(zero_idx) != 0 :
            B[zero_idx] = 1.0
        B = torch.diag(torch.pow(B, -1))

        R = D @ H_image @ self.hgcn_weight_image.weight @ B @ H_image.T
        image_hyper_emb = item_image_emb.clone()
        for _ in range(self.n_hyper_layers) :
            image_hyper_emb = torch.sigmoid(R @ image_hyper_emb)


        D = torch.sum(H_text, axis = 1)
        zero_idx = torch.where(D == 0)[0]
        if len(zero_idx) != 0 :
            D[zero_idx] = 1.0
        D = torch.diag(torch.pow(D, -1))

        B = torch.sum(H_text, axis = 0)
        zero_idx = torch.where(B == 0)[0]
        if len(zero_idx) != 0 :
            B[zero_idx] = 1.0
        B = torch.diag(torch.pow(B, -1))

        R = D @ H_text @ self.hgcn_weight_text.weight @ B @ H_text.T
        text_hyper_emb = item_text_emb.clone()
        for _ in range(self.n_hyper_layers) :
            text_hyper_emb = torch.sigmoid(R @ text_hyper_emb)


        if self.a_feat is not None:
            D = torch.sum(H_audio, axis = 1)
            zero_idx = torch.where(D == 0)[0]
            if len(zero_idx) != 0 :
                D[zero_idx] = 1.0
            D = torch.diag(torch.pow(D, -1))

            B = torch.sum(H_audio, axis = 0)
            zero_idx = torch.where(B == 0)[0]
            if len(zero_idx) != 0 :
                B[zero_idx] = 1.0
            B = torch.diag(torch.pow(B, -1))

            R = D @ H_audio @ self.hgcn_weight_audio.weight @ B @ H_audio.T
            audio_hyper_emb = item_audio_emb.clone()
            for _ in range(self.n_hyper_layers) :
                audio_hyper_emb = torch.sigmoid(R @ audio_hyper_emb)

        # Inter-Modalit Generation
        if self.a_feat is not None:
            # 3-modality historical behavior preserved (pairwise-sum OT couplings)
            dist_it2a = torch.cdist(image_hyper_emb + text_hyper_emb, audio_hyper_emb) ** 2
            ot_it2a = sinkhorn_algorithm2(dist_it2a, 5.0, 10); del dist_it2a
            dist_ia2t = torch.cdist(image_hyper_emb + audio_hyper_emb, text_hyper_emb) ** 2
            ot_ia2t = sinkhorn_algorithm2(dist_ia2t, 5.0, 10); del dist_ia2t
            dist_ta2i = torch.cdist(text_hyper_emb + audio_hyper_emb, image_hyper_emb) ** 2
            ot_ta2i = sinkhorn_algorithm2(dist_ta2i, 5.0, 10); del dist_ta2i

            image_rep = (ot_ta2i  + self.skh_delta_image) @ image_hyper_emb
            text_rep = (ot_ia2t  + self.skh_delta_text) @ text_hyper_emb
            audio_rep = (ot_it2a  + self.skh_delta_audio) @ audio_hyper_emb
        else :
            dist = torch.cdist(image_hyper_emb, text_hyper_emb, p=2) ** 2
            ot_i2t = sinkhorn_algorithm2(dist, 5.0, 10)
            ot_t2i = sinkhorn_algorithm2(dist.T, 5.0, 10)

            # ot_i2t = sinkhorn_algorithm(image_hyper_emb, text_hyper_emb)
            # ot_t2i = sinkhorn_algorithm(text_hyper_emb, image_hyper_emb)

            image_rep = (ot_t2i  + self.skh_delta_image) @ image_hyper_emb
            text_rep = (ot_i2t  + self.skh_delta_text) @ text_hyper_emb

        image_emb_final = torch.cat([image_hyper_emb, image_rep], dim = 1)
        text_emb_final = torch.cat([text_hyper_emb, text_rep], dim = 1)
        if self.a_feat is not None:
            audio_emb_final = torch.cat([audio_hyper_emb, audio_rep], dim = 1)

        image_enh_emb = (self.bn(self.image_enhance(image_emb_final)))
        text_enh_emb = (self.bn(self.text_enhance(text_emb_final)))
        if self.a_feat is not None:
            audio_enh_emb = (self.bn(self.audio_enhance(audio_emb_final)))

        if self.a_feat is not None:
            final_user_emb = torch.cat([user_embeddings, F.normalize(user_image_emb), F.normalize(user_text_emb), F.normalize(user_audio_emb)], dim = 1)
            final_item_emb = torch.cat([item_embeddings, F.normalize(image_enh_emb), F.normalize(text_enh_emb), F.normalize(audio_enh_emb)], dim = 1)
        else :
            final_user_emb = torch.cat([user_embeddings, F.normalize(user_image_emb), F.normalize(user_text_emb)], dim = 1)
            final_item_emb = torch.cat([item_embeddings, F.normalize(image_enh_emb), F.normalize(text_enh_emb)], dim = 1)

        score = torch.matmul(final_user_emb[users], final_item_emb.transpose(0, 1))

        return score

    def bpr_loss(self, users, pos_items, neg_items):
        if len(pos_items.shape) == 2 :
            pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
            neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        else :
            pos_scores = torch.einsum("ik, ijk -> ij", users, pos_items)
            neg_scores =torch.einsum("ik, ijk -> ij", users, neg_items)


        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

# Graph Convolution Layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, adj, features):
        D = torch.diag(torch.sum(adj, dim=1))  # Degree matrix
        adj_norm = torch.inverse(D) @ adj  # Normalized adjacency matrix
        return self.fc(adj_norm @ features)

# Hypergraph Convolution
class HyperGraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HyperGraphConvolution, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, hypergraph, features):
        H_norm = self.normalize_hypergraph(hypergraph)
        return self.fc(H_norm @ features)

    def normalize_hypergraph(self, hypergraph):
        Dv = torch.diag(torch.sum(hypergraph, dim=1))
        De = torch.diag(torch.sum(hypergraph, dim=0))
        return torch.inverse(Dv) @ hypergraph @ torch.inverse(De)

@torch.no_grad()
def sinkhorn_algorithm(a, b, epsilon=1.0, max_iter=50):
    """
    Compute the Sinkhorn distance between two distributions using optimal transport.
    Args:
        a: Tensor of size (batch_size, feature_dim), representing source modality features.
        b: Tensor of size (batch_size, feature_dim), representing target modality features.
        epsilon: Regularization parameter for Sinkhorn distance.
        max_iter: Maximum iterations for convergence.
    Returns:
        Optimal transport cost between `a` and `b`.
    """
    n = a.size(0)
    M = torch.cdist(a, b, p=2) ** 2  # Euclidean distance matrix between `a` and `b`

    K = torch.exp(-M / epsilon)  # Kernel (similarity) matrix
    u = torch.ones(n, device = M.device) / n  # Marginal for a
    v = torch.ones(n, device = M.device) / n  # Marginal for b

    # Sinkhorn iterations
    for _ in range(max_iter):
        u = 1.0 / (K @ (v.unsqueeze(1))).squeeze(1)
        v = 1.0 / (K.T @ (u.unsqueeze(1))).squeeze(1)

    # Optimal transport cost
    transport_cost = u.unsqueeze(1) * K * v.unsqueeze(0) * M
    return transport_cost

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, adj, features):
        D = torch.diag(torch.sum(adj, dim=1))  # Degree matrix
        adj_norm = torch.inverse(D) @ adj  # Normalized adjacency matrix
        return self.fc(adj_norm @ features)

@torch.no_grad()
def sinkhorn_algorithm2(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q
