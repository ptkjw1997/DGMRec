import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from collections import defaultdict
import math
from scipy.sparse import lil_matrix
import random
import json

class GUME(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GUME, self).__init__(config, dataset)
        self.sparse = True
        self.bm_loss = config['bm_loss']
        self.um_loss = config['um_loss']
        self.vt_loss = config['vt_loss']
        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.bm_temp = config['bm_temp']
        self.um_temp = config['um_temp']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_image_user.weight)
        
        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        if self.a_feat is not None:
            self.extended_audio_user = nn.Embedding(self.n_users, self.embedding_dim)
            nn.init.xavier_uniform_(self.extended_audio_user.weight)

        # self.dataset_path = os.path.abspath(os.getcwd()+config['data_path'] + config['dataset'])
        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.data_name = config['dataset']

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
        self.missing_generation = config['missing_generation']

        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)

            if self.a_feat is not None:
                # 3-modality historical behavior preserved: the tiktok tree
                # (a) loads a cached adj file if present and (b) applies the
                # missing-modal mask AFTER the kNN sparsification.
                if os.path.exists(image_adj_file):
                    image_adj = torch.load(image_adj_file)
                else:
                    image_adj = build_sim(self.image_embedding.weight.detach())
                    image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                    if self.missing_modal :
                        image_adj[self.missing_items_v, :] = image_adj[:, self.missing_items_v] = 0.0
                        image_adj[self.missing_items_v, self.missing_items_v] = 1.0
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
            else:
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

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

            if self.a_feat is not None:
                # 3-modality historical behavior preserved: cached adj file +
                # missing-modal mask applied AFTER the kNN sparsification.
                if os.path.exists(text_adj_file):
                    text_adj = torch.load(text_adj_file)
                else:
                    text_adj = build_sim(self.text_embedding.weight.detach())
                    text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                    if self.missing_modal :
                        text_adj[self.missing_items_t, :] = text_adj[:, self.missing_items_t] = 0.0
                        text_adj[self.missing_items_t, self.missing_items_t] = 1.0
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
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                if self.missing_modal :
                    text_adj[self.missing_items_t, :] = text_adj[:, self.missing_items_t] = 0.0
                    text_adj[self.missing_items_t, self.missing_items_t] = 1.0
                # text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj)

                # if self.new_items :
                #     text_adj = build_sim(self.text_embedding.weight.detach())
                #     text_adj[self.new_items_set, :] = 0.0
                #     text_adj[:, self.new_items_set] = 0.0
                #     text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                #     text_adj = compute_normalized_laplacian(text_adj)
                #     self.text_original_adj_newitems = text_adj.cuda()

                text_adj = text_adj.to_sparse_coo()
                self.text_original_adj = text_adj.cuda()

                if self.new_items :
                    text_adj = build_sim(self.text_embedding.weight.detach())
                    text_adj[self.new_items_set, :] = 0.0
                    text_adj[:, self.new_items_set] = 0.0
                    text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                    text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo()
                    self.text_original_adj_newitems = text_adj.cuda()

        if self.a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)

            audio_adj = build_sim(self.audio_embedding.weight.detach())
            audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
            if self.missing_modal :
                audio_adj[self.missing_items_a, :] = audio_adj[:, self.missing_items_a] = 0.0
                audio_adj[self.missing_items_a, self.missing_items_a] = 1.0
            audio_adj = compute_normalized_laplacian(audio_adj)


            audio_adj = audio_adj.to_sparse_coo()
            self.audio_original_adj = audio_adj.cuda()

            if self.new_items :
                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj[self.new_items_set, :] = 0.0
                audio_adj[:, self.new_items_set] = 0.0
                # audio_adj[self.new_items_set, self.new_items_set] = 1.0
                audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
                audio_adj = compute_normalized_laplacian(audio_adj).to_sparse_coo()
                self.audio_original_adj_newitems = audio_adj.cuda()

        #  Enhancing User-Item Graph
        if self.a_feat is not None:
            if self.new_items :
                self.inter = self.find_inter(self.image_original_adj_newitems, self.text_original_adj_newitems, self.audio_original_adj_newitems)
            else :
                self.inter = self.find_inter(self.image_original_adj, self.text_original_adj, self.audio_original_adj)
        else:
            if self.new_items :
                self.inter = self.find_inter(self.image_original_adj_newitems, self.text_original_adj_newitems)
            else :
                self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        
        
        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )
        
        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )

        if self.a_feat is not None:
            self.audio_reduce_dim = nn.Linear(self.a_feat.shape[1], self.embedding_dim)
            self.audio_trans_dim = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
            self.audio_space_trans = nn.Sequential(
                self.audio_reduce_dim,
                self.audio_trans_dim
            )

        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=-1)
                
        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        if self.a_feat is not None:
            self.audio_behavior = nn.Sequential(
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
        else:
            # 2-modality (Amazon) masks: keys {all, t, v}
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))

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

            # NOTE: the tiktok tree additionally branched on config['missing_new_n']
            # (values 0/1/2/3) right here. The released tiktok runs used
            # missing_new_n = -1, so none of those branches were ever taken;
            # they are dropped in this unified file and 'missing_new_n' is not
            # a config key anymore. (3-modality historical behavior preserved)
        elif config['missing_imputation'] == 2 and self.a_feat is not None :
            # 3-modality historical behavior preserved: the tiktok tree accepted
            # missing_imputation == 2 as a no-op.
            pass
        else :
            if self.a_feat is not None:
                assert False, f"Missing Imputation Must bo 0 or 1 or 2, Not {config['missing_imputation']}"
            assert False, f"Missing Imputation Must bo 0 or 1, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']
    
    def find_inter(self, image_adj, text_adj, audio_adj=None):
        inter_file = os.path.join(self.dataset_path, 'inter.json')

        if audio_adj is not None:
            # 3-modality historical behavior preserved: the tiktok tree caches
            # `inter.json` on disk (loading it back on later runs, with string
            # keys) and actually runs the intersection loop. Note the historical
            # quirk that ado_sim is collected but NOT used in the intersection.
            if os.path.exists(inter_file):
                with open(inter_file) as f:
                    inter = json.load(f)
            else:
                j = 0
                inter = defaultdict(list)
                img_sim = []
                txt_sim = []
                ado_sim = []
                for i in range(0,len(image_adj._indices()[0])):
                    img_id = image_adj._indices()[0][i]
                    txt_id = text_adj._indices()[0][i]
                    ado_id = audio_adj._indices()[0][i]
                    assert img_id == txt_id
                    assert txt_id == ado_id
                    id = img_id.item()
                    img_sim.append(image_adj._indices()[1][j].item())
                    txt_sim.append(text_adj._indices()[1][j].item())
                    ado_sim.append(audio_adj._indices()[1][j].item())

                    if len(img_sim)==10 and len(txt_sim)==10 and len(ado_sim) == 10:
                        it_inter = list(set(img_sim) & set(txt_sim))
                        inter[id] = [v for v in it_inter if v != id]
                        img_sim = []
                        txt_sim = []
                        ado_sim = []

                    j += 1

                with open(inter_file, "w") as f:
                    json.dump(inter, f)

            return inter

        j = 0
        inter = defaultdict(list)
        # img_sim = []
        # txt_sim = []
        # for i in range(0,len(image_adj._indices()[0])):
        #     img_id = image_adj._indices()[0][i]
        #     txt_id = text_adj._indices()[0][i]
        #     assert img_id == txt_id
        #     id = img_id.item()
        #     img_sim.append(image_adj._indices()[1][j].item())
        #     txt_sim.append(text_adj._indices()[1][j].item())
            
        #     if len(img_sim)==10 and len(txt_sim)==10:
        #         it_inter = list(set(img_sim) & set(txt_sim))
        #         inter[id] = [v for v in it_inter if v != id]
        #         img_sim = []
        #         txt_sim = []
            
        #     j += 1
        
        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)
        
        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1]*len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items,self.n_items), dtype=np.int32)
        return item_adj
    
    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def forward(self, adj, train=False):
        #  Encoding Multiple Modalities

        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.image_space_trans(self.image_embedding.weight))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.text_space_trans(self.text_embedding.weight))
        if self.a_feat is not None:
            audio_item_embeds = torch.multiply(self.item_id_embedding.weight, self.audio_space_trans(self.audio_embedding.weight))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

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

        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)
        
        explicit_image_item = self.conv_ii(image_org_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)
        
        extended_image_embeds = self.conv_ui(adj, self.extended_image_user.weight, explicit_image_item) 

        explicit_text_item = self.conv_ii(text_org_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        
        extended_text_embeds = self.conv_ui(adj, self.extended_text_user.weight, explicit_text_item)

        if self.a_feat is not None:
            explicit_audio_item = self.conv_ii(audio_org_adj, audio_item_embeds)
            explicit_audio_user = torch.sparse.mm(self.R, explicit_audio_item)
            explicit_audio_embeds = torch.cat([explicit_audio_user, explicit_audio_item], dim=0)

            extended_audio_embeds = self.conv_ui(adj, self.extended_audio_user.weight, explicit_audio_item)

            extended_it_embeds = (extended_image_embeds + extended_text_embeds + extended_audio_embeds) / 3.0
        else:
            extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2

        # Attributes Separation for Better Integration
        if self.a_feat is not None:
            image_weights, text_weights, audio_weights = torch.split(
                self.softmax(
                    torch.cat([
                        self.separate_coarse(explicit_image_embeds),
                        self.separate_coarse(explicit_text_embeds),
                        self.separate_coarse(explicit_audio_embeds)
                    ], dim=-1)
                ),
                1,
                dim=-1
            )
            coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds + audio_weights * explicit_audio_embeds
        else:
            image_weights, text_weights = torch.split(
                self.softmax(
                    torch.cat([
                        self.separate_coarse(explicit_image_embeds),
                        self.separate_coarse(explicit_text_embeds)
                    ], dim=-1)
                ),
                1,
                dim=-1
            )
            coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds

        fine_grained_image = torch.multiply(self.image_behavior(extended_id_embeds), (explicit_image_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(self.text_behavior(extended_id_embeds), (explicit_text_embeds - coarse_grained_embeds))
        if self.a_feat is not None:
            fine_grained_audio = torch.multiply(self.audio_behavior(extended_id_embeds), (explicit_audio_embeds - coarse_grained_embeds))

            integration_embeds = (fine_grained_image + fine_grained_text + fine_grained_audio + coarse_grained_embeds) / 4
        else:
            integration_embeds = (fine_grained_image + fine_grained_text + coarse_grained_embeds) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train:
            if self.a_feat is not None:
                return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (explicit_image_embeds, explicit_text_embeds, explicit_audio_embeds)
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (explicit_image_embeds, explicit_text_embeds)

        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

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

        if self.a_feat is not None:
            # 3-modality historical behavior preserved (tiktok tree)
            if self.new_items and np.isin(neg_items.detach().cpu().numpy(), self.new_items_set).sum() != 0 :
                assert False, "New Item Error !!"

            if self.missing_modal :
                t_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_t) # t 있는 애들
                v_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_v) # v 있는 애들
                a_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_a) # a 있는 애들

                tv_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_v))
                ta_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_a))
                va_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(self.missing_items_a, self.missing_items_v))

                tva_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(np.union1d(self.missing_items_t, self.missing_items_v), self.missing_items_a))
            else :
                t_index = pos_items.detach().cpu().numpy()
                v_index = pos_items.detach().cpu().numpy()
                a_index = pos_items.detach().cpu().numpy()

                tv_index = pos_items.detach().cpu().numpy()
                ta_index = pos_items.detach().cpu().numpy()
                va_index = pos_items.detach().cpu().numpy()

                tva_index = pos_items.detach().cpu().numpy()
        else:
            if self.missing_modal :
                t_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_t)
                v_index = np.setdiff1d(pos_items.detach().cpu().numpy(), self.missing_items_v)
                tv_index = np.setdiff1d(pos_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_v))
            else :
                t_index = pos_items.detach().cpu().numpy()
                v_index = pos_items.detach().cpu().numpy()
                tv_index = pos_items.detach().cpu().numpy()

        embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)

        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        if self.a_feat is not None:
            explicit_image_embeds, explicit_text_embeds, explicit_audio_embeds = embeds_3
        else:
            explicit_image_embeds, explicit_text_embeds = embeds_3

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        if self.a_feat is not None:
            # 3-modality historical behavior preserved: pairwise alignment over
            # the fully-complete (tva) items, averaged over the 3 pairs.
            vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds[tva_index], explicit_text_embeds[tva_index])
            vt_loss += self.vt_loss * self.align_vt(explicit_audio_embeds[tva_index], explicit_text_embeds[tva_index])
            vt_loss += self.vt_loss * self.align_vt(explicit_image_embeds[tva_index], explicit_audio_embeds[tva_index])
            vt_loss /= 3.0
        else:
            vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds[tv_index], explicit_text_embeds[tv_index])

        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings)

        if self.a_feat is not None:
            bm_loss = self.bm_loss * (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + self.InfoNCE(integration_items[tva_index], extended_id_items[tva_index], self.bm_temp))
        else:
            bm_loss = self.bm_loss * (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + self.InfoNCE(integration_items[tv_index], extended_id_items[tv_index], self.bm_temp))

        al_loss = vt_loss + bm_loss
        
        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)

        # Enhancing User Modality Representation
        c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)
        
        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2
        
        return bpr_loss + al_loss + um_loss + reg_loss
    
    
    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss
    
    def align_vt(self,embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        
        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()
        
        return vt_loss
    
    def full_sort_predict(self, interaction):
        assert not self.training
        user = interaction[0]

        all_embeds = self.forward(self.norm_adj)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores