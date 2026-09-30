import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class MILK(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MILK, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        assert self.embedding_dim == self.feat_embed_dim
        self.penalty_coeff = config['penalty_coeff']
        self.align_coeff = config['align_coeff']
        self.reg_coeff = config['reg_coeff']
        self.alpha = config['alpha']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.new_items = config['new_items']
        if config['new_items'] :
            self.new_items_set = np.load(f"../data/{config['dataset']}/new_items.npy")
            self.old_items_set = np.setdiff1d(np.arange(self.n_items), self.new_items_set)
        else :
            self.new_items_set = self.old_items_set = np.arange(self.n_items)

        self.missing_modal = config['missing_modal']
        if config['missing_modal'] :
            self.preprocess_missing_modal(config)
        if config['normalize'] :
            self.v_feat = F.normalize(self.v_feat)
            self.t_feat = F.normalize(self.t_feat)
            if self.a_feat is not None :
                self.a_feat = F.normalize(self.a_feat)

        if self.v_feat is not None :
            if self.a_feat is None :
                self.v_feat = F.normalize(self.v_feat)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze = False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None :
            if self.a_feat is None :
                self.t_feat = F.normalize(self.t_feat)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze = False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        if self.a_feat is not None :
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze = False)
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.embedding_dim)

        
        self.fusion_trs = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(5)])

        self.final_item = None
        self.final_user = None
        self.activation = nn.Sigmoid()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])


        if self.a_feat is not None :
            self.modality_count = torch.ones(self.n_items) / 3.0
            if self.missing_modal :
                self.modality_count[self.missing_items['all']] = 0
                self.modality_count[self.missing_items['tv']] = 1
                self.modality_count[self.missing_items['ta']] = 1
                self.modality_count[self.missing_items['va']] = 1
                self.modality_count[self.missing_items['t']] = 1/2
                self.modality_count[self.missing_items['v']] = 1/2
                self.modality_count[self.missing_items['a']] = 1/2
        else :
            self.modality_count = torch.ones(self.n_items) / 2.0
            if self.missing_modal :
                self.modality_count[self.missing_items['all']] = 0
                self.modality_count[self.missing_items['t']] = 1.0
                self.modality_count[self.missing_items['v']] = 1.0
        self.modality_count = self.modality_count.to(self.device)

        self.align = nn.MSELoss(reduction = 'mean')
    def preprocess_missing_modal(self, config) :

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        if 'a' in self.missing_items :
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t'],
                                                    self.missing_items['tv'], self.missing_items['ta']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v'],
                                                    self.missing_items['tv'], self.missing_items['va']))
            self.missing_items_a = np.concatenate((self.missing_items['all'], self.missing_items['a'],
                                                    self.missing_items['ta'], self.missing_items['va']))
        else :
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))

        self.impute_missing_features(config['missing_imputation'])

    def forward(self) :
        pass

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        image_emb = self.image_trs(self.image_embedding.weight)
        text_emb = self.text_trs(self.text_embedding.weight)
        if self.a_feat is not None :
            audio_emb = self.audio_trs(self.audio_embedding.weight)

        if self.missing_modal :
            index_tv = np.setdiff1d(np.setdiff1d(pos_items.cpu().detach().numpy(), self.missing_items['t']), self.missing_items['v'])
            if self.a_feat is not None :
                index_ta = np.setdiff1d(np.setdiff1d(pos_items.cpu().detach().numpy(), self.missing_items['t']), self.missing_items['a'])
                index_va = np.setdiff1d(np.setdiff1d(pos_items.cpu().detach().numpy(), self.missing_items['v']), self.missing_items['a'])
        else :
            index_tv = pos_items.cpu().detach().numpy()
            if self.a_feat is not None :
                index_ta = pos_items.cpu().detach().numpy()
                index_va = pos_items.cpu().detach().numpy()

        align_loss = self.align(image_emb[index_tv], text_emb[index_tv])
        if self.a_feat is not None :
            align_loss += self.align(audio_emb[index_ta], text_emb[index_ta])
            align_loss += self.align(image_emb[index_va], audio_emb[index_va])

        if self.a_feat is not None :
            env_ratio = [[1/3, 1/3, 1/3]]
            lam_1, lam_2, lam_3 = np.random.dirichlet([self.alpha, self.alpha, self.alpha])
            env_ratio.append([lam_1, lam_2, lam_3])
            env_ratio.append([lam_3, lam_1, lam_2])
            env_ratio.append([lam_2, lam_3, lam_1])
        else :
            env_ratio = [[1/2, 1/2]]
            lam_1, lam_2  = np.random.dirichlet([self.alpha, self.alpha])
            env_ratio.append([lam_1, lam_2])
            env_ratio.append([lam_2, lam_1])

        bpr_loss, reg_loss = [], []
        user_emb = self.user_embedding.weight
        for env in env_ratio :
            if self.a_feat is not None :
                item_emb = image_emb * env[0] + text_emb * env[1] + audio_emb * env[2]
            else :
                item_emb = image_emb * env[0] + text_emb * env[1]
            item_emb = torch.einsum("ij, i -> ij", item_emb, self.modality_count)
            bpr_loss.append(self.bpr_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items]))

            reg_loss.append(self.reg_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items]))

        bpr_loss_exp = torch.stack(bpr_loss).mean()
        bpr_loss_var = torch.stack(bpr_loss).var()
        losses = [torch.stack(bpr_loss).sum(), self.reg_coeff * reg_loss[0], self.align_coeff * align_loss, self.penalty_coeff * bpr_loss_var]
        return sum(losses)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding.weight

        image_emb = self.image_trs(self.image_embedding.weight)
        text_emb = self.text_trs(self.text_embedding.weight)

        if self.a_feat is not None :
            audio_emb = self.audio_trs(self.audio_embedding.weight)
            item_emb = (image_emb + text_emb + audio_emb)
        else :
            item_emb = (image_emb + text_emb)
        item_emb = torch.einsum("ij, i -> ij", item_emb, self.modality_count)
        user_e = user_emb[user, :]

        score = torch.matmul(user_e, item_emb.transpose(0, 1))
        return score

    def predict(self, interaction):
        return super().predict(interaction)
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss
    
    def reg_loss(self, users, pos_items, neg_items) :
        
        return (1 / 2) * (users.norm(2).pow(2) + pos_items.norm(2).pow(2) + neg_items.norm(2).pow(2)) / float(len(users))
