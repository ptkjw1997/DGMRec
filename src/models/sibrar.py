# coding: utf-8

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class SIBRAR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SIBRAR, self).__init__(config, dataset)

        self.embedding_dim = 64
        self.feat_embed_dim = 64

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

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
        self.missing_generation = config['missing_generation']

        if self.a_feat is not None :
            self.modality_count = torch.ones(self.n_items) / 4.0
            if self.missing_modal :
                self.modality_count[self.missing_items['all']] = 1.0
                self.modality_count[self.missing_items['t']] = 1.0 / 3.0
                self.modality_count[self.missing_items['v']] = 1.0 / 3.0
                self.modality_count[self.missing_items['a']] = 1.0 / 3.0

                self.modality_count[self.missing_items['tv']] = 2.0 / 3.0
                self.modality_count[self.missing_items['ta']] = 2.0 / 3.0
                self.modality_count[self.missing_items['va']] = 2.0 / 3.0
        else :
            self.modality_count = torch.ones(self.n_items) / 3.0
            if self.missing_modal :
                self.modality_count[self.missing_items['all']] = 1.0
                self.modality_count[self.missing_items['t']] = 0.5
                self.modality_count[self.missing_items['v']] = 0.5
        self.modality_count = self.modality_count.to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained((self.v_feat), freeze=False)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained((self.t_feat), freeze=False)

        if self.a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained((self.a_feat), freeze=False)

        self.fc1 = nn.Linear(64, 64)
        self.b_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()

        self.temp = config['temp']
        self.lamb = config['lamb']

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        if self.a_feat is not None:
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.feat_embed_dim)

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



    def forward(self):
        pass

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeddings = self.user_embedding(users)

        id_embeddings = self.item_id_embedding.weight

        image_embeddings, text_embeddings = self.image_trs(self.image_embedding.weight), self.text_trs(self.text_embedding.weight)
        if self.a_feat is not None:
            audio_embeddings = self.audio_trs(self.audio_embedding.weight)

        id_embeddings = (self.fc1(self.dropout(id_embeddings)))
        id_embeddings = self.fc2(self.relu(id_embeddings))

        image_embeddings = (self.fc1(self.dropout(image_embeddings)))
        image_embeddings = self.fc2(self.relu(image_embeddings))

        text_embeddings = (self.fc1(self.dropout(text_embeddings)))
        text_embeddings = self.fc2(self.relu(text_embeddings))

        if self.a_feat is not None:
            audio_embeddings = (self.fc1(self.dropout(audio_embeddings)))
            audio_embeddings = self.fc2(self.relu(audio_embeddings))

        image_embeddings, text_embeddings = F.normalize(image_embeddings), F.normalize(text_embeddings)
        if self.a_feat is not None:
            audio_embeddings = F.normalize(audio_embeddings)


        pos_id_embeddings, neg_id_embeddings = id_embeddings[pos_items], id_embeddings[neg_items]
        pos_image_embeddings, neg_image_embeddings = image_embeddings[pos_items], image_embeddings[neg_items]
        pos_text_embeddings, neg_text_embeddings = text_embeddings[pos_items], text_embeddings[neg_items]

        if self.a_feat is not None:
            # 3-modality historical behavior preserved: multinomial draws 3 of 4 modalities but only the
            # first two samples are used downstream (RNG consumption must match the tiktok tree)
            pos_audio_embeddings, neg_audio_embeddings = audio_embeddings[pos_items], audio_embeddings[neg_items]

            pos_item_embeddings = torch.stack([pos_id_embeddings, pos_image_embeddings, pos_text_embeddings, pos_audio_embeddings], dim = 1)
            neg_item_embeddings = torch.stack([neg_id_embeddings, neg_image_embeddings, neg_text_embeddings, neg_audio_embeddings], dim = 1)

            weight = torch.ones(4).expand(users.shape[0], -1)
            modal_indice = torch.multinomial(weight, num_samples = 3, replacement = False)
        else:
            pos_item_embeddings = torch.stack([pos_id_embeddings, pos_image_embeddings, pos_text_embeddings], dim = 1)
            neg_item_embeddings = torch.stack([neg_id_embeddings, neg_image_embeddings, neg_text_embeddings], dim = 1)

            weight = torch.ones(3).expand(users.shape[0], -1)
            modal_indice = torch.multinomial(weight, num_samples = 2, replacement = False)

        pos_item_modal_1 = pos_item_embeddings[torch.arange(users.shape[0]), modal_indice[:, 0], :]
        pos_item_modal_2 = pos_item_embeddings[torch.arange(users.shape[0]), modal_indice[:, 1], :]

        neg_item_modal_1 = neg_item_embeddings[torch.arange(users.shape[0]), modal_indice[:, 0], :]
        neg_item_modal_2 = neg_item_embeddings[torch.arange(users.shape[0]), modal_indice[:, 1], :]

        pos_item_final = (pos_item_modal_1 + pos_item_modal_2)
        neg_item_final = (neg_item_modal_1 + neg_item_modal_2)
        # pos_cnt, neg_cnt = self.modality_count[pos_items], self.modality_count[neg_items]

        # pos_item_final = torch.einsum("ij, i -> ij", pos_item_final, pos_cnt)
        # neg_item_final = torch.einsum("ij, i -> ij", neg_item_final, neg_cnt)

        batch_mf_loss = self.bpr_loss(user_embeddings, pos_item_final, neg_item_final)

        pos_score = torch.einsum("ij, ij -> i", pos_item_modal_1, pos_item_modal_2) / self.temp
        neg_score_1 = torch.einsum("ij, ij -> i", pos_item_modal_1, neg_item_modal_2) / self.temp
        neg_score_2 = torch.einsum("ij, ij -> i", pos_item_modal_2, neg_item_modal_1) / self.temp

        info_loss = 0.0
        info_loss += -torch.log(torch.exp(pos_score)) + torch.logaddexp(pos_score, neg_score_1)
        info_loss += -torch.log(torch.exp(pos_score)) + torch.logaddexp(pos_score, neg_score_2)

        # print(f"BPR : {batch_mf_loss:.4f} | INFO : {info_loss.mean():.4f}")
        if info_loss.mean() < 0.049 :
            print("check")
        return batch_mf_loss + info_loss.mean() * self.lamb

    def full_sort_predict(self, interaction):
        user = interaction[0]

        user_embeddings = self.user_embedding(user)

        id_embeddings = self.item_id_embedding.weight

        image_embeddings, text_embeddings = self.image_trs(self.image_embedding.weight), self.text_trs(self.text_embedding.weight)
        if self.a_feat is not None:
            audio_embeddings = self.audio_trs(self.audio_embedding.weight)

        id_embeddings = (self.fc1(self.dropout(id_embeddings)))
        id_embeddings = self.fc2(self.relu(id_embeddings))

        # image_embeddings = self.b_norm(self.fc1(self.dropout(image_embeddings)))
        image_embeddings = (self.fc1(self.dropout(image_embeddings)))
        image_embeddings = self.fc2(self.relu(image_embeddings))

        # text_embeddings = self.b_norm(self.fc1(self.dropout(text_embeddings)))
        text_embeddings = (self.fc1(self.dropout(text_embeddings)))
        text_embeddings = self.fc2(self.relu(text_embeddings))

        if self.a_feat is not None:
            audio_embeddings = (self.fc1(self.dropout(audio_embeddings)))
            audio_embeddings = self.fc2(self.relu(audio_embeddings))

        image_embeddings, text_embeddings = F.normalize(image_embeddings), F.normalize(text_embeddings)
        if self.a_feat is not None:
            audio_embeddings = F.normalize(audio_embeddings)

        if self.a_feat is not None:
            item_embeddings = torch.mean(torch.stack([id_embeddings, image_embeddings, text_embeddings, audio_embeddings], dim = 1), dim = 1)
        else:
            item_embeddings = torch.mean(torch.stack([id_embeddings, image_embeddings, text_embeddings], dim = 1), dim = 1)
        scores = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
        return scores

