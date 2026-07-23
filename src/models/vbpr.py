# coding: utf-8

r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, DiceLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F

class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.idx = 0

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

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
        self.missing_generation = config['missing_generation']

        self.item_raw_features = self.t_feat


        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss() # DiceLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

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

            self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(np.union1d(self.missing_items_v, self.missing_items_t), self.missing_items_a))
        else :
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))
            self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t))

            self.items_tv = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_t, self.missing_items_v))

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
        else :
            assert False, f"Missing Imputation Must bo 0 or 1, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']
        
    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_embeddings = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]

        loss_bpr = self.bpr_loss(user_e, pos_e, neg_e)

        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = loss_bpr + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
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
