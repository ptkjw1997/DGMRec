# coding: utf-8
# 
"""
Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback, MM 2020
"""
import math
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from SAGEConv import SAGEConv
#from GATConv import GATConv
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import add_self_loops, dropout_adj
# from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
# from torch.utils.checkpoint import checkpoint
##########################################################################
def add_self_loops(edge_index, num_nodes):
    """Add self-loops to the edge index."""
    loops = torch.arange(0, num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops], dim=1)
    return edge_index

def remove_self_loops(edge_index):
    """Remove self-loops from the edge index."""
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index

def softmax(values, index, num_nodes):
    """Manually implement softmax over edge indices."""
    exp_values = torch.exp(values - torch.max(values))
    sum_values = torch.zeros(num_nodes, device=values.device).scatter_add_(0, index, exp_values)
    return exp_values / sum_values[index]

def filter_adj(row, col, edge_attr, mask):
    """Filter edges based on the mask."""
    row = row[mask]
    col = col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    return row, col, edge_attr

def dropout_edge(edge_index, edge_attr=None, p=0.5, force_undirected=False, num_nodes=None, training=True):
    """Drop edges based on probability."""
    if p < 0. or p > 1.:
        raise ValueError(f"Dropout probability has to be between 0 and 1 (got {p})")

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index

    mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax as pyg_softmax


class SAGEConv(MessagePassing):
    """Original PyG-based GRCN conv (restored from the commented reference
    implementation; the interim hand-written port mis-aligned routing
    weights and under-performed)."""
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr='add')
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j, size_i, edge_index_i):
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        self.alpha = pyg_softmax(self.alpha, edge_index_i, num_nodes=size_i)
        return x_j * self.alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        if self.has_norm:
            x = F.normalize(x)
        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)
        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)
        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)
        return x + x_hat_1 + x_hat_2


# class SAGEConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='mean', **kwargs):
#         super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def forward(self, x, edge_index, weight_vector, size=None):
#         self.weight_vector = weight_vector
#         return self.propagate(edge_index, size=size, x=x)

#     def message(self, x_j):
#         return x_j * self.weight_vector

#     def update(self, aggr_out):
#         return aggr_out

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)

# class GATConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, self_loops=False):
#         super(GATConv, self).__init__(aggr='add')#, **kwargs)
#         self.self_loops = self_loops
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def forward(self, x, edge_index, size=None):
#         edge_index, _ = remove_self_loops(edge_index)
#         if self.self_loops:
#             edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         return self.propagate(edge_index, size=size, x=x)


#     def message(self,  x_i, x_j, size_i ,edge_index_i):
#         #print(edge_index_i, x_i, x_j)
#         self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
#         #print(self.alpha)
#         #print(edge_index_i,size_i)
#         # alpha = F.tanh(alpha)
#         # self.alpha = F.leaky_relu(self.alpha)
#         # alpha = torch.sigmoid(alpha)
#         self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)
#         # Sample attention coefficients stochastically.
#         # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return x_j*self.alpha.view(-1,1)
#         # return x_j * alpha.view(-1, self.heads, 1)

#     def update(self, aggr_out):
#         return aggr_out



# class EGCN(torch.nn.Module):
#     def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
#         super(EGCN, self).__init__()
#         self.num_user = num_user
#         self.num_item = num_item
#         self.dim_E = dim_E
#         self.aggr_mode = aggr_mode
#         self.has_act = has_act
#         self.has_norm = has_norm
#         self.id_embedding = nn.Parameter( nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
#         self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)         
#         self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

#     def forward(self, edge_index, weight_vector):
#         x = self.id_embedding
#         edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)

#         if self.has_norm:
#             x = F.normalize(x) 

#         x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector) 

#         if self.has_act:
#             x_hat_1 = F.leaky_relu_(x_hat_1)

#         x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
#         if self.has_act:
#             x_hat_2 = F.leaky_relu_(x_hat_2)

#         return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(self, features, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, is_word=False):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)
        self.is_word = is_word

        if is_word:
            self.word_tensor = torch.LongTensor(features).cuda()
            self.features = nn.Embedding(torch.max(features[1])+1, dim_C)
            nn.init.xavier_normal_(self.features.weight)

        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.MLP = nn.Linear(self.dim_feat, self.dim_C)
            #print('MLP weight',self.MLP.weight)
            nn.init.xavier_normal_(self.MLP.weight)
            #print(self.MLP.weight)

    def forward(self, edge_index):
        #print(self.features)
        features = F.leaky_relu(self.MLP(self.features))
        #print('features',features)
        
        if self.has_norm:
            preference = F.normalize(self.preference)
            features = F.normalize(features)
            #print(preference,features)

        for i in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            #print(x,edge_index)
            x_hat_1 = self.conv_embed_1(x, edge_index) 
            preference = preference + x_hat_1[:self.num_user]

            if self.has_norm:
                preference = F.normalize(preference)

        x = torch.cat((preference, features), dim=0)
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)

        x_hat_1 = self.conv_embed_1(x, edge_index) 

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCN(GeneralRecommender):
    def __init__(self,  config, dataset):
        super(GRCN, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        dim_C = config['latent_embedding']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         # not used
        self.aggr_mode = 'add'
        self.weight_mode = 'confid'
        self.fusion_mode = 'concat'
        has_id = True
        has_act= False
        has_norm= True
        is_word = False
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']
        self.dropout = 0
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        #self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0
        self.id_gcn = EGCN(num_user, num_item, dim_x, self.aggr_mode, has_act, has_norm)
        self.pruning = True

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

        num_model = 0
        if self.v_feat is not None:
            self.v_gcn = CGCN(self.v_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm)
            num_model += 1

        if self.a_feat is not None:
            self.a_gcn = CGCN(self.a_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm)
            num_model += 1

        if self.t_feat is not None:
            self.t_gcn = CGCN(self.t_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm, is_word)
            num_model += 1

        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, num_model))))

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).to(self.device)
        
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
            self.complete_items = np.setdiff1d(np.arange(self.n_items),
                                               np.union1d(np.union1d(self.missing_items_v, self.missing_items_t), self.missing_items_a))
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
        elif config['missing_imputation'] == 2 and self.a_feat is not None :
            # NOTE: historical 3-modality (tiktok) branch accepts missing_imputation == 2 (no-op)
            pass
        elif self.a_feat is not None :
            assert False, f"Missing Imputation Must bo 0 or 1 or 2, Not {config['missing_imputation']}"
        else :
            assert False, f"Missing Imputation Must bo 0 or 1, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def dropout_adj(self, edge_index, edge_attr = None, p = 0.5, force_undirected = False, num_nodes = None, training = True):
        if p < 0. or p > 1.:
            raise ValueError(f'Dropout probability has to be between 0 and 1 '
                            f'(got {p}')

        if not training or p == 0.0:
            return edge_index, edge_attr

        row, col = edge_index

        mask = torch.rand(row.size(0), device=edge_index.device) >= p

        if force_undirected:
            mask[row > col] = False

        row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

        if force_undirected:
            edge_index = torch.stack(
                [torch.cat([row, col], dim=0),
                torch.cat([col, row], dim=0)], dim=0)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        else:
            edge_index = torch.stack([row, col], dim=0)

        return edge_index, edge_attr
    
    def forward(self):
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = self.dropout_adj(self.edge_index, p=self.dropout)
        # edge_index = self.dropout_edge(self.edge_index, p=0.5, training=self.training)
        #print('edge_index: ', edge_index)

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(edge_index)
            weight = weight_v
            content_rep = v_rep
            #print('weight_v is: ', weight)
            #print('content_rep: ',content_rep)

        if self.a_feat is not None:
            num_modal += 1
            a_rep, weight_a = self.a_gcn(edge_index)
            if weight is None:
                weight = weight_a
                content_rep = a_rep
            else:
                content_rep = torch.cat((content_rep, a_rep), dim=1)
                if self.weight_mode == 'mean':
                    weight = weight + weight_a
                else:
                    weight = torch.cat((weight, weight_a), dim=1)

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t   
                conetent_rep = t_rep
            else:
                content_rep = torch.cat((content_rep,t_rep),dim=1)
                if self.weight_mode == 'mean':  
                    weight  = weight+  weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)   

        if self.weight_mode == 'mean':
            weight = weight/num_modal
        elif self.weight_mode == 'max':
            weight, _ = torch.max(weight, dim = 1)
            weight = weight.view(-1, 1)
        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]), dim = 0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim = 1)
            weight = weight.view(-1, 1)
            

        if self.pruning:
            weight = torch.relu(weight)
            


        id_rep = self.id_gcn(edge_index, weight)
        #print('id_rep is: ',id_rep)

        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
            
        elif self.fusion_mode  == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            # representation = (id_rep+v_rep+a_rep+t_rep)/4
            representation = (id_rep+v_rep+t_rep)/3

        self.result = representation
        #print('representation is: ',representation)
        return representation

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_gcn.id_embedding[user_tensor]**2 + self.id_gcn.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        reg_content_loss = torch.zeros(1).cuda() 
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor]**2).mean()
        if self.a_feat is not None:
            reg_content_loss = reg_content_loss + (self.a_gcn.preference[user_tensor]**2).mean()
        if self.t_feat is not None:
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor]**2).mean()

        reg_confid_loss = (self.model_specific_conf**2).mean()
        
        reg_loss = reg_embedding_loss + reg_content_loss

        reg_loss = self.reg_weight * reg_loss
        # debug print removed (tensor __format__ incompatible with torch>=2.4)

        return loss + reg_loss
        
    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix