# coding: utf-8
"""
MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. 
In ACM MM`19,
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch_geometric is only needed for the 3-modality (tiktok) path, which uses a
# MessagePassing-based BaseModel (see BaseModelPyG below). The 2-modality path
# never touches it, so the import is guarded to keep 2-modality runs working in
# environments without torch_geometric.
try:
    from torch_geometric.nn.conv import MessagePassing
    import torch_geometric
except ImportError:
    MessagePassing = None
    torch_geometric = None

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class MMGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGCN, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         # not used
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        if self.a_feat is None:
            # 2-modality (Amazon) tree only: the 3-modality tiktok tree had no
            # new-items / missing-modal preprocessing in MMGCN at all.
            # 3-modality historical behavior preserved
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

        # The 3-modality tiktok tree used a torch_geometric MessagePassing
        # BaseModel (different init bounds, no bias, no output normalization);
        # the 2-modality tree uses the vectorized BaseModel below.
        # 3-modality historical behavior preserved
        base_model_cls = BaseModel if self.a_feat is None else BaseModelPyG

        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256, device=self.device,
                             base_model_cls=base_model_cls)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, device=self.device,
                             base_model_cls=base_model_cls)
            self.num_modal += 1

        if self.a_feat is not None:
            self.a_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.a_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, device=self.device,
                             base_model_cls=base_model_cls)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(self.device)
        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(self.device)
        
    def preprocess_missing_modal(self, config) :

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t']))
        self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v']))
        self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_v, self.missing_items_t))

        self.items_tv = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_t, self.missing_items_v))
        
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
        else :
            assert False, f"Missing Imputation Must bo 0 or 1, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)
        if self.a_feat is not None:
            if representation is None:
                representation = self.a_gcn(self.a_feat, self.id_embedding)
            else:
                representation += self.a_gcn(self.a_feat, self.id_embedding)
        representation /= self.num_modal

        self.result = representation
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
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)))) # weight 就是label

        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        if self.t_feat is not None:
            reg_embedding_loss += (self.t_gcn.preference**2).mean()
        if self.a_feat is not None:
            reg_embedding_loss += (self.a_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        if self.a_feat is None:
            # the tiktok tree had no per-step loss print
            # (3-modality historical behavior preserved)
            print(f"Loss : {loss:.4f}")
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None, device='cpu', base_model_cls=None):
        super(GCN, self).__init__()
        if base_model_cls is None:
            base_model_cls = BaseModel
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent))))

            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = base_model_cls(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat))))

            self.conv_embed_1 = base_model_cls(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = base_model_cls(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = base_model_cls(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))  # equation 5
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))  # equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))  # equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)

        return x

class BaseModel(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add'):
        super(BaseModel, self).__init__()
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """Perform the forward pass."""
        # Linear transformation
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias

        # Aggregate neighbors (vectorized; equivalent to `out[i] += x[j]`
        # over all edges, which as a Python loop is ~1000x slower)
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])

        if self.aggr == 'mean':
            degree = torch.bincount(row, minlength=x.size(0))
            degree[degree == 0] = 1  # Avoid division by zero
            out = out / degree.view(-1, 1)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

if MessagePassing is not None:
    class BaseModelPyG(MessagePassing):
        # 3-modality historical behavior preserved: this is the tiktok tree's
        # BaseModel verbatim (torch_geometric uniform init, no bias, no output
        # normalization), used only when a_feat is not None.
        def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
            super(BaseModelPyG, self).__init__(aggr=aggr, **kwargs)
            self.aggr = aggr
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.normalize = normalize
            self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

            self.reset_parameters()

        def reset_parameters(self):
            torch_geometric.nn.inits.uniform(self.in_channels, self.weight)

        def forward(self, x, edge_index, size=None):
            x = torch.matmul(x, self.weight)
            return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

        def message(self, x_j, edge_index, size):
            return x_j

        def update(self, aggr_out):
            return aggr_out

        def __repr(self):
            return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
else:
    BaseModelPyG = None