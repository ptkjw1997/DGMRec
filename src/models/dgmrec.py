# coding: utf-8
r"""DGMRec: Disentangled Generative Multimodal Recommendation."""
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_neighbourhood
from utils.mi_estimator import CLUBSample


def compute_normalized_laplacian(adj):
    if not adj.is_sparse:
        adj = adj.to_sparse()

    rowsum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    indices = adj._indices()
    values = adj._values()
    row, col = indices[0], indices[1]
    new_values = values * d_inv_sqrt[row] * d_inv_sqrt[col]

    return torch.sparse.FloatTensor(indices, new_values, adj.shape)


def build_knn_graph_sparse(context_feats, topk, mask_idx=None, chunk=2048):
    f = F.normalize(context_feats, p=2, dim=-1)
    n = f.size(0)
    rows, cols, vals = [], [], []
    for s in range(0, n, chunk):
        sim = f[s:s + chunk] @ f.T
        v, i = torch.topk(sim, topk, dim=-1)
        r = torch.arange(s, s + sim.size(0), device=f.device).unsqueeze(1).expand_as(i)
        rows.append(r.reshape(-1))
        cols.append(i.reshape(-1))
        vals.append(v.reshape(-1))
        del sim
    row, col, val = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    if mask_idx is not None and len(mask_idx) > 0:
        m = torch.zeros(n, dtype=torch.bool, device=f.device)
        m[torch.as_tensor(np.asarray(mask_idx), device=f.device)] = True
        keep = ~(m[row] | m[col])
        row, col, val = row[keep], col[keep], val[keep]
        mi = torch.nonzero(m).squeeze(1)
        row = torch.cat([row, mi])
        col = torch.cat([col, mi])
        val = torch.cat([val, torch.ones_like(mi, dtype=val.dtype)])
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n)).coalesce()


SPARSE_KNN_THRESHOLD = 30000

MODALITY_NAMES = {'v': 'image', 't': 'text', 'a': 'audio'}


class DGMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DGMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = 3
        self.n_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']

        feats = {'v': self.v_feat, 't': self.t_feat, 'a': self.a_feat}
        self.mods = [m for m in ('v', 't', 'a') if feats[m] is not None]
        n_mods = len(self.mods)
        if n_mods > 2 :
            self.mod_pairs = list(zip(self.mods, self.mods[1:] + self.mods[:1]))
        else :
            self.mod_pairs = [(self.mods[0], self.mods[1])]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.n_nodes = self.n_users + self.n_items
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.norm_adj = self.norm_adj.to(self.device)
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.new_items = config['new_items']
        if config['new_items'] :
            self.new_items_set = np.load(f"../data/{config['dataset']}/new_items.npy")
            self.old_items_set = np.setdiff1d(np.arange(self.n_items), self.new_items_set)
        else :
            self.new_items_set = self.old_items_set = np.arange(self.n_items)

        self.missing_modal = config['missing_modal']
        self.missing_imputation = config['missing_imputation']
        self.missing_items_m = {m: np.array([], dtype=np.int64) for m in self.mods}
        self.writeback_items_m = {m: np.array([], dtype=np.int64) for m in self.mods}
        if config['missing_modal'] :
            self.preprocess_missing_modal(config, feats)

        self.mm_adj, self.mm_adj_infer = {}, {}
        for m in self.mods :
            emb = nn.Embedding.from_pretrained(feats[m], freeze = False).to(self.device)
            setattr(self, f'{MODALITY_NAMES[m]}_embedding', emb)
            self.build_mm_graph(m, emb.weight.detach())
        torch.cuda.empty_cache()

        dim = self.embedding_dim
        for m in self.mods :
            self._set(m, 'encoder', nn.Linear(feats[m].shape[1], dim).to(self.device))
        self.shared_encoder = nn.Linear(dim, dim).to(self.device)
        for m in self.mods :
            nn.init.xavier_uniform_(self._get(m, 'encoder').weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        for m in self.mods :
            self._set(m, 'encoder_s', nn.Linear(feats[m].shape[1], dim).to(self.device))
        for m in self.mods :
            nn.init.xavier_uniform_(self._get(m, 'encoder_s').weight)

        for m in self.mods :
            setattr(self, f'user_{MODALITY_NAMES[m]}_prefer', nn.Embedding(self.n_users, dim))
        for m in self.mods :
            nn.init.xavier_uniform_(getattr(self, f'user_{MODALITY_NAMES[m]}_prefer').weight)

        for m in self.mods :
            self._set(m, 'g_filter_trans', nn.Linear(dim, dim, bias = False))
        for m in self.mods :
            nn.init.xavier_uniform_(self._get(m, 'g_filter_trans').weight)

        for m in self.mods :
            self._set(m, 's_filter_trans', nn.Linear(dim, dim, bias = False))
        for m in self.mods :
            nn.init.xavier_uniform_(self._get(m, 's_filter_trans').weight)

        for m in self.mods :
            self._set(m, 'decoder', nn.Linear(dim * 2, feats[m].shape[1]).to(self.device))
        for m in self.mods :
            nn.init.xavier_uniform_(self._get(m, 'decoder').weight)

        self.act_g = nn.Tanh()

        self.refresh_adj_counter = 0

        for m in self.mods :
            gen = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
            gen.apply(self.init_weight)
            self._set(m, 'gen', gen)

        for m in self.translate_order() :
            trans = nn.Sequential(nn.Linear(dim * (n_mods - 1), dim), nn.Tanh(), nn.Linear(dim, dim))
            trans.apply(self.init_weight)
            self._set(m, 'translator', trans)

        self.additive = config['additive']
        self.avg_lambda = config['avg_lambda']
        self.infer_adj_update = config['infer_adj_update']

        self.interModal, self.interModalTemp = config['interModal'], config['interModalTemp']
        self.intraModal, self.intraModalTemp = config['intraModal'], config['intraModalTemp']
        self.alignBM, self.alignBMTemp = config['alignBM'], config['alignBMTemp']
        self.recon = config['recon']
        self.reg = config['reg']
        self.sampler = config['sampler']

    def _get(self, m, part) :
        return getattr(self, f'{MODALITY_NAMES[m]}_{part}')

    def _set(self, m, part, module) :
        setattr(self, f'{MODALITY_NAMES[m]}_{part}', module)

    def translate_order(self) :
        return [m for m in ('t', 'v', 'a') if m in self.mods]

    def others(self, m) :
        return [o for o in self.mods if o != m]

    def init_weight(self, layer) :
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def build_mm_graph(self, m, feat) :
        mask_idx = self.missing_items_m[m] if self.missing_modal else None
        if self.n_items > SPARSE_KNN_THRESHOLD :
            adj = compute_normalized_laplacian(build_knn_graph_sparse(feat, self.knn_k, mask_idx=mask_idx))
        else :
            adj = build_knn_neighbourhood(build_sim(feat), topk=self.knn_k)
            if mask_idx is not None :
                adj[mask_idx, :] = adj[:, mask_idx] = 0.0
                adj[mask_idx, mask_idx] = 1.0
            adj = compute_normalized_laplacian(adj).to_sparse_coo()

        if self.new_items :
            adj_new = build_sim(feat)
            adj_new[self.new_items_set, :] = adj_new[:, self.new_items_set] = 0.0
            adj_new[self.new_items_set, self.new_items_set] = 1.0
            adj_new = build_knn_neighbourhood(adj_new, topk=self.knn_k)
            adj_new = compute_normalized_laplacian(adj_new).to_sparse_coo()
            self.mm_adj_infer[m] = adj.clone()
            self.mm_adj[m] = adj_new.cuda()
        else :
            self.mm_adj[m] = self.mm_adj_infer[m] = adj

    def rebuild_mm_graph(self, m) :
        feat = self._get(m, 'embedding').weight.detach()
        if self.n_items > SPARSE_KNN_THRESHOLD :
            adj = build_knn_graph_sparse(feat, self.knn_k)
        else :
            adj = build_knn_neighbourhood(build_sim(feat), topk=self.knn_k)
        return compute_normalized_laplacian(adj).cpu()

    def _to_scipy(self, tensor):
        if tensor.is_sparse:
            tensor = tensor.detach().cpu()
            indices = tensor._indices().numpy()
            values = tensor._values().numpy()
            shape = tensor.shape
            return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
        else:
            return sp.csr_matrix(tensor.detach().cpu().numpy())

    def _to_tensor(self, matrix):
        coo = matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(self.device)

    def update_adj(self):
        torch.cuda.empty_cache()
        for m in self.mods :
            index = self.missing_items_m[m]
            if self.new_items :
                index = np.intersect1d(index, self.old_items_set)

            old_adj = self._to_scipy(self.mm_adj[m])
            with torch.no_grad() :
                new_adj = self._to_scipy(self.rebuild_mm_graph(m))

            old_adj[index] = new_adj[index].multiply(self.avg_lambda) + old_adj[index].multiply(1 - self.avg_lambda)
            self.mm_adj[m] = self._to_tensor(old_adj)

            del old_adj, new_adj
            torch.cuda.empty_cache()

    def update_adj_infer(self) :
        assert self.new_items == 1, "Error"
        with torch.no_grad() :
            for m in self.mods :
                index = np.intersect1d(self.missing_items_m[m], self.new_items_set)

                old_adj = self.mm_adj_infer[m].cpu().to_dense()
                torch.cuda.empty_cache()

                new_adj = build_knn_neighbourhood(build_sim(self._get(m, 'embedding').weight.detach()), topk=self.knn_k)
                new_adj = compute_normalized_laplacian(new_adj).cpu().to_dense()

                old_adj[index] = new_adj[index] * self.avg_lambda + old_adj[index] * (1 - self.avg_lambda)
                self.mm_adj_infer[m] = old_adj.to_sparse_coo()
                del new_adj

            torch.cuda.empty_cache()
            for m in self.mods :
                self.mm_adj_infer[m] = self.mm_adj_infer[m].to(self.device)

    def preprocess_missing_modal(self, config, feats) :
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_ratio = config['missing_ratio']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        groups = [k for k in self.missing_items if k != 'all']
        for m in self.mods :
            self.missing_items_m[m] = np.concatenate([self.missing_items['all']] + [self.missing_items[k] for k in groups if m in k])
            self.writeback_items_m[m] = np.concatenate([self.missing_items[k] for k in groups if m in k])
            setattr(self, f'missing_items_{m}', self.missing_items_m[m])

        self.impute_missing_features(config['missing_imputation'])

    def item_filter(self, m) :
        return torch.sparse.mm(self.adj.t(), F.tanh(self._get(m, 'g_filter_trans')(self.user_embedding.weight))) * self.num_inters[self.n_users:]

    @torch.no_grad()
    def generate_features(self, graphs) :
        g, _ = self.mge()
        filters = {m: self.item_filter(m) for m in self.mods}

        g_gen = {m: self._get(m, 'translator')(torch.concat([g[o] for o in self.others(m)], dim = 1)) for m in self.translate_order()}
        for _ in range(self.n_mm_layers) :
            for m in self.mods :
                g_gen[m] = torch.sparse.mm(graphs[m], g_gen[m])

        s_gen = {m: self._get(m, 'gen')(filters[m]) for m in self.mods}
        for _ in range(self.n_mm_layers) :
            for m in self.mods :
                s_gen[m] = torch.sparse.mm(graphs[m], s_gen[m])

        return {m: self._get(m, 'decoder')(self.perturb(torch.concat([g_gen[m], s_gen[m]], dim = 1))) for m in self.mods}

    def generate_missing_modal(self) :
        recon = self.generate_features(self.mm_adj)
        with torch.no_grad() :
            for m in self.mods :
                index = self.writeback_items_m[m]
                if self.new_items :
                    index = np.intersect1d(index, self.old_items_set)
                self._get(m, 'embedding').weight[index] = recon[m][index]

    def generate_missing_modal_infer(self) :
        assert self.new_items == 1, "Error"
        recon = self.generate_features(self.mm_adj_infer)
        with torch.no_grad() :
            for m in self.mods :
                index = np.intersect1d(self.writeback_items_m[m], self.new_items_set)
                self._get(m, 'embedding').weight[index] = recon[m][index]

    def init_mi_estimator(self) :
        if not self.sampler :
            return
        params = []
        for m in self.mods :
            for side in ('item', 'user') :
                est = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
                setattr(self, f'{side}_{MODALITY_NAMES[m]}_estimator', est)
                params += list(est.parameters())

        self.optimizer_club = torch.optim.Adam(params, lr = 1e-4)

    def item_estimator(self, m) :
        return getattr(self, f'item_{MODALITY_NAMES[m]}_estimator')

    def pre_epoch_processing(self) :
        if self.sampler :
            g, s = self.mge()
            for _ in range(5) :
                for m in self.mods :
                    self.item_estimator(m).train()

                item_rand_idx = torch.randperm(self.n_items)[:2048]

                loss_mi = 0.0
                for m in self.mods :
                    loss_mi += self.item_estimator(m).learning_loss(s[m][item_rand_idx], g[m][item_rand_idx])

                self.optimizer_club.zero_grad()
                loss_mi.backward(retain_graph = True)
                self.optimizer_club.step()

            for m in self.mods :
                self.item_estimator(m).eval()

        self.refresh_adj_counter += 1
        if self.missing_modal :
            self.generate_missing_modal()
            if self.refresh_adj_counter % 5 == 0 :
                self.update_adj()

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        _rows, _cols = zip(*data_dict.keys())
        A = sp.coo_matrix((list(data_dict.values()), (list(_rows), list(_cols))), shape=A.shape, dtype=np.float32)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return sumArr, torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def cge(self, user_emb, item_emb, adj) :
        ego_embeddings = torch.cat((user_emb, item_emb), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        user_embeddings, item_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        del ego_embeddings, side_embeddings

        return user_embeddings, item_embedding

    def mge(self) :
        g = {m: F.sigmoid(self.shared_encoder(self.act_g(self._get(m, 'encoder')(self._get(m, 'embedding').weight)))) for m in self.mods}
        s = {m: F.sigmoid(self._get(m, 'encoder_s')(self._get(m, 'embedding').weight)) for m in self.mods}
        return g, s

    def propagate(self, filters, feats, graphs) :
        items = {m: torch.einsum("ij, ij -> ij", filters[m], feats[m]) for m in self.mods}
        for _ in range(self.n_mm_layers) :
            for m in self.mods :
                items[m] = torch.sparse.mm(graphs[m], items[m])
        users = {m: torch.sparse.mm(self.adj, items[m]) * self.num_inters[:self.n_users] for m in self.mods}
        return users, items

    def fuse(self, cf_emb, g, s) :
        n_mods = len(self.mods)
        mm_emb = sum(g[m] for m in self.mods) / n_mods
        for m in self.mods :
            mm_emb = mm_emb + s[m]
        return cf_emb + mm_emb / (n_mods + 1)

    def calculate_loss(self, interaction) :
        users, pos_items, neg_items = interaction

        user_embeddings, item_embedding = self.cge(self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj)
        item_g, item_s = self.mge()

        all_items, _ = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        all_items = all_items.detach().cpu().numpy()

        observed = {m: np.setdiff1d(all_items, self.missing_items_m[m]) for m in self.mods}
        observed_all = np.setdiff1d(all_items, np.concatenate([self.missing_items_m[m] for m in self.mods]))

        loss_interModal = 0.0
        if self.interModal :
            for m1, m2 in self.mod_pairs :
                loss_interModal += self.InfoNCE_v2(item_g[m1][observed_all], item_g[m2][observed_all], temperature = self.interModalTemp)

        filters = {m: self.item_filter(m) for m in self.mods}
        user_g, item_g = self.propagate(filters, item_g, self.mm_adj)

        loss_additive = 0.0
        if self.missing_modal :
            for m in self.mods :
                loss_additive += F.mse_loss(item_s[m][observed[m]], self._get(m, 'gen')(self.perturb(filters[m]))[observed[m]])
            for m in self.translate_order() :
                source = torch.concat([item_g[o] for o in self.others(m)], dim = 1)
                loss_additive += F.mse_loss(item_g[m][observed_all], self._get(m, 'translator')(self.perturb(source))[observed_all])

        user_s, item_s = self.propagate(filters, item_s, self.mm_adj)

        loss_sampler = 0.0
        if self.sampler :
            for m in self.mods :
                loss_sampler += self.item_estimator(m)(item_s[m], item_g[m])

        if self.interModal :
            for m1, m2 in self.mod_pairs :
                loss_interModal += self.InfoNCE_v2(user_g[m1][users], user_g[m2][users], temperature = self.interModalTemp)

        user_g_sum = sum(user_g[m] for m in self.mods)
        item_g_sum = sum(item_g[m] for m in self.mods)

        loss_intraModal = self.InfoNCE_v2(user_embeddings[users], item_embedding[pos_items], temperature = self.intraModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_g_sum[users], item_g_sum[pos_items], temperature = self.interModalTemp)
        for m in self.mods :
            loss_intraModal += self.InfoNCE_v2(user_s[m][users], item_s[m][pos_items], temperature = self.intraModalTemp)

        loss_alignBM = self.InfoNCE_v2(item_embedding[pos_items], item_g_sum[pos_items], temperature = self.alignBMTemp)
        loss_alignBM += self.InfoNCE_v2(user_embeddings[users], user_g_sum[users], temperature = self.alignBMTemp)

        user_emb = self.fuse(user_embeddings, user_g, user_s)
        item_emb = self.fuse(item_embedding, item_g, item_s)
        loss_main_bpr = self.bpr_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items])

        loss_reg = self.reg_loss(user_embeddings[users], item_embedding[pos_items], item_embedding[neg_items]) * 1e-5
        for m in self.mods :
            loss_reg += self.reg_loss((item_g[m] + item_s[m])[pos_items]) * self.reg

        loss_recon = 0.0
        for m in self.mods :
            recon = self._get(m, 'decoder')(self.perturb(torch.concat([item_g[m], item_s[m]], dim = 1).detach()))
            loss_recon += F.mse_loss(recon, self._get(m, 'embedding').weight)

        loss_interModal *= self.interModal
        loss_intraModal *= self.intraModal
        loss_alignBM *= self.alignBM
        loss_recon *= self.recon
        loss_sampler *= self.sampler

        return loss_main_bpr + loss_reg + loss_recon + loss_sampler + loss_interModal + loss_intraModal + loss_alignBM + loss_additive * self.additive

    def perturb(self, x) :
        noise = torch.rand_like(x).to(self.device)
        x = x + torch.sign(x) * F.normalize(noise, dim = -1) * 0.1

        return x

    def InfoNCE_v2(self, view1, view2, temperature = 0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def forward(self) :
        pass

    def full_sort_predict(self, interaction) :
        users, _ = interaction

        user_embeddings, item_embedding = self.cge(self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj)
        item_g, item_s = self.mge()

        filters = {m: self.item_filter(m) for m in self.mods}
        graphs = self.mm_adj_infer if self.new_items else self.mm_adj
        user_g, item_g = self.propagate(filters, item_g, graphs)
        user_s, item_s = self.propagate(filters, item_s, graphs)

        user_emb = self.fuse(user_embeddings, user_g, user_s)
        item_emb = self.fuse(item_embedding, item_g, item_s)

        score = user_emb[users] @ item_emb.T
        return score

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss
