# coding: utf-8
r"""DGMRec: Disentangled Generative Multimodal Recommendation.

Item modality features are disentangled into a general (modality-shared)
part and a specific (modality-unique) part. General features of missing
modalities are translated from the observed modality, specific features are
generated from collaborative signals, and the reconstructed raw features are
written back into the feature tables so that the modality kNN graphs can be
refreshed during training.
"""
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_neighbourhood
from utils.utils import compute_normalized_laplacian as compute_normalized_laplacian_dense
from utils.mi_estimator import CLUBSample


def compute_normalized_laplacian(adj):
    """Symmetrically normalize an adjacency matrix: D^-1/2 * A * D^-1/2.

    Accepts a dense or sparse torch tensor and returns a torch sparse tensor.
    """
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
    """Chunked kNN cosine graph as a torch sparse tensor.

    Equivalent to build_sim -> build_knn_neighbourhood (-> row/col masking of
    `mask_idx` with self-loops) but never materialises the dense n x n
    similarity matrix; required for large item sets (e.g. Electronics, where
    dense n^2 exceeds the CUDA nonzero INT_MAX limit)."""
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


# datasets larger than this use the chunked sparse kNN builder
SPARSE_KNN_THRESHOLD = 30000


class DGMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DGMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = 3  # user-item GCN depth, fixed in all released configs
        self.n_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']

        # Collaborative filtering embeddings
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

        self.complete_items = np.arange(self.n_items)
        self.missing_modal = config['missing_modal']
        self.missing_imputation = config['missing_imputation']
        if config['missing_modal'] :
            self.preprocess_missing_modal(config)

        # Modality kNN graphs (training graph + inference graph for new items)
        #
        # 2-modality (Amazon) datasets keep the sparse graphs of the original
        # amazon tree; 3-modality datasets (a_feat present, e.g. tiktok) keep
        # the dense graphs of the original tiktok tree so that the numerical
        # behavior of both historical trees is reproduced exactly.
        if self.a_feat is not None :
            # ---- 3-modality historical path (dense adjacency) ----
            if self.v_feat is not None :
                self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze = False).to(self.device)

                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                if self.missing_modal :
                    image_adj[self.missing_items_v, :] = image_adj[:, self.missing_items_v] = 0.0
                    image_adj[self.missing_items_v, self.missing_items_v] = 1.0
                self.image_adj = compute_normalized_laplacian_dense(image_adj)
                self.image_adj_infer = self.image_adj.clone()

                if self.new_items :
                    image_adj = build_sim(self.image_embedding.weight.detach())
                    image_adj[self.new_items_set, :] = image_adj[:, self.new_items_set] = 0.0
                    image_adj[self.new_items_set, self.new_items_set] = 1.0
                    image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                    image_adj = compute_normalized_laplacian_dense(image_adj)
                    self.image_adj = image_adj.cuda()

            if self.t_feat is not None :
                self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze = False).to(self.device)

                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                if self.missing_modal :
                    text_adj[self.missing_items_t, :] = text_adj[:, self.missing_items_t] = 0.0
                    text_adj[self.missing_items_t, self.missing_items_t] = 1.0
                self.text_adj = compute_normalized_laplacian_dense(text_adj)
                self.text_adj_infer = self.text_adj.clone()

                if self.new_items :
                    text_adj = build_sim(self.text_embedding.weight.detach())
                    text_adj[self.new_items_set, :] = text_adj[:, self.new_items_set] = 0.0
                    text_adj[self.new_items_set, self.new_items_set] = 1.0
                    text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                    text_adj = compute_normalized_laplacian_dense(text_adj)
                    self.text_adj = text_adj.cuda()

            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze = False).to(self.device)

            audio_adj = build_sim(self.audio_embedding.weight.detach())
            audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
            if self.missing_modal :
                audio_adj[self.missing_items_a, :] = audio_adj[:, self.missing_items_a] = 0.0
                audio_adj[self.missing_items_a, self.missing_items_a] = 1.0
            self.audio_adj = compute_normalized_laplacian_dense(audio_adj)
            self.audio_adj_infer = self.audio_adj.clone()

            if self.new_items :
                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj[self.new_items_set, :] = audio_adj[:, self.new_items_set] = 0.0
                audio_adj[self.new_items_set, self.new_items_set] = 1.0
                audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
                audio_adj = compute_normalized_laplacian_dense(audio_adj)
                self.audio_adj = audio_adj.cuda()
        else :
            # ---- 2-modality path (sparse adjacency) ----
            if self.v_feat is not None :
                self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze = False).to(self.device)

                if self.n_items > SPARSE_KNN_THRESHOLD:
                    image_adj = build_knn_graph_sparse(
                        self.image_embedding.weight.detach(), self.knn_k,
                        mask_idx=self.missing_items_v if self.missing_modal else None)
                    self.image_adj = compute_normalized_laplacian(image_adj)
                else:
                    image_adj = build_sim(self.image_embedding.weight.detach())
                    image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                    if self.missing_modal :
                        image_adj[self.missing_items_v, :] = image_adj[:, self.missing_items_v] = 0.0
                        image_adj[self.missing_items_v, self.missing_items_v] = 1.0
                    self.image_adj = compute_normalized_laplacian(image_adj).to_sparse_coo()

                if self.new_items :
                    image_adj = build_sim(self.image_embedding.weight.detach())
                    image_adj[self.new_items_set, :] = image_adj[:, self.new_items_set] = 0.0
                    image_adj[self.new_items_set, self.new_items_set] = 1.0
                    image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                    image_adj = compute_normalized_laplacian(image_adj).to_sparse_coo()
                    self.image_adj_infer = self.image_adj.clone()
                    self.image_adj = image_adj.cuda()
                else :
                    self.image_adj_infer = self.image_adj
                del image_adj

            if self.t_feat is not None :
                self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze = False).to(self.device)

                if self.n_items > SPARSE_KNN_THRESHOLD:
                    text_adj = build_knn_graph_sparse(
                        self.text_embedding.weight.detach(), self.knn_k,
                        mask_idx=self.missing_items_t if self.missing_modal else None)
                    self.text_adj = compute_normalized_laplacian(text_adj)
                else:
                    text_adj = build_sim(self.text_embedding.weight.detach())
                    text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                    if self.missing_modal :
                        text_adj[self.missing_items_t, :] = text_adj[:, self.missing_items_t] = 0.0
                        text_adj[self.missing_items_t, self.missing_items_t] = 1.0
                    self.text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo()

                if self.new_items :
                    text_adj = build_sim(self.text_embedding.weight.detach())
                    text_adj[self.new_items_set, :] = text_adj[:, self.new_items_set] = 0.0
                    text_adj[self.new_items_set, self.new_items_set] = 1.0
                    text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                    text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo()
                    self.text_adj_infer = self.text_adj.clone()
                    self.text_adj = text_adj.cuda()
                else :
                    self.text_adj_infer = self.text_adj
                del text_adj

        torch.cuda.empty_cache()

        # Modality encoders: general (modality-shared) and specific branches
        self.image_encoder  = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder   = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        if self.a_feat is not None :
            self.audio_encoder = nn.Linear(self.a_feat.shape[1], self.embedding_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder.weight); nn.init.xavier_uniform_(self.text_encoder.weight)
        if self.a_feat is not None :
            nn.init.xavier_uniform_(self.audio_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        self.image_encoder_s  = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder_s   = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        if self.a_feat is not None :
            self.audio_encoder_s = nn.Linear(self.a_feat.shape[1], self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder_s.weight); nn.init.xavier_uniform_(self.text_encoder_s.weight)
        if self.a_feat is not None :
            nn.init.xavier_uniform_(self.audio_encoder_s.weight)

        # Per-user modality preference tables (kept in the parameter set / state_dict)
        self.user_image_prefer = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_text_prefer  = nn.Embedding(self.n_users, self.embedding_dim)
        if self.a_feat is not None :
            self.user_audio_prefer = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image_prefer.weight); nn.init.xavier_uniform_(self.user_text_prefer.weight)
        if self.a_feat is not None :
            nn.init.xavier_uniform_(self.user_audio_prefer.weight)

        # User-preference filters aggregated onto items
        self.image_g_filter_trans = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        self.text_g_filter_trans  = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        if self.a_feat is not None :
            self.audio_g_filter_trans = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        nn.init.xavier_uniform_(self.image_g_filter_trans.weight); nn.init.xavier_uniform_(self.text_g_filter_trans.weight)
        if self.a_feat is not None :
            nn.init.xavier_uniform_(self.audio_g_filter_trans.weight)

        self.image_s_filter_trans = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        self.text_s_filter_trans  = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        if self.a_feat is not None :
            self.audio_s_filter_trans = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        nn.init.xavier_uniform_(self.image_s_filter_trans.weight); nn.init.xavier_uniform_(self.text_s_filter_trans.weight)
        if self.a_feat is not None :
            nn.init.xavier_uniform_(self.audio_s_filter_trans.weight)

        # Decoders back to the raw feature spaces
        self.image_decoder = nn.Linear(self.embedding_dim * 2, self.v_feat.shape[1]).to(self.device)
        self.text_decoder  = nn.Linear(self.embedding_dim * 2, self.t_feat.shape[1]).to(self.device)
        if self.a_feat is not None :
            self.audio_decoder = nn.Linear(self.embedding_dim * 2, self.a_feat.shape[1]).to(self.device)
        nn.init.xavier_uniform_(self.image_decoder.weight); nn.init.xavier_uniform_(self.text_decoder.weight)
        if self.a_feat is not None :
            nn.init.xavier_uniform_(self.audio_decoder.weight)

        self.act_g = nn.Tanh()

        self.refresh_adj_counter = 0

        # Generators for the specific features of missing modalities
        self.image_gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.image_gen.apply(self.init_weight)

        self.text_gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.text_gen.apply(self.init_weight)

        if self.a_feat is not None :
            self.audio_gen = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.audio_gen.apply(self.init_weight)

        # Cross-modal translators for the general features. In the 3-modality
        # setting each modality is translated from the concatenation of the
        # two other modalities (input dim doubled), as in the historical
        # tiktok tree.
        trans_in_dim = self.embedding_dim * 2 if self.a_feat is not None else self.embedding_dim
        self.image2text = nn.Sequential(
            nn.Linear(trans_in_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.image2text.apply(self.init_weight)

        self.text2image = nn.Sequential(
            nn.Linear(trans_in_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.text2image.apply(self.init_weight)

        if self.a_feat is not None :
            self.text2audio = nn.Sequential(
                nn.Linear(trans_in_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.text2audio.apply(self.init_weight)

        # Hyper-parameters (see configs/best/DGMRec_*.json)
        self.additive = config['additive']
        self.avg_lambda = config['avg_lambda']
        self.infer_adj_update = config['infer_adj_update']

        self.interModal, self.interModalTemp, self.interModalDist = config['interModal'], config['interModalTemp'], config['interModalDist']
        self.intraModal, self.intraModalTemp = config['intraModal'], config['intraModalTemp']
        self.alignBM, self.alignBMTemp = config['alignBM'], config['alignBMTemp']
        self.recon = config['recon']
        self.reg = config['reg']
        self.sampler = config['sampler']

    def init_weight(self, layer) :
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def init_mi_estimator(self) :
        # The CLUB estimators only feed the `sampler`-weighted loss term.
        # When sampler == 0 (historical tiktok objective) they are skipped
        # entirely so that no random state is consumed by their creation,
        # training or forward passes.
        if not self.sampler :
            return
        self.item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
        self.user_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
        self.item_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()
        self.user_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).cuda()

        params = list(self.item_image_estimator.parameters()) + list(self.user_image_estimator.parameters()) + \
                    list(self.item_text_estimator.parameters()) + list(self.user_text_estimator.parameters())

        self.optimizer_club = torch.optim.Adam(params, lr = 1e-4)

    def pre_epoch_processing(self) :
        # Train the CLUB mutual-information estimators on the current
        # general/specific item representations (skipped when sampler == 0,
        # see init_mi_estimator).
        if self.sampler :
            if self.a_feat is not None :
                item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s = self.mge()
            else :
                item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

            for _ in range(5) :
                self.item_image_estimator.train(); self.item_text_estimator.train()

                item_rand_idx = torch.randperm(self.n_items)[:2048]

                loss_mi = 0.0
                loss_mi += self.item_image_estimator.learning_loss(item_image_s[item_rand_idx], item_image_g[item_rand_idx])
                loss_mi += self.item_text_estimator.learning_loss(item_text_s[item_rand_idx], item_text_g[item_rand_idx])

                self.optimizer_club.zero_grad()
                loss_mi.backward(retain_graph = True)
                self.optimizer_club.step()

            self.item_image_estimator.eval(); self.item_text_estimator.eval()

        # Regenerate the missing raw features and periodically refresh the
        # modality kNN graphs from them.
        self.refresh_adj_counter += 1
        if self.missing_modal :
            self.generate_missing_modal()
            if self.refresh_adj_counter % 5 == 0 :
                self.update_adj()

    def generate_missing_modal(self) :
        if self.a_feat is not None :
            # 3-modality historical path (tiktok tree)
            item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s = self.mge()
            item_image_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
            item_text_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
            item_audio_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.audio_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

            with torch.no_grad() :
                # General part: translate each modality from the two others.
                item_it, item_ta, item_ai = torch.concat([item_image_g, item_text_g], dim = 1), torch.concat([item_text_g, item_audio_g], dim = 1), torch.concat([item_audio_g, item_image_g], dim = 1)
                item_text_g, item_image_g, item_audio_g = self.image2text(item_ai), self.text2image(item_ta), self.text2audio(item_it)
                for _ in range(self.n_mm_layers) :
                    item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
                    item_text_g  = torch.sparse.mm(self.text_adj, item_text_g)
                    item_audio_g  = torch.sparse.mm(self.audio_adj, item_audio_g)

                # Specific part: generate from the collaborative filter signal.
                item_image_s, item_text_s = self.image_gen(item_image_filter), self.text_gen(item_text_filter)
                item_audio_s = self.audio_gen(item_audio_filter)
                for _ in range(self.n_mm_layers) :
                    item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
                    item_text_s  = torch.sparse.mm(self.text_adj, item_text_s)
                    item_audio_s  = torch.sparse.mm(self.audio_adj, item_audio_s)

                item_image_recon = self.image_decoder(self.perturb(torch.concat([item_image_g, item_image_s], dim = 1)))
                item_text_recon = self.text_decoder(self.perturb(torch.concat([item_text_g, item_text_s], dim = 1)))
                item_audio_recon = self.audio_decoder(self.perturb(torch.concat([item_audio_g, item_audio_s], dim = 1)))

            with torch.no_grad() :
                if self.new_items :
                    t_index = np.intersect1d(self.missing_items['t'], self.old_items_set)
                    v_index = np.intersect1d(self.missing_items['v'], self.old_items_set)
                    a_index = np.intersect1d(self.missing_items['a'], self.old_items_set)
                else :
                    t_index = self.missing_items['t']
                    v_index = self.missing_items['v']
                    a_index = self.missing_items['a']
                self.text_embedding.weight[t_index] = item_text_recon[t_index]
                self.image_embedding.weight[v_index] = item_image_recon[v_index]
                self.audio_embedding.weight[a_index] = item_audio_recon[a_index]
            return

        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
        item_image_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        with torch.no_grad() :
            # General part: translate from the observed modality.
            item_text_g, item_image_g = self.image2text(item_image_g), self.text2image(item_text_g)
            for _ in range(self.n_mm_layers) :
                item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
                item_text_g  = torch.sparse.mm(self.text_adj, item_text_g)

            # Specific part: generate from the collaborative filter signal.
            item_image_s, item_text_s = self.image_gen(item_image_filter), self.text_gen(item_text_filter)
            for _ in range(self.n_mm_layers) :
                item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
                item_text_s  = torch.sparse.mm(self.text_adj, item_text_s)

            item_image_recon = self.image_decoder(self.perturb(torch.concat([item_image_g, item_image_s], dim = 1)))
            item_text_recon = self.text_decoder(self.perturb(torch.concat([item_text_g, item_text_s], dim = 1)))

        with torch.no_grad() :
            if self.new_items :
                t_index = np.intersect1d(self.missing_items['t'], self.old_items_set)
                v_index = np.intersect1d(self.missing_items['v'], self.old_items_set)
            else :
                t_index = self.missing_items['t']
                v_index = self.missing_items['v']
            self.text_embedding.weight[t_index] = item_text_recon[t_index]
            self.image_embedding.weight[v_index] = item_image_recon[v_index]

    def generate_missing_modal_infer(self) :
        if self.a_feat is not None :
            # 3-modality historical path (tiktok tree): uses the inference
            # graphs; index selection follows the historical code verbatim.
            item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s = self.mge()
            item_image_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
            item_text_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
            item_audio_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.audio_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

            with torch.no_grad() :
                item_it, item_ta, item_ai = torch.concat([item_image_g, item_text_g], dim = 1), torch.concat([item_text_g, item_audio_g], dim = 1), torch.concat([item_audio_g, item_image_g], dim = 1)
                item_text_g, item_image_g, item_audio_g = self.image2text(item_ai), self.text2image(item_ta), self.text2audio(item_it)
                for _ in range(self.n_mm_layers) :
                    item_image_g = torch.sparse.mm(self.image_adj_infer, item_image_g)
                    item_text_g  = torch.sparse.mm(self.text_adj_infer, item_text_g)
                    item_audio_g  = torch.sparse.mm(self.audio_adj_infer, item_audio_g)

                item_image_s, item_text_s = self.image_gen(item_image_filter), self.text_gen(item_text_filter)
                item_audio_s = self.audio_gen(item_audio_filter)
                for _ in range(self.n_mm_layers) :
                    item_image_s = torch.sparse.mm(self.image_adj_infer, item_image_s)
                    item_text_s  = torch.sparse.mm(self.text_adj_infer, item_text_s)
                    item_audio_s  = torch.sparse.mm(self.audio_adj_infer, item_audio_s)

                item_image_recon = self.image_decoder(self.perturb(torch.concat([item_image_g, item_image_s], dim = 1)))
                item_text_recon = self.text_decoder(self.perturb(torch.concat([item_text_g, item_text_s], dim = 1)))
                item_audio_recon = self.audio_decoder(self.perturb(torch.concat([item_audio_g, item_audio_s], dim = 1)))

            with torch.no_grad() :
                if self.new_items :
                    t_index = np.intersect1d(self.missing_items['t'], self.old_items_set)
                    v_index = np.intersect1d(self.missing_items['v'], self.old_items_set)
                    a_index = np.intersect1d(self.missing_items['a'], self.old_items_set)
                else :
                    t_index = self.missing_items['t']
                    v_index = self.missing_items['v']
                    a_index = self.missing_items['a']
                self.text_embedding.weight[t_index] = item_text_recon[t_index]
                self.image_embedding.weight[v_index] = item_image_recon[v_index]
                self.audio_embedding.weight[a_index] = item_audio_recon[a_index]
            return

        # New-item variant of generate_missing_modal: uses the inference
        # graphs and only fills features of new items.
        assert self.new_items == 1, "Error"
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
        item_image_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        with torch.no_grad() :
            item_text_g, item_image_g = self.image2text(item_image_g), self.text2image(item_text_g)
            for _ in range(self.n_mm_layers) :
                item_image_g = torch.sparse.mm(self.image_adj_infer, item_image_g)
                item_text_g  = torch.sparse.mm(self.text_adj_infer, item_text_g)

            item_image_s, item_text_s = self.image_gen(item_image_filter), self.text_gen(item_text_filter)
            for _ in range(self.n_mm_layers) :
                item_image_s = torch.sparse.mm(self.image_adj_infer, item_image_s)
                item_text_s  = torch.sparse.mm(self.text_adj_infer, item_text_s)

            item_image_recon = self.image_decoder(self.perturb(torch.concat([item_image_g, item_image_s], dim = 1)))
            item_text_recon = self.text_decoder(self.perturb(torch.concat([item_text_g, item_text_s], dim = 1)))

        with torch.no_grad() :
            if self.new_items :
                t_index = np.intersect1d(self.missing_items['t'], self.new_items_set)
                v_index = np.intersect1d(self.missing_items['v'], self.new_items_set)
            else :
                t_index = self.missing_items['t']
                v_index = self.missing_items['v']
            self.text_embedding.weight[t_index] = item_text_recon[t_index]
            self.image_embedding.weight[v_index] = item_image_recon[v_index]

    def _to_scipy(self, tensor):
        """torch (sparse or dense) tensor -> scipy CSR matrix."""
        if tensor.is_sparse:
            tensor = tensor.detach().cpu()
            indices = tensor._indices().numpy()
            values = tensor._values().numpy()
            shape = tensor.shape
            return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
        else:
            return sp.csr_matrix(tensor.detach().cpu().numpy())

    def _to_tensor(self, matrix):
        """scipy sparse matrix -> torch sparse tensor on the model device."""
        coo = matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(self.device)

    def update_adj(self):
        """Blend the rows of the modality kNN graphs that belong to items with
        a generated modality with the graph rebuilt from the current features:
        new_row = avg_lambda * rebuilt_row + (1 - avg_lambda) * old_row.

        Row updates are done in scipy (CPU) to support row slicing on sparse
        matrices and to avoid materialising dense n x n graphs on the GPU.
        The 3-modality (dense-graph) path blends rows in place as in the
        historical tiktok tree."""
        if self.a_feat is not None :
            # 3-modality historical path (dense graphs, raw per-modality
            # missing index sets)
            with torch.no_grad() :
                if self.new_items :
                    t_index = np.intersect1d(self.missing_items['t'], self.old_items_set)
                    v_index = np.intersect1d(self.missing_items['v'], self.old_items_set)
                    a_index = np.intersect1d(self.missing_items['a'], self.old_items_set)
                else :
                    t_index = self.missing_items['t']
                    v_index = self.missing_items['v']
                    a_index = self.missing_items['a']

            with torch.no_grad() :
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian_dense(image_adj)

                self.image_adj[v_index] = image_adj[v_index] * self.avg_lambda + self.image_adj[v_index] * (1 - self.avg_lambda)

                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian_dense(text_adj)

                self.text_adj[t_index] = text_adj[t_index] * self.avg_lambda + self.text_adj[t_index] * (1 - self.avg_lambda)

                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
                audio_adj = compute_normalized_laplacian_dense(audio_adj)

                self.audio_adj[a_index] = audio_adj[a_index] * self.avg_lambda + self.audio_adj[a_index] * (1 - self.avg_lambda)
            return

        torch.cuda.empty_cache()

        if self.new_items:
            t_index = np.intersect1d(self.missing_items_t, self.old_items_set)
            v_index = np.intersect1d(self.missing_items_v, self.old_items_set)
        else:
            t_index = self.missing_items_t
            v_index = self.missing_items_v

        # -------------------- image graph --------------------
        if isinstance(self.image_adj, torch.Tensor):
            self_image_adj_scipy = self._to_scipy(self.image_adj)
        else:
            self_image_adj_scipy = self.image_adj

        with torch.no_grad():
            if self.n_items > SPARSE_KNN_THRESHOLD:
                image_adj = build_knn_graph_sparse(self.image_embedding.weight.detach(), self.knn_k)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            image_adj_tensor = compute_normalized_laplacian(image_adj).cpu()

        image_adj_scipy = self._to_scipy(image_adj_tensor)

        update_rows = image_adj_scipy[v_index].multiply(self.avg_lambda) + \
                      self_image_adj_scipy[v_index].multiply(1 - self.avg_lambda)
        self_image_adj_scipy[v_index] = update_rows
        self.image_adj = self._to_tensor(self_image_adj_scipy)

        del image_adj, image_adj_tensor, image_adj_scipy, self_image_adj_scipy, update_rows
        torch.cuda.empty_cache()

        # -------------------- text graph --------------------
        if isinstance(self.text_adj, torch.Tensor):
            self_text_adj_scipy = self._to_scipy(self.text_adj)
        else:
            self_text_adj_scipy = self.text_adj

        with torch.no_grad():
            if self.n_items > SPARSE_KNN_THRESHOLD:
                text_adj = build_knn_graph_sparse(self.text_embedding.weight.detach(), self.knn_k)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            text_adj_tensor = compute_normalized_laplacian(text_adj).cpu()

        text_adj_scipy = self._to_scipy(text_adj_tensor)

        update_rows_t = text_adj_scipy[t_index].multiply(self.avg_lambda) + \
                        self_text_adj_scipy[t_index].multiply(1 - self.avg_lambda)
        self_text_adj_scipy[t_index] = update_rows_t
        self.text_adj = self._to_tensor(self_text_adj_scipy)

        del text_adj, text_adj_tensor, text_adj_scipy, self_text_adj_scipy, update_rows_t
        torch.cuda.empty_cache()

    def update_adj_infer(self) :
        """Same row-blending as update_adj, applied to the inference graphs
        (new-item evaluation)."""
        if self.a_feat is not None :
            # 3-modality historical path (dense graphs, raw per-modality
            # missing index sets)
            with torch.no_grad() :
                if self.new_items :
                    t_index = np.intersect1d(self.missing_items['t'], self.old_items_set)
                    v_index = np.intersect1d(self.missing_items['v'], self.old_items_set)
                    a_index = np.intersect1d(self.missing_items['a'], self.old_items_set)
                else :
                    t_index = self.missing_items['t']
                    v_index = self.missing_items['v']
                    a_index = self.missing_items['a']

            with torch.no_grad() :
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian_dense(image_adj)

                self.image_adj_infer[v_index] = image_adj[v_index] * self.avg_lambda + self.image_adj_infer[v_index] * (1 - self.avg_lambda)

                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian_dense(text_adj)

                self.text_adj_infer[t_index] = text_adj[t_index] * self.avg_lambda + self.text_adj_infer[t_index] * (1 - self.avg_lambda)

                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
                audio_adj = compute_normalized_laplacian_dense(audio_adj)

                self.audio_adj_infer[a_index] = audio_adj[a_index] * self.avg_lambda + self.audio_adj_infer[a_index] * (1 - self.avg_lambda)
            return

        assert self.new_items == 1, "Error"
        with torch.no_grad() :
            if self.new_items :
                t_index = np.intersect1d(self.missing_items_t, self.new_items_set)
                v_index = np.intersect1d(self.missing_items_v, self.new_items_set)
            else :
                t_index = self.missing_items_t
                v_index = self.missing_items_v

        with torch.no_grad() :
            self.image_adj_infer = self.image_adj_infer.cpu().to_dense()
            torch.cuda.empty_cache()

            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            # local laplacian returns sparse; densify for row-wise blending below
            image_adj = compute_normalized_laplacian(image_adj).cpu().to_dense()

            self.image_adj_infer[v_index] = image_adj[v_index] * self.avg_lambda + self.image_adj_infer[v_index] * (1 - self.avg_lambda)
            self.image_adj_infer = self.image_adj_infer.to_sparse_coo()
            del image_adj

            self.text_adj_infer = self.text_adj_infer.cpu().to_dense()
            torch.cuda.empty_cache()

            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            # local laplacian returns sparse; densify for row-wise blending below
            text_adj = compute_normalized_laplacian(text_adj).cpu().to_dense()

            self.text_adj_infer[t_index] = text_adj[t_index] * self.avg_lambda + self.text_adj_infer[t_index] * (1 - self.avg_lambda)
            self.text_adj_infer = self.text_adj_infer.to_sparse_coo()
            del text_adj

            torch.cuda.empty_cache()
            self.image_adj_infer = self.image_adj_infer.to(self.device)
            self.text_adj_infer = self.text_adj_infer.to(self.device)

    def preprocess_missing_modal(self, config) :
        """Load the fixed missing-item masks and impute the missing raw
        features (0: zeros, 1: mean of the observed features)."""
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        self.missing_items = np.load(os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy"), allow_pickle = True).item()

        if 'a' in self.missing_items :
            # 3-modality masks (keys: all/t/v/a/tv/ta/va)
            self.missing_items_t = np.concatenate((self.missing_items['all'], self.missing_items['t'],
                                                    self.missing_items['tv'], self.missing_items['ta']))
            self.missing_items_v = np.concatenate((self.missing_items['all'], self.missing_items['v'],
                                                    self.missing_items['tv'], self.missing_items['va']))
            self.missing_items_a = np.concatenate((self.missing_items['all'], self.missing_items['a'],
                                                    self.missing_items['ta'], self.missing_items['va']))

            self.complete_items = np.setdiff1d(np.arange(self.n_items), np.union1d(np.union1d(self.missing_items_v, self.missing_items_t), self.missing_items_a))
        else :
            # 2-modality masks (keys: all/t/v)
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
            assert False, f"Missing Imputation Must be 0 or 1, Not {config['missing_imputation']}"
        self.missing_imputation = config['missing_imputation']

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
        # scipy>=1.12 removed dok_matrix._update; build a COO matrix directly instead
        _rows, _cols = zip(*data_dict.keys())
        A = sp.coo_matrix((list(data_dict.values()), (list(_rows), list(_cols))), shape=A.shape, dtype=np.float32)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
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

        return sumArr, torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def cge(self, user_emb, item_emb, adj) :
        # Collaborative graph embedding (LightGCN-style propagation)
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
        # Modality graph embedding: general (shared encoder) and specific parts
        item_image_g = F.sigmoid(self.shared_encoder(self.act_g(self.image_encoder(self.image_embedding.weight))))
        item_text_g  = F.sigmoid(self.shared_encoder(self.act_g(self.text_encoder(self.text_embedding.weight))))
        if self.a_feat is not None :
            item_audio_g = F.sigmoid(self.shared_encoder(self.act_g(self.audio_encoder(self.audio_embedding.weight))))

        item_image_s = F.sigmoid(self.image_encoder_s(self.image_embedding.weight))
        item_text_s  = F.sigmoid(self.text_encoder_s(self.text_embedding.weight))
        if self.a_feat is not None :
            item_audio_s = F.sigmoid(self.audio_encoder_s(self.audio_embedding.weight))

        if self.a_feat is not None :
            return item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s
        return item_image_g, item_text_g, item_image_s, item_text_s

    def calculate_loss(self, interaction) :
        if self.a_feat is not None :
            return self.calculate_loss_3mod(interaction)

        users, pos_items, neg_items = interaction

        user_embeddings, item_embedding = self.cge(self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj)
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        all_items, _ = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)

        # in-batch items whose text / image / both modalities are observed
        if self.missing_modal :
            t_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_t)
            v_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_v)
            tv_index = np.setdiff1d(all_items.detach().cpu().numpy(), np.union1d(self.missing_items_t, self.missing_items_v))
        else :
            t_index = all_items.detach().cpu().numpy()
            v_index = all_items.detach().cpu().numpy()
            tv_index = all_items.detach().cpu().numpy()

        # Inter-modal alignment of the general representations
        # (skipped entirely when the weight is 0)
        if self.interModal :
            loss_interModal = self.InfoNCE_v2(item_image_g[tv_index], item_text_g[tv_index], temperature = self.interModalTemp)
        else :
            loss_interModal = 0.0

        # General embeddings, filtered by user preferences and propagated on
        # the modality kNN graphs
        item_image_g_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_g_filter  = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        item_image_g = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_g)
        item_text_g  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_g)

        for _ in range(self.n_mm_layers) :
            item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
            item_text_g  = torch.sparse.mm(self.text_adj, item_text_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g  = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]

        # Generation losses: teach the generators/translators to reproduce
        # the specific and general representations of observed modalities
        loss_additive = 0.0
        if self.missing_modal :
            loss_additive += F.mse_loss(item_image_s[v_index], self.image_gen(self.perturb(item_image_g_filter))[v_index])
            loss_additive += F.mse_loss(item_text_s[t_index], self.text_gen(self.perturb(item_text_g_filter))[t_index])

            loss_additive += F.mse_loss(item_text_g[tv_index], self.image2text(self.perturb(item_image_g))[tv_index])
            loss_additive += F.mse_loss(item_image_g[tv_index], self.text2image(self.perturb(item_text_g))[tv_index])

        # Specific embeddings
        item_image_s = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_s)
        item_text_s  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_s)

        for _ in range(self.n_mm_layers) :
            item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
            item_text_s  = torch.sparse.mm(self.text_adj, item_text_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s  = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]

        image_embs = torch.concat([user_image_g + user_image_s, item_image_g + item_image_s], dim = 0)
        text_embs = torch.concat([user_text_g + user_text_s, item_text_g + item_text_s], dim = 0)

        user_image_final, item_image_final = torch.split(image_embs, [self.n_users, self.n_items], dim=0)
        user_text_final, item_text_final = torch.split(text_embs, [self.n_users, self.n_items], dim=0)

        # Disentanglement: CLUB upper bound between specific and general parts
        # (the estimators consume random state, so they are skipped entirely
        # when sampler == 0; see init_mi_estimator)
        loss_sampler = 0.0
        if self.sampler :
            loss_sampler += self.item_image_estimator(item_image_s, item_image_g)
            loss_sampler += self.item_text_estimator(item_text_s, item_text_g)

        if self.interModal :
            loss_interModal += self.InfoNCE_v2(user_image_g[users], user_text_g[users], temperature = self.interModalTemp)

        # User-item alignment (per representation space)
        loss_intraModal = self.InfoNCE_v2(user_embeddings[users], item_embedding[pos_items], temperature = self.intraModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_image_g[users] + user_text_g[users], item_image_g[pos_items] + item_text_g[pos_items], temperature = self.interModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_image_s[users], item_image_s[pos_items], temperature = self.intraModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_text_s[users], item_text_s[pos_items], temperature = self.intraModalTemp)

        # Behavior-modality alignment
        loss_alignBM = self.InfoNCE_v2(item_embedding[pos_items], item_image_g[pos_items] + item_text_g[pos_items], temperature = self.alignBMTemp)
        loss_alignBM += self.InfoNCE_v2(user_embeddings[users], user_image_g[users] + user_text_g[users], temperature = self.alignBMTemp)

        # BPR loss on the fused embeddings
        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3

        user_emb, pos_item_emb, neg_item_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]

        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        loss_reg = self.reg_loss(user_embeddings[users], item_embedding[pos_items], item_embedding[neg_items]) * 1e-5
        loss_reg += self.reg_loss(item_image_final[pos_items]) * self.reg
        loss_reg += self.reg_loss(item_text_final[pos_items]) * self.reg

        # Reconstruction of the raw modality features from [general, specific]
        image_final, text_final = torch.concat([item_image_g, item_image_s], dim = 1), torch.concat([item_text_g, item_text_s], dim = 1)
        item_image_recon = self.image_decoder(self.perturb(image_final.detach()))
        item_text_recon  = self.text_decoder(self.perturb(text_final.detach()))

        loss_recon = F.mse_loss(item_image_recon, self.image_embedding.weight)
        loss_recon += F.mse_loss(item_text_recon, self.text_embedding.weight)

        loss_interModal *= self.interModal
        loss_intraModal *= self.intraModal
        loss_alignBM *= self.alignBM
        loss_recon *= self.recon
        loss_sampler *= self.sampler

        del item_image_g, item_text_g, item_image_s, item_text_s

        return loss_main_bpr + loss_reg + loss_recon + loss_sampler + loss_interModal + loss_intraModal + loss_alignBM + loss_additive * self.additive

    def calculate_loss_3mod(self, interaction) :
        """Full DGMRec objective in the 3-modality (image/text/audio) setting.

        With the released tiktok configuration (configs/best/DGMRec_tiktok.json:
        sampler = 0, interModal = 0, interModalTemp = 0.4, intraModalTemp = 0.2,
        intraModal = alignBM = 0.01, alignBMTemp = 0.4, recon = reg = 0.1) this
        is bitwise identical to the historical tiktok tree's objective: the
        zero-weighted CLUB (sampler) and inter-modal InfoNCE terms are skipped
        so that they neither contribute to the loss nor consume random state.
        """
        users, pos_items, neg_items = interaction

        user_embeddings, item_embedding = self.cge(self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj)
        item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s = self.mge()

        all_items, _ = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)

        # in-batch items whose modalities are observed
        if self.missing_modal :
            t_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_t)
            v_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_v)
            a_index = np.setdiff1d(all_items.detach().cpu().numpy(), self.missing_items_a)

            tva_index = np.setdiff1d(all_items.detach().cpu().numpy(), np.union1d(np.union1d(self.missing_items_t, self.missing_items_v), self.missing_items_a))
        else :
            t_index = all_items.detach().cpu().numpy()
            v_index = all_items.detach().cpu().numpy()
            a_index = all_items.detach().cpu().numpy()

            tva_index = all_items.detach().cpu().numpy()

        # Inter-modal alignment of the general representations (pairwise);
        # weight 0 in the released tiktok configuration -> skipped
        if self.interModal :
            loss_interModal = self.InfoNCE_v2(item_image_g[tva_index], item_text_g[tva_index], temperature = self.interModalTemp)
            loss_interModal += self.InfoNCE_v2(item_text_g[tva_index], item_audio_g[tva_index], temperature = self.interModalTemp)
            loss_interModal += self.InfoNCE_v2(item_audio_g[tva_index], item_image_g[tva_index], temperature = self.interModalTemp)
        else :
            loss_interModal = 0.0

        # General embeddings, filtered by user preferences and propagated on
        # the modality kNN graphs
        item_image_g_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_g_filter  = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_audio_g_filter  = torch.sparse.mm(self.adj.t(), F.tanh(self.audio_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        item_image_g = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_g)
        item_text_g  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_g)
        item_audio_g  = torch.einsum("ij, ij -> ij", item_audio_g_filter, item_audio_g)

        for _ in range(self.n_mm_layers) :
            item_image_g = torch.sparse.mm(self.image_adj, item_image_g)
            item_text_g  = torch.sparse.mm(self.text_adj, item_text_g)
            item_audio_g  = torch.sparse.mm(self.audio_adj, item_audio_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g  = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]
        user_audio_g  = torch.sparse.mm(self.adj, item_audio_g) * self.num_inters[:self.n_users]

        # Generation losses: teach the generators/translators to reproduce
        # the specific and general representations of observed modalities
        loss_additive = 0.0
        if self.missing_modal :
            loss_additive += F.mse_loss(item_image_s[v_index], self.image_gen(self.perturb(item_image_g_filter))[v_index])
            loss_additive += F.mse_loss(item_text_s[t_index], self.text_gen(self.perturb(item_text_g_filter))[t_index])
            loss_additive += F.mse_loss(item_audio_s[a_index], self.audio_gen(self.perturb(item_audio_g_filter))[a_index])

            item_it, item_ta, item_ai = torch.concat([item_image_g, item_text_g], dim = 1), torch.concat([item_text_g, item_audio_g], dim = 1), torch.concat([item_audio_g, item_image_g], dim = 1)
            loss_additive += F.mse_loss(item_text_g[tva_index], self.image2text(self.perturb(item_ai))[tva_index])
            loss_additive += F.mse_loss(item_image_g[tva_index], self.text2image(self.perturb(item_ta))[tva_index])
            loss_additive += F.mse_loss(item_audio_g[tva_index], self.text2audio(self.perturb(item_it))[tva_index])

        # Specific embeddings
        item_image_s = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_s)
        item_text_s  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_s)
        item_audio_s  = torch.einsum("ij, ij -> ij", item_audio_g_filter, item_audio_s)
        for _ in range(self.n_mm_layers) :
            item_image_s = torch.sparse.mm(self.image_adj, item_image_s)
            item_text_s  = torch.sparse.mm(self.text_adj, item_text_s)
            item_audio_s  = torch.sparse.mm(self.audio_adj, item_audio_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s  = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]
        user_audio_s  = torch.sparse.mm(self.adj, item_audio_s) * self.num_inters[:self.n_users]

        image_embs = torch.concat([user_image_g + user_image_s, item_image_g + item_image_s], dim = 0)
        text_embs = torch.concat([user_text_g + user_text_s, item_text_g + item_text_s], dim = 0)
        audio_embs = torch.concat([user_audio_g + user_audio_s, item_audio_g + item_audio_s], dim = 0)

        user_image_final, item_image_final = torch.split(image_embs, [self.n_users, self.n_items], dim=0)
        user_text_final, item_text_final = torch.split(text_embs, [self.n_users, self.n_items], dim=0)
        user_audio_final, item_audio_final = torch.split(audio_embs, [self.n_users, self.n_items], dim=0)

        # Disentanglement: CLUB upper bound between specific and general parts
        # (weight 0 in the released tiktok configuration -> skipped, so no
        # random state is consumed; see init_mi_estimator)
        loss_sampler = 0.0
        if self.sampler :
            loss_sampler += self.item_image_estimator(item_image_s, item_image_g)
            loss_sampler += self.item_text_estimator(item_text_s, item_text_g)

        if self.interModal :
            loss_interModal += self.InfoNCE_v2(user_image_g[users], user_text_g[users], temperature = self.interModalTemp)

        # User-item alignment (per representation space)
        loss_intraModal = self.InfoNCE_v2(user_embeddings[users], item_embedding[pos_items], temperature = self.intraModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_image_g[users] + user_text_g[users] + user_audio_g[users], item_image_g[pos_items] + item_text_g[pos_items] + item_audio_g[pos_items], temperature = self.interModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_image_s[users], item_image_s[pos_items], temperature = self.intraModalTemp)
        loss_intraModal += self.InfoNCE_v2(user_text_s[users], item_text_s[pos_items], temperature = self.intraModalTemp)
        # NOTE: historical 3-modality behavior preserved - the audio term
        # aligns the general (not specific) audio representations
        loss_intraModal += self.InfoNCE_v2(user_audio_g[users], item_audio_g[pos_items], temperature = self.intraModalTemp)

        # Behavior-modality alignment
        loss_alignBM = self.InfoNCE_v2(item_embedding[pos_items], item_image_g[pos_items] + item_text_g[pos_items] + item_audio_g[pos_items], temperature = self.alignBMTemp)
        loss_alignBM += self.InfoNCE_v2(user_embeddings[users], user_image_g[users] + user_text_g[users] + user_audio_g[users], temperature = self.alignBMTemp)

        # BPR loss on the fused embeddings
        user_emb = user_embeddings + ((user_image_g + user_text_g + user_audio_g) / 3 + user_image_s + user_text_s + user_audio_s) / 4
        item_emb = item_embedding + ((item_image_g + item_text_g + item_audio_g) / 3 + item_image_s + item_text_s + item_audio_s) / 4

        user_emb, pos_item_emb, neg_item_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]

        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        loss_reg = self.reg_loss(user_embeddings[users], item_embedding[pos_items], item_embedding[neg_items]) * 1e-5
        loss_reg += self.reg_loss(item_image_final[pos_items]) * self.reg
        loss_reg += self.reg_loss(item_text_final[pos_items]) * self.reg
        loss_reg += self.reg_loss(item_audio_final[pos_items]) * self.reg

        # Reconstruction of the raw modality features from [general, specific]
        image_final, text_final = torch.concat([item_image_g, item_image_s], dim = 1), torch.concat([item_text_g, item_text_s], dim = 1)
        audio_final = torch.concat([item_audio_g, item_audio_s], dim = 1)
        item_image_recon = self.image_decoder(self.perturb(image_final.detach()))
        item_text_recon  = self.text_decoder(self.perturb(text_final.detach()))
        item_audio_recon  = self.audio_decoder(self.perturb(audio_final.detach()))

        loss_recon = F.mse_loss(item_image_recon, self.image_embedding.weight)
        loss_recon += F.mse_loss(item_text_recon, self.text_embedding.weight)
        loss_recon += F.mse_loss(item_audio_recon, self.audio_embedding.weight)

        loss_interModal *= self.interModal
        loss_intraModal *= self.intraModal
        loss_alignBM *= self.alignBM
        loss_recon *= self.recon
        loss_sampler *= self.sampler

        del item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s

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
        if self.a_feat is not None :
            return self.full_sort_predict_3mod(interaction)

        users, _ = interaction

        user_embeddings, item_embedding = self.cge(self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj)
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        item_image_g_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_g_filter  = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        item_image_g = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_g)
        item_text_g  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_g)

        for _ in range(self.n_mm_layers) :
            item_image_g = torch.sparse.mm(self.image_adj_infer, item_image_g)
            item_text_g  = torch.sparse.mm(self.text_adj_infer, item_text_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g  = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]

        item_image_s = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_s)
        item_text_s  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_s)

        for _ in range(self.n_mm_layers) :
            item_image_s = torch.sparse.mm(self.image_adj_infer, item_image_s)
            item_text_s  = torch.sparse.mm(self.text_adj_infer, item_text_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s  = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]

        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3

        user_emb, pos_item_emb = user_emb[users], item_emb

        score = user_emb @ pos_item_emb.T
        return score

    def full_sort_predict_3mod(self, interaction) :
        users, _ = interaction

        user_embeddings, item_embedding = self.cge(self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj)
        item_image_g, item_text_g, item_audio_g, item_image_s, item_text_s, item_audio_s = self.mge()

        item_image_g_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_g_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_audio_g_filter  = torch.sparse.mm(self.adj.t(), F.tanh(self.audio_g_filter_trans(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        item_image_g = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_g)
        item_text_g  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_g)
        item_audio_g  = torch.einsum("ij, ij -> ij", item_audio_g_filter, item_audio_g)
        for _ in range(self.n_mm_layers) :
            item_image_g = torch.sparse.mm(self.image_adj_infer, item_image_g)
            item_text_g  = torch.sparse.mm(self.text_adj_infer, item_text_g)
            item_audio_g  = torch.sparse.mm(self.audio_adj_infer, item_audio_g)
        user_image_g = torch.sparse.mm(self.adj, item_image_g) * self.num_inters[:self.n_users]
        user_text_g  = torch.sparse.mm(self.adj, item_text_g) * self.num_inters[:self.n_users]
        user_audio_g  = torch.sparse.mm(self.adj, item_audio_g) * self.num_inters[:self.n_users]

        item_image_s = torch.einsum("ij, ij -> ij", item_image_g_filter, item_image_s)
        item_text_s  = torch.einsum("ij, ij -> ij", item_text_g_filter, item_text_s)
        item_audio_s  = torch.einsum("ij, ij -> ij", item_audio_g_filter, item_audio_s)
        for _ in range(self.n_mm_layers) :
            item_image_s = torch.sparse.mm(self.image_adj_infer, item_image_s)
            item_text_s  = torch.sparse.mm(self.text_adj_infer, item_text_s)
            item_audio_s  = torch.sparse.mm(self.audio_adj_infer, item_audio_s)
        user_image_s = torch.sparse.mm(self.adj, item_image_s) * self.num_inters[:self.n_users]
        user_text_s  = torch.sparse.mm(self.adj, item_text_s) * self.num_inters[:self.n_users]
        user_audio_s  = torch.sparse.mm(self.adj, item_audio_s) * self.num_inters[:self.n_users]

        user_emb = user_embeddings + ((user_image_g + user_text_g + user_audio_g) / 3 + user_image_s + user_text_s + user_audio_s) / 4
        item_emb = item_embedding + ((item_image_g + item_text_g + item_audio_g) / 3 + item_image_s + item_text_s + item_audio_s) / 4

        user_emb, pos_item_emb = user_emb[users], item_emb

        score = user_emb @ pos_item_emb.T
        return score

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss
