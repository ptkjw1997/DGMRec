# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        self.batch_size = config['train_batch_size']
        self.device = config['device']

        self.v_feat, self.t_feat, self.a_feat = None, None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if config['audio_feature_file']:
                a_feat_file_path = os.path.join(dataset_path, config['audio_feature_file'])
                if os.path.isfile(a_feat_file_path):
                    self.a_feat = torch.from_numpy(np.load(a_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                        self.device)

            assert self.v_feat is not None or self.t_feat is not None or self.a_feat is not None, 'Features all NONE'
            self.train_inter = dataloader.inter_matrix(form='csr').astype(np.float32)

    def impute_missing_features(self, mode):
        for m in ('v', 't', 'a'):
            feat = getattr(self, f'{m}_feat')
            if feat is None:
                continue
            missing = np.asarray(getattr(self, f'missing_items_{m}'), dtype=np.int64)
            observed = np.setdiff1d(self.old_items_set, missing)
            if mode == 0:
                feat[missing] = 0.0
            elif mode == 1:
                feat[missing] = feat[observed].mean(dim=0)
            elif mode == 2:
                feat[missing] = self._nn_injection(feat, missing, observed)
            else:
                raise ValueError(f'missing_imputation must be 0, 1 or 2, not {mode}')

    def _nn_injection(self, feat, missing, observed):
        inter = self.train_inter.tocsc(copy=True)
        inter.data[:] = 1.0
        co = (inter[:, missing].T @ inter[:, observed]).tocsr()
        co.data[:] = 1.0
        counts = np.asarray(co.sum(axis=1)).ravel()
        neighbor_sum = torch.from_numpy(co @ feat[observed].cpu().numpy()).to(feat.device)
        imputed = feat[observed].mean(dim=0).expand(len(missing), -1).clone()
        has = torch.from_numpy(counts > 0).to(feat.device)
        counts = torch.from_numpy(counts).to(feat.device).unsqueeze(1)
        imputed[has] = neighbor_sum[has] / counts[has]
        return imputed
