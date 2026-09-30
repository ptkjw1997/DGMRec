# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss

class DiceLoss(nn.Module):
    def __init__(self, beta=1, smooth = 1e-5):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, target):
        tp = torch.sum(inputs * target) 
        fp = torch.sum(inputs) - tp
        fn = torch.sum(target) - tp
        score = ((1 + self.beta ** 2) * tp + self.smooth) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        loss = 1 - torch.mean(score)
        return loss
    
def MSELoss(a, b, weight: float = 0.05):
    return F.mse_loss(a, b) * weight
