import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def divided_by_maximum(labels):
    return labels / torch.max(labels)

def sigmoid(labels):
    labels = np.array(labels)
    return 1 / (1 + np.exp(-labels))

def norm_function(norm, labels):
    return globals()[norm](labels)

def cosine_similarity_mse_loss(embedding1, embedding2, labels):
    #cosine similarity between the pairs of embeddings using torch.nn
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # calculate mean square error
    squared_difference = (labels - cos_sim) ** 2
    loss = squared_difference.mean()
    # loss = F.mse_loss(cos_sim, labels)

    return loss

def cosine_similarity_mse_norm(embedding1, embedding2, labels, norm):
    labels_norm = norm_function(norm, labels)
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels_norm - cos_sim) ** 2
    loss = squared_difference.mean()

    return loss

def cosent_loss(embedding1, embedding2, labels, tau=20.0):
    # Input preparation...
    labels = (labels[:, None] < labels[None, :]).float()

    # Normalization of Logits...
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    # Cosine Similarity Calculation...
    # The dot product of these pairs gives the cosine similarity, scaled by a factor of tau to control the sharpness of similarity scores...
    y_pred = torch.sum(embedding1 * embedding2, dim=1) * tau

    # Pairwise cosine similarity difference calculation...
    y_pred = y_pred[:, None] - y_pred[None, :]

    y_pred = (y_pred - (1 - labels) * 1e12).view(-1)

    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def categorical_crossentropy(y_true, y_pred):
    return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)

def in_batch_negative_loss(embedding1, embedding2, labels, tau=20.0, negative_weights=0.0):
    device = labels.device
    y_pred = torch.empty((2 * embedding1.shape[0], embedding1.shape[1]), device=device)
    y_pred[0::2] = embedding1
    y_pred[1::2] = embedding2
    y_true = labels.repeat_interleave(2).unsqueeze(1)

    def make_target_matrix(y_true):
        idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
        y_true = y_true.int()
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

        idxs_1 *= y_true.T
        idxs_1 += (y_true.T == 0).int() * -2

        idxs_2 *= y_true
        idxs_2 += (y_true == 0).int() * -1

        y_true = (idxs_1 == idxs_2).float()
        return y_true

    neg_mask = make_target_matrix(y_true == 0)

    y_true = make_target_matrix(y_true)

    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T
    similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
    similarities = similarities * tau

    if negative_weights > 0:
        similarities += neg_mask * negative_weights

    return categorical_crossentropy(y_true, similarities).mean()

def angle_loss(embedding1, embedding2, labels, tau=1.0):
    # Input preparation...
    labels = (labels[:, None] < labels[None, :]).float()

    # Chunking into real and imaginary parts...
    y_pred_re1, y_pred_im1 = torch.chunk(embedding1, 2, dim=1)
    y_pred_re2, y_pred_im2 = torch.chunk(embedding2, 2, dim=1)

    a = y_pred_re1
    b = y_pred_im1
    c = y_pred_re2
    d = y_pred_im2

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    y_pred = torch.concat((re, im), dim=1)
    y_pred = torch.abs(torch.sum(y_pred, dim=1)) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - labels) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def cosent_ibn_angle(embedding1, embedding2, labels, w_cosent=1, w_ibn=1, w_angle=1, tau_cosent=20.0, tau_ibn=20.0, tau_angle=1.0):
    return w_cosent * cosent_loss(embedding1, embedding2, labels, tau_cosent) + w_ibn * in_batch_negative_loss(embedding1, embedding2, labels, tau_ibn) + w_angle * angle_loss(embedding1, embedding2, labels, tau_angle)