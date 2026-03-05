import torch
import numpy as np
import torch.nn.functional as F


_loss_registry = {}

def register_loss(name):
    def decorator(func):
        _loss_registry[name] = func
        return func
    return decorator

def divided_by_maximum(labels):
    return labels / torch.max(labels)

def sigmoid(labels):
    labels = np.array(labels)
    return 1 / (1 + np.exp(-labels))

NORM_FUNCTIONS = {
    "divided_by_maximum": divided_by_maximum,
    "sigmoid": sigmoid,
    "none": lambda x: x,  # No normalization
    "minmax": lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8),
}

@register_loss("cosine_similarity_mse_norm")
def cosine_similarity_mse_norm(embedding1, embedding2, labels, norm):
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels_norm - cos_sim) ** 2
    loss = squared_difference.mean()
    #print("MSE",loss)
    return loss

@register_loss("cosine_similarity_mse_norm_adj")
def cosine_similarity_mse_norm(embedding1, embedding2, labels, norm):
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    # MSE loss...
    squared_difference = (labels_norm - cos_sim - 0.5) ** 2
    loss = squared_difference.mean()
    #print("MSE",loss)
    return loss

@register_loss("cosine_similarity_mse_mean")
def cosine_similarity_mse_mean(embedding1, embedding2, labels, norm, mean = 0):
    alpha = 0.5
    mean = torch.tensor(mean, device=embedding1.device, dtype=embedding1.dtype)
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    squared_difference = (labels_norm - (cos_sim - mean)) ** 2

    target_mean = labels_norm.mean().detach()
    mean_penalty = (cos_sim.mean() - target_mean).abs()

    loss = squared_difference.mean() + alpha * mean_penalty

    return loss

def covariance_matrix(x):
    x = x.unsqueeze(1)
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (x.size(0) - 1)
    return cov

@register_loss("cosine_similarity_mse_covar")
def cosine_similarity_mse_covar(embedding1, embedding2, labels, norm):
    alpha = 1
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    squared_difference = (labels_norm - cos_sim) ** 2


    covar_cos_sim = covariance_matrix(cos_sim)
    covar_labels_norm = covariance_matrix(labels_norm)
    covar_loss = torch.norm(covar_cos_sim - covar_labels_norm, p='fro') ** 2 / (4 * 1 * 1)
    mse_loss = squared_difference.mean()
    loss = mse_loss + alpha * covar_loss

    return loss

def kl_divergence(p, q, eps=1e-12):
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)))

def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

@register_loss("Batch_JS_div")
def Batch_JS_div(embedding1, embedding2, labels, norm):
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    cos_prob = (cos_sim + 1) / 2
    label_prob = (labels_norm + 1) / 2

    cos_prob = cos_prob / cos_prob.sum(dim=-1, keepdim=True)
    label_prob = label_prob / label_prob.sum(dim=-1, keepdim=True)
    
    js_loss = js_divergence(label_prob, cos_prob).mean()
    loss = js_loss

    return loss

@register_loss("Batch_KL_div")
def Batch_KL_div(embedding1, embedding2, labels, norm):
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    cos_prob = (cos_sim + 1) / 2
    label_prob = (labels_norm + 1) / 2

    cos_prob = cos_prob / cos_prob.sum(dim=-1, keepdim=True)
    label_prob = label_prob / label_prob.sum(dim=-1, keepdim=True)

    KL_loss = kl_divergence(label_prob, cos_prob).mean()
    loss = KL_loss
    return loss

def to_histogram(x, num_bins=20, value_range=(0, 1)):
    # x: [B, N]
    B, N = x.shape

    bin_edges = torch.linspace(
        value_range[0],
        value_range[1],
        steps=num_bins + 1,
        device=x.device
    )

    bin_idx = torch.bucketize(x, bin_edges) - 1
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    hist = torch.zeros(
        B, num_bins,
        device=x.device,
        dtype=x.dtype  # keep consistent
    )

    hist.scatter_add_(
        1,
        bin_idx,
        torch.ones_like(x, dtype=hist.dtype)
    )

    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-12)
    return hist

@register_loss("cosine_similarity_Wasserstein")
def cosine_similarity_Wasserstein(embedding1, embedding2, labels, norm):
    alpha = 1
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    squared_difference = (labels_norm - cos_sim) ** 2
    
    mse_loss = squared_difference.mean()

    cos_prob = (cos_sim + 1) / 2
    label_prob = (labels_norm + 1) / 2

    p = cos_prob / cos_prob.sum()
    q = label_prob / label_prob.sum()

    p_hist = to_histogram(cos_prob, num_bins=20, value_range=(0,1))
    q_hist = to_histogram(labels_norm, num_bins=20, value_range=(0,1))

    Fp = torch.cumsum(p, dim=0)
    Fq = torch.cumsum(q, dim=0)

    W_loss = torch.sum(torch.abs(Fp - Fq))
    loss = mse_loss + alpha * W_loss
    return loss

def rbf_kernel(x, y, sigmas = [0.1, 0.5, 1.0]):
  
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist_sq = x_norm + y_norm - 2 * torch.mm(x, y.t())
    return sum(torch.exp(-dist_sq / (2*s**2)) for s in sigmas)

def mmd(x, y):

    Kxx = rbf_kernel(x, x)
    Kyy = rbf_kernel(y, y)
    Kxy = rbf_kernel(x, y)

    m = x.size(0)
    n = y.size(0)

    mmd_sq = (Kxx.sum() - Kxx.trace()) / (m * (m - 1)) \
           + (Kyy.sum() - Kyy.trace()) / (n * (n - 1)) \
           - 2 * Kxy.mean()

    return torch.sqrt(mmd_sq + 1e-8)

@register_loss("cosine_similarity_mmd")
def cosine_similarity_mmd(embedding1, embedding2, labels, norm):
    alpha = 0.2
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)

    cos_sim = F.cosine_similarity(embedding1, embedding2)

    labels = labels.to(cos_sim.device).to(cos_sim.dtype)
    squared_difference = (labels_norm - cos_sim) ** 2
    
    mse_loss = squared_difference.mean()


    mmd_loss = mmd(cos_sim, labels)
    loss = mse_loss + alpha * mmd_loss
    return loss

def euclidean_distance(x, y, eps):
    return torch.sqrt(torch.sum((x - y) ** 2, dim=1) + eps)

def cosine_similarity(x, y, eps):
    return F.cosine_similarity(x, y, dim=1)

@register_loss("triplet")
def triplet(embedding1, embedding2, embedding3, margin, minimum, eps, distance):
    if distance == 'Eucliden':
      cal_dis = euclidean_distance
    elif distance == 'cos_sim':
      cal_dis = cosine_similarity
    dis_ap = cal_dis(embedding1, embedding2, eps)
    dis_an = cal_dis(embedding1, embedding3, eps)
    loss_per_sample = torch.clamp(dis_ap - dis_an + margin, minimum)
    return torch.mean(loss_per_sample)

@register_loss("cosent_loss")
def cosent_loss(embedding1, embedding2, labels, tau=20.0):
    labels = (labels[:, None] < labels[None, :]).float()
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    y_pred = torch.sum(embedding1 * embedding2, dim=1) * tau

    y_pred = y_pred[:, None] - y_pred[None, :]

    y_pred = (y_pred - (1 - labels) * 1e12).view(-1)

    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def categorical_crossentropy(y_true, y_pred):
    return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)

@register_loss("in_batch_negative_loss")
# Modify from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py#L166
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

@register_loss("angle_loss")
def angle_loss(embedding1, embedding2, labels, tau=1.0):
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

@register_loss("cosent_ibn_angle")
def cosent_ibn_angle(embedding1, embedding2, labels, w_cosent=1, w_ibn=1, w_angle=1, tau_cosent=20.0, tau_ibn=20.0, tau_angle=1.0):
    return w_cosent * cosent_loss(embedding1, embedding2, labels, tau_cosent) + w_ibn * in_batch_negative_loss(embedding1, embedding2, labels, tau_ibn) + w_angle * angle_loss(embedding1, embedding2, labels, tau_angle)

@register_loss("ibn_JSD")
def ibn_JSD(embedding1, embedding2, labels, w_ibn = 1, w_JSD = 1, tau_ibn=20.0):
    return w_ibn * in_batch_negative_loss(embedding1, embedding2, labels, tau_ibn) + w_JSD * Batch_JS_div(embedding1, embedding2, labels, 'divided_by_maximum')

def get_loss(name, **kwargs):
    """Get loss function by name with parameters"""
    if name not in _loss_registry:
        raise ValueError(f"Unknown loss: {name}. Available: {list(_loss_registry.keys())}")
    
    def loss_wrapper(*args):
        return _loss_registry[name](*args, **kwargs)
    
    return loss_wrapper