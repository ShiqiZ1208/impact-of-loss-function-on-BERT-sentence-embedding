import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import pearsonr

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
    "zero_one": lambda x: x/5,
    "neg_one": lambda x:((x/5)*2-1)
}

@register_loss("cosine_similarity_mse_norm")
def cosine_similarity_mse_norm(embedding1, embedding2, labels, norm):
    #print(norm)
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)
    #print(labels_norm)
    # Calculating the cosine similarity between the pairs of embeddings...
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    # MSE loss...
    squared_difference = (labels_norm - cos_sim) ** 2
    loss = squared_difference.mean()

    #print("MSE",loss)
    return loss

def scale_labels_to_target(labels, target_mean, target_variance, eps=1e-8):
    target_std = target_variance ** 0.5
    labels_mean = labels.mean()
    labels_std = labels.std()
    
    z = (labels - labels_mean) / (labels_std + eps)
    
    scaled_labels = z * target_std + target_mean
    return scaled_labels

@register_loss('mean_adjust_MSE')
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

def kl_divergence(p, q, eps=1e-12):
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)))

def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

@register_loss("Batch_JS_div")
def Batch_JS_div(embedding1, embedding2, labels, norm, tau):
    #print(max(labels)-min(labels))
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)

    cos_sim = F.cosine_similarity(embedding1, embedding2)
    cos_sim = norm_func(cos_sim)
    cos_prob = F.softmax(cos_sim/tau, dim=0)
    label_prob = F.softmax(labels_norm/tau, dim=0)

    
    js_loss = js_divergence(label_prob, cos_prob)
    loss = js_loss

    return loss


@register_loss("softmax_MSE")
def softmax_MSE(embedding1, embedding2, labels, norm):
    labels_norm = NORM_FUNCTIONS[norm](labels)

    cos_sim = F.cosine_similarity(embedding1, embedding2)
    cos_sim = NORM_FUNCTIONS[norm](cos_sim)
    eps = 1e-8

    cos_prob = F.softmax(cos_sim, dim=0)
    label_prob = F.softmax(labels_norm, dim=0)

    M = 0.5 * (cos_prob + label_prob)
    squared_difference = (label_prob - cos_prob) ** 2
    loss = (1/8) * (squared_difference / (M + eps)).mean()
    return loss


@register_loss("Batch_KL_div")
def Batch_KL_div(embedding1, embedding2, labels, norm, tau =1.0 ):
    #norm_func = NORM_FUNCTIONS[norm]
    #labels_norm = norm_func(labels)
    eps = 1e-8
    labels_norm = labels
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + eps)
    labels_norm = (labels_norm - labels_norm.min()) / (labels_norm.max() - labels_norm.min() + eps)
    cos_prob = F.softmax(cos_sim/tau, dim=0)
    label_prob = F.softmax(labels_norm/tau, dim=0)

    KL_loss = kl_divergence(label_prob, cos_prob)
    loss = KL_loss
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
        #print(y_true[:8, :8])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

        idxs_1 *= y_true.T
        idxs_1 += (y_true.T == 0).int() * -2

        idxs_2 *= y_true
        idxs_2 += (y_true == 0).int() * -1

        y_true = (idxs_1 == idxs_2).float()
        #print(y_true[:8, :8])
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
def angle_loss(embedding1, embedding2, labels, tau=20.0):
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
    return w_cosent * cosent_loss(embedding1, embedding2, labels, tau_cosent) + w_ibn * in_batch_negative_loss_equivalent(embedding1, embedding2, labels, tau_ibn) + w_angle * angle_loss(embedding1, embedding2, labels, tau_angle)



def build_label_matrix(labels, N):
    label_matrix = torch.zeros(2*N, 2*N, device=labels.device)
    #label_matrix = torch.full((2*N, 2*N), float('-inf'), device=labels.device)
    label_matrix[:N, N:] = torch.diag(labels)   # top right block
    label_matrix[N:, :N] = torch.diag(labels)   # bottom left block
    return label_matrix

def js_divergence_matrix(p, q, eps=1e-12):
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    return (0.5 * kl_pm + 0.5 * kl_qm).mean()

@register_loss("contrastive_JSD")
def contrastive_JSD_loss(embedding1, embedding2, labels, norm = 'divided_by_maximum', tau=20.0):
    norm_func = NORM_FUNCTIONS[norm]
    labels_norm = norm_func(labels)

    N = embedding1.shape[0]

    all_embeddings = torch.cat([embedding1, embedding2], dim=0)  # (2N, D)
    sim_matrix = F.cosine_similarity(
        all_embeddings.unsqueeze(1),   # (2N, 1, D)
        all_embeddings.unsqueeze(0),   # (1, 2N, D)
        dim=-1
    )  # (2N, 2N)

    mask = torch.eye(2*N, dtype=torch.bool, device=embedding1.device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    label_matrix = build_label_matrix(labels_norm, N)
    label_matrix = label_matrix.masked_fill(mask, float('-inf'))

    neighbor_dist = F.softmax(sim_matrix, dim=-1)

    label_dist = F.softmax(label_matrix, dim=-1)

    # per-row JSD (don't average yet)
    p = torch.clamp(label_dist, min=1e-12)
    q = torch.clamp(neighbor_dist, min=1e-12)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    jsd_per_row = 0.5 * kl_pm + 0.5 * kl_qm  # shape (2N,)

    # graded weight: repeat labels_norm for both halves of the 2N rows
    row_weight = torch.cat([labels_norm, labels_norm], dim=0)  # shape (2N,)

    loss = (jsd_per_row * row_weight).mean()

    return loss


def ibn_to_jsd_format(embedding1, embedding2, labels, threshold=1.0):
    N = embedding1.shape[0]
    device = labels.device

    label_matrix = torch.zeros((2*N, 2*N), device=device, dtype=torch.float32)

    pos_mask = (labels >= threshold).float()
    label_matrix[:N, N:] = torch.diag(pos_mask)
    label_matrix[N:, :N] = torch.diag(pos_mask)

    label_matrix.fill_diagonal_(0)
    return label_matrix

@register_loss('ibn')
def in_batch_negative_loss_equivalent(embedding1, embedding2, labels, tau=20, threshold = 1.0):
    #print(labels[0])
    device = labels.device
    N = embedding1.shape[0]

    y_pred = torch.cat([embedding1, embedding2], dim=0)
    y_pred = F.normalize(y_pred, dim=1, p=2)


    cos_sim = y_pred @ y_pred.T
    cos_sim = cos_sim - torch.eye(2*N, device=device) * 1e12
    cos_sim = cos_sim * tau


    label_matrix = ibn_to_jsd_format(embedding1, embedding2, labels, threshold)
    label_matrix = (label_matrix != 0).float()

    loss = -(F.log_softmax(cos_sim, dim=1) * label_matrix).sum(dim=1).mean()

    return loss

@register_loss("cosent_batch_jsd")
def cosent_batch_jsd(embedding1, embedding2, labels, norm='divided_by_maximum',
                      tau_cosent=20.0, w_jsd=0.3):
    cosent_l = cosent_loss(embedding1, embedding2, labels, tau_cosent)
    jsd_l = Batch_JS_div(embedding1, embedding2, labels, norm)

    scale = cosent_l.detach() / (jsd_l.detach() + 1e-8)

    return cosent_l + w_jsd * scale * jsd_l


@register_loss("pearson_loss")
def pearson_loss(embedding1, embedding2, labels, eps=1e-8):
    cos_sim = F.cosine_similarity(embedding1, embedding2)

    cos_sim_centered = cos_sim - cos_sim.mean()
    labels_centered = labels - labels.mean()

    numerator = (cos_sim_centered * labels_centered).sum()
    denominator = torch.sqrt((cos_sim_centered ** 2).sum()) * torch.sqrt((labels_centered ** 2).sum())

    pearson_score = numerator / (denominator + eps)
    loss = 1 - pearson_score

    return loss

def get_loss(name, **kwargs):
    """Get loss function by name with parameters"""
    if name not in _loss_registry:
        raise ValueError(f"Unknown loss: {name}. Available: {list(_loss_registry.keys())}")
    
    def loss_wrapper(*args):
        return _loss_registry[name](*args, **kwargs)
    
    return loss_wrapper