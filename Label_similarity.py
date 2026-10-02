from datapreprocess import get_sts_dataset, STSDataset
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import umap
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import ConfusionMatrixDisplay

def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)
    
def generate_distribution(model_name, pooling, loss_name, train_dataset, test_dataset_name, cosine_similarity, labels):
    '''
    generate distribution graph using matplot
    input: model, tokenizer, data, spearman, loop num. loss name, display_label = False(if true will generate true label distribution)
    '''
    directory = f"prediction_distribution/{model_name}_use_{pooling}/{loss_name}_on_{train_dataset}"
    cosine_similarity = to_np(cosine_similarity)
    labels = to_np(labels)
    if labels.max() > 1:
        labels = (labels - labels.min()) / (labels.max() - labels.min())
    model_name = model_name.replace("/", "_")

    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory (including parent dirs if needed)
        print(f"Created directory: {directory}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    lo = min(labels.min(), cosine_similarity.min())
    bins = np.linspace(lo, 1.0, 51)

    # light histograms in the background
    sns.histplot(labels, bins=bins, stat="density", color="steelblue", alpha=0.2, edgecolor=None)
    sns.histplot(cosine_similarity,    bins=bins, stat="density", color="darkorange", alpha=0.2, edgecolor=None)

    # smooth density curves on top
    sns.kdeplot(labels, fill=True, color="steelblue", alpha=0.35, linewidth=2,
                clip=(0, 1), bw_adjust=0.8, label="gold (rescaled 0–1)")
    sns.kdeplot(cosine_similarity, fill=True, color="darkorange", alpha=0.35, linewidth=2,
                clip=(lo, 1), bw_adjust=0.8, label="cosine similarity")

    plt.xlabel("similarity")
    plt.ylabel("density")
    plt.title(f"{test_dataset_name}: {loss_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{directory}/{test_dataset_name}_distribution.png", dpi=150)
    plt.close()
        

def generate_random_pair_distribution(cos_sim, loss_name, model_id, pooling, train_name, test_name):
      directory = f"Anistropy_Anlysis/random_embedding/{model_id}_use_{pooling}/{loss_name}_on_{train_name}"
      if not os.path.exists(directory):
          os.makedirs(directory)
          print(f"Created directory: {directory}")

      plt.hist(cos_sim, bins=100, density=True, alpha=0.7, label=model_id)
      plt.title(f"Anisotropy: {model_id}")
      plt.axvline(cos_sim.mean(), linestyle='--', label=f"mean={cos_sim.mean():.4f}")
      plt.xlabel("Cosine Similarity (random pairs)")
      plt.ylabel("Density")
      plt.xlim(-1, 1)
      plt.ylim(0, 10)
      plt.savefig(f'{directory}/{test_name}_random_Embedding.png')
      plt.close()

def plot_isoscore(n_comp=100, path="./Anistropy/chart/all.npz", out_dir="./Anistropy/chart"):
    data = np.load(path)
    os.makedirs(out_dir, exist_ok=True)

    evrs, dims = {}, {}
    for key in data.files:
        emb = data[key]
        if emb.shape[0] < 1000:
            print(f"warning: {key} has only {emb.shape[0]} embeddings; spectrum tail will be unreliable")
        n = min(n_comp, emb.shape[1], emb.shape[0] - 1)
        evrs[key] = PCA(n_components=n).fit(emb).explained_variance_ratio_
        dims[key] = emb.shape[1]

    plt.figure(figsize=(7, 5))
    for key, evr in evrs.items():
        plt.plot(range(1, len(evr) + 1), evr, marker='o', ms=3, label=key)
    plt.yscale('log')
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio (log)")
    plt.title("Scree (flatter = more isotropic)")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scree_all.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for key, evr in evrs.items():
        plt.plot(range(1, len(evr) + 1), np.cumsum(evr), lw=2, label=key)
    n_max = max(len(e) for e in evrs.values())
    for D in sorted(set(dims.values())):
        plt.plot(range(1, n_max + 1), np.arange(1, n_max + 1) / D, 'k--', alpha=0.5,
                 label=f"perfect isotropy (D={D})")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance")
    plt.title("Cumulative (closer to dashed = more isotropic)")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cumulative_all.png", dpi=150)
    plt.close()


def plot_clusters(embeddings, labels, pred, model_id, pooling, loss_name, train_name,
                  test_name, n_points=5000, seed=None):
    # output folder, e.g. cluster/bert-base-uncased_use_mean/Batch_JS_div_on_STS-B
    directory = f"cluster/{model_id}_use_{pooling}/{loss_name}_on_{train_name}"
    os.makedirs(directory, exist_ok=True)

    # L2-normalize so distances behave like cosine distance
    X = np.asarray(embeddings, dtype=np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    y = np.asarray(labels)

    # subsample so the plot stays readable and UMAP stays fast
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_points, len(X)), replace=False)
    X, y, pred = X[idx], y[idx], np.asarray(pred)[idx]

    # 2D projection for plotting
    Z = umap.UMAP(n_components=2, metric="cosine", init = "pca", random_state=seed).fit_transform(X)

    # one graph: each color = one k-means cluster
    colors = list(plt.get_cmap("tab10").colors)

    # pick the 5 largest k-means clusters
    ids, counts = np.unique(pred, return_counts=True)
    order = np.lexsort((ids, -counts))
    show = ids[np.argsort(-counts)[:5]]

    plt.figure(figsize=(8, 7))
    # background: all other points in light gray
    rest = ~np.isin(pred, show)
    plt.scatter(Z[rest, 0], Z[rest, 1], s=2, color="lightgray", alpha=0.4)

    # foreground: the 5 selected clusters
    for i, c in enumerate(show):
        mask = pred == c
        plt.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.8,
                    color=colors[i], label=str(c + 1))

    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=8, markerscale=4, ncol=2 if len(np.unique(pred)) > 25 else 1)
    plt.title(f"{test_name}: K-means clusters ({loss_name}, {model_id})")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()

    save_path = os.path.join(directory, f"{test_name}_clusters.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    #print(f"Saved cluster plot: {save_path}")
    plt.close()

def plot_SP_threshold(test_sims, test_labels, thr, model_id, pooling, loss_name,
                      train_name, test_name):
    directory = f"sentence_pair/{model_id}_use_{pooling}/{loss_name}_on_{train_name}"
    os.makedirs(directory, exist_ok=True)

    sims = np.asarray(test_sims, dtype=float)
    labels = np.asarray(test_labels).astype(int)

    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 201)
    lo, hi = sims.min(), sims.max()
    pad = 0.05 * (hi - lo)
    pos, neg = sims[labels == 1], sims[labels == 0]

    sns.histplot(pos, bins=bins, stat="density", color="#1baf7a", alpha=0.2, edgecolor=None)
    sns.histplot(neg, bins=bins, stat="density", color="#eb6834", alpha=0.2, edgecolor=None)
    sns.kdeplot(x=pos, fill=False, linewidth=2, color="#1baf7a",
                bw_adjust=0.5, clip=(-1, 1), label="positive")
    sns.kdeplot(x=neg, fill=False, linewidth=2, color="#eb6834",
                bw_adjust=0.5, clip=(-1, 1), label="negative")

    plt.axvline(thr, color="k", linestyle="--", label=f"τ = {thr:.2f} (from dev)")
    plt.xlim(max(-1, lo - pad), min(1, hi + pad))
    plt.xlabel("cosine similarity"); plt.ylabel("density")
    plt.title(f"{test_name}: {loss_name} ({model_id})")
    plt.legend()
    save_path = os.path.join(directory, f"{test_name}_SP_threshold.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_confusion(y_true, y_pred, model_id, pooling, loss_name, train_name,
                   test_name, class_names=None):
    directory = f"classification/{model_id}_use_{pooling}/{loss_name}_on_{train_name}"
    os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize="true", display_labels=class_names,
        cmap="Blues", values_format=".2f", colorbar=False, ax=ax)
    ax.set_title(f"{test_name}: {loss_name} ({model_id})")
    plt.xticks(rotation=45, ha="right")

    save_path = os.path.join(directory, f"{test_name}_confusion.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

