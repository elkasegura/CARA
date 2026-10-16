from ..UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import numpy as np
import torch
import scipy.io
from pathlib import Path
from ..MODELS.DGA import DGA
from ..UTILS import unsupervised_evaluation

def normalize_zscore(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std + 1e-8
    return X_norm


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "Prostate-GE.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)      # (n, d)
    y = mat["Y"][:, 0].astype(int)       # (n,)

    n_samples, d_features = X.shape
    n_clusters = len(np.unique(y))

    #X = normalize_zscore(X)

    dga = DGA(
        hidden=50,
        gamma=0.01,
        lam=0.01,
        lr=1e-3
    )

    # Paper Sec. 4.1: B = N
    dga.fit(X, epochs=50, B=n_samples)

    num_fea = 30
    idx, scores = dga.select_top_features(num_fea)  # sort g_k (paper)
    selected_features = X[:, idx]                   # (n, num_fea)

    repeats = 20
    nmi_scores, acc_scores = [], []

    for _ in range(repeats):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=selected_features,
            n_clusters=n_clusters,
            y=y
        )
        nmi_scores.append(nmi)
        acc_scores.append(acc)

    print("Dataset: Prostate-GE")
    print("Selected features:", num_fea)
    print("NMI:", np.mean(nmi_scores), "±", np.std(nmi_scores, ddof=1))
    print("ACC:", np.mean(acc_scores), "±", np.std(acc_scores, ddof=1))


if __name__ == "__main__":
    main()
