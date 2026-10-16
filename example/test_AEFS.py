from ..UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import numpy as np
import torch
import scipy.io
from pathlib import Path

from ..MODELS.AEFS import AEFS
from ..UTILS import unsupervised_evaluation


def main():
    # --- Load data ---
    data_path = Path(__file__).resolve().parents[1] / "data" / "Prostate-GE.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)        # (n, d)
    y = mat["Y"][:, 0].astype(int)         # (n,)
    n_clusters = len(np.unique(y))


    
    # AEFS settings EXACTLY as in the paper
    # - hidden ∈ {128, 256, 512, 1024}
    # - sigmoid encoder, linear decoder (handled inside AEFS)

    hidden = 1024        # one valid paper value
    alpha = 1   # sparsity parameter (to be tuned in grid in experiments)
    beta = 0.001   # weight decay
    lr = 1e-3
    epochs = 100

    aefs = AEFS(
        hidden=hidden,
        alpha=alpha,
        beta=beta,
        lr=lr
    )

    aefs.fit(X, epochs=epochs)

    num_fea = 90


    idx, scores = aefs.select_top_features(num_fea)
    selected_features = X[:, idx]


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

    std_nmi = np.std(nmi_scores, ddof=1)
    std_acc = np.std(acc_scores, ddof=1)

    
    print("Hidden size:", hidden)
    print("Selected features:", num_fea)
    print("NMI:", np.mean(nmi_scores), "±", std_nmi)
    print("ACC:", np.mean(acc_scores), "±", std_acc)


if __name__ == "__main__":
    main()
