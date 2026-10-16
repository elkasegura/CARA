from ..UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
from ..MODELS.RFAE import cal  
from ..UTILS import unsupervised_evaluation

def normalize_zscore(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std
    return X_norm

def main():

    data_path = Path(__file__).resolve().parents[1] / "data" / "Prostate-GE.mat"
    mat = scipy.io.loadmat(data_path)


    X = mat["X"].astype(np.float32)      # (n, d)
    y = mat["Y"][:, 0].astype(int)       # (n,) SOLO para evaluación

    X = normalize_zscore(X)

    num_fea = 300
    
    n_clusters = len(np.unique(y))

    # -----------------------------
    feature_idx = cal(
        p_data_arr=X,
        p_label_arr_onehot=y,     # (realmente son labels, no onehot)
        datasetname="Prostate-GE",
        p_key_feture_number=num_fea,
        p_epochs_number=1000,
        p_batch_size_value=128,
        clf=False,
        p_seed=0,
        device=0  # CPU
    )

    feature_idx = np.array(feature_idx, dtype=int)
    X_selected = X[:, feature_idx]

    nmi_scores, acc_scores = [], []

    for _ in range(30):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=X_selected,
            n_clusters=n_clusters,
            y=y
        )
        nmi_scores.append(nmi)
        acc_scores.append(acc)

    print("Dataset: Prostate-GE")
    print("Selected features:", len(feature_idx))
    print("NMI:", np.mean(nmi_scores), "±", np.std(nmi_scores, ddof=1))
    print("ACC:", np.mean(acc_scores), "±", np.std(acc_scores, ddof=1))


if __name__ == "__main__":
    main()