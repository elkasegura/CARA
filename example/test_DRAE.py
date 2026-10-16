from ..UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import numpy as np
import torch
import scipy.io
from pathlib import Path
from ..MODELS.DRAE import DRAE
from ..UTILS import unsupervised_evaluation
from ..UTILS.conf_models import _drae_config_for_num_features


def main():
    # --- Load data ---
    data_path = Path(__file__).resolve().parents[1] / "data" / "colon.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(float)     # normalmente (n, d)
    y = mat["Y"][:, 0].astype(int)      # (n,)

    # DRAE espera (d, n)
    XT = X.T                             # (d, n)             
    n_clusters = len(np.unique(y))

     #  Select top features 
    num_fea = 50   

    #  Train DRAE 

    cfg = _drae_config_for_num_features(dataset="colon", num_fea=num_fea)  # ajusta según el número de características que quieras seleccionar
    drae = DRAE(
        d_hidden=cfg["d_hidden"],
        n_clusters=n_clusters, 
        alpha=1e8, 
        beta=cfg["beta"], 
        gamma=cfg["gamma"], 
        zeta=cfg["zeta"], 
        lam=cfg["lam"])

    drae.fit(XT, epochs=100)

    #  Feature ranking correcto 
    # feature_ranking() espera W 2D (d,k), no un vector de índices
    idx, scores = drae.select_top_genes(num_fea)      # (d, k)

    selected_features = X[:, idx]                  # (n, num_fea)

    #  Evaluate: repetir SOLO KMeans/evaluación 
    repeats = 20
    nmi_total, acc_total = 0.0, 0.0
    nmi_scores, acc_scores = [], []

    for _ in range(repeats):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=selected_features,
            n_clusters=n_clusters,   # usa el número real de clases/cluster
            y=y
        )
        nmi_total += nmi
        acc_total += acc

        nmi_scores.append(nmi)
        acc_scores.append(acc)

    std_nmi = np.std(nmi_scores, ddof=1)
    std_acc = np.std(acc_scores, ddof=1)

    
    print("Selected features:", num_fea)
    print("NMI:", np.mean(nmi_scores), "±", std_nmi)
    print("ACC:", np.mean(acc_scores), "±", std_acc)

if __name__ == "__main__":
    main()
