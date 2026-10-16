from ..UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import numpy as np
import torch
import scipy.io
from pathlib import Path
from ..MODELS.SMLAE import SMLAE
from ..UTILS import unsupervised_evaluation


def main():
    # --- Load data ---
    data_path = Path(__file__).resolve().parents[1] / "data" / "GLI-85.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)     # normalmente (n, d)
    y = mat["Y"][:, 0].astype(int)      # (n,)

    # DRAE espera (d, n)
    XT = X.T                             # (d, n)
    d, n = XT.shape
    n_clusters1 = len(np.unique(y))
    # A veces SMLAE usa esta fórmula para n_clusters
    n_clusters = np.sqrt(n/2)

    # --- Train SMLAE  ---
    smlae = SMLAE(n_clusters=n_clusters, d_hidden=50, n_neighbors=5, alpha=1e-6, beta=0, gamma=0.00001, omega=1e-3, rho1=0.1, rho2=1e-5)

    smlae.fit(XT, epochs=100)

    # --- Select top features ---
    num_fea = 10    

    # Feature ranking correcto 
    idx, scores = smlae.select_top_genes(num_fea)      # (d, k)

    selected_features = X[:, idx]                  # (n, num_fea)

    # --- Evaluate: repetir SOLO KMeans/evaluación ---
    repeats = 300
    nmi_total, acc_total = [], []

    for _ in range(repeats):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=selected_features,
            n_clusters=n_clusters1,   # usa el número real de clases/cluster
            y=y
        )
        nmi_total.append(nmi)
        acc_total.append(acc)

    nmi_mean = np.mean(nmi_total)
    acc_mean = np.mean(acc_total)

    std_nmi = np.std(nmi_total, ddof=1)
    std_acc = np.std(acc_total, ddof=1)     

    print("Dataset: Prostate-GE")
    print("Selected features:", num_fea)
    print("NMI:", nmi_mean, "±", std_nmi)
    print("ACC:", acc_mean, "±", std_acc)

if __name__ == "__main__":
    main()
