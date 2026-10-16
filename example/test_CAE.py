from ..UTILS.utils import set_seed
set_seed(seed=42, deterministic=True)

import numpy as np
import scipy.io
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ..MODELS.CAE import ConcreteAutoencoderFeatureSelector
from ..UTILS import unsupervised_evaluation

def normalize_zscore(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std
    return X_norm


def main():
    # --- Load data ---
    data_path = Path(__file__).resolve().parents[1] / "data" / "Prostate-GE.mat"
    mat = scipy.io.loadmat(data_path)

    X = mat["X"].astype(np.float32)     # (n, d)
    y = mat["Y"][:, 0].astype(int)      # (n,)

    X = normalize_zscore(X)

    num_fea = 30
    n_clusters = len(np.unique(y))
    d = X.shape[1]                      # dimensión original a reconstruir

    # output_function mapea (n, K) -> (n, d)
    def output_function(selected_features):
        model = Sequential([
            Dense(3*num_fea//2, activation="relu", input_shape=(selected_features.shape[1],)),
            Dense(d, activation="linear")
        ])
        return model(selected_features)

    cae = ConcreteAutoencoderFeatureSelector(
        num_fea,
        output_function=output_function,
        num_epochs=300,
        batch_size=None,
        learning_rate=0.001,
        start_temp=10.0,
        min_temp=0.1,
        tryout_limit=5
    )

    cae.fit(X=X)

    idx = cae.get_support(indices=True).astype(int)
    selected_features = X[:, idx]

    # Evaluación (KMeans repeats)
    repeats = 20
    nmi_scores, acc_scores = [], []

    for _ in range(repeats):
        nmi, acc = unsupervised_evaluation.evaluation(
            X_selected=selected_features,
            n_clusters=n_clusters,
            y=y
        )
        nmi_scores.append(float(nmi))
        acc_scores.append(float(acc))
        
    print("Dataset: Prostate-GE")
    print("Selected features:", num_fea)
    print("idx:", idx)
    print("Unique idx:", len(np.unique(idx)), "/", len(idx))
    print("NMI:", np.mean(nmi_scores), "±", np.std(nmi_scores,))
    print("ACC:", np.mean(acc_scores), "±", np.std(acc_scores, ddof=1))


if __name__ == "__main__":
    main()