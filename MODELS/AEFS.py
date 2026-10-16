# ====================================================================================
# MODELO AEFS: Autoencoder Inspired Unsupervised Feature Selection
# Authors: Kai Han, Yunhe Wang, Chao Zhang, Chao Li, Chao Xu
# Conference: IEEE ICASSP 2018
# Pages: 2941--2945
# ====================================================================================

# Parameters
# hidden: tamaño hidden del autoencoder
# alpha: weight of L2,1 regularizer (α)
# beta: weight of weight decay regularizer (β)
# lr: learning rate del optimizador SGD

#Grid {
#       "hidden": [128,256,512,1024]
#        "alpha_grid": [0.001,0.01,0.05,0.1,0.5,1,10]
#       "beta_grid": [0.001,0.01,0.05,0.1,0.5,1,10]
#        num_fea: [10, 20,30,40,50,60,70,80,90,100]
#      }

# 1) Normalization of X
#    - Input data X ∈ R^{m×d}
#    - Z-score normalization is applied feature-wise:
#         X ← (X − μ) / (σ+1e-12)
#    - This ensures comparability across features for ℓ2,1 regularization
#
# 2) Autoencoder initialization
#    - One-hidden-layer autoencoder is constructed:
#         Encoder:  X → H = sigmoid(X W(1))
#         Decoder:  H → X_hat = H W(2)
#    - Bias terms are disabled to strictly follow the paper formulation
#    - W(1) ∈ R^{d×h}, W(2) ∈ R^{h×d}
#
# 3) Forward pass
#    - Compute latent representation:
#         H = sigmoid(X W(1))
#    - Reconstruct the input:
#         X_hat = H W(2)
#
# 4) Objective function (Eq. 4)
#    - Minimize:
#         (1 / 2m) ||X − X_hat||_F^2          (reconstruction loss)
#       + α ||W(1)||_{2,1}                   (feature selection term)
#       + (β / 2)(||W(1)||_F^2 + ||W(2)||_F^2) (weight decay)
#
# 5) Optimization via proximal gradient descent
#
#    5.1) Smooth loss optimization
#         - The non-sparse part of the objective is:
#              J⁻ = (1 / 2m) ||X − X_hat||_F^2
#                 + (β / 2)(||W(1)||_F^2 + ||W(2)||_F^2)
#         - Gradients are computed using backpropagation
#
#    5.2) Decoder update
#         - W(2) is updated using standard gradient descent:
#              W(2) ← W(2) − η ∇W(2) J⁻
#
#    5.3) Encoder update (feature selection step)
#         - Take a gradient step on W(1):
#              W(1) ← W(1) − η ∇W(1) J⁻
#         - Apply row-wise group soft-thresholding:
#              w_i ← max(1 − αη / ||w_i||₂, 0) · w_i
#         - Entire rows of W(1) are driven to zero, removing features
#
# 6) Iteration
#    - Steps 3–5 are repeated for a fixed number of epochs
#
# 7) Feature scoring and selection
#    - After convergence, feature importance is computed as:
#         score_i = ||W(1)_i||₂
#    - Features are ranked in descending order of score
#    - The top-k features are selected as the final subset
#
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class AEFSCore(nn.Module):
    """
    f(X) = σ1(X W(1)),   σ1(x) = 1 / (1 + exp(-x)) sigmoid
    X_hat = f(X) W(2),   σ2(x) = x
    """
    def __init__(self, d_in: int, h: int):
        super().__init__()
        self.enc = nn.Linear(d_in, h, bias=False)
        self.dec = nn.Linear(h, d_in, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.act(self.enc(X))
        X_hat = self.dec(H)
        return X_hat


def zscore_norm(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True, unbiased=False)
    return (X - mu) / (std + 1e-12)


def group_soft_threshold_rows(W: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Group soft-thresholding (row-wise):
      w_new = max(1 - λ/||w||2, 0) * w
    """
    row_norm = torch.norm(W, p=2, dim=1, keepdim=True)
    scale = torch.clamp(1.0 - lam / (row_norm + 1e-12), min=0.0)
    return scale * W


def aefs_loss(
    X: torch.Tensor,
    X_hat: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    alpha: float,
    beta: float
) -> torch.Tensor:
    """
    AEFS loss function (Eq. 4)
    J(Θ) = (1/(2m)) ||X - X_hat||_F^2
         + α ||W(1)||_{2,1}
         + (β/2) * ( ||W(1)||_F^2 + ||W(2)||_F^2 )
    """
    m = X.shape[0]

    recon = 0.5 * torch.norm(X - X_hat, p="fro").pow(2) / m
    l21 = alpha * torch.norm(W1, p=2, dim=1).sum()
    weight_decay = 0.5 * beta * (
        torch.norm(W1, p="fro").pow(2) +
        torch.norm(W2, p="fro").pow(2)
    )

    return recon + l21 + weight_decay



class AEFS:
    """
    Proximal optimization exactly as described in Sec. 3 of the paper.
    """
    def __init__(
        self,
        hidden: int,
        alpha: float,
        beta: float,
        lr: float = 1e-3
    ):
        self.d_in = None
        self.hidden = int(hidden)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lr = float(lr)

        self.net = None  # se construye en fit() cuando conocemos d

    def fit(self, X: np.ndarray, epochs: int = 100, normalize: bool = True, verbose: bool = True):
        X_t = torch.tensor(X, dtype=torch.float32)

        if normalize:
            X_t = zscore_norm(X_t)

        m, d = X_t.shape

        # Inferimos d_in desde X y construimos la red una sola vez
        if self.net is None:
            self.d_in = int(d)
            self.net = AEFSCore(self.d_in, self.hidden)
        else:
            if d != self.d_in:
                raise ValueError(f"Dimensión incorrecta: X tiene d={d}, modelo d_in={self.d_in}")

        for ep in range(1, epochs + 1):
            X_hat = self.net(X_t)

            # Paper notation:
            # W(1) ∈ R^{d×h}, W(2) ∈ R^{h×d}
            W1 = self.net.enc.weight.T
            W2 = self.net.dec.weight.T

            loss_full = aefs_loss(X_t, X_hat, W1, W2, self.alpha, self.beta)

            # Smooth part J^-(Θ)
            recon = 0.5 * torch.norm(X_t - X_hat, p="fro").pow(2) / m
            weight_decay = 0.5 * self.beta * (
                torch.norm(W1, p="fro").pow(2) +
                torch.norm(W2, p="fro").pow(2)
            )
            loss_smooth = recon + weight_decay

            self.net.zero_grad(set_to_none=True)
            loss_smooth.backward()

            with torch.no_grad():
                # Gradient step for W(2)
                self.net.dec.weight -= self.lr * self.net.dec.weight.grad

                # Proximal step for W(1)
                W1_t = self.net.enc.weight - self.lr * self.net.enc.weight.grad
                W1_new_T = group_soft_threshold_rows(W1_t.T, self.alpha * self.lr)
                self.net.enc.weight.copy_(W1_new_T.T)

            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"Epoch {ep}/{epochs} | Loss={float(loss_full):.6f}")

    def feature_scores(self) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() antes de feature_scores().")

        with torch.no_grad():
            W1 = self.net.enc.weight.T
            scores = torch.norm(W1, p=2, dim=1)
        return scores.detach().cpu().numpy()

    def select_top_features(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.feature_scores()
        idx = np.argsort(scores)[::-1][:k]
        return idx, scores[idx]