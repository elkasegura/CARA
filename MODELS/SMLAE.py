# ====================================================================================
# MODELO SMLAE: Sparse Manifold Learning AutoEncoder (SMLAE)
# Authors: Moslemi & Jamshidi, 
# Information Processing and Management 62 (2025) 103923
# ====================================================================================

#Parametros del modelo SMLAE:
# d_hidden: dimensión del espacio latente (número de neuronas en la capa oculta).
# n_neighbors: número de vecinos para construir el Laplaciano kNN.
# alpha: peso del término de esparcidad en la función de pérdida .
# beta: peso del término de regularización de pesos en la función de pérdida (weight decay).
# gamma: peso del término de preservación de la estructura del manifold en la función de pérdida
# omega: peso del término de preservación de la estructura latente en la función de pérdida.
# rho1: tasa de aprendizaje para la actualización de los pesos del codificador (W1).
# rho2: tasa de aprendizaje para la actualización de los pesos del decodificador (W2).
# N : numero de epocas de entrenamiento

#SMLAE

# grids = {
#        "hidden_grid": [10, 20, 30, 40, 50],
#        "num_fea_grid": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#        "alpha_grid": [1e-6, 1e-5, 1e-4 1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3, 1e4, 1e-5, 1e6],
#        "beta_grid": [1e-6, 1e-5, 1e-4 1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3, 1e4, 1e-5, 1e6], 
#        "gamma_grid": [1e-6, 1e-5, 1e-4 1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3, 1e4, 1e-5, 1e6],
#        "omega_grid": [1e-6, 1e-5, 1e-4 1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3, 1e4, 1e-5, 1e6],
#    }


# ====================================================================================
# 1) Normalization of X
#    - X ∈ R^{d×n} z-score 

#   - Each feature (row) is standardized using z-score:
#       X ← (X − mean(X)) / (std(X) + ε)
#       where:
#         X ∈ R^{d×n}
#         mean(X) ∈ R^{d×1}
#         std(X) ∈ R^{d×1}
#

# 2) Graph construction and Laplacian

#   - A kNN graph is constructed over the samples (columns of X).
#     Each sample x_i ∈ R^{d}.
#   - The affinity matrix A is defined using a Gaussian kernel:
#       A_ij = exp( −||x_i − x_j||^2 / (2σ^2) )
#       A ∈ R^{n×n}
#   - The degree matrix D is computed as:
#       D_ii = Σ_j A_ij
#       D ∈ R^{n×n} (diagonal)
#   - The graph Laplacian is defined as:
#       Lx = D − A
#       Lx ∈ R^{n×n} Lx contiene la estructura del manifold de los datos.
#

# 3) Initialization

#   - Initialize the autoencoder weights:
#       W1 ∈ R^{d×k},  W2 ∈ R^{k×d}
#       b1 ∈ R^{k},    b2 ∈ R^{d}
#       b1 = 0, b2 = 0
#   - Determine the size c of H:
#       c = floor( sqrt(n / 2) )   (as suggested in the paper)
#   - Initialize auxiliary variables:
#       H ∈ R^{n×c},   H ≥ 0  (random positive values)
#       Y ∈ R^{n×c}    (Lagrange multiplier)
#       Q ∈ R^{d×d}    (diagonal matrix for sparsity)
#

# Main iteration

#   Repeat for a fixed number of epochs:
#
# 4) Update of Q (ℓ2,1 approximation via IRLS)
#   - Compute the ℓ2 norm of each row of W1:
#       w_i ∈ R^{1×k},   ||w_i||_2
#   - Define:
#       Q = diag( 1 / (2||w_i||_2 + ε) )
#       Q ∈ R^{d×d}
#   - Q penalizes rows of W1 with low relevance,
#     promoting feature selection.
#
# 5) Autoencoder step (update of W1 and W2)
#   - Forward pass:
#       Z = σ(W1^T X + b1)
#       Z ∈ R^{k×n}
#       X_hat = W2^T Z + b2
#       X_hat ∈ R^{d×n}
#   - Minimize the objective function (Eq. 5 in the paper):
#       (i)  Reconstruction error:
#            (1/(2n)) ||X − X_hat||_F^2
#            ||·||_F^2 → scalar
#       (ii) Sparsity penalty:
#            α Tr(W1^T Q W1)
#            W1^T Q W1 ∈ R^{k×k}, Tr(·) → scalar
#       (iii) Weight regularization Weiht decay:
#            (β/2)(||W1||_F^2 + ||W2||_F^2)
#            ||·||_F^2 → scalar
#       (iv) Laplacian regularization on the encoder:
#            (γ/2) Tr(W1^T X Lx X^T W1)
#            X Lx X^T ∈ R^{d×d}
#       (v)  Laplacian regularization on the decoder:
#            (ω/2) Tr(W2^T Z (H H^T) Z^T W2)
#            H H^T ∈ R^{n×n}
#   - Update W1 and W2 using gradient descent
#     (with gradient clipping for stability).

# 6) Graph step (update of H and Y)

#   - Update H using the multiplicative rule derived from KKT:
#       H_ij ← H_ij * ( (2 Lx H)_ij / ( (4 H H^T H)_ij + ε ) )
#       Lx H ∈ R^{n×c},   H H^T H ∈ R^{n×c}
#   - Project to enforce non-negativity:
#       H ← max(H, ε)
#   - Update the Lagrange multiplier Y:
#       Y = 2 Lx H − 4 H H^T H
#       Y ∈ R^{n×c}
#       Y ← max(Y, 0)
#       Y_ij = 0 if H_ij > ε  

# 7) Repeat until convergence

#   - Alternate the autoencoder and graph steps
#     until the objective function stabilizes.
# ====================================================================================

import numpy as np
import torch
import torch.nn as nn


class SMLAECore(nn.Module):
    """
    Notación alineada a X ∈ R^{d×n}:
      Z     = σ( W1^T X + b1 )     con W1 ∈ R^{d×k}
      X_hat = W2^T Z + b2          con W2 ∈ R^{k×d}  (decoder lineal)
    """
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden

        self.W1 = nn.Parameter(torch.empty(d_in, d_hidden))
        self.W2 = nn.Parameter(torch.empty(d_hidden, d_in))

        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.b2 = nn.Parameter(torch.zeros(d_in))

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, X: torch.Tensor):
        """
        X: (d, n)
        returns:
          Z: (k, n)
          X_hat: (d, n)
        """
        Z = torch.sigmoid(self.W1.t() @ X + self.b1.unsqueeze(1))   # (k, n)
        X_hat = self.W2.t() @ Z + self.b2.unsqueeze(1)              # (d, n)
        return Z, X_hat


def zscore_features(X: torch.Tensor) -> torch.Tensor:
    """
    Z-score por feature (fila), como Algorithm 1.
    X: (d, n)
    """
    mu = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True, unbiased=False)
    return (X - mu) / (std + 1e-12)


# -------------------------
# Laplacian (kNN + Gaussian kernel weights)
# -------------------------
def build_laplacian_knn_gaussian(
    X: torch.Tensor,
    n_neighbors: int = 5,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Construye Lx = D - A con kNN y kernel Gaussiano.
    X: (d, n)  (columnas = samples)
    return Lx: (n, n)
    """
    Xn = X.t()  # (n, d)
    n = Xn.shape[0]

    # dist^2 completa (n,n)
    sq_norms = (Xn ** 2).sum(dim=1, keepdim=True)  # (n,1) ||x_i||^2
    dist2 = sq_norms + sq_norms.t() - 2.0 * (Xn @ Xn.t())  # (n,n) ||x_i - x_j||^2
    dist2 = dist2.clamp_min(0.0)  # evitar negativos numéricos

    k = min(n_neighbors, n)
    knn_dist2, knn_idx = torch.topk(dist2, k=k, largest=False, dim=1)

    # varianza de distancias (sin el self-neighbor)
    if k > 1:
        d2_edges = knn_dist2[:, 1:].reshape(-1)  # excluir self-distances
    else:
        d2_edges = knn_dist2.reshape(-1)

    var = torch.var(d2_edges, unbiased=False).clamp_min(eps)  # σ^2

    A = torch.zeros((n, n), device=X.device, dtype=X.dtype)
    for i in range(n):
        for t in range(k):
            j = int(knn_idx[i, t].item())
            if i == j:
                continue
            w = torch.exp(-dist2[i, j] / (2.0 * var))
            A[i, j] = w
            A[j, i] = w  # simétrico

    D = torch.diag(A.sum(dim=1))
    Lx = D - A
    return Lx


# -------------------------
# Q (diagonal): Q = diag( 1 / (2||w_i||_2) )
# -------------------------
@torch.no_grad()
def update_Q_from_W1(W1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Paper: Q = diag( 1 / (2||w_i||_2) )
    W1: (d, k)
    return Q: (d, d) diagonal
    """
    row_norms = torch.sqrt((W1 ** 2).sum(dim=1) + eps)
    q_diag = 1.0 / (2.0 * row_norms)
    q_diag = q_diag.clamp_min(eps)
    return torch.diag(q_diag)


# -------------------------
# Eq.(5) Loss
# -------------------------
def smlae_loss_eq05_with_Q(
    X: torch.Tensor,
    model: SMLAECore,
    Lx: torch.Tensor,
    H: torch.Tensor,
    Q: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
) -> torch.Tensor:
    """
    Implementa (5) en forma estable.
    Nota: NO añadimos aquí ||HH^T - Lx||_F^2 ni Tr(Y^T H); el papel los usa en el Lagrangiano,
    mientras que aquí actualizamos H (y Y) aparte.
    """
    d, n = X.shape
    Z, X_hat = model(X)

    # 1/(2n) ||X - X_hat||_F^2
    rec = ((X - X_hat) ** 2).sum() / (2.0 * n)

    W1, W2 = model.W1, model.W2

    # alpha * Tr(W1^T Q W1)
    sp = alpha * torch.trace(W1.t() @ Q @ W1)

    # (beta/2) (||W1||_F^2 + ||W2||_F^2)
    wd = (beta / 2.0) * ((W1 ** 2).sum() + (W2 ** 2).sum())

    # (gamma/2) Tr(W1^T X Lx X^T W1)
    A = X @ Lx @ X.t()                      # (d, d)
    lp = (gamma / 2.0) * torch.trace(W1.t() @ A @ W1)

    # (omega/2) Tr(W2^T Z H H^T Z^T W2)
    HHT = H @ H.t()                         # (n, n)
    B = Z @ HHT @ Z.t()                     # (k, k)
    zlp = (omega / 2.0) * torch.trace(W2.t() @ B @ W2)

    return rec + sp + wd + lp + zlp


# -------------------------
# Eq.(15) update
# -------------------------
@torch.no_grad()
def update_H_eq15_literal_paper(H: torch.Tensor, Lx: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Paper Eq.(15) (literal):
      H_ij <- H_ij * ( (2 Lx H)_ij / ( (4 H H^T H)_ij + eps ) )

    Para mantener H >= 0 en práctica, proyectamos con clamp_min(eps).
    """
    num = 2.0 * (Lx @ H)                               # (n, c) puede tener negativos
    den = 4.0 * (H @ (H.t() @ H))                      # (n, c) >= 0
    H = H * (num / (den + eps))
    return H.clamp_min(eps)


@torch.no_grad()
def update_Y_from_stationarity_and_KKT(H: torch.Tensor, Lx: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Estacionariedad:
      -2 Lx H + 4 H H^T H + Y = 0  =>  Y = 2 Lx H - 4 H H^T H

    Imponemos:
      - Y >= 0 (proyección)
      - Y_ij H_ij = 0 (aprox): si H_ij > eps, forzamos Y_ij = 0
    """
    Y = 2.0 * (Lx @ H) - 4.0 * (H @ (H.t() @ H))
    Y = torch.clamp(Y, min=0.0)  # Y >= 0
    Y = Y.masked_fill(H > eps, 0.0)  # complementary slackness (aprox.)
    return Y


@torch.no_grad()
def update_H_and_Y_paper_style(
    H: torch.Tensor,
    Lx: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-12
):
    """
    Un paso paper-like:
      1) Update H con Eq.(15) literal + proyección a H>=0
      2) Update Y vía estacionariedad + proyección + slackness aproximada
    """
    H = update_H_eq15_literal_paper(H, Lx, eps=eps)
    Y = update_Y_from_stationarity_and_KKT(H, Lx, eps=eps)
    return H, Y


class SMLAE:
    """
    Versión alineada al paper, incluyendo Y explícito (multiplicador de Lagrange para H>=0).

    - Q correcto: diag(1/(2||w_i||_2))
    - H con dimensión (n, c) donde c = floor(sqrt(n/2)) si n_clusters=None
    - update H: Eq.(15) literal (con proyección H>=0 para estabilidad)
    - update Y: vía estacionariedad + proyección + complementary slackness (aprox.)
    - gradient clipping para estabilidad
    """

    def __init__(
        self,
        d_hidden: int,
        n_neighbors: int = 5,
        alpha: float = 1e-6,
        beta: float = 1e-3,
        gamma: float = 1e-3,
        omega: float = 1e-3,
        rho1: float = 1e-6,
        rho2: float = 1e-6,
        device: str = "cpu",
        n_clusters: int | None = None,
        clip_grad_norm: float | None = 1.0,
        eps_hy: float = 1e-12,
    ):
        self.d_hidden = d_hidden
        self.n_neighbors = n_neighbors

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

        self.rho1 = rho1
        self.rho2 = rho2
        self.device = device

        self.n_clusters = n_clusters
        self.clip_grad_norm = clip_grad_norm
        self.eps_hy = eps_hy

        self.model: SMLAECore | None = None
        self.H: torch.Tensor | None = None
        self.Y: torch.Tensor | None = None
        self.Q: torch.Tensor | None = None
        self.Lx: torch.Tensor | None = None
        self.W1: torch.Tensor | None = None

    def fit(self, X_input, epochs: int = 100):
        """
        X_input: (d, n) (XT = X.T)
        """
        X = torch.tensor(X_input, dtype=torch.float32, device=self.device)
        d, n = X.shape

        # Step 1: Z-score
        X = zscore_features(X)

        # Step 2: Laplacian
        self.Lx = build_laplacian_knn_gaussian(X, n_neighbors=self.n_neighbors)

        # c según paper: floor(sqrt(n/2))
        c = self.n_clusters if self.n_clusters is not None else int(np.sqrt(n / 2.0))
        c = max(2, min(c, n))
        c = int(c)

        # Step 3: init W1,W2,H,Q,Y
        self.model = SMLAECore(d_in=d, d_hidden=self.d_hidden).to(self.device)
        self.H = torch.rand((n, c), device=self.device).clamp_min(self.eps_hy)
        self.Y = torch.zeros((n, c), device=self.device)  # paper-like
        self.Q = torch.eye(d, device=self.device)

        #beta_t = float(self.beta)

        for ep in range(epochs + 1):
            # update Q (paper)
            self.Q = update_Q_from_W1(self.model.W1)

            self.model.zero_grad(set_to_none=True)

            loss = smlae_loss_eq05_with_Q(
                X=X,
                model=self.model,
                Lx=self.Lx,
                H=self.H,
                Q=self.Q,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                omega=self.omega,
            )
            loss.backward()

            # Gradient clipping (evita explosión)
            if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            with torch.no_grad():
                # GD como en paper (10)-(11)
                self.model.W1 -= self.rho1 * self.model.W1.grad
                self.model.W2 -= self.rho2 * self.model.W2.grad

                # update H y Y (paper-like)
                self.H, self.Y = update_H_and_Y_paper_style(self.H, self.Lx, self.Y, eps=self.eps_hy)

            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep}/{epochs} | Loss={float(loss):.6f}")

        self.W1 = self.model.W1.detach().cpu()
        return self

    def get_gene_score(self):
        """
        score = ||w_i||_2 por fila de W1 (paper: rank rows by Euclidean norm)
        """
        W1 = self.model.W1.detach().cpu().numpy()
        return np.linalg.norm(W1, axis=1)

    def select_top_genes(self, num_fea: int):
        scores = self.get_gene_score()
        idx = np.argsort(scores)[::-1][:num_fea]
        return idx, scores[idx]
