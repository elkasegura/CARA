# ====================================================================================
# MODELO DRAE "Deep Robust Autoencoder for Unsupervised Feature Selection"
# Authors: Yunzhi Ling, Feiping Nie, Weizhong Yu, Xuelong Li
# IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 1, JANUARY 2025
# ====================================================================================

#Parametros del modelo DRAE:
# d_hidden: número de neuronas en la capa oculta del AE
# n_clusters: número de clusters
# alpha: parámetro grande para la proyección de V (en el paper el parametro viene fijado a 10^8) para mantener la ortogonalidad de V, con propiedades V^T V = I e V >= 0
# beta: peso del término de selección de features L21
# gamma: parámetro para los pesos s_i (robustez)
# zeta: peso del término de clustering
# lam: peso del weight decay  
# T: epochs: número de épocas de entrenamiento

#DRAE

# grids = {
#        "hidden_grid": [10, 20, 30, 40, 50],
#        "num_fea_grid": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#        "zeta_grid": [1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3],
#        "lam_grid": [1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3],
#        "beta_grid": [1e-3, 1e-2, 1e-1, 0.0, 1e1, 1e2, 1e3],
#        "gamma_grid": [1e5, 1e4, 1e3, 1e2],
#    }


#Prostate 
#NUM_FEATURES = 10 : DRAE(d_in=d_in, d_hidden=20, n_clusters=n_clusters, alpha=1e8, beta=0.0001, gamma=10000, zeta=0.0001, lam=10)
#NUM_FEATURES = 20 : DRAE(d_in=d_in, d_hidden=20, n_clusters=n_clusters, alpha=1e8, beta=0.1, gamma=10000, zeta=0.01, lam=10)
#NUM_FEATURES = 60 : DRAE(d_in=d_in, d_hidden=20, n_clusters=n_clusters, alpha=1e8, beta=0.01, gamma=10000, zeta=0.1, lam=10)
#NUM_FEATURES = 70 : DRAE(d_in=d_in, d_hidden=20, n_clusters=n_clusters, alpha=1e8, beta=0.01, gamma=10000, zeta=0, lam=10)
#NUM_FEATURES = 120 : DRAE(d_in=d_in, d_hidden=20, n_clusters=n_clusters, alpha=1e8, beta=0.01, gamma=10000, zeta=0, lam=10)

# 1) Normalization of X
#    - X ∈ R^{d×n} z-score 

# 2) Autoencoder (AE) initialization
#    - W(1), W(2) ~ N(0, 0.005)
#    - b1 = 0, b2 = 0

# 3) Clustering initialization
#    - V is initialized using k-means (only as an initialization step)
#    - Z is obtained by a forward pass of the AE: Z = f(X)
#    - U is computed using Eq. (25):
#         U = Z V (V^T V)^{-1}
#    - V is updated via orthogonal projection (Eq. 28–29):
#         P = (1/alpha) Z^T U + Q
#         P = M Σ N^T   (SVD)
#         V = M N^T
#    - Non-negativity is enforced (Eq. 32):
#         Q = max(V, 0)


# Main iteration 


# 4) Autoencoder step (robust learning + feature selection)
#    - Using the current AE, compute:
#         Z = f(X)
#         X_hat = g(Z)
#    - Compute s (Eq. 16): sample weights for robustness
#    - Compute D (Eq. 18): feature weights via IRLS
#    - In this sub-step, s and D are FIXED
#    - Update W(1) and W(2) by minimizing P (Eq. 18) using L-BFGS
#
#      IMPORTANT:
#      Backpropagation INCLUDES the clustering term,
#      because the hidden-layer error contains:
#         ζ (Z − U V^T)
#      (see Eq. 19–20), which allows the clustering objective
#      to directly influence the update of W(1)

# 5) Clustering step
#    - Recompute Z using the updated AE parameters
#    - Update U using Eq. (25)
#    - Update V and Q using Eq. (28–29) and Eq. (32)

# 6) Repeat
#    - Repeat Steps 4 and 5 until convergence
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


#Este el AE del DRAE, donde se construye el AE con una sola capa oculta 
class DRAECore(nn.Module):
    """
    Paper:
      X ∈ R^{d×n}
      W(1) ∈ R^{d×k}, b1 ∈ R^{k}
      W(2) ∈ R^{k×d}, b2 ∈ R^{d}

      z = tanh( W(1)^T x + b1 )  => Z ∈ R^{k×n}
      x_hat = W(2)^T z + b2      => X_hat ∈ R^{d×n}
    """
    def __init__(self, d_in: int, d_hidden: int):             # funcion de inicializacion y construccion del modelo
        super().__init__()                                    # esto es para construir nn.Module
        self.d_in = d_in                                      # atributo (d_in = numero de features) de la clase para construir el modelo
        self.d_hidden = d_hidden                              # atributo (d_hidden = numero de neuronas en la capa oculta) de la clase para construir el modelo

        # W1: (d,k), b1: (k,)
        self.W1 = nn.Parameter(torch.empty(d_in, d_hidden))   # atributo y parametro entrenable (W1 = pesos de la capa de entrada a la oculta) de la clase para construir el modelo
        self.b1 = nn.Parameter(torch.zeros(d_hidden))         # atributo y parametro entrenable (b1 = bias de la capa oculta) de la clase para construir el modelo

        # W2: (k,d), b2: (d,)
        self.W2 = nn.Parameter(torch.empty(d_hidden, d_in))   # atributo y parametro entrenable (W2 = pesos de la capa oculta a la de salida) de la clase para construir el modelo
        self.b2 = nn.Parameter(torch.zeros(d_in))             # atributo y parametro entrenable (b2 = bias de la capa de salida) de la clase para construir el modelo
       
        # Paper init: w_ij^(l) ~ N(0, η), η=0.005
        nn.init.normal_(self.W1, mean=0.0, std=0.005)         # inicializacion de los pesos W1 con Paper init: w_ij^(l) ~ N(0, η), η=0.005
        nn.init.normal_(self.W2, mean=0.0, std=0.005)         # inicializacion de los pesos W2 con Paper init: w_ij^(l) ~ N(0, η), η=0.005

    def forward(self, X: torch.Tensor):
        # Z = tanh(W1^T X + b1)
        Z = torch.tanh(self.W1.t() @ X + self.b1.unsqueeze(1))  # (k,n)     #contruccion de la capa oculta con funcion de activacion tanh

        # X_hat = W2^T Z + b2
        X_hat = self.W2.t() @ Z + self.b2.unsqueeze(1)          # (d,n)     #contruccion de la capa de salida (reconstruccion) linear
        return Z, X_hat


# CALCULO DE S Y D
# s_i (Eq. 16) y D (Eq. 18)


def update_s_eq16(X: torch.Tensor, X_hat: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Eq. (16):
      s_i = exp( -||x_i - g(f(x_i))||_2^2 / γ ) / Σ_j exp( -||x_j - g(f(x_j))||_2^2 / γ )
    """
    err = torch.sum((X - X_hat) ** 2, dim=0)  # (n,)  # ||x_i - xhat_i||^2 para cada muestra i
    logits = -err / gamma                     # logits = -||x_i - xhat_i||^2 / gamma
    #logits = logits - torch.max(logits)       # estabilidad numérica s > 0, restamos max para evitar overflow en exp, que significa que el mayor valor de logits sea 0
    s = torch.exp(logits)                     # la parte de arriba de s_i
    s = s / torch.sum(s)                      # propiedad de s_1 + ... + s_n = 1, la parte de abajo de s_i
    return s                                  # (n,)

def update_D_eq18(W1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Eq. (18) IRLS:
      D = Diag(d),  d_ii = 1 / (2||w_i||_2 + eps)
    donde w_i es la i-ésima fila de W(1).
    """
    row_norm = torch.norm(W1, p=2, dim=1)  # esto calcula ||w_i||_2 para cada fila i de W1  # (d,)
    d = 1.0 / (2.0 * row_norm + eps)       # esto calcula d_ii = 1 / (2||w_i||_2 + eps) para cada i  # (d,)
    return d                               # (d,)



# Funcion Objetivo del paper (Eq. 18) con S y D fijos, 


def drae_objective_P_eq18(
    X: torch.Tensor,    # (d, n) 
    model: nn.Module,
    U: torch.Tensor,
    V: torch.Tensor,
    s: torch.Tensor,      # fijo en este paso (Eq. 16)
    d_diag: torch.Tensor, # fijo en este paso (Eq. 18)
    beta: float,
    zeta: float,
    lam: float,
    gamma: float
) -> torch.Tensor:
    """
    Eq. (18):
      P = Tr((X - g(f(X))) S (X - g(f(X)))^T)
          + β Tr(W(1)^T D W(1))
          + ζ ||Z - U V^T||_F^2
          + λ Σ_{i=1}^2 ||W(i)||_F^2

    Donde:
      S = Diag(s),   D = Diag(d)
    """
    Z, X_hat = model(X)  # Z:(k,n), X_hat:(d,n)

    #  Reconstrucción robusta 
    # Tr((X-Xhat) S (X-Xhat)^T) == Σ_i s_i ||x_i - xhat_i||^2
    rec_per_sample = torch.sum((X - X_hat) ** 2, dim=0)  # (n,)
    rec = torch.sum(rec_per_sample * s)                  # escalar

    #  Entropia de s para evitar colapso a un solo punto
    # H(s) = gamma*( Σ_i s_i log(s_i))                                
    ent = gamma * torch.sum((s) * torch.log(s + 1e-8))  # se le añade un pequeño valor para evitar log(0)
    rec = rec + ent

    #  Feature selection IRLS 
    # β Tr(W1^T D W1) = β Σ_i d_i ||w_i||^2
    W1 = model.W1  # (d,k)
    fs = beta * torch.sum(d_diag.unsqueeze(1) * (W1 ** 2))

    #  Clustering term 
    # ζ ||Z - U V^T||_F^2
    UVt = U @ V.t()               # (k,n)
    cl = zeta * torch.sum((Z - UVt) ** 2)

    #  Weight decay 
    # (λ)(||W1||_F^2 + ||W2||_F^2)
    W2 = model.W2
    wd = lam * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2))
    #print(rec.item(), fs.item(), cl.item(), wd.item())

    return rec + fs + cl + wd



# Updates U, V, Q 

class DRAE:
    def __init__(
        self,
        #d_in: int,
        d_hidden: int = 10,
        n_clusters: int = 2,
        alpha: float = 1e8, # Paper: α grande (Alg.1 α=10^8)
        beta: float = 1e-3,
        gamma: float = 1e-2,
        zeta: float = 1e2,
        lam: float = 1e4,
        device: str | None = None,
        eps_irls: float = 1e-8,
    ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = None  # DRAECore
        
        self.d_hidden = d_hidden
        self.n_clusters = n_clusters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.lam = float(lam)
        self.eps_irls = float(eps_irls)

        # full-batch L-BFGS (el paper lo usa)
        #self.opt = torch.optim.LBFGS(self.model.parameters())
        self.opt = None

        self.U = None  # (k,c)
        self.V = None  # (n,c)
        self.Q = None  # (n,c)

        # IRLS: D = I al inicio
        self.d_diag = None  # (d,) vector diag de D

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        # Algoritmo 1 step 1: zero mean, unit variance por feature
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        return (X - mean) / (std+1e-8)

    def _init_UVQ(self, X_np: np.ndarray):
        """
        Algoritmo 1 step 2:
          - Inicializar Q=0
          - Inicializar V con k-means SOLO para arranque
          - Calcular Z inicial con el AE
          - Actualizar U con Eq. (25): U = Z V (V^T V)^(-1)
          - Proyectar V al manifold con Eq. (28)-(29) inmediatamente
        """
        d, n = X_np.shape

        # IRLS: D = I (paper + tu comentario)
        self.d_diag = torch.ones(d, dtype=torch.float32, device=self.device)  # D=I => diag = 1

        # V init por k-means (solo inicialización)
        X_samples = X_np.T  # (n,d)
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X_samples)  # (n,)

        V0 = np.eye(self.n_clusters, dtype=np.float32)[labels]  # one-hot (n,c)
        self.V = torch.tensor(V0, dtype=torch.float32, device=self.device)

        # Q=0
        self.Q = torch.zeros_like(self.V)

        # Z inicial, aqui se hace forward al modelo AE para obtener Z
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Z, _ = self.model(X_t)  # (k,n)

        # Eq. (25): U = Z V (V^T V)^(-1)
        self._update_U(Z)

        # Proyectar V usando Eq. (28)-(29) ya desde el inicio
        self._update_V_and_Q(Z)

    def _update_U(self, Z: torch.Tensor):
        """
        Eq. (25):
          U = Z V (V^T V)^(-1)
        """
        Z_np = Z.detach().cpu().numpy()              # (k,n)
        V_np = self.V.detach().cpu().numpy()         # (n,c)

        VtV = V_np.T @ V_np                          # (c,c)
        VtV_inv = np.linalg.inv(VtV + 1e-6 * np.eye(self.n_clusters, dtype=np.float32))

        U_np = Z_np @ V_np @ VtV_inv                 # (k,c)
        self.U = torch.tensor(U_np, dtype=torch.float32, device=self.device)

    

    def _update_V_and_Q(self, Z: torch.Tensor):
        """
        Eq. (28):
          min_{V^T V = I} ||V - P||_F^2
        con:
          P = (1/α) Z^T U + Q

        SVD:
          P = M Σ N^T
        Eq. (29):
          V = M N^T

        Eq. (32):
          Q = max(V, 0)
        """
        Z_np = Z.detach().cpu().numpy()              # (k,n)
        U_np = self.U.detach().cpu().numpy()         # (k,c)
        Q_np = self.Q.detach().cpu().numpy()         # (n,c)

        # P = (1/α) Z^T U + Q
        P = (1.0 / self.alpha) * (Z_np.T @ U_np) + Q_np  # (n,c)

         # seguridad mínima: eliminar NaN / Inf
        if not np.isfinite(P).all():
          P[~np.isfinite(P)] = 0.0

        # P = M Σ N^T
        M_svd, S_svd, Nt = np.linalg.svd(P, full_matrices=False)

        # V = M N^T
        V_np = M_svd @ Nt  # (n,c)

        # Q = max(V,0)
        Q_np = np.maximum(V_np, 0.0)

        self.V = torch.tensor(V_np, dtype=torch.float32, device=self.device)
        self.Q = torch.tensor(Q_np, dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray, epochs: int = 100):
        X = np.asarray(X, dtype=np.float32)
        X = self._normalize(X)
        d, n = X.shape

        self.model = DRAECore(d, self.d_hidden).to(self.device)
        self.opt = torch.optim.LBFGS( self.model.parameters(),
                        lr=0.5,                 # si explota: 0.1
                        max_iter=20,
                        history_size=10,
                        tolerance_grad=1e-7,
                        tolerance_change=1e-9,
                        line_search_fn="strong_wolfe")

        # build params
        _ = self.model(torch.tensor(X[:, :1], dtype=torch.float32, device=self.device))

        # init U,V,Q y D=I
        self._init_UVQ(X)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        for ep in range(1, epochs + 1):
            self.model.train()

            # ========= (A) Forward actual para actualizar s (Eq. 16) =========
            with torch.no_grad():
                Z_cur, X_hat_cur = self.model(X_t)
                s = update_s_eq16(X_t, X_hat_cur, self.gamma)  # (n,)
                # S = Diag(s) (no guardamos matriz completa, usamos s directamente)

            # ========= (B) IRLS: actualizar D usando W1 actual (Eq. 18) =========
            # Importante: D inicia como I, pero luego se actualiza iterativamente.
            with torch.no_grad():
                self.d_diag = update_D_eq18(self.model.W1, eps=self.eps_irls)  # (d,)

            # ========= (C) Optimizar W(1),W(2) con s y D FIJOS (Eq. 18) =========
            def closure():
                self.opt.zero_grad(set_to_none=True)
                P = drae_objective_P_eq18(
                    X=X_t,
                    model=self.model,
                    U=self.U,
                    V=self.V,
                    s=s.detach(),                 # fijo
                    d_diag=self.d_diag.detach(),   # fijo
                    beta=self.beta,
                    zeta=self.zeta,
                    lam=self.lam,
                    gamma=self.gamma
                )
                P = P.mean() #####
                if not torch.isfinite(P):
                    return torch.tensor(1e30, device=X_t.device, dtype=X_t.dtype, requires_grad=True) ####
                P.backward() #esta funcion calcula los gradientes
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) #####
                return P

            loss = self.opt.step(closure)

            # ========= (D) Recomputar Z tras update de AE =========
            with torch.no_grad():
                Z_new, _ = self.model(X_t)

            # ========= (E) Update U (Eq. 25) =========
            self._update_U(Z_new)

            # ========= (F) Update V,Q (Eq. 28-29, 32) =========
            self._update_V_and_Q(Z_new)

            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep}/{epochs} | Loss={float(loss):.6f}")

    def get_gene_scores(self):
        # score_i = ||w_i||_2  (norma por fila de W1)
        W1 = self.model.W1.detach().cpu().numpy()  # (d,k)
        scores = np.linalg.norm(W1, axis=1)        # (d,) esto suma por fila
        return scores

    def select_top_genes(self, p: int):
        scores = self.get_gene_scores()
        idx = np.argsort(scores)[::-1][:p]
        return idx, scores[idx]