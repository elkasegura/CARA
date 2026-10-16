# ====================================================================================
# MODELO DGA: "Differentiable Gated Autoencoders for Unsupervised Feature Selection"
# Authors: Zebin Chen, Jintang Bian, Bo Qiao, Xiaohua Xie
# Journal: Neurocomputing, Vol. 601, 2024, pp. 128202
# ====================================================================================

#Parameters
#hidden: tamaño hidden del autoencoder
#gamma: weight of reconstruction loss (γ)
#lambda: weight of regularizer (λ)
#lr: learning rate del optimizador SGD (beta en el paper)
#B: nº de muestras Monte Carlo (si B=None, aquí se usa B=N)
#epochs: nº de épocas de entrenamiento


#grid { SGD Optimizer
#       "beta_grid": learning_rate = 0.001
#       "hidden_grid": [10,20,30,40,50]
#       Monte Carlo sampling B = n, numero de muestras
#       "gamma_grid": [0.001,0.01,0.05,0.1,0.5,1,10]
#       "lambda_grid": [0.001,0.01,0.05,0.1,0.5,1,10]
#      }

# Dimensiones:
#   X_t  : (N, d)  -> N muestras, d features
#   h    : tamaño hidden del autoencoder
#   B    : nº de muestras Monte Carlo (si B=None, aquí se usa B=N)
#
# Parámetros a aprender:
#   θ    : pesos del autoencoder f_θ
#   μ    : (d,) parámetros de gates (inicializados en 0.5)
#
# Autoencoder usado:
#   Encoder: Linear(d -> h) + ReLU + Linear(h -> h) + ReLU
#   Decoder: Linear(h -> h) + ReLU + Linear(h -> d)


# ====================================================================================
# 0) Inicialización

# - Crear f_θ (autoencoder) con input d y hidden h
# - Crear μ = 0.5 * ones(d)  -> forma (d,)
# - Optimizador SGD sobre (θ, μ) con lr = beta (aquí self.lr) que es beta


# 1) Loop de entrenamiento por épocas

# En cada época:

# 1.1) Muestrear gates G con Eq.(6)

#   eps ~ N(0,1) con forma (B, d)
#   G = clamp( μ[None, :] + eps , 0, 1 )  -> forma (B, d)
#
#   mu.unsqueeze(0): (1, d)
#   eps           : (B, d)
#   G             : (B, d)

# 1.2) Pérdida global LG (Eq.(3))

#   X_hat = f_θ(X_t)                 -> (N, d)
#   LG = (1/N) * ||X_hat - X_t||_F^2 -> escalar

# 1.3) Pérdida informativeness LI (Eq.(2)) aproximada por Monte Carlo

# Aplicar gates a la entrada:
#   X_t.unsqueeze(0) : (1, N, d)
#   G.unsqueeze(1)   : (B, 1, d)
#   X_g = X_t * G    : (B, N, d)   (broadcast)
#
# Aplanar para pasar por el AE:
#   X_g_2d = reshape(X_g, (B*N, d))       -> (B*N, d)
#   X_hat_g_2d = f_θ(X_g_2d)             -> (B*N, d)
#   X_hat_g = reshape(X_hat_g_2d, (B,N,d))-> (B, N, d)
#
# Calcular error vs X original:
#   diff = X_hat_g - X_t[None, :, :]      -> (B, N, d)
#   LI_each[b] = (1/N) * sum_{i,k} diff[b,i,k]^2
#   LI_each -> (B,)

# 1.4) Loss principal L (Algorithm 1 - línea 9)

# Para cada muestra Monte Carlo b:
#   L_b = LI_each[b] + gamma * LG
# Promedio sobre b:
#   L = mean_b(L_b) -> escalar

# 1.5) Regularizador R (Algorithm 1 - línea 10, Eq.(9))

#   R = lambda * sum_k Phi( μ_k / sigma )
# Aquí sigma=1.0 (coherente con eps~N(0,1) en Eq.(6))
#
#   mu     : (d,)
#   Phi(mu): (d,)
#   sum    : escalar
#   R      : escalar

# 1.6) Loss total y actualización (Algorithm 1 - líneas 11–13)

#   loss = L + R         -> escalar
#   backward()           -> gradientes para θ y μ
#   step()               -> update SGD

# ====================================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal




# Autoencoder: 2 encoder + 2 decoder, ReLU en cada layer (paper)

class DGACore(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, d),
            # (paper dice "each layer contains one ReLU";
            # en AE típicos no se usa ReLU final para reconstrucción real-valued.
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(X))



# Eq.(6) gates: g = clamp(mu + eps, 0, 1) con eps~N(0,1)
# g_k = min(1, max(0, mu_k + epsilon_k)),  with  epsilon_k ~ N(0, 1)

def sample_gates_eq6(mu: torch.Tensor, B: int, normal: Normal) -> torch.Tensor:
    eps = normal.sample((B, mu.shape[0]))              # (B,d) en CPU
    g = torch.clamp(mu.unsqueeze(0) + eps, 0.0, 1.0)   # Eq.(6)
    return g



# Eq.(9) regularizador: sum_k Phi(mu_k / sigma)
# Para ser fiel, usa sigma=1 (consistente con Eq.(6) escrita).

def gate_regularizer_eq9(mu: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    x = mu / sigma
    Phi = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    return torch.sum(Phi)


class DGA:
    """
    DGA (Algorithm 1 style).
    - Usa AE (2 capas encoder + 2 decoder con ReLU internas)
    - Gates: g = clamp(mu + eps, 0, 1), eps~N(0,1)
    - Loss estable numéricamente: MSE global (mean) en vez de sumas enormes
    - Gradient clipping real con clip_grad_norm_
    """

    def __init__(
        self,
        hidden: int,
        gamma: float,
        lam: float,
        lr: float = 1e-3,
        optimizer: str = "sgd",      # "sgd" o "adam"
        max_grad_norm: float = 1.0,  # clipping
        seed: int | None = None,
    ):
        self.hidden = int(hidden)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.lr = float(lr)

        self.optimizer = optimizer.lower()
        self.max_grad_norm = float(max_grad_norm)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.normal = Normal(0.0, 1.0)

        self.d = None
        self.f_theta: DGACore | None = None
        self.mu: nn.Parameter | None = None
        self.opt: optim.Optimizer | None = None

        self.g_last: torch.Tensor | None = None

    def fit(self, X: np.ndarray, epochs: int = 50, B: int | None = None, verbose: bool = True):
        X_t = torch.tensor(X, dtype=torch.float32)  # CPU
        N, d = X_t.shape
        self.d = d

        if B is None:
            B = N

        self.f_theta = DGACore(d=d, h=self.hidden)
        self.mu = nn.Parameter(torch.full((d,), 0.5, dtype=torch.float32))

        params = list(self.f_theta.parameters()) + [self.mu]

        if self.optimizer == "adam":
            self.opt = optim.Adam(params, lr=self.lr)
        elif self.optimizer == "sgd":
            self.opt = optim.SGD(params, lr=self.lr)
        else:
            raise ValueError("optimizer debe ser 'sgd' o 'adam'")

        for ep in range(1, epochs + 1):
            self.opt.zero_grad()

            # Gates (Eq.6)
            G = sample_gates_eq6(self.mu, B, self.normal)  # (B,d)
            self.g_last = G.detach()

            # LG (estable: MSE global)
            X_hat = self.f_theta(X_t)  # (N,d)
            #definir como forb
            LG = (X_hat - X_t).pow(2).mean()

            # LI (MC)
            X_g = X_t.unsqueeze(0) * G.unsqueeze(1)  # (B,N,d) #Esto es para aplicar gates a cada muestra de entrada, pero es muy grande para pasar por el AE, así que se aplanan las muestras para pasarlas por el AE y luego se vuelven a dar forma (B,N,d)
            X_hat_g = self.f_theta(X_g.reshape(B * N, d)).reshape(B, N, d) #Aqui se aplanan las muestras para pasarlas por el AE y luego se vuelven a dar forma (B,N,d)

            # (B,) cada muestra MC tiene su error medio en N,d
            LI_each = (X_hat_g - X_t.unsqueeze(0)).pow(2).mean(dim=(1, 2)) # el pow es para el cuadrado

            # Alg line 9: promedio sobre b
            L = (LI_each + self.gamma * LG).mean()

            # Regularizador (Eq.9)
            R = self.lam * gate_regularizer_eq9(self.mu, sigma=1.0)

            loss = L + R

            # safety: corta si se rompe
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss no finita (NaN/Inf) en epoch {ep}. "
                    f"LG={LG.item():.6e}, L={L.item():.6e}, R={R.item():.6e}"
                )

            loss.backward()

            # Gradient clipping REAL
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)

            self.opt.step()

            if verbose and (ep == 1 or ep % 10 == 0):
                print(
                    f"Epoch {ep}/{epochs} |"
                    f"Loss={loss.item():.6f}"
                )

        return self

    def select_top_features(self, h: int):
        if self.g_last is None:
            raise RuntimeError("Debes llamar fit() antes de seleccionar features.")
        g_bar = self.g_last.mean(dim=0).cpu().numpy()  # (d,)
        idx = np.argsort(g_bar)[::-1][:h]
        return idx, g_bar[idx]

    def get_params(self):
        return {
            "mu": None if self.mu is None else self.mu.detach().cpu().numpy(),
            "theta": None
            if self.f_theta is None
            else {k: v.detach().cpu() for k, v in self.f_theta.state_dict().items()},
        }