"""Guan 2015 stepwise-specific + power-law NSB (Eq. 5, global k). See MODELS.md §3.5."""
import numpy as np

from .common import make_free_ligand, make_residual_vector

MODEL_NAME = "occupancy_decay"


def _unpack(ln_params):
    arr = np.asarray(ln_params)
    beta = np.exp(arr[0])
    gamma = arr[1]
    Ks = np.exp(arr[2:])
    return beta, gamma, Ks


def mole_fractions(L_free_M, ln_params, S, N):
    beta, gamma, Ks = _unpack(ln_params)
    n_max = S + N

    K_app = np.zeros(n_max + 1)
    for k in range(1, n_max + 1):
        denom = k ** gamma
        Kn_k = beta / denom if denom > 0 and np.isfinite(denom) else 0.0
        K_app[k] = (Ks[k - 1] if k <= S else 0.0) + Kn_k

    alpha = np.zeros(n_max + 1)
    alpha[0] = 1.0
    for j in range(1, n_max + 1):
        alpha[j] = alpha[j - 1] * L_free_M * K_app[j]

    total = alpha.sum()
    if total <= 0 or not np.isfinite(total):
        out = np.zeros(n_max + 1)
        out[0] = 1.0
        return out
    return alpha / total


def n_params(S):
    return S + 2


def initial_lnK(S, Kn=1e3, Ks=1e5, gamma=0.5):
    return np.concatenate(([np.log(Kn), gamma], np.log(np.full(S, Ks))))


def param_labels(S):
    return ["beta", "gamma"] + [f"Ks_{i+1}" for i in range(S)]


def partition_terms(ln_params, S, N):
    """w[i,j] = unnormalized (j specific, i-j nonspecific) contribution to peak i.

    Elementary-symmetric expansion of the per-step apparent product, with a
    step-dependent nonspecific term:
        w[i,j] = [x^j] prod_{k=1..i} (a_k x + b_k),  a_k = Ks[k-1] (k<=S) else 0,
        b_k = beta / k**gamma.  Sum_j w[i,j] = prod_k K_app[k] = Z[i].
    """
    beta, gamma, Ks = _unpack(ln_params)
    w = np.zeros((S + N + 1, S + 1))
    for i in range(S + N + 1):
        poly = np.array([1.0])
        for k in range(1, i + 1):
            a = Ks[k - 1] if k <= S else 0.0
            denom = k ** gamma
            b = beta / denom if denom > 0 and np.isfinite(denom) else 0.0
            new = np.zeros(len(poly) + 1)
            new[:len(poly)] += poly * b   # nonspecific at step k
            new[1:] += poly * a           # specific at step k
            poly = new
        for j in range(min(i, S) + 1):
            w[i, j] = poly[j]
    return w

free_ligand = make_free_ligand(MODEL_NAME, mole_fractions)
residual_vector = make_residual_vector(mole_fractions, free_ligand)
