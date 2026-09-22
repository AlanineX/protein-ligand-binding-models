"""Stepwise specific binding with a constant nonspecific association term."""
import numpy as np

from .common import make_free_ligand, make_residual_vector, stepwise_n_params

MODEL_NAME = "competing_adduct"


def _unpack(ln_params):
    arr = np.asarray(ln_params)
    beta = np.exp(arr[0])           # NSB amplitude, K_a units (M^-1)
    Ks = np.exp(arr[1:])            # specific Ka per site, K_a units
    return beta, Ks


def mole_fractions(L_free_M, ln_params, S, N):
    beta, Ks = _unpack(ln_params)
    n_max = S + N

    K_app = np.zeros(n_max + 1)
    for k in range(1, n_max + 1):
        K_app[k] = (Ks[k - 1] if k <= S else 0.0) + beta

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


def partition_terms(ln_params, S, N):
    """w[i,j] = unnormalized (j specific, i-j nonspecific) contribution to peak i.

    The apparent model has no explicit j-sum; recover it from the elementary-
    symmetric expansion of the per-step product:
        w[i,j] = [x^j] prod_{k=1..i} (a_k x + b_k),  a_k = Ks[k-1] (k<=S) else 0,
        b_k = beta (constant nonspecific).  Sum_j w[i,j] = prod_k K_app[k] = Z[i].
    """
    beta, Ks = _unpack(ln_params)
    w = np.zeros((S + N + 1, S + 1))
    for i in range(S + N + 1):
        poly = np.array([1.0])  # coeffs of x^0..x^deg
        for k in range(1, i + 1):
            a = Ks[k - 1] if k <= S else 0.0
            new = np.zeros(len(poly) + 1)
            new[:len(poly)] += poly * beta   # nonspecific at step k (x^0)
            new[1:] += poly * a              # specific at step k (x^1)
            poly = new
        for j in range(min(i, S) + 1):
            w[i, j] = poly[j]
    return w


n_params = stepwise_n_params


def initial_lnK(S, Kn=1e3, Ks=1e5):
    return np.concatenate(([np.log(Kn)], np.log(np.full(S, Ks))))


def param_labels(S):
    return ["beta"] + [f"Ks_{i+1}" for i in range(S)]

free_ligand = make_free_ligand(MODEL_NAME, mole_fractions)
residual_vector = make_residual_vector(mole_fractions, free_ligand)
