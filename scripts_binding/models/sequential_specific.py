"""Adair-type stepwise specific binding (no NSB). See MODELS.md §3.1."""
import numpy as np

from .common import make_free_ligand, make_residual_vector

MODEL_NAME = "sequential_specific"


def mole_fractions(L_free_M, ln_params, S, N):
    K = np.exp(np.asarray(ln_params))
    n = len(K)
    a = np.ones(n + 1)
    for i in range(1, n + 1):
        a[i] = (L_free_M ** i) * np.prod(K[:i])
    return a / a.sum()


def n_params(S):
    return S


def initial_lnK(S, Kn=None, Ks=1e5):
    return np.full(S, np.log(Ks))


def param_labels(S):
    return [f"Ks_{i+1}" for i in range(S)]

free_ligand = make_free_ligand(MODEL_NAME, mole_fractions)
residual_vector = make_residual_vector(mole_fractions, free_ligand)
