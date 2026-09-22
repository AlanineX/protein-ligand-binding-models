"""Daubenfeld 2006 binomial-specific + Poisson NSB. See MODELS.md §3.4."""
from math import comb, factorial

import numpy as np

from .common import make_free_ligand, make_residual_vector

MODEL_NAME = "shared_site"


def mole_fractions(L_free_M, ln_params, S, N):
    params = np.exp(np.asarray(ln_params))
    Kn = params[0]
    Ks = params[1]

    Z = np.zeros(S + N + 1)
    Z[0] = 1.0
    for i in range(1, S + N + 1):
        z_sum = 0.0
        for j in range(min(i, S) + 1):
            m = i - j
            z_sum += comb(S, j) * (Ks ** j) * (Kn ** m) / factorial(m)
        Z[i] = z_sum

    alpha = Z * (L_free_M ** np.arange(S + N + 1))
    return alpha / alpha.sum()


def partition_terms(ln_params, S, N):
    """w[i,j] = unnormalized (j specific, i-j nonspecific) contribution to peak i."""
    p = np.exp(np.asarray(ln_params))
    Kn, Ks = p[0], p[1]
    w = np.zeros((S + N + 1, S + 1))
    for i in range(S + N + 1):
        for j in range(min(i, S) + 1):
            m = i - j
            w[i, j] = comb(S, j) * (Ks ** j) * (Kn ** m) / factorial(m)
    return w


def n_params(_S):
    """Return the two shared affinities: nonspecific Kn and specific Ks."""
    return 2


def initial_lnK(S, Kn=1e3, Ks=1e5):
    return np.log([Kn, Ks])


def param_labels(S):
    return ["Kn", "Ks"]

free_ligand = make_free_ligand(MODEL_NAME, mole_fractions)
residual_vector = make_residual_vector(mole_fractions, free_ligand)
