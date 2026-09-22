"""Stepwise-specific + Poisson NSB (hybrid, no paper). See MODELS.md §3.3, §8.4."""
from math import factorial

import numpy as np
from scipy.optimize import brentq

MODEL_NAME = "stochastic_adduct"


def mole_fractions(L_free_M, ln_params, S, N):
    params = np.exp(np.asarray(ln_params))
    Kn = params[0]
    Ks = params[1:]

    Z = np.zeros(S + N + 1)
    Z[0] = 1.0
    for i in range(1, S + N + 1):
        z_sum = 0.0
        for j in range(min(i, S) + 1):
            prod_ks = np.prod(Ks[:j]) if j > 0 else 1.0
            m = i - j
            z_sum += prod_ks * (Kn ** m) / factorial(m)
        Z[i] = z_sum

    alpha = Z * (L_free_M ** np.arange(S + N + 1))
    return alpha / alpha.sum()


def partition_terms(ln_params, S, N):
    """w[i,j] = unnormalized (j specific, i-j nonspecific) contribution to peak i."""
    p = np.exp(np.asarray(ln_params))
    Kn, Ks = p[0], p[1:]
    prod_ks = np.concatenate(([1.0], np.cumprod(Ks)))
    w = np.zeros((S + N + 1, S + 1))
    for i in range(S + N + 1):
        for j in range(min(i, S) + 1):
            m = i - j
            w[i, j] = prod_ks[j] * (Kn ** m) / factorial(m)
    return w


def n_params(S):
    return S + 1


def initial_lnK(S, Kn=1e3, Ks=1e5):
    return np.log(np.concatenate(([Kn], np.full(S, Ks))))


def param_labels(S):
    return ["Kn"] + [f"Ks_{i+1}" for i in range(S)]


def _balance(L_free_M, L_tot_M, P_tot_M, ln_params, S, N):
    F = mole_fractions(L_free_M, ln_params, S, N)
    return L_free_M - (L_tot_M - P_tot_M * np.dot(np.arange(len(F)), F))


def free_ligand(L_tot_M, P_tot_M, ln_params, S, N):
    if L_tot_M <= 0:
        return 0.0
    try:
        return brentq(_balance, 0, L_tot_M, args=(L_tot_M, P_tot_M, ln_params, S, N))
    except ValueError as e:
        import warnings
        warnings.warn(
            f"free_ligand({MODEL_NAME}): brentq failed at L_tot={L_tot_M:.3e} ({e}); "
            f"returning L_tot — mass balance may be violated.",
            RuntimeWarning, stacklevel=2,
        )
        return L_tot_M


def residual_vector(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history):
    res_list = []
    for L_tot, F_exp in zip(L_totals_M, F_exps):
        Lf = free_ligand(L_tot, P_tot_M, ln_params, S, N)
        Fc = mole_fractions(Lf, ln_params, S, N)
        res_list.append(Fc - F_exp)
    vec = np.concatenate(res_list)
    ssr_history.append(float(np.dot(vec, vec)))
    return vec
