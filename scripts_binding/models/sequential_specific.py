"""Adair-type stepwise specific binding (no NSB). See MODELS.md §3.1."""
import numpy as np
from scipy.optimize import brentq

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
        n_out = max(Fc.size, F_exp.size)
        if Fc.size < n_out:
            Fc = np.pad(Fc, (0, n_out - Fc.size), "constant")
        if F_exp.size < n_out:
            F_exp = np.pad(F_exp, (0, n_out - F_exp.size), "constant")
        res_list.append(Fc - F_exp)
    vec = np.concatenate(res_list)
    ssr_history.append(float(np.dot(vec, vec)))
    return vec
