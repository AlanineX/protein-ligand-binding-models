"""Stepwise specific binding with a constant nonspecific association term."""
import numpy as np
from scipy.optimize import brentq

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


def n_params(S):
    return S + 1


def initial_lnK(S, Kn=1e3, Ks=1e5):
    return np.concatenate(([np.log(Kn)], np.log(np.full(S, Ks))))


def param_labels(S):
    return ["beta"] + [f"Ks_{i+1}" for i in range(S)]


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
