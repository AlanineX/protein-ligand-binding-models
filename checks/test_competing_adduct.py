"""Synthetic recovery test for the Guan constant-NSB model."""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import competing_adduct as model
from scripts_binding.models import occupancy_decay


def synth(S, N, beta, Ks_true, P_tot_M, L_totals_M):
    ln_params = np.concatenate(([np.log(beta)], np.log(Ks_true)))
    F_list = []
    for L_tot in L_totals_M:
        L_free = model.free_ligand(L_tot, P_tot_M, ln_params, S, N)
        F_list.append(model.mole_fractions(L_free, ln_params, S, N))
    return F_list, ln_params


def main():
    print("=== Guan constant-NSB model test ===")

    S = 3
    N = 3
    beta_true = 2e3
    Ks_true = np.array([1e5, 2e4, 5e3])
    P_tot_M = 1e-6
    L_totals_M = np.array([1, 5, 10, 50, 100, 500, 1000]) * 1e-6

    F_exps, ln_true = synth(S, N, beta_true, Ks_true, P_tot_M, L_totals_M)
    lnK0 = ln_true.copy()
    lnK0[0] += np.log(0.5)
    lnK0[1:] += np.log(0.7)

    history = []
    res = least_squares(
        model.residual_vector, lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, history),
        method="lm",
    )
    beta_fit = np.exp(res.x[0])
    Ks_fit = np.exp(res.x[1:])
    rel_err = np.abs(np.r_[beta_fit, Ks_fit] - np.r_[beta_true, Ks_true]) / np.r_[beta_true, Ks_true]

    print(f"Fit converged: {res.success}, cost = {res.cost:.3e}")
    print(f"Fitted beta={beta_fit:.3e}, Ks={Ks_fit}")
    assert rel_err.max() < 0.01, "FAIL: constant-NSB parameters were not recovered"
    print("PASS: parameters recovered within 1%")

    ln_powerlaw_g0 = np.concatenate(([np.log(beta_true), 0.0], np.log(Ks_true)))
    for L_free in np.array([0.1, 1, 10, 100]) * 1e-6:
        F_const = model.mole_fractions(L_free, ln_true, S, N)
        F_power = occupancy_decay.mole_fractions(L_free, ln_powerlaw_g0, S, N)
        assert np.allclose(F_const, F_power, rtol=1e-12, atol=1e-14)
    print("PASS: constant model equals power-law model at gamma=0")


if __name__ == "__main__":
    main()
