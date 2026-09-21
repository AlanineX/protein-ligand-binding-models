"""Synthetic recovery test for sequential_specific (Adair)."""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import sequential_specific as model


def main():
    print("=== sequential_specific test ===")
    S = 3
    N = 0
    Ks_true = np.array([1e5, 4e4, 1e4])  # cooperativity pattern
    P_tot_M = 1e-6
    L_totals_M = np.array([1, 5, 10, 50, 100, 500, 1000]) * 1e-6

    ln_true = np.log(Ks_true)
    F_exps = np.vstack([
        model.mole_fractions(model.free_ligand(L, P_tot_M, ln_true, S, N), ln_true, S, N)
        for L in L_totals_M
    ])

    lnK0 = ln_true + np.log(0.3)
    history = []
    res = least_squares(
        model.residual_vector, lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, history),
        method="lm",
    )
    K_fit = np.exp(res.x)
    rel_err = np.abs(K_fit - Ks_true) / Ks_true
    print(f"Fit cost: {res.cost:.3e}, max rel err: {rel_err.max():.4e}")
    print("True Ks:", Ks_true)
    print("Fit Ks :", K_fit)
    assert rel_err.max() < 0.01, f"FAIL: max rel err {rel_err.max()}"
    print("PASS: parameters recovered within 1%")


if __name__ == "__main__":
    main()
