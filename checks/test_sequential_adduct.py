"""Synthetic recovery test for sequential_adduct (Shimon 2010)."""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import sequential_adduct as model


def main():
    print("=== sequential_adduct (Shimon) test ===")
    S = 3
    N = 3
    Ks_true = np.array([1e5, 2e4, 5e3])
    Kn_true = 1e3
    P_tot_M = 1e-6
    L_totals_M = np.array([1, 5, 10, 50, 100, 500, 1000]) * 1e-6

    ln_true = np.log(np.concatenate(([Kn_true], Ks_true)))
    F_exps = np.vstack([
        model.mole_fractions(model.free_ligand(L, P_tot_M, ln_true, S, N), ln_true, S, N)
        for L in L_totals_M
    ])

    lnK0 = ln_true + np.log(0.5)
    history = []
    res = least_squares(
        model.residual_vector, lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, history),
        method="lm",
    )
    K_fit = np.exp(res.x)
    rel_err = np.abs(K_fit - np.exp(ln_true)) / np.exp(ln_true)
    print(f"Fit cost: {res.cost:.3e}, max rel err: {rel_err.max():.4e}")
    assert rel_err.max() < 0.01, f"FAIL: max rel err {rel_err.max()}"
    print("PASS: parameters recovered within 1%")

    # Shimon Eq. 1 structural check: I_{j+1}/I_j = Kn[S] when j > S
    L_free = model.free_ligand(50e-6, P_tot_M, ln_true, S, N)
    F = model.mole_fractions(L_free, ln_true, S, N)
    # For j > S (=3), ratio I_{j+1}/I_j should equal Kn * L_free
    if F[S + 1] > 0 and F[S + 2] > 0:
        ratio = F[S + 2] / F[S + 1]
        expected = Kn_true * L_free
        rel = abs(ratio - expected) / expected
        print(f"Shimon Eq. 1 check: I_{S+2}/I_{S+1}={ratio:.4e}, Kn[S]={expected:.4e}, rel err={rel:.2e}")
        assert rel < 1e-9, f"Eq. 1 identity broken: {rel}"
        print("PASS: Shimon Eq. 1 ratio identity holds")


if __name__ == "__main__":
    main()
