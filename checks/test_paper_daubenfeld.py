"""Daubenfeld 2006 Table 1 regression.

Fit shared_site against Daubenfeld's CK+ADP R_i table
and check K_s recovery vs paper's abstract K_{1,diss,ADP} = 11.8 ± 1.5 µM.

Paper uses sequential (macroscopic) dissociation constant K_{1,diss};
our K_s is the microscopic site association constant. For S=2 equivalent
independent sites: K_s = 1 / (2 · K_{1,diss}) ≈ 4.24e4 M^-1 (see
11_notes.md §9).
"""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import shared_site as model


# Daubenfeld Table 1 (p. 1243): CK + ADP, [P]_tot = 4 µM, R_i = I_i/I_0
# Columns: [ADP]_total (µM), R_1, R_2, R_3, R_4
TABLE_1 = [
    (1,  0.06, 0.0,  0.0,  0.0),
    (2,  0.14, 0.01, 0.0,  0.0),
    (4,  0.31, 0.02, 0.0,  0.0),
    (6,  0.45, 0.09, 0.0,  0.0),
    (8,  0.56, 0.14, 0.0,  0.0),
    (10, 0.74, 0.22, 0.0,  0.0),
    (12, 0.80, 0.29, 0.02, 0.0),
    (16, 1.13, 0.57, 0.10, 0.0),
    (20, 1.33, 0.70, 0.16, 0.0),
    (40, 2.73, 2.73, 0.87, 0.24),
    (60, 3.61, 5.02, 2.20, 0.90),
]

S = 2
N = 4  # covers R_4
P_TOT_M = 4e-6


def _table_to_fexp():
    L_totals = []
    F_exps = []
    for row in TABLE_1:
        ratio = np.array([1.0, row[1], row[2], row[3], row[4]])  # R_0..R_4
        F = ratio / ratio.sum()
        F_pad = np.zeros(S + N + 1)
        F_pad[: len(F)] = F
        L_totals.append(row[0] * 1e-6)
        F_exps.append(F_pad)
    return np.array(L_totals), np.array(F_exps)


def main():
    print("=== Daubenfeld Table 1 regression ===")
    L_totals_M, F_exps = _table_to_fexp()

    lnK0 = np.log([1e3, 5e4])  # Kn=1e3 (Kd~1mM), Ks=5e4 (Kd~20uM)
    history = []
    res = least_squares(
        model.residual_vector, lnK0,
        args=(L_totals_M, P_TOT_M, F_exps, S, N, history),
        method="lm",
    )
    Kn_fit, Ks_fit = np.exp(res.x)
    # Convert micro Ks → macroscopic K_{1,diss} for S=2 sites
    K1_diss_uM = 1e6 / (2 * Ks_fit)
    Kn_Kd_uM = 1e6 / Kn_fit

    print(f"Fit cost: {res.cost:.3e}")
    print(f"Fitted Ks (micro): {Ks_fit:.3e} M^-1 → K_{{1,diss}} = {K1_diss_uM:.2f} µM")
    print(f"Fitted Kn: {Kn_fit:.3e} M^-1 → Kd_n = {Kn_Kd_uM:.1f} µM")

    # Paper abstract: K_{1,diss,ADP} = 11.8 ± 1.5 µM (averaged over 8 high-conc rows)
    # We fit all 11 rows globally, so allow somewhat looser tolerance.
    paper_K1 = 11.8
    tol = 4.0  # µM — within ~30% of paper mean
    assert abs(K1_diss_uM - paper_K1) < tol, (
        f"FAIL: K_{{1,diss}} = {K1_diss_uM:.2f} µM, paper = {paper_K1} µM, tol = {tol} µM"
    )
    print(f"PASS: K_{{1,diss}} within {tol} µM of paper value {paper_K1} µM")


if __name__ == "__main__":
    main()
