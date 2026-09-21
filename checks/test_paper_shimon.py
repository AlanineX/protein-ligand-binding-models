"""Shimon 2010 Table 1 mass-balance check.

Published Table 1 lists corrected C_0, C_1, C_2 per [ADP] row for CK dimer
(2 specific ADP sites). Because C_i is the specific-only population at
stoichiometry i, mass balance requires
    C_0 + C_1 + C_2 = [P]_tot = 4 µM
at every row. This is a cheap sanity check on the paper's transcription
and on our understanding of what C_i means.
"""
import numpy as np

# Shimon Table 1 (p. 1647): [ADP]/µM, C_0, C_1, C_2 in µM, [P]_tot = 4 µM
TABLE_1 = [
    (0,   4.00, 0.00, 0.00),
    (5,   3.28, 0.72, 0.00),
    (10,  2.31, 1.38, 0.31),
    (15,  2.28, 1.51, 0.21),
    (20,  1.71, 1.81, 0.48),
    (30,  1.09, 1.89, 1.02),
    (40,  0.87, 1.71, 1.42),
    (50,  0.29, 1.50, 2.22),
    (100, 0.13, 1.41, 2.46),
]

P_TOT_UM = 4.00


def main():
    print("=== Shimon Table 1 mass-balance check ===")
    max_err = 0.0
    for L, c0, c1, c2 in TABLE_1:
        total = c0 + c1 + c2
        err = abs(total - P_TOT_UM)
        print(f"  [ADP]={L:>4} µM  C0+C1+C2 = {total:.2f} µM  (diff = {total - P_TOT_UM:+.3f})")
        max_err = max(max_err, err)
    # Paper rounds each C_i to 2 decimals, so tolerance accordingly
    assert max_err < 0.05, f"FAIL: max row-sum deviation = {max_err:.3f} µM"
    print(f"\nPASS: all rows sum to {P_TOT_UM} µM within 0.05 µM (paper rounding)")


if __name__ == "__main__":
    main()
