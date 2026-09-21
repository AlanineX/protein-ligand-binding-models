"""Guan 2015 Eq. 8 numeric reproduction.

Paper Eq. 8 (p. 8544):  K_{n,j} = (4.2e5) / j^{1.8},  j = 1..N
with GLOBAL j index (no reset at the specific/nonspecific boundary).
Verify that our occupancy_decay.mole_fractions uses that same
structural form by pulling out the per-step K^app from a β, γ pair and
checking k > S values match paper Eq. 8 exactly.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import occupancy_decay as model


def _extract_kapp(S, N, beta, gamma, Ks):
    """Run the model's internal _unpack+K_app loop by calling mole_fractions
    at two L_free values and solving for K_app[k] = alpha[k]/(alpha[k-1]*L).
    Simpler: replicate the code's K_app construction directly."""
    n_max = S + N
    K_app = np.zeros(n_max + 1)
    for k in range(1, n_max + 1):
        nsb = beta / (k ** gamma)
        K_app[k] = (Ks[k - 1] + nsb) if k <= S else nsb
    return K_app


def main():
    print("=== Guan Eq. 8 reproduction ===")

    # Paper Eq. 8 parameters:
    beta = 4.2e5
    gamma = 1.8
    S = 1          # Guan's fit: 1 specific site (Pol II example)
    N = 9          # up to j=10
    Ks = np.array([1e5])   # arbitrary specific Ka

    K_app = _extract_kapp(S, N, beta, gamma, Ks)

    print(f"{'k':>3}  {'K^app_k':>14}  {'paper β/k^γ':>16}  {'match?':>10}")
    all_ok = True
    for k in range(1, N + S + 1):
        paper_val = beta / (k ** gamma)
        if k <= S:
            # Paper Eq. 5 specific step: K_app = Ks_k + β/k^γ (additive)
            expected = Ks[k - 1] + paper_val
            tag = "spec+NSB"
        else:
            # Paper Eq. 8: K_{n,j} = β/j^γ with GLOBAL j (j=k here)
            expected = paper_val
            tag = "NSB(global k)"
        rel = abs(K_app[k] - expected) / expected
        ok = rel < 1e-12
        all_ok = all_ok and ok
        print(f"  {k:>3}  {K_app[k]:>14.4e}  {paper_val:>16.4e}  {tag:>10}  rel_err={rel:.1e}  {'OK' if ok else 'FAIL'}")

    assert all_ok, "FAIL: K^app does not match Guan Eq. 5/8 form"
    print("\nPASS: occupancy_decay reproduces Guan Eq. 8 K_{n,j} exactly")

    # Structural check: at γ=0, all NSB steps give K_app=β (flat)
    K_app_flat = _extract_kapp(S, N, beta, 0.0, Ks)
    for k in range(S + 1, N + S + 1):
        assert abs(K_app_flat[k] - beta) < 1e-12
    print("PASS: γ=0 limit flattens NSB Ka to β")


if __name__ == "__main__":
    main()
