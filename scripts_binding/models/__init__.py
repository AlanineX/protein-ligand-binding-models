"""Binding-model registry."""

from . import (
    competing_adduct,
    occupancy_decay,
    sequential_adduct,
    sequential_specific,
    shared_site,
    stochastic_adduct,
)

REGISTRY = {
    sequential_specific.MODEL_NAME: sequential_specific,
    sequential_adduct.MODEL_NAME: sequential_adduct,
    competing_adduct.MODEL_NAME: competing_adduct,
    stochastic_adduct.MODEL_NAME: stochastic_adduct,
    occupancy_decay.MODEL_NAME: occupancy_decay,
    shared_site.MODEL_NAME: shared_site,
}


def deconvolve_terms(F_vals, weights, S):
    """Split each apparent peak F_i into (j specific, i-j nonspecific) parts.

    Generic across any adduct model exposing ``partition_terms`` -> weights[i,j].
    Returns (frac_within, contrib): frac_within[i,j] sums to 1 over j (the
    composition of peak i), contrib[i,j] = F_vals[i] * frac_within[i,j].
    """
    import numpy as np

    num_species = len(F_vals)
    frac_within = np.zeros((num_species, S + 1))
    contrib = np.zeros((num_species, S + 1))
    for i in range(num_species):
        max_j = min(i, S)
        denom = weights[i, :max_j + 1].sum()
        if denom <= 0:
            continue
        frac = weights[i, :max_j + 1] / denom
        frac_within[i, :max_j + 1] = frac
        contrib[i, :max_j + 1] = F_vals[i] * frac
    return frac_within, contrib
