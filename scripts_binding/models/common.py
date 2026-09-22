"""Shared mass-balance and residual functions for binding models."""

import warnings

import numpy as np
from scipy.optimize import brentq


def make_free_ligand(model_name, mole_fractions):
    """Bind the common ligand mass-balance solver to one model."""
    def balance(free, total, protein, params, specific, nonspecific):
        fractions = mole_fractions(free, params, specific, nonspecific)
        bound = np.dot(np.arange(len(fractions)), fractions)
        return free - (total - protein * bound)

    def free_ligand(total, protein, params, specific, nonspecific):
        if total <= 0:
            return 0.0
        try:
            return brentq(
                balance, 0, total,
                args=(total, protein, params, specific, nonspecific),
            )
        except ValueError as exc:
            warnings.warn(
                f"free_ligand({model_name}) failed at L_tot={total:.3e} ({exc}); "
                "using total ligand, so mass balance may be violated.",
                RuntimeWarning,
                stacklevel=2,
            )
            return total

    return free_ligand


def make_residual_vector(mole_fractions, free_ligand):
    """Bind the common observed-minus-calculated residual loop to one model."""
    def residual_vector(params, totals, protein, observations, specific, nonspecific, history):
        residuals = []
        for total, observed in zip(totals, observations):
            free = free_ligand(total, protein, params, specific, nonspecific)
            calculated = mole_fractions(free, params, specific, nonspecific)
            width = max(calculated.size, observed.size)
            calculated = np.pad(calculated, (0, width - calculated.size))
            observed = np.pad(observed, (0, width - observed.size))
            residuals.append(calculated - observed)
        vector = np.concatenate(residuals)
        history.append(float(np.dot(vector, vector)))
        return vector

    return residual_vector


def stepwise_n_params(specific):
    """One nonspecific affinity plus one affinity per specific step."""
    return specific + 1


def stepwise_initial_lnK(specific, Kn=1e3, Ks=1e5):
    """Initial log affinities for sequential-adduct model families."""
    return np.log(np.concatenate(([Kn], np.full(specific, Ks))))


def stepwise_param_labels(specific):
    """Labels for sequential-adduct model families."""
    return ["Kn"] + [f"Ks_{index + 1}" for index in range(specific)]
