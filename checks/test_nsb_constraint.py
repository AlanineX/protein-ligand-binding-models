"""Regression checks for the default weaker-NSB optimizer transform."""
import os
import sys
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.core.fitting import _NsbWeakerThanSpecificTransform, _least_squares_fit
from scripts_binding.models import REGISTRY


NSB_MODELS = [
    "sequential_adduct",
    "competing_adduct",
    "stochastic_adduct",
    "occupancy_decay",
    "shared_site",
]


def test_transform_enforces_weaker_nsb():
    S = 7
    for model_name in NSB_MODELS:
        transform = _NsbWeakerThanSpecificTransform(model_name, S)
        theta = np.linspace(-2.0, 2.0, transform.n)
        raw = transform.to_raw(theta)

        if model_name == "occupancy_decay":
            beta_ln = raw[0]
            gamma = raw[1]
            specific_ln = raw[2:]
            assert gamma >= 0.0
            assert np.all(specific_ln > beta_ln)
        elif model_name == "shared_site":
            kn_ln, ks_ln = raw
            assert ks_ln > kn_ln
        else:
            kn_ln = raw[0]
            specific_ln = raw[1:]
            assert np.all(specific_ln > kn_ln)


def test_transform_round_trip_preserves_valid_raw_values():
    S = 4
    valid_raw_by_model = {
        "sequential_adduct": np.log([1e3, 1e5, 8e4, 5e4, 2e4]),
        "competing_adduct": np.log([1e3, 1e5, 8e4, 5e4, 2e4]),
        "stochastic_adduct": np.log([1e3, 1e5, 8e4, 5e4, 2e4]),
        "occupancy_decay": np.array([np.log(1e3), 0.5, *np.log([1e5, 8e4, 5e4, 2e4])]),
        "shared_site": np.log([1e3, 1e5]),
    }
    for model_name, raw in valid_raw_by_model.items():
        transform = _NsbWeakerThanSpecificTransform(model_name, S)
        theta = transform.to_theta(raw)
        raw_round_trip = transform.to_raw(theta)
        assert np.allclose(raw_round_trip, raw, rtol=1e-10, atol=1e-10)


def test_constrained_fit_can_fallback_to_unconstrained():
    model_name = "sequential_adduct"
    model = REGISTRY[model_name]
    S, N = 2, 1
    P_tot = 1e-6
    L_totals = np.array([0.0, 20e-6, 50e-6, 100e-6])
    truth = np.log([1e3, 1e5, 5e4])
    F_exps = np.vstack([
        model.mole_fractions(model.free_ligand(L, P_tot, truth, S, N), truth, S, N)
        for L in L_totals
    ])
    cfg = SimpleNamespace(
        constrain_nsb_weaker_than_specific=True,
        nsb_constraint_multistart_n=2,
        nsb_constraint_max_nfev=1,
        nsb_constraint_fallback_to_unconstrained=True,
        nsb_constraint_log_margin=1e-6,
    )
    fit, lnK_opt, _history, transform, n_starts = _least_squares_fit(
        model, model_name, model.initial_lnK(S),
        L_totals, P_tot, F_exps, S, N, cfg,
    )
    assert transform is None
    assert n_starts == 3
    assert fit.binding_constraint_fallback_used is True
    assert fit.binding_fit_algorithm == "least_squares_after_constraint_fallback"
    assert np.all(np.isfinite(lnK_opt))


def main():
    test_transform_enforces_weaker_nsb()
    test_transform_round_trip_preserves_valid_raw_values()
    test_constrained_fit_can_fallback_to_unconstrained()
    print("PASS: default weaker-NSB optimizer transform")


if __name__ == "__main__":
    main()
