"""Row-resample bootstrap for any model in REGISTRY.

Skipped silently for N_data < 5 (resamples become degenerate). Failed fits
within the bootstrap loop are counted but excluded from quantile reporting.
Returns Jacobian-based and bootstrap-based summaries side-by-side, so the
caller can flag disagreement.
"""
import warnings

import numpy as np

from ..models.metadata import parameter_values_from_optimizer
from .config import RunConfig
from .fitting import _least_squares_fit

_MIN_N_DATA = 5


def bootstrap_fit(model, L_totals, F_exps, P_tot, S, N, n_boot=200, seed=None, cfg=None):
    """Row-resample (with replacement) bootstrap.

    Returns dict {param_label: {p50, lo95, hi95, n_success}}, or None if
    N_data < _MIN_N_DATA. `param_label` follows model.param_labels(S);
    binomial returns a scalar Ks; gamma is reported in linear space.
    """
    if len(L_totals) < _MIN_N_DATA:
        return None
    if cfg is None:
        cfg = RunConfig(input_unit="M", output_unit="uM")

    rng = np.random.default_rng(seed)
    n = len(L_totals)
    target_len = S + N + 1
    # NaN-pad missing cells so 'no measurement' is distinguished from zero
    F_padded_full = np.array([
        np.concatenate([f[:target_len],
                         np.full(max(target_len - len(f), 0), np.nan)])[:target_len]
        for f in F_exps
    ])
    labels = model.param_labels(S)
    samples = {lbl: [] for lbl in labels}
    n_success = 0
    failures = {}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Lb = L_totals[idx]
        Fb = F_padded_full[idx]
        try:
            _fit, lnK_opt, _history, _transform, _n_starts = _least_squares_fit(
                model, model.MODEL_NAME, model.initial_lnK(S),
                Lb, P_tot, Fb, S, N, cfg, verbose=0, max_nfev=2000,
            )
            for lbl, val in zip(labels, parameter_values_from_optimizer(labels, lnK_opt)):
                samples[lbl].append(val)
            n_success += 1
        except Exception as exc:  # noqa: BLE001 - retain successful bootstrap draws
            name = type(exc).__name__
            failures[name] = failures.get(name, 0) + 1

    if failures:
        warnings.warn(f"Bootstrap skipped failed fits: {failures}", RuntimeWarning, stacklevel=2)

    out = {}
    for lbl, vals in samples.items():
        if not vals:
            out[lbl] = {"p50": np.nan, "lo95": np.nan, "hi95": np.nan, "n_success": 0}
            continue
        v = np.array(vals)
        out[lbl] = {
            "p50": float(np.median(v)),
            "lo95": float(np.percentile(v, 2.5)),
            "hi95": float(np.percentile(v, 97.5)),
            "n_success": len(v),
        }
    out["__meta__"] = {"n_boot": n_boot, "n_success": n_success, "n_data": n}
    return out


def format_kd_ci(boot, label):
    """For a (Ka-domain) parameter `label`, return a {p50,lo95,hi95}_uM dict
    converting boot Ka samples to Kd µM. `label` must not be 'gamma'."""
    if boot is None or label not in boot:
        return {"p50_Kd_uM": np.nan, "lo95_Kd_uM": np.nan, "hi95_Kd_uM": np.nan, "n_success": 0}
    s = boot[label]
    if s["n_success"] == 0 or s["p50"] <= 0:
        return {"p50_Kd_uM": np.nan, "lo95_Kd_uM": np.nan, "hi95_Kd_uM": np.nan, "n_success": s["n_success"]}
    # Kd = 1/Ka in M, then ×1e6 → µM. Note: high Ka → low Kd, so percentile bounds invert.
    p50 = 1e6 / s["p50"]
    lo = 1e6 / s["hi95"] if s["hi95"] > 0 else np.nan
    hi = 1e6 / s["lo95"] if s["lo95"] > 0 else np.nan
    return {"p50_Kd_uM": p50, "lo95_Kd_uM": lo, "hi95_Kd_uM": hi, "n_success": s["n_success"]}
