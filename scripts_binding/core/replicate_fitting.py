"""Weighted fitting of pooled replicate bound-state fractions."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..models import REGISTRY
from ..models.metadata import (
    base_model_name,
    is_dimensionless_param,
    is_sequential_specific_model,
    ka_kd_from_optimizer,
)
from .fitting import (
    _constrain_nsb_for_model,
    _constrained_initial_raws,
    _nan_pad,
    _nsb_constraint_fallback_to_unconstrained,
    _nsb_constraint_max_nfev,
    _NsbWeakerThanSpecificTransform,
    _param_bounds,
    _resolve_S_N,
    _trim_low_pop_species,
    load_binding_csv,
)

# Floor for the per-cell SE so 1/σ doesn't blow up when reps agree closely.
# F_i is a mole fraction in [0, 1]. The floor of 1% reflects the realistic
# noise on a single-rep nMS mole-fraction measurement (replicate-derived SE
# from n=3 underestimates true noise when the 3 reps happen to agree by
# chance); the regression's own residual variance recovers the absolute
# scale via absolute_sigma=False semantics in the covariance step.
_SE_FLOOR = 0.01


def _build_union_L_grid(rep_data, atol_M=1e-9):
    """Union of all L_total values across reps with tolerance matching.

    Replicates can use different concentration grids (each titration is
    independent and may sample different [L] points). We build the union
    so every measured cell contributes to the fit; cells where < 2 reps
    overlap have NaN σ and zero weight.
    """
    all_L = np.concatenate([rep[0] for rep in rep_data])
    # Sort and de-dup with tolerance
    L_sorted = np.sort(all_L)
    L_union = [L_sorted[0]]
    for L in L_sorted[1:]:
        if L - L_union[-1] > atol_M:
            L_union.append(L)
    return np.array(L_union)


def _project_rep_onto_union(L_rep, F_rep, L_union, num_species, atol_M=1e-9):
    """Place rep's F values into a (n_union_L, num_species) array,
    NaN where rep didn't measure at that L."""
    F_padded = _nan_pad(F_rep, num_species)
    out = np.full((len(L_union), num_species), np.nan)
    for i, L in enumerate(L_rep):
        # Find matching union index within tolerance
        diff = np.abs(L_union - L)
        j = int(np.argmin(diff))
        if diff[j] <= atol_M:
            out[j] = F_padded[i]
    return out


def _pool_replicates(rep_csv_paths, cfg, model_name, S_override=None):
    """Pool measurements on their observed concentration grid and compute means and SEs."""
    rep_data = []
    for path in rep_csv_paths:
        _df, L_totals_M, I_cols, F_exps = load_binding_csv(path, cfg)
        if is_sequential_specific_model(model_name):
            F_exps, I_cols = _trim_low_pop_species(F_exps, I_cols, cfg.min_species_frac)
        rep_data.append((np.asarray(L_totals_M), np.asarray(F_exps), list(I_cols)))

    if not rep_data:
        raise ValueError("No replicate CSVs supplied")

    # Determine model dimensions from the widest rep.
    max_i_per_rep = [len(I) - 1 for _, _, I in rep_data]
    max_i = max(max_i_per_rep)
    S_eff, N_eff = _resolve_S_N(cfg, model_name, max_i, S_override)
    num_species_model = S_eff + N_eff + 1

    # Union L grid (sorted, with tolerance matching).
    L_union = _build_union_L_grid(rep_data)

    # Project each rep onto the union grid; NaN where rep didn't measure.
    projected = np.array([
        _project_rep_onto_union(L_rep, F_rep, L_union, num_species_model)
        for (L_rep, F_rep, _) in rep_data
    ])
    n_reps = projected.shape[0]

    # Per-cell n_valid, mean, and SE/√n_valid across the reps that actually
    # measured at each cell. Use ddof=1; require ≥2 valid reps for SE.
    F_mean = np.nanmean(projected, axis=0)

    F_se = np.full_like(F_mean, np.nan)
    n_L = projected.shape[1]
    for c in range(n_L):
        for s in range(num_species_model):
            col = projected[:, c, s]
            valid = col[~np.isnan(col)]
            if len(valid) >= 2:
                F_se[c, s] = np.std(valid, ddof=1) / np.sqrt(len(valid))
    # Floor to avoid 1/0 blow-up when reps agree exactly. NaN preserved
    # for cells with < 2 valid reps so the residual masks them out.
    finite = np.isfinite(F_se)
    F_se[finite] = np.maximum(F_se[finite], _SE_FLOOR)

    return L_union, F_mean, F_se, num_species_model, S_eff, N_eff, n_reps


def _weighted_residual(ln_params, L_totals, P_tot, F_mean, F_se, model, S, N,
                       history):
    """Weighted residual vector: (F_calc - F_exp) / σ, with NaN cells → 0.

    Cells where F_se is NaN (only 0 or 1 valid replicate) carry zero weight,
    so they neither contribute to chi² nor to the Jacobian.
    """
    res_blocks = []
    for L_tot, F_exp_row, F_se_row in zip(L_totals, F_mean, F_se):
        Lf = model.free_ligand(L_tot, P_tot, ln_params, S, N)
        Fc = model.mole_fractions(Lf, ln_params, S, N)
        weighted = (Fc - F_exp_row) / F_se_row
        weighted = np.where(np.isnan(weighted), 0.0, weighted)
        res_blocks.append(weighted)
    vec = np.concatenate(res_blocks)
    history.append(float(np.dot(vec, vec)))
    return vec


def fit_replicates_weighted(rep_csv_paths, model_name, cfg, S_override=None):
    """Fit one weighted model to pooled replicate fractions."""
    model = REGISTRY[base_model_name(model_name)]

    L_ref, F_mean, F_se, num_species_model, S_eff, N_eff, n_reps = (
        _pool_replicates(rep_csv_paths, cfg, model_name, S_override)
    )
    F_mean = _nan_pad(F_mean, num_species_model)
    F_se = _nan_pad(F_se, num_species_model)

    # Initial parameters and bounds
    lnK0 = model.initial_lnK(S_eff)

    history = []
    transform = None
    fit_constraint_fallback_used = False
    fit_algorithm = "least_squares"
    fit_constraint = ""
    constrained_attempt_statuses = []
    constrained_attempt_max_nfev = np.nan
    if _constrain_nsb_for_model(cfg, model_name):
        transform = _NsbWeakerThanSpecificTransform(
            model_name, S_eff, getattr(cfg, "nsb_constraint_log_margin", 1e-6),
        )
        theta_bounds = transform.theta_bounds()
        constrained_attempt_max_nfev = _nsb_constraint_max_nfev(cfg)
        residual_len = len(_weighted_residual(
            lnK0, L_ref, cfg.p_total_m, F_mean, F_se, model, S_eff, N_eff, [],
        ))

        def constrained_residual(theta):
            raw = transform.to_raw(theta)
            lo, hi = _param_bounds(model_name, S_eff)
            if not transform.is_raw_valid(raw):
                violation = np.maximum(lo - raw, 0.0) + np.maximum(raw - hi, 0.0)
                penalty_scale = 1e6 * (1.0 + float(np.sum(violation)))
                penalty = np.full(residual_len, penalty_scale, dtype=float)
                history.append(float(np.dot(penalty, penalty)))
                return penalty
            return _weighted_residual(
                raw, L_ref, cfg.p_total_m, F_mean, F_se, model, S_eff, N_eff, history,
            )

        starts = []
        max_starts = max(1, int(getattr(cfg, "nsb_constraint_multistart_n", 24)))
        for raw in [lnK0, *_constrained_initial_raws(model_name, S_eff, max_starts)]:
            try:
                theta = transform.to_theta(raw)
                lo, hi = theta_bounds
                theta = np.clip(theta, lo + 1e-9, hi - 1e-9)
                if transform.is_raw_valid(transform.to_raw(theta)):
                    starts.append(theta)
            except (ValueError, OverflowError) as exc:
                print(f"[NSB start] skipped invalid initialization: {exc}")
                continue
            if len(starts) >= max_starts:
                break
        if not starts:
            starts = [
                np.clip(
                    transform.to_theta(lnK0),
                    theta_bounds[0] + 1e-9,
                    theta_bounds[1] - 1e-9,
                )
            ]

        candidates = []
        for theta0 in starts:
            local_history = []

            def residual_with_local_history(theta, local_history=local_history):
                before = len(history)
                res = constrained_residual(theta)
                local_history.extend(history[before:])
                return res

            fit_try = least_squares(
                residual_with_local_history, theta0,
                bounds=theta_bounds, method="trf", verbose=0,
                max_nfev=constrained_attempt_max_nfev, x_scale="jac",
            )
            chi2_try = float(np.dot(fit_try.fun, fit_try.fun))
            candidates.append({
                "fit": fit_try,
                "chi2": chi2_try,
                "history": local_history,
                "success": bool(fit_try.success) and np.isfinite(chi2_try),
            })
        successful = [item for item in candidates if item["success"]]
        pool = successful if successful else candidates
        best = min(pool, key=lambda item: item["chi2"]) if pool else None
        if best is None:
            raise RuntimeError(f"{model_name} constrained optimization produced no fit candidates.")

        constrained_attempt_statuses = sorted({int(item["fit"].status) for item in candidates})
        if not successful and _nsb_constraint_fallback_to_unconstrained(cfg):
            history = []
            transform = None
            fit = least_squares(
                _weighted_residual, lnK0,
                args=(L_ref, cfg.p_total_m, F_mean, F_se, model, S_eff, N_eff, history),
                bounds=_param_bounds(model_name, S_eff), method="trf", verbose=0,
            )
            lnK_opt = fit.x
            fit_constraint_fallback_used = True
            fit_algorithm = "least_squares_after_constraint_fallback"
            fit_constraint = "unconstrained_after_failed_nsb_constraint"
        else:
            fit = best["fit"]
            history = best["history"]
            lnK_opt = transform.to_raw(fit.x)
            fit_algorithm = "nsb_constrained_multistart_least_squares"
            fit_constraint = "nonspecific_kd_greater_than_specific_kd"
    else:
        bounds = _param_bounds(model_name, S_eff)
        fit = least_squares(
            _weighted_residual, lnK0,
            args=(L_ref, cfg.p_total_m, F_mean, F_se, model, S_eff, N_eff, history),
            bounds=bounds, method="trf", verbose=0,
        )
        lnK_opt = fit.x
    param_names = model.param_labels(S_eff)
    _param_values, Ka_opt_M, Kd_opt_M = ka_kd_from_optimizer(param_names, lnK_opt)
    chi2 = float(np.dot(fit.fun, fit.fun))
    # n_obs counts cells that contributed (had finite F_se and finite F_mean)
    valid_mask = np.isfinite(F_se) & np.isfinite(F_mean)
    n_obs = int(np.sum(valid_mask))

    # Jacobian-derived covariance under absolute_sigma=False semantics
    # (matching scripts_binding/core/reporting.py::compute_uncertainties).
    # The replicate-derived σ_F is a noisy estimate from only n_reps; we
    # use it as a relative weight between cells, and let the regression's
    # own residual variance set the absolute scale:
    #     cov = (chi² / (n_obs − p)) · (J^T J)^−1
    # This converges to absolute_sigma=True when σ_F estimates are
    # accurate (chi²_red → 1) and corrects them gracefully when they are
    # not.
    e_lnK = e_Ka = e_Kd = cov_lnK = None
    n_par = len(lnK_opt)
    try:
        if n_obs > n_par:
            sig2 = chi2 / (n_obs - n_par)
            cov_theta = sig2 * np.linalg.inv(fit.jac.T @ fit.jac)
            if transform is not None:
                G = transform.jacobian_raw_wrt_theta(fit.x)
                cov_lnK = G @ cov_theta @ G.T
            else:
                cov_lnK = cov_theta
            diag = np.diag(cov_lnK)
            if np.all(diag >= 0):
                e_lnK = np.sqrt(diag)
                e_Ka = np.full(n_par, np.nan, dtype=float)
                e_Kd = np.full(n_par, np.nan, dtype=float)
                for i, name in enumerate(param_names):
                    if is_dimensionless_param(name):
                        continue
                    e_Ka[i] = Ka_opt_M[i] * e_lnK[i]
                    e_Kd[i] = Kd_opt_M[i] * e_lnK[i]
    except np.linalg.LinAlgError:
        pass

    # Convert to output units (μM)
    Kd_uM = Kd_opt_M * cfg.scale_m_to_out
    Ka_M_inv = Ka_opt_M
    e_Kd_uM = e_Kd * cfg.scale_m_to_out if e_Kd is not None else None
    e_Ka_M_inv = e_Ka if e_Ka is not None else None

    return {
        "param_names": param_names,
        "Kd_uM": Kd_uM, "e_Kd_uM": e_Kd_uM,
        "Ka_M_inv": Ka_M_inv, "e_Ka_M_inv": e_Ka_M_inv,
        "lnK_opt": lnK_opt, "e_lnK": e_lnK, "cov_lnK": cov_lnK,
        "chi2": chi2, "n_obs": n_obs, "n_params": len(lnK_opt),
        "S_eff": S_eff, "N_eff": N_eff, "n_reps": n_reps,
        "L_totals_M": L_ref, "F_mean": F_mean, "F_se": F_se,
        "fit_success": bool(fit.success),
        "fit_algorithm": fit_algorithm,
        "fit_constraint": fit_constraint,
        "fit_constraint_fallback_used": fit_constraint_fallback_used,
        "fit_constrained_attempt_statuses": constrained_attempt_statuses,
        "fit_constrained_attempt_max_nfev": constrained_attempt_max_nfev,
    }
