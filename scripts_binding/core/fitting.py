"""Registry-driven fitting — pure fit_file + separate plot_fit_results."""
import os
import re
import sys
import numpy as np
import pandas as pd
from .csv_io import read_csv
from scipy.optimize import OptimizeResult, least_squares

from ..models import REGISTRY, deconvolve_terms
from ..models import sequential_adduct
from ..models.metadata import (
    DEFAULT_NSB_MODELS,
    base_model_name,
    is_sequential_specific_model,
    ka_kd_from_optimizer,
    normalize_model_name,
    output_model_name,
)
from .reporting import (
    print_results_table,
    save_kd_csv,
    print_per_point_summary,
)

# Conservative log-space Ka bounds: Kd ∈ [~10 fM, ~10 M] (plus slack).
# Prevents runaway in sparse-data fits (see MODELS.md §7).
_LN_KA_LO, _LN_KA_HI = np.log(0.1), np.log(1e15)
_GAMMA_LO, _GAMMA_HI = -5.0, 10.0
_SOFTPLUS_CLIP = 30.0


def _param_bounds(model_name, S):
    """(lower, upper) arrays for least_squares — matches each model's ln_params layout."""
    model_name = base_model_name(model_name)
    if model_name == "occupancy_decay":
        n = S + 2
        lo = np.full(n, _LN_KA_LO)
        hi = np.full(n, _LN_KA_HI)
        lo[1], hi[1] = _GAMMA_LO, _GAMMA_HI
        return lo, hi
    n = REGISTRY[model_name].n_params(S)
    return np.full(n, _LN_KA_LO), np.full(n, _LN_KA_HI)


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > _SOFTPLUS_CLIP, x, np.log1p(np.exp(x)))


def _inv_softplus(y):
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 1e-12)
    return np.where(y > _SOFTPLUS_CLIP, y, np.log(np.expm1(y)))


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _constrain_nsb_for_model(cfg, model_name):
    return (
        bool(getattr(cfg, "constrain_nsb_weaker_than_specific", True))
        and base_model_name(model_name) in DEFAULT_NSB_MODELS
    )


def _nsb_constraint_max_nfev(cfg, max_nfev=None):
    if max_nfev is not None:
        return max_nfev
    value = int(getattr(cfg, "nsb_constraint_max_nfev", 2000))
    return max(1, value)


def _nsb_constraint_fallback_to_unconstrained(cfg):
    return bool(getattr(cfg, "nsb_constraint_fallback_to_unconstrained", True))


def _ordinary_least_squares_fit(model, model_name, lnK0, L_totals_M, P_tot_M,
                                F_exps, S, N, verbose=0, max_nfev=None,
                                algorithm_label="least_squares",
                                constraint_label=""):
    ssr_history = []
    lnK0 = np.asarray(lnK0, dtype=float)
    if lnK0.size == 0:
        fun = _make_nan_safe_residual(model)(
            lnK0, L_totals_M, P_tot_M, F_exps, S, N, ssr_history,
        )
        fit = OptimizeResult(
            x=lnK0,
            fun=fun,
            jac=np.zeros((len(fun), 0), dtype=float),
            cost=0.5 * float(np.dot(fun, fun)),
            optimality=0.0,
            active_mask=np.array([], dtype=float),
            nfev=1,
            njev=0,
            status=0,
            success=True,
            message="No fitted parameters; evaluated fixed null model.",
        )
        fit.binding_fit_algorithm = "fixed_no_parameter_model"
        fit.binding_fit_constraint = constraint_label
        fit.binding_constraint_fallback_used = False
        return fit, fit.x, ssr_history, None, 0
    fit = least_squares(
        _make_nan_safe_residual(model), lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, ssr_history),
        bounds=_param_bounds(model_name, S),
        verbose=verbose,
        max_nfev=max_nfev,
    )
    fit.binding_fit_algorithm = algorithm_label
    fit.binding_fit_constraint = constraint_label
    fit.binding_constraint_fallback_used = (
        algorithm_label == "least_squares_after_constraint_fallback"
    )
    return fit, fit.x, ssr_history, None, 1


class _NsbWeakerThanSpecificTransform:
    """Map free optimizer coordinates to ln-parameters with weaker NSB.

    The fitted models use association constants.  The biological constraint
    $K_{d,n} > K_{d,s}$ is therefore enforced as $K_n < K_{s,i}$.
    For occupancy_decay, beta is the first-step nonspecific association term;
    gamma is additionally constrained to be nonnegative so the model remains
    an occupancy-decay model.
    """

    def __init__(self, model_name, S, log_margin=1e-6):
        self.model_name = base_model_name(model_name)
        self.S = int(S)
        self.log_margin = float(log_margin)
        self.n = REGISTRY[self.model_name].n_params(self.S)

    def to_raw(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self.model_name == "occupancy_decay":
            raw = np.empty(self.S + 2, dtype=float)
            raw[0] = theta[0]
            raw[1] = _softplus(theta[1])
            raw[2:] = theta[0] + self.log_margin + _softplus(theta[2:])
            return raw
        if self.model_name == "shared_site":
            return np.array([
                theta[0],
                theta[0] + self.log_margin + _softplus(theta[1]),
            ], dtype=float)
        raw = np.empty(self.S + 1, dtype=float)
        raw[0] = theta[0]
        raw[1:] = theta[0] + self.log_margin + _softplus(theta[1:])
        return raw

    def to_theta(self, raw_params):
        raw = np.asarray(raw_params, dtype=float)
        if self.model_name == "occupancy_decay":
            theta = np.empty(self.S + 2, dtype=float)
            theta[0] = raw[0]
            theta[1] = _inv_softplus(max(raw[1], 1e-8))
            theta[2:] = _inv_softplus(np.maximum(raw[2:] - raw[0] - self.log_margin, 1e-8))
            return theta
        if self.model_name == "shared_site":
            return np.array([
                raw[0],
                _inv_softplus(max(raw[1] - raw[0] - self.log_margin, 1e-8)),
            ], dtype=float)
        theta = np.empty(self.S + 1, dtype=float)
        theta[0] = raw[0]
        theta[1:] = _inv_softplus(np.maximum(raw[1:] - raw[0] - self.log_margin, 1e-8))
        return theta

    def jacobian_raw_wrt_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self.model_name == "occupancy_decay":
            G = np.zeros((self.S + 2, self.S + 2), dtype=float)
            G[0, 0] = 1.0
            G[1, 1] = float(_sigmoid(theta[1]))
            G[2:, 0] = 1.0
            G[np.arange(2, self.S + 2), np.arange(2, self.S + 2)] = _sigmoid(theta[2:])
            return G
        if self.model_name == "shared_site":
            G = np.zeros((2, 2), dtype=float)
            G[0, 0] = 1.0
            G[1, 0] = 1.0
            G[1, 1] = float(_sigmoid(theta[1]))
            return G
        G = np.zeros((self.S + 1, self.S + 1), dtype=float)
        G[0, 0] = 1.0
        G[1:, 0] = 1.0
        G[np.arange(1, self.S + 1), np.arange(1, self.S + 1)] = _sigmoid(theta[1:])
        return G

    def theta_bounds(self):
        lo = np.full(self.n, -30.0)
        hi = np.full(self.n, 30.0)
        lo[0], hi[0] = _LN_KA_LO, _LN_KA_HI
        if self.model_name == "occupancy_decay":
            lo[1], hi[1] = -30.0, float(_inv_softplus(_GAMMA_HI))
        return lo, hi

    def is_raw_valid(self, raw_params):
        raw = np.asarray(raw_params, dtype=float)
        lo, hi = _param_bounds(self.model_name, self.S)
        return bool(
            np.all(np.isfinite(raw))
            and np.all(raw >= lo - 1e-9)
            and np.all(raw <= hi + 1e-9)
        )


def _ka_from_kd_uM(kd_uM):
    return 1e6 / float(kd_uM)


def _constrained_initial_raws(model_name, S, max_starts):
    model_name = base_model_name(model_name)
    model = REGISTRY[model_name]
    starts = [model.initial_lnK(S)]
    kd_n_values = [150.0, 300.0, 500.0, 1000.0, 3000.0, 8000.0]
    profiles = [
        np.linspace(0.15, 0.95, max(S, 1)),
        np.geomspace(0.08, 0.95, max(S, 1)),
        np.array([0.15, 0.25, 0.40, 0.60, 0.80, 0.90, 0.95][:S] or [0.5]),
        np.full(max(S, 1), 0.70),
    ]
    for kd_n in kd_n_values:
        for profile in profiles:
            frac = np.asarray(profile[:S], dtype=float)
            if len(frac) < S:
                frac = np.pad(frac, (0, S - len(frac)), mode="edge")
            kd_s = np.maximum(kd_n * frac, 1e-4)
            ka_n = _ka_from_kd_uM(kd_n)
            ka_s = np.array([_ka_from_kd_uM(v) for v in kd_s], dtype=float)
            if model_name == "occupancy_decay":
                for gamma in (0.0, 0.5, 1.0):
                    starts.append(np.concatenate(([np.log(ka_n), gamma], np.log(ka_s))))
            elif model_name == "shared_site":
                starts.append(np.array([np.log(ka_n), np.log(float(np.median(ka_s)))]))
            else:
                starts.append(np.concatenate(([np.log(ka_n)], np.log(ka_s))))
            if len(starts) >= max_starts:
                return starts[:max_starts]
    return starts[:max_starts]


def _fit_covariance_raw(fit, raw_params, Ka_opt_M, Kd_opt_M, ssr_final, n_obs,
                        param_names, transform=None):
    p = len(raw_params)
    if n_obs <= p:
        print("Warning: Not enough data points to compute uncertainties.")
        return False, None, None, None
    try:
        sig2 = ssr_final / (n_obs - p)
        cov_theta = sig2 * np.linalg.inv(fit.jac.T @ fit.jac)
        if transform is not None:
            G = transform.jacobian_raw_wrt_theta(fit.x)
            cov_raw = G @ cov_theta @ G.T
        else:
            cov_raw = cov_theta
        diag = np.diag(cov_raw)
        if np.any(diag < 0):
            raise ValueError("negative covariance diagonal")
        std_param = np.sqrt(diag)
        std_Ka_M = np.full(p, np.nan, dtype=float)
        std_Kd_M = np.full(p, np.nan, dtype=float)
        for i, name in enumerate(param_names):
            from ..models.metadata import is_dimensionless_param
            if is_dimensionless_param(name):
                continue
            std_Ka_M[i] = Ka_opt_M[i] * std_param[i]
            std_Kd_M[i] = Kd_opt_M[i] * std_param[i]
        return True, std_param, std_Ka_M, std_Kd_M
    except np.linalg.LinAlgError:
        print("Warning: Could not compute uncertainties (Jacobian matrix is singular).")
    except ValueError as exc:
        print(f"Warning: Could not compute uncertainties ({exc}).")
    return False, None, None, None


def _least_squares_fit(model, model_name, lnK0, L_totals_M, P_tot_M, F_exps, S, N,
                       cfg, verbose=0, max_nfev=None):
    """Run the configured optimizer and return fit, raw ln-params, history, transform."""
    if not _constrain_nsb_for_model(cfg, model_name):
        return _ordinary_least_squares_fit(
            model, model_name, lnK0, L_totals_M, P_tot_M, F_exps, S, N,
            verbose=verbose, max_nfev=max_nfev,
        )

    max_starts = max(1, int(getattr(cfg, "nsb_constraint_multistart_n", 24)))
    constrained_max_nfev = _nsb_constraint_max_nfev(cfg, max_nfev)
    transform = _NsbWeakerThanSpecificTransform(
        model_name, S, getattr(cfg, "nsb_constraint_log_margin", 1e-6),
    )
    theta_bounds = transform.theta_bounds()
    residual_len = int(np.prod(F_exps.shape))

    def residual(theta, history):
        raw = transform.to_raw(theta)
        if not transform.is_raw_valid(raw):
            lo, hi = _param_bounds(model_name, S)
            violation = np.maximum(lo - raw, 0.0) + np.maximum(raw - hi, 0.0)
            penalty_scale = 1e6 * (1.0 + float(np.sum(violation)))
            penalty = np.full(residual_len, penalty_scale, dtype=float)
            history.append(float(np.dot(penalty, penalty)))
            return penalty
        local_hist = []
        res = _make_nan_safe_residual(model)(
            raw, L_totals_M, P_tot_M, F_exps, S, N, local_hist,
        )
        history.append(float(np.dot(res, res)))
        return res

    starts = []
    for raw in [lnK0, *_constrained_initial_raws(model_name, S, max_starts)]:
        try:
            theta = transform.to_theta(raw)
            lo, hi = theta_bounds
            theta = np.clip(theta, lo + 1e-9, hi - 1e-9)
            if transform.is_raw_valid(transform.to_raw(theta)):
                starts.append(theta)
        except Exception:
            continue
        if len(starts) >= max_starts:
            break
    if not starts:
        starts = [np.clip(transform.to_theta(lnK0), theta_bounds[0] + 1e-9, theta_bounds[1] - 1e-9)]

    candidates = []
    for idx, theta0 in enumerate(starts):
        history = []
        fit_try = least_squares(
            lambda theta: residual(theta, history),
            theta0,
            bounds=theta_bounds,
            max_nfev=constrained_max_nfev,
            x_scale="jac",
            verbose=verbose if len(starts) == 1 else 0,
        )
        ssr_try = float(np.dot(fit_try.fun, fit_try.fun))
        candidates.append({
            "ssr": ssr_try,
            "fit": fit_try,
            "history": history,
            "start_index": idx,
            "success": bool(fit_try.success) and np.isfinite(ssr_try),
        })
    successful = [item for item in candidates if item["success"]]
    pool = successful if successful else candidates
    best = min(pool, key=lambda item: item["ssr"]) if pool else None
    if best is None:
        raise RuntimeError(f"{model_name} constrained optimization produced no fit candidates.")

    if not successful and _nsb_constraint_fallback_to_unconstrained(cfg):
        status_values = sorted({int(item["fit"].status) for item in candidates})
        print(
            f"[NSB constraint fallback] {model_name}: no constrained start converged "
            f"within max_nfev={constrained_max_nfev} (statuses={status_values}); "
            "retrying without the NSB constraint."
        )
        fit, raw, history, _transform, n_unconstrained_starts = _ordinary_least_squares_fit(
            model, model_name, lnK0, L_totals_M, P_tot_M, F_exps, S, N,
            verbose=verbose, max_nfev=max_nfev,
            algorithm_label="least_squares_after_constraint_fallback",
            constraint_label="unconstrained_after_failed_nsb_constraint",
        )
        fit.binding_constrained_attempt_statuses = status_values
        fit.binding_constrained_attempt_max_nfev = constrained_max_nfev
        return fit, raw, history, None, len(starts) + n_unconstrained_starts

    fit = best["fit"]
    raw = transform.to_raw(fit.x)
    fit.binding_fit_algorithm = "nsb_constrained_multistart_least_squares"
    fit.binding_fit_constraint = "nonspecific_kd_greater_than_specific_kd"
    fit.binding_constraint_fallback_used = False
    fit.binding_constrained_attempt_statuses = sorted({int(item["fit"].status) for item in candidates})
    fit.binding_constrained_attempt_max_nfev = constrained_max_nfev
    print(
        f"[NSB constraint] {model_name}: enforced Kd,n > fitted specific Kd "
        f"using {len(starts)} start(s), max_nfev={constrained_max_nfev}; "
        f"selected start {best['start_index'] + 1}."
    )
    if model_name == "occupancy_decay":
        print("[NSB constraint] occupancy_decay: constrained gamma >= 0.")
    return fit, raw, best["history"], transform, len(starts)


def parse_ligand_conc(entry, scale_l_in_to_m):
    """Numeric ligand conc from a string/number entry → Molar."""
    if pd.isna(entry):
        return np.nan
    if isinstance(entry, (int, float)):
        return float(entry) * scale_l_in_to_m
    try:
        match = re.search(r"[\d\.]+", str(entry))
        if match:
            return float(match.group(0)) * scale_l_in_to_m
    except (ValueError, TypeError):
        pass
    return np.nan


def load_binding_csv(data_path, cfg):
    """→ df, L_totals_M, I_cols, F_exps (row-normalized).

    Empty / missing intensity cells are kept as NaN (NOT filled with 0) so
    downstream fitting can distinguish 'no measurement' from a literal 0
    measurement of zero abundance.

    If the CSV omits the [L]=0 apo baseline row (F_0=1, F_i=0 for i>0), we
    auto-insert one so every fit starts from the same well-defined apo state.
    """
    df = read_csv(data_path).dropna(subset=["Entry"])
    I_cols = sorted([c for c in df.columns if c.startswith("I")], key=lambda c: int(c[1:]))

    # Auto-insert apo baseline [L]=0 row if missing (raw conc, before scaling).
    entry_num = pd.to_numeric(df["Entry"], errors="coerce")
    if not (entry_num.min() <= 1e-12):
        apo_row = {c: 0.0 for c in df.columns}
        apo_row["Entry"] = 0.0
        if I_cols:
            apo_row[I_cols[0]] = 1.0
            for c in I_cols[1:]:
                apo_row[c] = 0.0
        df = pd.concat([pd.DataFrame([apo_row]), df], ignore_index=True)
        df = df.sort_values("Entry", kind="mergesort").reset_index(drop=True)
        print(f"[load_binding_csv] auto-inserted apo baseline [L]=0, F_0=1 for {data_path}",
              file=sys.stderr)

    L_totals_M = df.iloc[:, 0].apply(lambda e: parse_ligand_conc(e, cfg.scale_l_in_to_m)).values

    I_vals = df[I_cols].values
    # Row-normalize using only non-NaN cells per row, preserving NaN positions
    row_sums = np.nansum(I_vals, axis=1)
    F_exps = (I_vals.T / np.maximum(row_sums, 1e-12)).T
    return df, L_totals_M, I_cols, F_exps


def _bic(rss, n_obs, n_par):
    if rss <= 0 or n_obs <= n_par:
        return np.inf
    return n_obs * np.log(rss / n_obs) + n_par * np.log(n_obs)


def _nan_pad(F_exps, target_len):
    """Pad each row of F_exps to target_len with NaN (NOT 0)."""
    if F_exps.shape[1] >= target_len:
        return F_exps[:, :target_len]
    pad = np.full((F_exps.shape[0], target_len - F_exps.shape[1]), np.nan)
    return np.hstack([F_exps, pad])


def _zero_pad(F_vals, target_len):
    """Pad calculated model fractions to target_len with zero-valued absent species."""
    if F_vals.shape[1] >= target_len:
        return F_vals[:, :target_len]
    pad = np.zeros((F_vals.shape[0], target_len - F_vals.shape[1]))
    return np.hstack([F_vals, pad])


def _make_nan_safe_residual(model):
    """Wrap model.residual_vector so NaN positions become 0-residual."""
    base_fn = model.residual_vector

    def wrapped(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history):
        local_hist = []
        res = base_fn(ln_params, L_totals_M, P_tot_M, F_exps, S, N, local_hist)
        res = np.where(np.isnan(res), 0.0, res)
        ssr_history.append(float(np.dot(res, res)))
        return res

    return wrapped


def fit_quick(model, L_totals_M, F_exps, P_tot_M, S, N, cfg=None):
    """Stats-only fit for auto-S BIC scans."""
    num_species_model = S + N + 1
    F_exps_padded = _nan_pad(F_exps, num_species_model)
    n_obs_valid = int(np.sum(~np.isnan(F_exps_padded)))

    try:
        fit, lnK_opt, ssr_history, _transform, n_starts = _least_squares_fit(
            model, model.MODEL_NAME, model.initial_lnK(S),
            L_totals_M, P_tot_M, F_exps_padded, S, N, cfg, verbose=0,
        )
    except Exception as e:
        print(f"  [Auto-S] S={S} failed: {e}")
        return None

    rss = float(np.dot(fit.fun, fit.fun))  # NaN-cleaned by wrapper
    n_par = len(fit.x)
    return {
        "S": S, "N": N, "n_params": n_par, "n_obs": n_obs_valid,
        "SSR": rss, "BIC": _bic(rss, n_obs_valid, n_par), "lnK_opt": lnK_opt,
        "n_optimizer_starts": n_starts,
    }


def auto_select_S(data_path, cfg, model_name):
    """Return S ∈ 0..max_i with lowest BIC for the given model."""
    model = REGISTRY[base_model_name(model_name)]
    df, L_totals_M, I_cols, F_exps = load_binding_csv(data_path, cfg)
    max_i = len(I_cols) - 1

    print(f"\n{'='*60}")
    print(f"[Auto-S | {model_name}] Scanning S = 0..{max_i} for: {os.path.basename(data_path)}")
    print(f"{'='*60}")

    results = []
    for S in range(0, max_i + 1):
        info = fit_quick(model, L_totals_M, F_exps, cfg.p_total_m, S, max_i - S, cfg)
        if info is not None:
            results.append(info)

    if not results:
        print("[Auto-S] All fits failed. Falling back to S=1.")
        return 1

    best = min(results, key=lambda r: r["BIC"])
    header = f"  {'S':>3} {'N':>3} {'params':>6} {'SSR':>12} {'BIC':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        marker = " <-- best" if r["S"] == best["S"] else ""
        print(f"  {r['S']:>3} {r['N']:>3} {r['n_params']:>6} {r['SSR']:>12.4e} {r['BIC']:>12.2f}{marker}")

    print(f"\n[Auto-S] Selected S = {best['S']} (BIC = {best['BIC']:.2f})\n")
    return best["S"]


def _resolve_S_N(cfg, model_name, max_i, S_override):
    """Effective (S, N) from config + data width."""
    model_name = normalize_model_name(model_name, S_override)
    model_s_overrides = getattr(cfg, "model_s_overrides", {}) or {}
    model_n_overrides = getattr(cfg, "model_n_overrides", {}) or {}

    if is_sequential_specific_model(model_name):
        if S_override is not None:
            S_eff = int(S_override)
        elif model_name in model_s_overrides:
            S_eff = int(model_s_overrides[model_name])
        elif model_name == "sequential_specific":
            S_eff = max_i
        elif re.fullmatch(r"sequential_specific_s\d+", model_name):
            S_eff = int(model_name.rsplit("s", 1)[1])
        else:
            S_eff = int(cfg.s)
        if S_eff > max_i:
            print(f"[Info] {model_name} uses S={S_eff}; data columns end at I{max_i}. "
                  "Missing higher species are treated as unmeasured, not zero-filled.")
        return S_eff, 0

    S_use = S_override if S_override is not None else cfg.s
    if model_name in model_s_overrides:
        S_use = int(model_s_overrides[model_name])
    S_eff = S_use
    if S_eff > max_i:
        if cfg.auto_adjust_s:
            print(f"[Warning] S={S_use} exceeds I0..I{max_i}. Clamping to {max_i}.")
            S_eff = max_i
        else:
            raise ValueError(f"S={S_use} exceeds I0..I{max_i}.")

    if model_name in model_n_overrides:
        N_eff = int(model_n_overrides[model_name])
        if S_eff + N_eff != max_i:
            print(f"[Warning] S+N={S_eff + N_eff} does not match max_i={max_i}.")
    elif cfg.n_override is not None:
        N_eff = cfg.n_override
        if S_eff + N_eff != max_i:
            print(f"[Warning] S+N={S_eff + N_eff} does not match max_i={max_i}.")
    else:
        N_eff = max_i - S_eff

    if N_eff < 0:
        raise ValueError(f"Computed N={N_eff} is negative.")
    return S_eff, N_eff


def _trim_low_pop_species(F_exps, I_cols, min_frac):
    """Drop trailing species with max finite frac < min_frac; preserve missing cells."""
    trimmed = []
    n_steps = len(I_cols) - 1
    while n_steps > 1:
        col = F_exps[:, n_steps]
        max_frac = np.nanmax(col) if np.isfinite(col).any() else 0.0
        if max_frac < min_frac:
            trimmed.append(f"I{n_steps} (max={max_frac:.4f})")
            F_exps = F_exps[:, :n_steps]
            I_cols = I_cols[:n_steps]
            n_steps -= 1
        else:
            break
    if trimmed:
        print(f"  [Trimmed] Dropped species with max frac < {min_frac}: {', '.join(trimmed)}")
    row_sums = np.nansum(F_exps, axis=1)
    F_exps = np.divide(
        F_exps, row_sums[:, None],
        out=np.full_like(F_exps, np.nan, dtype=float),
        where=row_sums[:, None] > 0,
    )
    return F_exps, I_cols


def compute_deconvolution(model, df, L_totals_M, F_exps, F_calcs, lnK_opt, S, N, cfg, out_dir, stem):
    """Compute (j spec, m NSB) decomposition and return arrays for plotting.

    Returns dict with: L_vals_M, contrib_all, frac_within_all, F_source, entries,
    or None if the model has no deconvolution weights.
    """
    if not hasattr(model, "partition_terms"):
        return None

    weights_ij = model.partition_terms(lnK_opt, S, N)

    if cfg.deconv_use_grid:
        L_vals_M = np.linspace(L_totals_M.min(), L_totals_M.max(), cfg.deconv_grid_points)
        L_free_vals = np.array([model.free_ligand(L, cfg.p_total_m, lnK_opt, S, N) for L in L_vals_M])
        F_source = np.vstack([model.mole_fractions(Lf, lnK_opt, S, N) for Lf in L_free_vals])
        source_key = "calc_grid"
        entries = [f"{v * cfg.scale_m_to_out:.3g}" for v in L_vals_M]
    else:
        L_vals_M = L_totals_M
        entries = df["Entry"].tolist()
        if cfg.deconv_source.lower() == "exp":
            F_source = F_exps
            source_key = "exp"
        else:
            F_source = F_calcs
            source_key = "calc"

    n_points = len(L_vals_M)
    num_species = S + N + 1
    frac_within_all = np.zeros((n_points, num_species, S + 1))
    contrib_all = np.zeros((n_points, num_species, S + 1))
    for idx in range(n_points):
        frac_within, contrib = deconvolve_terms(F_source[idx], weights_ij, S)
        frac_within_all[idx] = frac_within
        contrib_all[idx] = contrib

    rows = []
    for idx in range(n_points):
        L_out = L_vals_M[idx] * cfg.scale_m_to_out
        for i in range(num_species):
            max_j = min(i, S)
            for j in range(max_j + 1):
                rows.append({
                    "Entry": entries[idx],
                    f"L_tot({cfg.output_unit})": L_out,
                    "i_total": i, "j_specific": j, "m_nonspecific": i - j,
                    f"F_{source_key}": F_source[idx][i],
                    "fraction_within_i": frac_within_all[idx][i, j],
                    "fraction_total": contrib_all[idx][i, j],
                })
    if not getattr(cfg, "compact_outputs", False):
        deconv_csv = cfg.deconv_csv_path or os.path.join(out_dir, f"{stem}_deconv.csv")
        pd.DataFrame(rows).to_csv(deconv_csv, index=False)
        print(f"Saved deconvolution table to: {deconv_csv}")

    if cfg.report_ligand_conc:
        L_out_all = L_vals_M * cfg.scale_m_to_out
        print("\n--- Deconvolution report ---")
        for target in cfg.report_ligand_conc:
            idx = int(np.argmin(np.abs(L_out_all - target)))
            print(f"\nTarget {target} {cfg.output_unit} -> using {L_out_all[idx]:.3g} {cfg.output_unit} (Entry={entries[idx]})")
            for i in range(1, num_species):
                max_j = min(i, S)
                parts = [f"{j} spec + {i-j} non: {100.0*frac_within_all[idx][i,j]:.1f}%" for j in range(max_j + 1)]
                print(f"  I{i}: " + "; ".join(parts))

    return {
        "L_vals_M": L_vals_M,
        "contrib_all": contrib_all,
        "frac_within_all": frac_within_all,
        "F_source": F_source,
    }


def fit_file(data_path, out_dir, cfg, model_name, S_override=None):
    """Fit one titration CSV. Writes Kd CSV. Returns info dict (no plotting).

    The returned dict contains all arrays needed by plot_fit_results().
    """
    model_name = normalize_model_name(model_name, S_override)
    model = REGISTRY[base_model_name(model_name)]
    is_specific = is_sequential_specific_model(model_name)

    ssr_history = []
    df, L_totals_M, I_cols, F_exps = load_binding_csv(data_path, cfg)

    if is_specific and getattr(cfg, "trim_specific_low_pop", True) and cfg.min_species_frac > 0:
        F_exps, I_cols = _trim_low_pop_species(F_exps, I_cols, cfg.min_species_frac)

    max_i = len(I_cols) - 1
    S_eff, N_eff = _resolve_S_N(cfg, model_name, max_i, S_override)
    num_species_model = S_eff + N_eff + 1
    num_species_fit = max(num_species_model, F_exps.shape[1]) if is_specific else num_species_model

    if not is_specific and F_exps.shape[1] > num_species_model:
        raise ValueError(
            f"{model_name} with S={S_eff}, N={N_eff} models I0..I{num_species_model - 1}, "
            f"but data contain I0..I{F_exps.shape[1] - 1}. Increase N or remove extra data columns."
        )

    F_exps = _nan_pad(F_exps, num_species_fit)

    print(f"\n=== [{model_name}] Processing: {data_path} ===")
    print(f"Model parameters: S={S_eff}, N={N_eff}, n_params={model.n_params(S_eff)}")

    lnK0 = model.initial_lnK(S_eff)
    print(f"--- Starting optimization ({model_name}) ---")
    fit, lnK_opt, ssr_history, transform, n_starts = _least_squares_fit(
        model, model_name, lnK0,
        L_totals_M, cfg.p_total_m, F_exps, S_eff, N_eff, cfg,
        verbose=2,
    )
    print(f"--- Optimization finished ({model_name}) ---\n")

    param_names = model.param_labels(S_eff)
    param_values, Ka_opt_M, Kd_opt_M = ka_kd_from_optimizer(param_names, lnK_opt)
    ssr_final = float(np.dot(fit.fun, fit.fun))
    n_obs = int(np.sum(np.isfinite(F_exps)))

    has_errors, std_param, std_Ka_M, std_Kd_M = _fit_covariance_raw(
        fit, lnK_opt, Ka_opt_M, Kd_opt_M, ssr_final, n_obs,
        param_names=param_names, transform=transform,
    )

    # Jacobian rank check — flag unidentifiable fits
    try:
        sv = np.linalg.svd(fit.jac, compute_uv=False)
        if sv[0] > 0:
            rank_eff = int(np.sum(sv > sv[0] * 1e-8))
            cond = sv[0] / max(sv[-1], sv[0] * 1e-300)
            if rank_eff < len(sv):
                print(f"[FitQualityWarning] Jacobian rank-deficient: "
                      f"{rank_eff}/{len(sv)} effective dof (condition {cond:.2e}). "
                      f"Kd values for unidentified parameters are not meaningful. "
                      f"Consider running scripts/batch_kd_scan.py with --bootstrap 200 "
                      f"to quantify identifiability via row-resampling.")
    except Exception:
        pass

    print_results_table(
        param_names, lnK_opt, Ka_opt_M, Kd_opt_M, has_errors, cfg,
        std_param=std_param, std_Ka_M=std_Ka_M, std_Kd_M=std_Kd_M,
    )

    L_free_list = np.array([model.free_ligand(L, cfg.p_total_m, lnK_opt, S_eff, N_eff) for L in L_totals_M])
    F_calcs_model = np.vstack([model.mole_fractions(Lf, lnK_opt, S_eff, N_eff) for Lf in L_free_list])
    F_calcs = _zero_pad(F_calcs_model, num_species_fit)

    if cfg.debug_validate and model_name == "sequential_adduct" and len(L_totals_M) > 0:
        if cfg.debug_ligand_conc is not None:
            L_out_all = L_totals_M * cfg.scale_m_to_out
            idx = int(np.argmin(np.abs(L_out_all - cfg.debug_ligand_conc)))
        else:
            idx = min(cfg.debug_index, len(L_totals_M) - 1)
        sequential_adduct.debug_validate_point(
            idx, L_totals_M[idx], L_free_list[idx], lnK_opt, F_calcs[idx], S_eff, N_eff, cfg
        )

    print_per_point_summary(df, L_totals_M, F_exps, L_free_list, F_calcs, num_species_model, cfg)

    stem = os.path.splitext(os.path.basename(data_path))[0]
    if not cfg.brief and not getattr(cfg, "compact_outputs", False):
        kd_csv = os.path.join(out_dir, f"{stem}_kd.csv")
        save_kd_csv(
            param_names, lnK_opt, Ka_opt_M, Kd_opt_M, has_errors,
            std_param, std_Ka_M, std_Kd_M, kd_csv, cfg,
        )

    return {
        "model_name": model_name,
        "data_path": data_path,
        "df": df,
        "L_totals_M": L_totals_M,
        "F_exps": F_exps,
        "F_calcs": F_calcs,
        "param_values": param_values,
        "Ka_M_inv": Ka_opt_M,
        "Kd_out": Kd_opt_M * cfg.scale_m_to_out,
        "std_param": std_param,
        "std_Ka_M": std_Ka_M,
        "std_Kd_out": None if std_Kd_M is None else std_Kd_M * cfg.scale_m_to_out,
        "param_names": param_names,
        "num_species": num_species_fit,
        "num_species_model": num_species_model,
        "observed_max_stoichiometry": max_i,
        "S_eff": S_eff, "N_eff": N_eff,
        "stem": stem, "lnK_opt": lnK_opt,
        "ssr_history": list(ssr_history),
        "SSR": ssr_final, "n_obs": n_obs, "n_params": len(lnK_opt),
        "dof": n_obs - len(lnK_opt),
        "fit_success": bool(fit.success),
        "fit_status": int(fit.status),
        "fit_message": str(fit.message),
        "fit_cost": float(fit.cost),
        "fit_optimality": float(fit.optimality),
        "fit_algorithm": getattr(fit, "binding_fit_algorithm", (
            "nsb_constrained_multistart_least_squares"
            if transform is not None else "least_squares"
        )),
        "fit_constraint": getattr(fit, "binding_fit_constraint", (
            "nonspecific_kd_greater_than_specific_kd"
            if transform is not None else ""
        )),
        "fit_constraint_fallback_used": bool(
            getattr(fit, "binding_constraint_fallback_used", False)
        ),
        "fit_constrained_attempt_statuses": getattr(
            fit, "binding_constrained_attempt_statuses", []
        ),
        "fit_constrained_attempt_max_nfev": getattr(
            fit, "binding_constrained_attempt_max_nfev", np.nan
        ),
        "n_optimizer_starts": int(n_starts),
        "out_dir": out_dir,
    }


def plot_fit_results(info, cfg):
    """Generate fit curves and deconvolution plot from fit_file output."""
    if not cfg.save_plots and not cfg.show_plots:
        return

    # Imports deferred so pure fitting paths don't touch matplotlib.
    import matplotlib.pyplot as plt
    from .plotting import safe_savefig, plot_species_curves, plot_deconv_byconc

    model = REGISTRY[base_model_name(info["model_name"])]
    is_specific = is_sequential_specific_model(info["model_name"])
    S_eff, N_eff = info["S_eff"], info["N_eff"]
    lnK_opt = info["lnK_opt"]
    L_totals_M = info["L_totals_M"]
    F_exps = info["F_exps"]
    stem = info["stem"]
    out_dir = info["out_dir"]
    model_output_stem = output_model_name(info["model_name"], getattr(cfg, "model_output_names", {}))

    L_grid_M = np.linspace(L_totals_M.min(), L_totals_M.max(), 300)
    F_grid_model = np.vstack([
        model.mole_fractions(model.free_ligand(L, cfg.p_total_m, lnK_opt, S_eff, N_eff), lnK_opt, S_eff, N_eff)
        for L in L_grid_M
    ])
    F_grid = _zero_pad(F_grid_model, info["num_species"])
    n_specific_for_plot = None if is_specific else S_eff

    ext = getattr(cfg, "plot_format", "svg")
    fit_svg = os.path.join(out_dir, f"{stem}_{model_output_stem}_fit.{ext}")
    plot_species_curves(
        L_grid_M, F_grid, info["num_species"], cfg, output_svg=fit_svg,
        title=f"{info['model_name']}: global fit", n_specific=n_specific_for_plot,
        data_L_M=L_totals_M, data_F=F_exps,
        model_species_count=info["num_species_model"],
        legend_kd_values=(info["Kd_out"] if is_specific and getattr(cfg, "show_kd_in_legend", False) else None),
        legend_kd_unit=cfg.output_unit,
    )

    if cfg.deconv_enable and not is_specific:
        dec = compute_deconvolution(
            model, info["df"], L_totals_M, F_exps, info["F_calcs"],
            lnK_opt, S_eff, N_eff, cfg, out_dir, stem,
        )
        if dec is not None:
            outline_totals = None if cfg.deconv_use_grid else F_exps
            fig = plot_deconv_byconc(
                dec["L_vals_M"] * cfg.scale_m_to_out, dec["contrib_all"], S_eff, N_eff,
                "Deconvoluted fraction of apparent", cfg,
                outline_totals=outline_totals, outline_label="Frac_expt",
            )
            if cfg.save_plots:
                safe_savefig(fig, os.path.join(out_dir, f"{stem}_{model_output_stem}_deconv.{ext}"), cfg.max_image_dim)
            if cfg.show_plots:
                plt.show()
            else:
                plt.close(fig)
