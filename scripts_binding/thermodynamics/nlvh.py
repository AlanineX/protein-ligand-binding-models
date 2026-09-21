"""Van't Hoff thermodynamic analysis (linear and non-linear).

Free parameters are the natural thermodynamic constants ΔH and ΔS (and ΔCp
under NLVH), referenced to T0 = 298.15 K. ΔG and ΔG-derived quantities are
computed from the fitted parameters via ΔG(T0) = ΔH(T0) - T0·ΔS(T0); the
intercept lnK_0 is a derived quantity, not a free parameter.

Provides:
    statistical_correction  - macroscopic Kd(uM) -> ln(Ka_intrinsic)
    lvh_equation            - linear VH model (2 params: ΔH, ΔS)
    nlvh_equation           - non-linear VH model (3 params: ΔH, ΔS, ΔCp at T0)
    fit_lvh                 - linear VH curve-fit wrapper
    fit_nlvh                - NLVH curve-fit wrapper
"""

import numpy as np
from scipy.optimize import curve_fit

R_KJ = 8.3144598e-3  # kJ / (mol K)  — CODATA 2014


# ---------------------------------------------------------------------------
# Statistical correction
# ---------------------------------------------------------------------------

def statistical_correction(kd_uM, n_sites):
    """
    Convert an array of macroscopic sequential Kd values (uM) into
    ln(Ka_intrinsic) values for N equivalent independent binding sites.

    Ka_intrinsic,i = Ka_macro,i * i / (N - i + 1)
                   = [1 / (Kd_i * 1e-6)] * i / (N - i + 1)

    Parameters
    ----------
    kd_uM : array-like, shape (n_sites,)
        Macroscopic Kd values in micro-molar.
    n_sites : int
        Total number of equivalent independent sites N.

    Returns
    -------
    ln_ka : ndarray, shape (n_sites,)
        ln(Ka_intrinsic) for each sequential binding event.
    """
    N = n_sites
    kd = np.asarray(kd_uM, dtype=float)
    ln_ka = np.full(len(kd), np.nan)

    for idx in range(len(kd)):
        if np.isnan(kd[idx]) or kd[idx] <= 0:
            continue
        i = idx + 1  # 1-indexed site number
        if i > N:
            continue  # site index exceeds N — correction undefined
        ka_macro = 1.0 / (kd[idx] * 1e-6)       # M^-1
        ka_intr = ka_macro * i / (N - i + 1)
        ln_ka[idx] = np.log(ka_intr)

    return ln_ka


# ---------------------------------------------------------------------------
# LVH model
# ---------------------------------------------------------------------------

def lvh_equation(T, dH, dS):
    """
    Linear van't Hoff equation (ΔCp = 0):

        ln K(T) = ΔS / R  -  ΔH / (R T)

    Equivalently: ΔG(T) = ΔH - T ΔS, ln K = -ΔG/(RT).

    Parameters
    ----------
    T  : temperature (K), scalar or array
    dH : enthalpy (kJ/mol), constant with temperature
    dS : entropy (kJ/mol/K), constant with temperature
    """
    return dS / R_KJ - dH / (R_KJ * T)


# ---------------------------------------------------------------------------
# NLVH model
# ---------------------------------------------------------------------------

def nlvh_equation(T, dH, dS, Cp, T0):
    """
    Non-linear van't Hoff equation (ΔCp ≠ 0):

        ΔH(T)   = ΔH(T0) + ΔCp · (T - T0)
        ΔS(T)   = ΔS(T0) + ΔCp · ln(T / T0)
        ΔG(T)   = ΔH(T) - T · ΔS(T)
        ln K(T) = -ΔG(T) / (R T)

    ΔH, ΔS are the free parameters at the reference T0; ΔCp parameterises
    their temperature dependence.

    Parameters
    ----------
    T  : temperature (K), scalar or array
    dH : enthalpy at T0 (kJ/mol)
    dS : entropy at T0 (kJ/mol/K)
    Cp : heat-capacity change (kJ/mol/K)
    T0 : reference temperature (K)
    """
    T = np.asarray(T, dtype=float)
    dH_T = dH + Cp * (T - T0)
    dS_T = dS + Cp * np.log(T / T0)
    dG_T = dH_T - T * dS_T
    return -dG_T / (R_KJ * T)


# ---------------------------------------------------------------------------
# Shared uncertainty propagation
# ---------------------------------------------------------------------------

def _derived_at_T0(dH, dS, T0, var_dH, var_dS, cov_dH_dS):
    """Derive ΔG, -TΔS, lnK_0, Kd_uM at T0 with full covariance propagation.

    At T = T0 the curvature term ΔCp drops out (ΔH(T0) = ΔH, ΔS(T0) = ΔS),
    so ΔG(T0) = ΔH - T0·ΔS depends only on dH and dS. The same propagation
    therefore serves LVH and NLVH.

        var(ΔG)  = var(dH) + T0² var(dS) - 2 T0 cov(dH, dS)
        σ(-TΔS)  = T0 · σ(dS)     [-TΔS = -T0·dS at T0]
        var(lnK0) = var(dH)/(R T0)² + var(dS)/R² - 2 cov(dH, dS) / (R² T0)

    Returns
    -------
    dict with: dG, e_dG, minus_TdS, e_minus_TdS, lnK0, e_lnK0, Kd_uM
    """
    dG = dH - T0 * dS
    var_dG = var_dH + (T0 ** 2) * var_dS - 2.0 * T0 * cov_dH_dS
    e_dG = np.sqrt(var_dG) if var_dG >= 0 else float("nan")

    minus_TdS = -T0 * dS
    e_minus_TdS = T0 * np.sqrt(var_dS)

    lnK0 = dS / R_KJ - dH / (R_KJ * T0)
    var_lnK0 = (var_dH / (R_KJ ** 2 * T0 ** 2)
                + var_dS / R_KJ ** 2
                - 2.0 * cov_dH_dS / (R_KJ ** 2 * T0))
    e_lnK0 = np.sqrt(var_lnK0) if var_lnK0 >= 0 else float("nan")

    Kd_uM = (1.0 / np.exp(lnK0)) * 1e6

    return dict(dG=dG, e_dG=e_dG,
                minus_TdS=minus_TdS, e_minus_TdS=e_minus_TdS,
                lnK0=lnK0, e_lnK0=e_lnK0,
                Kd_uM=Kd_uM)


def _gof_metrics(y_obs, y_pred, lnKa_err, k):
    """Compute R², χ², BIC, AICc for a fit with k free parameters.

    χ² uses the absolute_sigma=True convention (squared standardised
    residuals). When weights are unavailable, χ² falls back to SSR.
    """
    n = len(y_obs)
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if (lnKa_err is not None
            and np.all(np.isfinite(lnKa_err))
            and np.all(lnKa_err > 0)):
        chi2 = float(np.sum(((y_obs - y_pred) / lnKa_err) ** 2))
    else:
        chi2 = ss_res

    BIC = chi2 + k * np.log(n)
    if n - k - 1 > 0:
        AICc = chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    else:
        AICc = np.nan
    return R2, chi2, BIC, AICc


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_lvh(T_arr, lnKa_arr, lnKa_err, T0):
    """
    Fit the linear van't Hoff equation (ΔCp = 0) to ln(Ka) data.

    Free parameters: ΔH, ΔS. ln K_0 (the value at T0) and ΔG(T0) are
    derived, not fitted. See module docstring for the rationale.

    Parameters
    ----------
    T_arr     : 1-D array of temperatures (K)
    lnKa_arr  : 1-D array of mean ln(Ka) values
    lnKa_err  : 1-D array of std-err of ln(Ka) (may contain 0 or NaN)
    T0        : reference temperature (K)

    Returns
    -------
    dict with keys: dH, e_dH, dS, e_dS, Cp, e_Cp,
                    dG, e_dG, minus_TdS, e_minus_TdS,
                    Kd_uM, lnK0, e_lnK0,
                    R2, chi2, BIC, AICc, k_params, T0, lnKa_pred
    """
    # Initial guesses: ΔH = -10 kJ/mol, then ΔS chosen to match observed
    # rep mean at T0 (lnK0_init).
    if T0 in T_arr:
        lnK0_init = lnKa_arr[T_arr == T0][0]
    else:
        lnK0_init = float(np.interp(T0, T_arr, lnKa_arr))
    dH_init = -10.0
    dS_init = R_KJ * lnK0_init + dH_init / T0

    sigma_kw = {}
    if (lnKa_err is not None
            and np.all(np.isfinite(lnKa_err))
            and np.all(lnKa_err > 0)):
        sigma_kw = dict(sigma=lnKa_err, absolute_sigma=True)

    popt, pcov = curve_fit(lvh_equation, T_arr, lnKa_arr,
                           p0=[dH_init, dS_init], **sigma_kw)

    dH, dS = popt
    var_dH = pcov[0, 0]
    var_dS = pcov[1, 1]
    cov_dH_dS = pcov[0, 1]
    e_dH = np.sqrt(var_dH)
    e_dS = np.sqrt(var_dS)

    derived = _derived_at_T0(dH, dS, T0, var_dH, var_dS, cov_dH_dS)

    y_pred = lvh_equation(T_arr, dH, dS)
    k = 2  # LVH free params: ΔH, ΔS
    R2, chi2, BIC, AICc = _gof_metrics(lnKa_arr, y_pred, lnKa_err, k)

    return dict(
        dH=dH, e_dH=e_dH,
        dS=dS, e_dS=e_dS,
        Cp=0.0, e_Cp=0.0,
        **derived,
        R2=R2, chi2=chi2, BIC=BIC, AICc=AICc, k_params=k, T0=T0,
        lnKa_pred=y_pred,
    )


def fit_nlvh(T_arr, lnKa_arr, lnKa_err, T0):
    """
    Fit the NLVH equation to temperature-dependent ln(Ka) data.

    Free parameters: ΔH, ΔS at T0, and ΔCp. ΔG(T0) and lnK_0 are derived.

    Parameters
    ----------
    T_arr     : 1-D array of temperatures (K)
    lnKa_arr  : 1-D array of mean ln(Ka) values
    lnKa_err  : 1-D array of std-err of ln(Ka) (may contain 0 or NaN)
    T0        : reference temperature (K)

    Returns
    -------
    dict with same keys as fit_lvh plus a fitted Cp, e_Cp.
    """
    if T0 in T_arr:
        lnK0_init = lnKa_arr[T_arr == T0][0]
    else:
        lnK0_init = float(np.interp(T0, T_arr, lnKa_arr))
    dH_init = -10.0
    dS_init = R_KJ * lnK0_init + dH_init / T0
    Cp_init = 0.5

    def _model(T, dH, dS, Cp):
        return nlvh_equation(T, dH, dS, Cp, T0)

    sigma_kw = {}
    if (lnKa_err is not None
            and np.all(np.isfinite(lnKa_err))
            and np.all(lnKa_err > 0)):
        sigma_kw = dict(sigma=lnKa_err, absolute_sigma=True)

    popt, pcov = curve_fit(_model, T_arr, lnKa_arr,
                           p0=[dH_init, dS_init, Cp_init], **sigma_kw)

    dH, dS, Cp = popt
    var_dH = pcov[0, 0]
    var_dS = pcov[1, 1]
    var_Cp = pcov[2, 2]
    cov_dH_dS = pcov[0, 1]
    e_dH = np.sqrt(var_dH)
    e_dS = np.sqrt(var_dS)
    e_Cp = np.sqrt(var_Cp)

    # At T = T0 the ΔCp curvature term vanishes, so ΔG(T0), -TΔS(T0), lnK0
    # depend only on (dH, dS) and propagate identically to LVH.
    derived = _derived_at_T0(dH, dS, T0, var_dH, var_dS, cov_dH_dS)

    y_pred = _model(T_arr, dH, dS, Cp)
    k = 3  # NLVH free params: ΔH, ΔS, ΔCp
    R2, chi2, BIC, AICc = _gof_metrics(lnKa_arr, y_pred, lnKa_err, k)

    return dict(
        dH=dH, e_dH=e_dH,
        dS=dS, e_dS=e_dS,
        Cp=Cp, e_Cp=e_Cp,
        **derived,
        R2=R2, chi2=chi2, BIC=BIC, AICc=AICc, k_params=k, T0=T0,
        lnKa_pred=y_pred,
    )
