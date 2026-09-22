"""Thermodynamic fits from temperature-dependent Kd tables."""
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_kd_csv
from .nlvh import fit_lvh, fit_nlvh, statistical_correction


def _reorder_sites_kn_last(data):
    """Move Kn (column with subscript-n in label) to the last position.

    Returns (data_ref, kn_col_index_or_None). Modifies data in place.
    """
    labels = data["site_labels"]
    kn_idx = next((i for i, lab in enumerate(labels) if "\u2099" in lab), None)
    if kn_idx is None:
        return data, None
    if kn_idx == len(labels) - 1:
        return data, kn_idx

    order = [i for i in range(len(labels)) if i != kn_idx] + [kn_idx]
    data["site_labels"] = [labels[i] for i in order]
    for temp in data["kd_data"]:
        data["kd_data"][temp] = [rep[order] for rep in data["kd_data"][temp]]
    # Reorder paired σ arrays in lockstep so weighted-format inputs stay aligned
    if "kd_se_data" in data:
        for temp in data["kd_se_data"]:
            data["kd_se_data"][temp] = [se[order] for se in data["kd_se_data"][temp]]
    return data, len(labels) - 1


def _lnKa_per_replicate(data, n_for_correction, kn_col):
    """Build (temps_K, lnKa[temp, rep, site], temps_C) with statistical correction.

    Kn column (if present) gets no Wyman-Gill factor — it is a separate
    nonspecific interaction, not one of the equivalent sites.
    """
    temps_C = sorted(data["kd_data"].keys())
    temps_K = np.array([t + 273.15 for t in temps_C])
    n_cols = data["n_sites"]
    max_reps = max(len(data["kd_data"][tc]) for tc in temps_C)

    lnKa = np.full((len(temps_C), max_reps, n_cols), np.nan)
    for ti, tc in enumerate(temps_C):
        for ri, rep in enumerate(data["kd_data"][tc]):
            lnka = statistical_correction(rep, n_for_correction)
            if kn_col is not None and kn_col < len(rep):
                kd_val = rep[kn_col]
                if not np.isnan(kd_val) and kd_val > 0:
                    lnka[kn_col] = np.log(1.0 / (kd_val * 1e-6))
            lnKa[ti, ri, :] = lnka

    return temps_K, lnKa, np.array(temps_C)


def _lnKa_se_from_upstream(data, kn_col):
    """For weighted-format inputs, propagate upstream σ(Kd) to σ(lnKa).

    σ(lnKa) = σ(Kd) / Kd  (delta method on lnKa = -ln(Kd) + const).
    The Wyman-Gill conversion is a multiplicative factor on Ka, so it
    cancels in σ(lnKa). Returns array shape (n_temps, n_cols), or None
    if no upstream σ data is present.
    """
    if data.get("format") != "weighted":
        return None
    temps_C = sorted(data["kd_data"].keys())
    n_cols = data["n_sites"]
    se_arr = np.full((len(temps_C), n_cols), np.nan)
    for ti, tc in enumerate(temps_C):
        # Weighted format has exactly 1 row per temperature
        kd = data["kd_data"][tc][0]
        sd = data["kd_se_data"][tc][0]
        for si in range(n_cols):
            if (np.isfinite(kd[si]) and kd[si] > 0
                    and np.isfinite(sd[si]) and sd[si] > 0):
                se_arr[ti, si] = sd[si] / kd[si]
    return se_arr


def _short_label(raw):
    """Strip '(uM)' / '(μM)' suffixes from a site label."""
    for suf in ("(uM)", "(μM)"):
        raw = raw.replace(suf, "")
    return raw.strip()


def _format_four_significant_digits(x):
    """Format to 4 significant figures (%.4g). Auto scientific notation."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.nan
    return float(f"{x:.4g}")


def run_analysis(data, N, method, ref_temp_C=25.0, error_mode="covariance"):
    """Fit one dataset and return long-format site and replicate results."""
    data, kn_col = _reorder_sites_kn_last(data)
    T0 = ref_temp_C + 273.15
    fit_func = fit_lvh if method == "lvh" else fit_nlvh

    temps_K, lnKa, temps_C = _lnKa_per_replicate(data, N, kn_col)
    n_cols = data["n_sites"]
    max_reps = lnKa.shape[1]

    # Mean and SE-of-the-mean across replicates, per (temp, site).
    #
    # The variable name `lnKa_std` is kept for downstream-API compatibility,
    # but the quantity passed to curve_fit is SE = SD/sqrt(n_valid), NOT
    # the per-replicate sample standard deviation. Rationale: the y-value
    # we fit at each temperature is the replicate *mean* of ln K_a, whose
    # true uncertainty is the SE of the mean, not the sample SD. Using SD
    # as sigma overestimates each data-point uncertainty by sqrt(n).
    #
    # If the input dataset is in 'weighted' format (one row per T with
    # paired e_*(uM) columns from upstream replicate-weighted Kd fitting),
    # we use those σ_Kd values directly: σ(lnKa) = σ(Kd) / Kd. This
    # propagates uncertainty from the spectrum-level fit straight into
    # the van't Hoff regression without re-aggregating per-rep Kds.
    lnKa_mean = np.nanmean(lnKa, axis=1)
    # Sample SD across replicates (ddof=1) — used for the van't Hoff plot
    # error bars when reporting under the per-rep avg±SD convention.
    with np.errstate(invalid="ignore"):
        lnKa_sd = np.nanstd(lnKa, axis=1, ddof=1)
    lnKa_std = np.full_like(lnKa_mean, np.nan)   # name retained; now SE not SD
    upstream_se = _lnKa_se_from_upstream(data, kn_col)
    if upstream_se is not None:
        lnKa_std = upstream_se          # use upstream σ_Kd / Kd directly
    else:
        for ti in range(len(temps_C)):
            for si in range(n_cols):
                vals = lnKa[ti, :, si]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= 2:
                    sd = np.std(valid, ddof=1)
                    lnKa_std[ti, si] = sd / np.sqrt(len(valid))
                elif len(valid) == 1:
                    lnKa_std[ti, si] = 0.0

    if error_mode not in ("covariance", "replicate", "sd"):
        raise ValueError(
            f"error_mode must be 'covariance', 'replicate', or 'sd', got {error_mode!r}"
        )

    rows = []
    for si in range(n_cols):
        is_kn = (si == kn_col)
        if not is_kn and (si + 1) > N:
            continue  # i > N → statistical correction undefined
        label = _short_label(data["site_labels"][si])

        # Per-replicate fits (unweighted). Cache the fit dicts so we can
        # compute replicate-spread σ later if error_mode='replicate'.
        rep_fits = []
        for ri in range(max_reps):
            y = lnKa[:, ri, si]
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                continue
            try:
                res = fit_func(temps_K[mask], y[mask], None, T0)
            except Exception as exc:  # noqa: BLE001 - continue other replicates
                warnings.warn(f"Site {si + 1} replicate {ri + 1} fit failed: {exc}", RuntimeWarning, stacklevel=2)
                continue
            rep_fits.append(res)
            rows.append({
                "site": si + 1, "label": label, "is_kn": is_kn,
                "N": N, "method": method, "replicate": f"rep{ri + 1}",
                "T0_K": T0, "n_temps": int(mask.sum()),
                "dH_kJmol": _format_four_significant_digits(res["dH"]), "e_dH": None,
                "dS_kJmolK": _format_four_significant_digits(res["dS"]), "e_dS": None,
                "Cp_kJmolK": _format_four_significant_digits(res["Cp"]), "e_Cp": None,
                "dG_kJmol": _format_four_significant_digits(res["dG"]), "e_dG": None,
                "mTdS_kJmol": _format_four_significant_digits(res["minus_TdS"]), "e_mTdS": None,
                "Kd_uM": _format_four_significant_digits(res["Kd_uM"]),
                "R2": _format_four_significant_digits(res["R2"]),
            })

        # Weighted mean-fit (primary result). The same fit is used for both
        # error modes; only the e_* fields written to the mean row differ.
        y_m = lnKa_mean[:, si]
        yerr_m = lnKa_std[:, si]
        mask_m = ~np.isnan(y_m)
        if mask_m.sum() < 3:
            continue
        try:
            mf = fit_func(temps_K[mask_m], y_m[mask_m], yerr_m[mask_m], T0)
        except Exception as exc:  # noqa: BLE001 - continue other sites
            warnings.warn(f"Site {si + 1} mean fit failed: {exc}", RuntimeWarning, stacklevel=2)
            continue

        if error_mode == "covariance":
            # σ(ΔH), σ(ΔS), σ(ΔCp) from pcov diagonal; σ(ΔG), σ(-TΔS) via
            # delta-method propagation through cov(ΔH, ΔS) — see
            # thermodynamics/nlvh.py::_derived_at_T0.
            e_dH, e_dS, e_Cp = mf["e_dH"], mf["e_dS"], mf["e_Cp"]
            e_dG, e_mTdS = mf["e_dG"], mf["e_minus_TdS"]
        elif error_mode == "replicate":
            # σ = SD / √n_reps of each fitted quantity across the per-rep fits.
            # Captures replicate-to-replicate reproducibility but with only
            # df = n_reps − 1, so estimates are noisy at n_reps = 3.
            e_dH = _rep_se(rep_fits, "dH")
            e_dS = _rep_se(rep_fits, "dS")
            e_Cp = _rep_se(rep_fits, "Cp")
            e_dG = _rep_se(rep_fits, "dG")
            e_mTdS = _rep_se(rep_fits, "minus_TdS")
        else:  # 'sd'
            # σ = plain SD across the per-replicate fit values (no √n divisor).
            # Matches the per-rep avg±SD convention used in §1.
            e_dH = _rep_sd(rep_fits, "dH")
            e_dS = _rep_sd(rep_fits, "dS")
            e_Cp = _rep_sd(rep_fits, "Cp")
            e_dG = _rep_sd(rep_fits, "dG")
            e_mTdS = _rep_sd(rep_fits, "minus_TdS")

        # For error_mode='sd', the central value is the per-replicate mean
        # of each quantity (matches avg±SD reporting elsewhere). For other
        # modes, the central value is the weighted mean-fit value.
        if error_mode == "sd":
            dH_c    = _rep_mean(rep_fits, "dH")
            dS_c    = _rep_mean(rep_fits, "dS")
            Cp_c    = _rep_mean(rep_fits, "Cp")
            dG_c    = _rep_mean(rep_fits, "dG")
            mTdS_c  = _rep_mean(rep_fits, "minus_TdS")
            Kd_c    = _rep_mean(rep_fits, "Kd_uM")
            R2_c    = _rep_mean(rep_fits, "R2")
            chi2_c  = _rep_mean(rep_fits, "chi2")
            BIC_c   = _rep_mean(rep_fits, "BIC")
            AICc_c  = _rep_mean(rep_fits, "AICc")
        else:
            dH_c, dS_c, Cp_c = mf["dH"], mf["dS"], mf["Cp"]
            dG_c, mTdS_c     = mf["dG"], mf["minus_TdS"]
            Kd_c, R2_c       = mf["Kd_uM"], mf["R2"]
            chi2_c           = mf.get("chi2", np.nan)
            BIC_c            = mf.get("BIC", np.nan)
            AICc_c           = mf.get("AICc", np.nan)

        rows.append({
            "site": si + 1, "label": label, "is_kn": is_kn,
            "N": N, "method": method, "replicate": "mean",
            "T0_K": T0, "n_temps": int(mask_m.sum()),
            "error_mode": error_mode,
            "dH_kJmol": _format_four_significant_digits(dH_c), "e_dH": _format_four_significant_digits(e_dH),
            "dS_kJmolK": _format_four_significant_digits(dS_c), "e_dS": _format_four_significant_digits(e_dS),
            "Cp_kJmolK": _format_four_significant_digits(Cp_c), "e_Cp": _format_four_significant_digits(e_Cp),
            "dG_kJmol": _format_four_significant_digits(dG_c), "e_dG": _format_four_significant_digits(e_dG),
            "mTdS_kJmol": _format_four_significant_digits(mTdS_c), "e_mTdS": _format_four_significant_digits(e_mTdS),
            "Kd_uM": _format_four_significant_digits(Kd_c),
            "R2": _format_four_significant_digits(R2_c),
            "chi2": _format_four_significant_digits(chi2_c),
            "BIC": _format_four_significant_digits(BIC_c),
            "AICc": _format_four_significant_digits(AICc_c),
        })

    return rows, temps_K, temps_C, lnKa_mean, lnKa_std, kn_col, lnKa_sd


def _rep_se(rep_fits, key):
    """Standard error of the mean across replicate fits for one quantity.

    Returns NaN if fewer than 2 replicate fits succeeded (df < 1).
    """
    vals = [r[key] for r in rep_fits if r is not None and key in r
            and np.isfinite(r[key])]
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _rep_sd(rep_fits, key):
    """Plain SD across replicate fits for one quantity (no √n divisor)."""
    vals = [r[key] for r in rep_fits if r is not None and key in r
            and np.isfinite(r[key])]
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals, ddof=1))


def _rep_mean(rep_fits, key):
    """Mean across replicate fits for one quantity. NaN if no valid fits."""
    vals = [r[key] for r in rep_fits if r is not None and key in r
            and np.isfinite(r[key])]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def run_grid(csv_path, n_values, methods=("nlvh", "lvh"), ref_temp_C=25.0,
             error_mode="covariance"):
    """Run `run_analysis` for every (N, method) combo on one CSV file.

    Returns (df_long, fits_by_key) where fits_by_key is a dict:
        (N, method) → (temps_K, temps_C, lnKa_mean, lnKa_std, rows_for_this_combo, kn_col)
    so the plotting code can re-fit and draw curves without re-running.

    `error_mode` is passed through to `run_analysis`; it switches between
    fit-covariance and replicate-spread error bars uniformly across LVH/NLVH.
    """
    all_rows = []
    fits_by_key = {}
    for N, method in product(n_values, methods):
        data = load_kd_csv(csv_path)
        rows, temps_K, temps_C, lnKa_mean, lnKa_std, kn_col, lnKa_sd = run_analysis(
            data, N, method, ref_temp_C=ref_temp_C, error_mode=error_mode,
        )
        for r in rows:
            r["dataset"] = Path(csv_path).stem
        all_rows.extend(rows)
        fits_by_key[(N, method)] = {
            "temps_K": temps_K, "temps_C": temps_C,
            "lnKa_mean": lnKa_mean, "lnKa_std": lnKa_std,
            "lnKa_sd": lnKa_sd,
            "rows": rows, "kn_col": kn_col,
            "site_labels": data["site_labels"],
        }
    df = pd.DataFrame(all_rows)
    return df, fits_by_key
