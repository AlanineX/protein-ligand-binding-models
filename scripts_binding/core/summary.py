"""Multi-replicate summary + compact all-model output writing."""
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..models import REGISTRY, deconvolve_terms
from ..models.metadata import (
    base_model_name,
    display_model_name,
    is_dimensionless_param,
    is_sequential_specific_model,
    model_role,
    output_parameter_name,
)
from .plotting import safe_savefig, plot_species_curves, plot_deconv_byconc


def _nanmean_no_warn(values, axis):
    """np.nanmean equivalent that returns NaN for empty slices without warning."""
    arr = np.asarray(values, dtype=float)
    count = np.sum(np.isfinite(arr), axis=axis)
    total = np.nansum(arr, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _nanstd_no_warn(values, axis):
    """np.nanstd equivalent that returns NaN for empty slices without warning."""
    arr = np.asarray(values, dtype=float)
    mean = np.expand_dims(_nanmean_no_warn(arr, axis), axis)
    count = np.sum(np.isfinite(arr), axis=axis)
    ss = np.nansum((arr - mean) ** 2, axis=axis)
    var = np.divide(ss, count, out=np.full_like(ss, np.nan, dtype=float), where=count > 0)
    return np.sqrt(var)


def compute_model_grid(L_grid_M, lnK_opt, S_eff, N_eff, p_total_m, model_name):
    """F_grid from any model at each L_grid_M value."""
    model = REGISTRY[base_model_name(model_name)]
    F_grid = []
    for L in L_grid_M:
        Lf = model.free_ligand(L, p_total_m, lnK_opt, S_eff, N_eff)
        F_grid.append(model.mole_fractions(Lf, lnK_opt, S_eff, N_eff))
    return np.array(F_grid)


def build_summary(all_L_tot, all_F_exp, all_Kd, all_stems, num_species_list,
                  all_lnK, all_S, all_N, model_name, out_dir, label, cfg):
    """Multi-file mean/std summary: fit plot, plus legacy Kd CSV outside compact mode."""
    ref_L = np.unique(np.concatenate([
        np.asarray(x, dtype=float)[np.isfinite(x)] for x in all_L_tot
    ]))
    if not len(ref_L):
        raise ValueError("No measured ligand concentrations for replicate summary")
    num_species = max(num_species_list)
    is_specific = is_sequential_specific_model(model_name)

    # Keep only measured replicate points at each concentration.
    F_exp_list = []
    for L_tot, F_exp in zip(all_L_tot, all_F_exp):
        observed = np.full((len(ref_L), num_species), np.nan)
        for row, concentration in zip(F_exp, L_tot):
            if not np.isfinite(concentration):
                continue
            index = np.searchsorted(ref_L, concentration)
            observed[index, :len(row)] = row
        F_exp_list.append(observed)

    F_arr = np.stack(F_exp_list, axis=0)
    F_exp_mean = _nanmean_no_warn(F_arr, axis=0)
    counts = np.sum(np.isfinite(F_arr), axis=0)
    squared = np.nansum((F_arr - F_exp_mean) ** 2, axis=0)
    F_exp_std = np.sqrt(np.divide(squared, counts - 1,
                                  out=np.full_like(squared, np.nan), where=counts > 1))

    # Model curves per run on a common grid
    L_grid_M = np.linspace(ref_L.min(), ref_L.max(), 300)
    calc_runs = []
    for lnK_opt, S_eff, N_eff in zip(all_lnK, all_S, all_N):
        F_grid_run = compute_model_grid(L_grid_M, lnK_opt, S_eff, N_eff, cfg.p_total_m, model_name)
        if F_grid_run.shape[1] < num_species:
            F_grid_run = np.pad(F_grid_run, ((0, 0), (0, num_species - F_grid_run.shape[1])), "constant")
        calc_runs.append(F_grid_run)
    calc_arr = np.stack(calc_runs, axis=0)
    F_calc_mean = _nanmean_no_warn(calc_arr, axis=0)
    F_calc_std = _nanstd_no_warn(calc_arr, axis=0)

    n_specific = None if is_specific else max(set(all_S), key=all_S.count)
    summary_stem = os.path.basename(cfg.csv_name_wildcard).replace("*", "").replace(".csv", "")
    summary_fit_svg = os.path.join(out_dir, summary_stem + "fit_summary.svg")
    model_species_count = max(S + N + 1 for S, N in zip(all_S, all_N))

    # Mean Kd (in output units) — pad shorter arrays with NaN. Calculate this
    # before plotting so sequential-specific legends can show Kd for each step.
    max_kd_len = max(len(kd) for kd in all_Kd)
    kd_padded = [np.pad(kd, (0, max_kd_len - len(kd)), constant_values=np.nan) if len(kd) < max_kd_len else kd
                 for kd in all_Kd]
    kd_values = np.stack(kd_padded, axis=1)
    mean_Kd_out = _nanmean_no_warn(kd_values, axis=1)

    plot_species_curves(L_grid_M, F_calc_mean, num_species, cfg, output_svg=summary_fit_svg,
                        n_specific=n_specific, data_L_M=ref_L, data_F=F_exp_mean,
                        data_err=F_exp_std, calc_std=F_calc_std,
                        model_species_count=model_species_count,
                        legend_kd_values=(mean_Kd_out if is_specific and getattr(cfg, "show_kd_in_legend", False) else None),
                        legend_kd_unit=cfg.output_unit)
    print(f"[{label} Summary] Wrote summary fit plot to {summary_fit_svg}")

    kd_unit = cfg.output_unit
    param_names = REGISTRY[base_model_name(model_name)].param_labels(n_specific if n_specific is not None else max_kd_len)
    if len(param_names) < max_kd_len:
        param_names += [f"p_{i}" for i in range(len(param_names), max_kd_len)]
    param_names = param_names[:max_kd_len]

    summary = {
        "Result": param_names,
        f"Mean_Kd_({kd_unit})": mean_Kd_out,
        f"Std_Kd_({kd_unit})": _nanstd_no_warn(kd_values, axis=1),
    }
    for col_idx, stem in enumerate(all_stems):
        summary[stem] = kd_values[:, col_idx]
    if not getattr(cfg, "compact_outputs", False):
        summary_file = os.path.join(out_dir, summary_stem + "stat_summary.csv")
        pd.DataFrame(summary).to_csv(summary_file, index=False, float_format="%.6e")
        print(f"[{label} Summary] Wrote summary Kd table to {summary_file}")

    return ref_L, F_exp_mean, F_exp_std, kd_values, mean_Kd_out, num_species


def build_deconv_summary(model_name, ref_L, F_exp_mean, F_exp_std, mean_lnK,
                         summary_S, num_species, out_dir, cfg):
    """Replicate-averaged specific/nonspecific deconvolution for any NSB model
    that exposes `partition_terms`. Writes the summary figure, plus legacy CSV
    outside compact mode.

    Uses the replicate-mean ln-parameters (averaged in ln space, so gamma and
    log-K are both handled) to reconstruct the model curves, then splits each
    apparent peak into (j specific, i-j nonspecific) via the model's partition
    terms.
    """
    model = REGISTRY[base_model_name(model_name)]
    summary_N = num_species - summary_S - 1

    F_calc_mean_ref = np.array([
        model.mole_fractions(
            model.free_ligand(L, cfg.p_total_m, mean_lnK, summary_S, summary_N),
            mean_lnK, summary_S, summary_N,
        )
        for L in ref_L
    ])
    if F_calc_mean_ref.shape[1] < num_species:
        F_calc_mean_ref = np.pad(F_calc_mean_ref, ((0, 0), (0, num_species - F_calc_mean_ref.shape[1])), "constant")

    weights_ij = model.partition_terms(mean_lnK, summary_S, summary_N)
    contrib_all = np.zeros((len(ref_L), num_species, summary_S + 1))
    frac_within_all = np.zeros((len(ref_L), num_species, summary_S + 1))
    for idx in range(len(ref_L)):
        frac_within, contrib = deconvolve_terms(F_calc_mean_ref[idx], weights_ij, summary_S)
        contrib_all[idx] = contrib
        frac_within_all[idx] = frac_within

    summary_stem = os.path.basename(cfg.csv_name_wildcard).replace("*", "").replace(".csv", "")
    summary_deconv_svg = os.path.join(out_dir, summary_stem + "deconv_summary.svg")
    fig = plot_deconv_byconc(
        ref_L * cfg.scale_m_to_out, contrib_all, summary_S, summary_N,
        "Deconvoluted fraction of apparent", cfg,
        outline_totals=F_exp_mean, outline_err=F_exp_std, outline_label="Frac_expt",
    )
    safe_savefig(fig, summary_deconv_svg, cfg.max_image_dim)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(f"[{model_name} Summary] Wrote summary deconvolution plot to {summary_deconv_svg}")

    csv_rows = []
    for idx in range(len(ref_L)):
        L_out = ref_L[idx] * cfg.scale_m_to_out
        for i in range(num_species):
            max_j = min(i, summary_S)
            for j in range(max_j + 1):
                csv_rows.append({
                    f"L_tot({cfg.output_unit})": L_out,
                    "i_total": i, "j_specific": j, "m_nonspecific": i - j,
                    "F_exp_mean": F_exp_mean[idx][i] if i < F_exp_mean.shape[1] else np.nan,
                    "F_exp_std": F_exp_std[idx][i] if i < F_exp_std.shape[1] else np.nan,
                    "F_calc_mean": F_calc_mean_ref[idx][i],
                    "fraction_within_i": frac_within_all[idx][i, j],
                    "fraction_total": contrib_all[idx][i, j],
                })
    deconv_df = pd.DataFrame(csv_rows)
    if not getattr(cfg, "compact_outputs", False):
        summary_deconv_csv = os.path.join(out_dir, summary_stem + "deconv_summary.csv")
        deconv_df.to_csv(summary_deconv_csv, index=False)
        print(f"[{model_name} Summary] Wrote summary deconvolution CSV to {summary_deconv_csv}")

    return summary_S, deconv_df


def _parse_run_context(cfg, stem):
    """Context columns shared by compact output tables."""
    system_name = getattr(cfg, "system_name", "") or ""
    parts = [p for p in system_name.split("_") if p]
    analyte = parts[0].upper() if parts else ""
    buffer = parts[1].upper() if len(parts) > 1 else os.path.basename(cfg.base_dir).upper()
    temp = getattr(cfg, "temperature_C", None)
    if temp is None:
        m_temp = re.search(r"_(\d+)C(?:_|$)", stem)
        temp = int(m_temp.group(1)) if m_temp else np.nan
    m_rep = re.search(r"_(\d+)$", stem)
    replicate_id = int(m_rep.group(1)) if m_rep else stem
    return {
        "analyte": analyte,
        "buffer": buffer,
        "temperature_C": temp,
        "replicate_id": replicate_id,
    }


def _kd_symbol(param_name):
    if param_name.startswith("Ks_"):
        return "K_d" + param_name.split("_", 1)[1]
    if param_name in ("Kn", "beta"):
        return "K_d,n"
    if param_name == "Ks":
        return "K_d"
    if param_name == "gamma":
        return "gamma"
    return param_name


def _parameter_class(param_name):
    if param_name == "gamma":
        return "empirical_shape"
    if param_name == "Ks":
        return "shared_specific_binding"
    if param_name.startswith("Ks_"):
        return "specific_binding"
    if param_name in ("Kn", "beta"):
        return "nonspecific_binding"
    return "other"


def _ci95_from_log_se(value, se_log_or_value, log_scale):
    if value is None or not np.isfinite(value):
        return np.nan, np.nan
    if se_log_or_value is None or not np.isfinite(se_log_or_value):
        return np.nan, np.nan
    half = 1.96 * float(se_log_or_value)
    if log_scale:
        if value <= 0:
            return np.nan, np.nan
        log_value = np.log(value)
        lo_log = log_value - half
        hi_log = log_value + half
        lo = 0.0 if lo_log < -745.0 else float(np.exp(lo_log))
        hi = np.inf if hi_log > 709.0 else float(np.exp(hi_log))
        return lo, hi
    return float(value - half), float(value + half)


def _metrics_from_result(d):
    ssr = float(d["SSR"])
    n = int(d["n_obs"])
    k = int(d["n_params"])
    bic = n * np.log(ssr / n) + k * np.log(n)
    aic = n * np.log(ssr / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    sst = d.get("SST")
    r2 = 1.0 - ssr / sst if sst and sst > 0 else np.nan
    dof = n - k
    rmse = np.sqrt(ssr / dof) if dof > 0 else np.nan
    return bic, aic, aicc, r2, rmse


def _extra_sum_squares_ftest(ssr_red, k_red, ssr_full, k_full, n):
    from scipy.stats import f as _f_dist
    if k_full <= k_red or n <= k_full or not ssr_full > 0:
        return np.nan, np.nan, np.nan, np.nan
    df1, df2 = k_full - k_red, n - k_full
    F = ((ssr_red - ssr_full) / df1) / (ssr_full / df2)
    p = float(_f_dist.sf(F, df1, df2)) if F > 0 else 1.0
    return F, df1, df2, p


def _build_fit_metrics(per_model_results, cfg):
    rows = []
    for model_name, results in per_model_results.items():
        for d in results:
            context = _parse_run_context(cfg, d["stem"])
            bic, aic, aicc, r2, rmse = _metrics_from_result(d)
            rows.append({
                **context,
                "model_name": model_name,
                "specific_site_count": d.get("S_eff", np.nan),
                "nonspecific_site_count": d.get("N_eff", np.nan),
                "observed_max_stoichiometry": d.get("observed_max_stoichiometry", np.nan),
                "n_parameters": d["n_params"],
                "n_observations": d["n_obs"],
                "degrees_of_freedom": d["n_obs"] - d["n_params"],
                "SSR": d["SSR"],
                "RMSE": rmse,
                "R2": r2,
                "AIC": aic,
                "AICc": aicc,
                "BIC": bic,
                "fit_success": d.get("fit_success", np.nan),
                "constraint_fallback_used": d.get("fit_constraint_fallback_used", False),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ordered = [
        "analyte", "buffer", "temperature_C", "replicate_id", "model_name",
        "specific_site_count", "nonspecific_site_count", "observed_max_stoichiometry",
        "n_parameters", "n_observations", "degrees_of_freedom", "SSR", "RMSE",
        "R2",
        "AIC", "AICc", "BIC", "fit_success", "constraint_fallback_used",
    ]
    return df[ordered]


def _build_fit_parameters(per_model_results, cfg):
    rows = []
    sd_lookup = {}
    for model_name, results in per_model_results.items():
        for d in results:
            context = _parse_run_context(cfg, d["stem"])
            labels = d.get("param_names", [])
            values = d.get("param_values", [])
            kd_vals = d.get("Kd_out", [])
            std_param = d.get("std_param")
            std_kd = d.get("std_Kd_out")
            if std_kd is None:
                std_kd = d.get("e_Kd_uM", [])
            for idx, label in enumerate(labels):
                is_dimless = is_dimensionless_param(label)
                parameter_value = values[idx] if idx < len(values) else np.nan
                kd_uM = kd_vals[idx] if idx < len(kd_vals) else np.nan
                se_log = std_param[idx] if std_param is not None and idx < len(std_param) else np.nan
                if is_dimless:
                    se_kd_uM = np.nan
                    sd_kd_uM = np.nan
                else:
                    se_kd_uM = std_kd[idx] if std_kd is not None and idx < len(std_kd) else np.nan
                    sd_kd_uM = se_kd_uM
                rows.append({
                    **context,
                    "model_name": model_name,
                    "parameter_name": label,
                    "parameter_value": parameter_value,
                    "Kd_uM": kd_uM,
                    "SD_Kd_uM": sd_kd_uM,
                    "SE_Kd_uM": se_kd_uM,
                })
                if np.isfinite(kd_uM):
                    key = (
                        context["analyte"], context["buffer"], context["temperature_C"],
                        model_name, label,
                    )
                    sd_lookup.setdefault(key, []).append(kd_uM)

    ordered = [
        "analyte", "buffer", "temperature_C", "replicate_id", "model_name",
        "parameter_name", "parameter_value", "Kd_uM", "SD_Kd_uM", "SE_Kd_uM",
    ]
    df = pd.DataFrame(rows, columns=ordered)

    # Fill SD with the across-replicate spread for matching model/parameter groups
    # when there are multiple replicates (typically 3). This is useful for quick
    # consistency checks alongside per-fit SE.
    if not df.empty:
        for key, vals in sd_lookup.items():
            vals_arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
            if len(vals_arr) <= 1:
                continue
            sd_val = float(np.std(vals_arr, ddof=1))
            mask = (
                (df["analyte"] == key[0])
                & (df["buffer"] == key[1])
                & (df["temperature_C"] == key[2])
                & (df["model_name"] == key[3])
                & (df["parameter_name"] == key[4])
            )
            df.loc[mask, "SD_Kd_uM"] = sd_val
    return df


def _build_nested_ftests(per_model_results, cfg):
    rows = []
    reference = getattr(cfg, "reference_model", "sequential_specific_s7")
    nested = set(getattr(cfg, "nested_ftest_models", []) or [])
    if reference == "sequential_specific_s7":
        for apparent_model in ("sequential_specific_s9", "sequential_specific_s10"):
            if apparent_model in per_model_results:
                nested.add(apparent_model)
    if reference not in per_model_results:
        return pd.DataFrame(columns=[
            "analyte", "buffer", "temperature_C", "replicate_id",
            "reduced_model_name", "full_model_name",
            "n_parameters_reduced", "n_parameters_full", "n_observations_used",
            "SSR_reduced", "SSR_full", "delta_SSR", "F_value", "df_numerator",
            "df_denominator", "p_value",
        ])
    ref_by_file = {d["stem"]: d for d in per_model_results[reference]}
    for full_model in nested:
        if full_model not in per_model_results:
            continue
        for full in per_model_results[full_model]:
            red = ref_by_file.get(full["stem"])
            if red is None:
                continue
            n_used = min(red["n_obs"], full["n_obs"])
            F, df1, df2, p = _extra_sum_squares_ftest(
                red["SSR"], red["n_params"], full["SSR"], full["n_params"], n_used
            )
            rows.append({
                **_parse_run_context(cfg, full["stem"]),
                "reduced_model_name": reference,
                "full_model_name": full_model,
                "n_parameters_reduced": red["n_params"],
                "n_parameters_full": full["n_params"],
                "n_observations_used": n_used,
                "SSR_reduced": red["SSR"],
                "SSR_full": full["SSR"],
                "delta_SSR": red["SSR"] - full["SSR"],
                "F_value": F,
                "df_numerator": df1,
                "df_denominator": df2,
                "p_value": p,
            })
    return pd.DataFrame(rows)


def _markdown_table(df, columns, max_rows=None):
    if df.empty:
        return "_No rows._\n"
    sub = df[columns].copy()
    if max_rows is not None:
        sub = sub.head(max_rows)
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    lines = [
        "| " + " | ".join(sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in sub.columns) + " |")
    return "\n".join(lines) + "\n"


def _round_sig(value, sig=4):
    if value is None or pd.isna(value):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(v):
        return value
    return float(f"{v:.{sig}g}")


def _round_numeric_columns(df, sig=4):
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].apply(_round_sig)
    return out


def _autosize_excel_columns(writer, df, sheet_name):
    if sheet_name not in writer.sheets:
        return
    worksheet = writer.sheets[sheet_name]
    min_width = 8
    max_width = 50
    pad = 2
    for col_idx, col in enumerate(df.columns):
        max_len = len(str(col))
        for value in df[col].tolist():
            if pd.isna(value):
                continue
            max_len = max(max_len, len(str(value)))
        width = max(min_width, min(max_width, max_len + pad))
        worksheet.set_column(col_idx, col_idx, width)


def _format_mean_sd(mean, sd, unit=""):
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        return f"{mean:.4g}{unit}"
    return f"{mean:.4g} ± {sd:.4g}{unit}"


def _display_output_unit(unit):
    return "µM" if str(unit).lower() == "um" else str(unit)


def _output_parameter_label(parameter_name):
    """Display-only labels for compact summary sheets."""
    return output_parameter_name(parameter_name)


def _format_summary_value(values, unit=""):
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return ""
    sd = vals.std(ddof=1) if len(vals) > 1 else np.nan
    return _format_mean_sd(vals.mean(), sd, unit)


def _build_model_summary_sheet(fit_metrics, fit_parameters, nested_ftests, cfg):
    if fit_metrics.empty:
        return pd.DataFrame(columns=["parameter"])

    model_names = fit_metrics["model_name"].drop_duplicates().tolist()
    model_columns = {
        model_name: display_model_name(model_name, getattr(cfg, "model_display_names", {}))
        for model_name in model_names
    }
    rows = []

    def add_row(label, values_by_model):
        rows.append({
            "parameter": label,
            **{model_columns[model_name]: values_by_model.get(model_name, "") for model_name in model_names},
        })

    kd_unit = f" {_display_output_unit(getattr(cfg, 'output_unit', 'uM'))}"
    params = fit_parameters.copy()
    if not params.empty:
        params["param_order"] = params["parameter_name"].map(_parameter_sort_key)
        params["display_parameter_name"] = params["parameter_name"].map(_output_parameter_label)
        ordered_params = (
            params[["display_parameter_name", "param_order"]]
            .drop_duplicates()
            .sort_values(["param_order", "display_parameter_name"])["display_parameter_name"]
            .drop_duplicates()
            .tolist()
        )
        for display_pname in ordered_params:
            values_by_model = {}
            for model_name in model_names:
                psub = params[
                    (params["model_name"] == model_name)
                    & (params["display_parameter_name"] == display_pname)
                ]
                if psub.empty:
                    continue
                kd_vals = pd.to_numeric(psub["Kd_uM"], errors="coerce")
                if kd_vals.notna().any():
                    values_by_model[model_name] = _format_summary_value(kd_vals, kd_unit)
                else:
                    values_by_model[model_name] = _format_summary_value(psub["parameter_value"])
            add_row(display_pname, values_by_model)

    metrics = fit_metrics.copy()
    for label, col in [
        ("SSR", "SSR"),
        ("RMSE", "RMSE"),
        ("R2", "R2"),
        ("AIC", "AIC"),
        ("AICc", "AICc"),
        ("BIC", "BIC"),
    ]:
        values_by_model = {}
        for model_name in model_names:
            sub = metrics[metrics["model_name"] == model_name]
            values_by_model[model_name] = _format_summary_value(sub[col])
        add_row(label, values_by_model)

    reference_model = getattr(cfg, "reference_model", "sequential_specific_s7")
    if not nested_ftests.empty:
        for label, col in [
            (f"F vs {display_model_name(reference_model, getattr(cfg, 'model_display_names', {}))}", "F_value"),
            (f"p vs {display_model_name(reference_model, getattr(cfg, 'model_display_names', {}))}", "p_value"),
        ]:
            values_by_model = {}
            for model_name in model_names:
                sub = nested_ftests[nested_ftests["full_model_name"] == model_name]
                values_by_model[model_name] = _format_summary_value(sub[col])
            add_row(label, values_by_model)
    else:
        reference_label = display_model_name(reference_model, getattr(cfg, "model_display_names", {}))
        add_row(f"F vs {reference_label}", {})
        add_row(f"p vs {reference_label}", {})

    return pd.DataFrame(rows, columns=["parameter", *[model_columns[m] for m in model_names]])


def _parameter_warning_tags(row):
    tags = []
    fit_ok = row.get("fit_success", True)
    if pd.isna(fit_ok) or bool(fit_ok):
        pass
    else:
        tags.append("fit_not_converged")

    kd = row.get("Kd_uM", np.nan)
    se = row.get("SE_Kd_uM", np.nan)
    if pd.isna(kd) or kd <= 0 or kd > 1e6:
        if np.isfinite(kd):
            tags.append("unphysical_kd")
    else:
        if not np.isfinite(se) or (se / kd > 3):
            tags.append("wide_or_unbounded")

    if not tags:
        return "ok"
    return ", ".join(tags)


def _aggregate_warning_tags(tag_values):
    """Combine multiple warning strings into a sorted unique summary."""
    tags = set()
    for tag_text in tag_values:
        if tag_text is None or pd.isna(tag_text):
            continue
        if isinstance(tag_text, str):
            for tag in tag_text.split(","):
                tag = tag.strip()
                if tag and tag != "ok":
                    tags.add(tag)
        else:
            if str(tag_text) != "ok":
                tags.add(str(tag_text))
    if not tags:
        return "ok"
    return ", ".join(sorted(tags))


def _parameter_sort_key(parameter_name):
    if parameter_name is None:
        return (99, 99, "")
    if isinstance(parameter_name, str):
        m = re.match(r"^Ks_(\d+)$", parameter_name)
        if m:
            return (0, int(m.group(1)), parameter_name)
        m = re.match(r"^Kd,(\d+)$", parameter_name)
        if m:
            return (0, int(m.group(1)), parameter_name)
        if parameter_name == "Ks":
            return (1, 0, parameter_name)
        if parameter_name == "Kd,avg":
            return (1, 0, parameter_name)
        if parameter_name == "Kn":
            return (2, 0, parameter_name)
        if parameter_name == "Kd,n":
            return (2, 0, parameter_name)
        if parameter_name == "beta":
            return (3, 0, parameter_name)
        if parameter_name == "gamma":
            return (4, 0, parameter_name)
    return (5, 0, str(parameter_name))


def _write_summary_report(out_dir, fit_metrics, fit_parameters, nested_ftests, cfg):
    report_path = os.path.join(out_dir, "summary_report.md")
    display_overrides = getattr(cfg, "model_display_names", {})

    def _display(model_name):
        return display_model_name(model_name, display_overrides)

    group_cols = ["model_name"]
    ranking = (
        fit_metrics.groupby(group_cols, dropna=False)
        .agg(
            AICc_mean=("AICc", "mean"),
            AICc_std=("AICc", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
        .sort_values("AICc_mean")
    )
    if not ranking.empty:
        ranking["AICc_rank"] = np.arange(1, len(ranking) + 1)
    ranking["AICc±SD"] = ranking.apply(
        lambda row: _format_mean_sd(row["AICc_mean"], row["AICc_std"]),
        axis=1,
    )
    ranking["R2_vs_sd"] = ranking.apply(
        lambda row: _format_mean_sd(row["R2_mean"], row["R2_std"]),
        axis=1,
    )
    if nested_ftests.empty:
        ranking["mean_p_value"] = np.nan
    else:
        p_means = (
            nested_ftests.groupby("full_model_name")["p_value"]
            .mean()
            .to_dict()
        )
        ranking["mean_p_value"] = ranking["model_name"].map(
            lambda model_name: p_means.get(model_name, np.nan)
        )

    model_order_map = {
        m: r for r, m in enumerate(ranking["model_name"].tolist(), start=1)
    } if not ranking.empty else {}

    fp_work = fit_parameters.copy()
    context_cols = [
        "analyte", "buffer", "temperature_C", "replicate_id", "model_name",
    ]
    if all(c in fit_metrics.columns for c in context_cols + ["fit_success"]):
        fp_work = fp_work.merge(
            fit_metrics[context_cols + ["fit_success"]],
            on=context_cols,
            how="left",
        )
    fp_work["warning_row_tag"] = fp_work.apply(_parameter_warning_tags, axis=1)
    fp_work["fit_success"] = fp_work["fit_success"].astype("boolean")

    failed_models = set(
        fp_work.loc[
            fp_work["fit_success"].notna() & ~fp_work["fit_success"],
            "model_name",
        ].tolist()
    )
    model_warning_by_param = (
        fp_work.groupby("model_name")["warning_row_tag"].apply(_aggregate_warning_tags).to_dict()
    )

    param_rows = []
    finite_params = fp_work[np.isfinite(fp_work["Kd_uM"])]
    for keys, sub in finite_params.groupby(["model_name", "parameter_name"], dropna=False):
        model_name, pname = keys
        tags = sorted(set(_parameter_warning_tags(r) for _, r in sub.iterrows()))
        param_rows.append({
                    "model_name": model_name,
                    "parameter_name": _output_parameter_label(pname),
                    "note": ", ".join(tags) if tags else "ok",
                    "Kd_uM_mean_sd": _format_mean_sd(sub["Kd_uM"].mean(), sub["Kd_uM"].std(ddof=1), " uM"),
                })
    param_summary = pd.DataFrame(param_rows)
    if not param_summary.empty:
        param_summary["model_rank"] = param_summary["model_name"].map(lambda m: model_order_map.get(m, 10_000))
        param_summary["param_order"] = param_summary["parameter_name"].map(_parameter_sort_key)
        param_summary = param_summary.sort_values(
            by=["model_rank", "param_order", "model_name", "parameter_name"]
        ).drop(columns=["model_rank", "param_order"])
        param_summary["model"] = param_summary["model_name"].map(_display)

    ranking_warning_map = {}
    for model_name, warning_tag in model_warning_by_param.items():
        if warning_tag != "ok":
            ranking_warning_map[model_name] = warning_tag
    for model_name in failed_models:
        existing = ranking_warning_map.get(model_name, "")
        merged = {t.strip() for t in existing.split(",") if t.strip()}
        merged.add("failed")
        ranking_warning_map[model_name] = ", ".join(sorted(merged))
    if "constraint_fallback_used" in fit_metrics.columns:
        fallback_models = set(
            fit_metrics.loc[
                fit_metrics["constraint_fallback_used"].fillna(False).astype(bool),
                "model_name",
            ].tolist()
        )
        for model_name in fallback_models:
            existing = ranking_warning_map.get(model_name, "")
            merged = {t.strip() for t in existing.split(",") if t.strip()}
            merged.add("constraint_fallback")
            ranking_warning_map[model_name] = ", ".join(sorted(merged))
    ranking["note"] = ranking["model_name"].map(
        lambda m: ranking_warning_map.get(m, "ok")
    )
    ranking["model"] = ranking["model_name"].map(_display)

    if nested_ftests.empty:
        ftest_summary = pd.DataFrame()
    else:
        ftest_alpha = float(getattr(cfg, "ftest_alpha", 0.05))
        ftest_rows = []
        for model_name, sub in nested_ftests.groupby("full_model_name", sort=False):
            pvals = ", ".join(f"{p:.3g}" for p in sub["p_value"])
            ftest_rows.append({
                "full_model_name": _display(model_name),
                "n_reps": len(sub),
                "n_significant_reps": int((sub["p_value"] < ftest_alpha).sum()),
                "replicate_p_values": pvals,
            })
        ftest_summary = pd.DataFrame(ftest_rows)

    warnings = []
    warnings.extend(["### Convergence issues"])
    failed = fit_metrics[~fit_metrics["fit_success"].astype(bool)]
    if not failed.empty:
        warnings.append("Convergence failures by model:")
        for model_name, sub in failed.groupby("model_name", sort=False):
            warnings.append(f"- {_display(model_name)}: {len(sub)} model-replicate fit(s) did not converge.")
    else:
        warnings.append("No convergence failures detected in replicate-level fits.")

    if "constraint_fallback_used" in fit_metrics.columns:
        fallback = fit_metrics[
            fit_metrics["constraint_fallback_used"].fillna(False).astype(bool)
        ]
        if not fallback.empty:
            warnings.append("")
            warnings.append("Constraint fallback was used after the bounded constrained fit did not converge:")
            for model_name, sub in fallback.groupby("model_name", sort=False):
                warnings.append(f"- {_display(model_name)}: {len(sub)} model-replicate fit(s) rerun unconstrained.")

    warnings.extend(["", "### Interval / stability issues"])
    unstable = fit_parameters[
        np.isfinite(fit_parameters["Kd_uM"]) &
        (
            ~np.isfinite(fit_parameters["SE_Kd_uM"]) |
            ((fit_parameters["Kd_uM"] > 0) & (fit_parameters["SE_Kd_uM"] / fit_parameters["Kd_uM"] > 3))
        )
    ]
    if not unstable.empty:
        warnings.append("Parameters with wide/unbounded uncertainty (SE/mean > 3 or missing SE):")
        for model_name, sub in unstable.groupby("model_name", sort=False):
            warnings.append(f"- {_display(model_name)}: {len(sub)} parameter/replicate fits.")
    else:
        warnings.append("No unstable interval patterns were detected.")

    warnings.extend(["", "### Physical sanity checks"])
    extreme = fit_parameters[
        np.isfinite(fit_parameters["Kd_uM"]) &
        ((fit_parameters["Kd_uM"] <= 0) | (fit_parameters["Kd_uM"] > 1e6))
    ]
    if not extreme.empty:
        warnings.append("Out-of-range $K_d$ estimates:")
        for model_name, sub in extreme.groupby("model_name", sort=False):
            warnings.append(f"- {_display(model_name)}: {len(sub)} $K_d$ value(s) outside 0 < Kd <= 1e6 uM.")
    else:
        warnings.append("No out-of-range $K_d$ values detected.")

    warnings.extend(["", "### Interpretation"])
    if bool(getattr(cfg, "constrain_nsb_weaker_than_specific", True)):
        warnings.extend([
            "NSB candidate fits used the default biological constraint "
            "that the nonspecific dissociation constant must be weaker than "
            "the fitted specific constants: $K_{d,n}$ is constrained to be "
            "larger than fitted specific $K_d$ values. For `occupancy_decay`, "
            "the first nonspecific association amplitude is constrained below "
            "the specific association terms and $\\gamma \\ge 0$. If no "
            "constrained start converges within `nsb_constraint_max_nfev`, "
            "the fit is rerun without the NSB constraint when "
            "`nsb_constraint_fallback_to_unconstrained: true`.",
        ])
    else:
        warnings.extend([
            "NSB candidate fits were run without the default "
            "$K_{d,n}$-weaker-than-specific constraint.",
        ])

    reference = getattr(cfg, "reference_model", "sequential_specific_s7")
    valid_ftest_models = (
        nested_ftests["full_model_name"].dropna().drop_duplicates().tolist()
        if not nested_ftests.empty else []
    )
    all_model_names = sorted(set(fit_metrics["model_name"].dropna().tolist()))
    non_nested = []
    for model_name in all_model_names:
        if model_name == reference:
            reason = "reference model; used as reduced model for F-tests"
        elif model_name in valid_ftest_models:
            continue
        else:
            reason = "not a valid nested extension of canonical reference"
        non_nested.append({"model_name": _display(model_name), "excluded_from_ftest_reason": reason})
    non_nested_df = pd.DataFrame(non_nested)

    context = _parse_run_context(cfg, "UNKNOWN_0")
    title = f"{context['analyte']} in {context['buffer']} {context['temperature_C']} C"
    lines = [
        f"# All-Models Summary Report: {title}",
        "",
        "This report is derived from the canonical workbook `all_models_results.xlsx`.",
        "",
        "## Model Ranking By AICc",
        _markdown_table(ranking, [
            "AICc_rank", "model", "R2_vs_sd", "AICc±SD", "mean_p_value", "note",
        ]),
        "## Parameter Summary",
        _markdown_table(param_summary, [
            "model", "parameter_name", "note", "Kd_uM_mean_sd",
        ], max_rows=80),
        "## Nested F-Test Summary",
        _markdown_table(ftest_summary, [
            "full_model_name", "n_reps", "n_significant_reps", "replicate_p_values",
        ]),
        "## F-Test Scope",
        "Valid nested F-test models: " + (", ".join(_display(m) for m in valid_ftest_models) if valid_ftest_models else "none") + ".",
        "",
        "Models outside this set are compared by AICc/BIC, not by the nested F-test:",
        _markdown_table(non_nested_df, ["model_name", "excluded_from_ftest_reason"]),
        "## Warnings",
        "\n".join(warnings),
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def _write_manifest(out_dir, per_model_results, cfg, workbook_path, report_path):
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    input_files = sorted({
        os.path.abspath(d.get("data_path", ""))
        for results in per_model_results.values()
        for d in results
        if d.get("data_path")
    })
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_version": "compact_all_models_v3",
        "system_name": getattr(cfg, "system_name", ""),
        "base_dir": os.path.abspath(cfg.base_dir),
        "input_files": input_files,
        "models": list(per_model_results.keys()),
        "reference_model": getattr(cfg, "reference_model", ""),
        "nested_ftest_models": list(getattr(cfg, "nested_ftest_models", []) or []),
        "constrain_nsb_weaker_than_specific": bool(
            getattr(cfg, "constrain_nsb_weaker_than_specific", True)
        ),
        "nsb_constraint_multistart_n": int(
            getattr(cfg, "nsb_constraint_multistart_n", 24)
        ),
        "nsb_constraint_max_nfev": int(
            getattr(cfg, "nsb_constraint_max_nfev", 2000)
        ),
        "nsb_constraint_fallback_to_unconstrained": bool(
            getattr(cfg, "nsb_constraint_fallback_to_unconstrained", True)
        ),
        "nsb_constraint_log_margin": float(
            getattr(cfg, "nsb_constraint_log_margin", 1e-6)
        ),
        "model_display_names": dict(getattr(cfg, "model_display_names", {}) or {}),
        "model_output_names": dict(getattr(cfg, "model_output_names", {}) or {}),
        "export_csv": bool(getattr(cfg, "export_csv", False)),
        "outputs": {
            "workbook": os.path.abspath(workbook_path),
            "summary_report": os.path.abspath(report_path),
            "figures_dir": os.path.abspath(os.path.join(out_dir, "figures")),
            "logs_dir": os.path.abspath(os.path.join(out_dir, "logs")),
        },
        "workbook_sheets": [
            "model_summary", "fit_metrics", "fit_parameters", "nested_ftests",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def _write_compact_outputs(per_model_results, out_dir, cfg):
    fit_metrics = _build_fit_metrics(per_model_results, cfg)
    fit_parameters = _build_fit_parameters(per_model_results, cfg)
    nested_ftests = _build_nested_ftests(per_model_results, cfg)
    model_summary = _build_model_summary_sheet(
        fit_metrics, fit_parameters, nested_ftests, cfg,
    )
    fit_metrics_out = _round_numeric_columns(fit_metrics, sig=4)
    fit_parameters_out = _round_numeric_columns(fit_parameters, sig=4)
    nested_ftests_out = _round_numeric_columns(nested_ftests, sig=4)
    fit_parameters_display = fit_parameters_out.copy()
    if "parameter_name" in fit_parameters_display:
        fit_parameters_display["parameter_name"] = fit_parameters_display["parameter_name"].map(_output_parameter_label)

    workbook_path = os.path.join(out_dir, "all_models_results.xlsx")
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        model_summary.to_excel(writer, sheet_name="model_summary", index=False)
        fit_metrics_out.to_excel(writer, sheet_name="fit_metrics", index=False)
        fit_parameters_display.to_excel(writer, sheet_name="fit_parameters", index=False)
        nested_ftests_out.to_excel(writer, sheet_name="nested_ftests", index=False)
        _autosize_excel_columns(writer, model_summary, "model_summary")
        _autosize_excel_columns(writer, fit_metrics_out, "fit_metrics")
        _autosize_excel_columns(writer, fit_parameters_display, "fit_parameters")
        _autosize_excel_columns(writer, nested_ftests_out, "nested_ftests")

    if getattr(cfg, "export_csv", False):
        fit_metrics_out.to_csv(
            os.path.join(out_dir, "fit_metrics.csv"),
            index=False,
            float_format="%.4g",
        )
        fit_parameters_display.to_csv(
            os.path.join(out_dir, "fit_parameters.csv"),
            index=False,
            float_format="%.4g",
        )
        nested_ftests_out.to_csv(
            os.path.join(out_dir, "nested_ftests.csv"),
            index=False,
            float_format="%.4g",
        )
        model_summary.to_csv(
            os.path.join(out_dir, "model_summary.csv"),
            index=False,
        )

    report_path = _write_summary_report(
        out_dir, fit_metrics_out, fit_parameters_out, nested_ftests_out, cfg,
    )
    manifest_path = _write_manifest(out_dir, per_model_results, cfg, workbook_path, report_path)

    print("\n" + "-" * 70)
    print("COMPACT ALL-MODELS OUTPUT")
    print("-" * 70)
    print(f"Saved workbook to:       {workbook_path}")
    print(f"Saved summary report to: {report_path}")
    print(f"Saved run manifest to:   {manifest_path}")
    if getattr(cfg, "export_csv", False):
        print("CSV export enabled: wrote one CSV per workbook sheet.")
    else:
        print("CSV export disabled: workbook is the canonical machine-readable output.")
    return {
        "model_summary": model_summary,
        "fit_metrics": fit_metrics_out,
        "fit_parameters": fit_parameters_out,
        "nested_ftests": nested_ftests_out,
    }


def compare_models_bic_aic(per_model_results, out_dir, cfg=None):
    """Write model-comparison outputs for independently fitted replicates.

    The broad model set is compared with per-replicate SSR, AICc, and BIC.
    Nested F-tests are restricted to the configured canonical reference model
    versus configured NSB models and are reported per replicate; p-values are
    not averaged across replicates.
    """
    from scipy.stats import f as _f_dist

    if getattr(cfg, "compact_outputs", False):
        return _write_compact_outputs(per_model_results, out_dir, cfg)

    def _metrics(ssr, sst, n, k):
        bic = n * np.log(ssr / n) + k * np.log(n)
        aic = n * np.log(ssr / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
        r2 = 1.0 - ssr / sst if sst and sst > 0 else np.nan
        rmse = np.sqrt(ssr / (n - k)) if n > k else np.nan
        return bic, aic, aicc, r2, rmse

    def _ftest(ssr_red, k_red, ssr_full, k_full, n):
        """Extra-sum-of-squares F: reduced (fewer k) vs full (more k)."""
        if k_full <= k_red or n <= k_full or not ssr_full > 0:
            return np.nan, np.nan, np.nan, np.nan
        df1, df2 = k_full - k_red, n - k_full
        F = ((ssr_red - ssr_full) / df1) / (ssr_full / df2)
        p = float(_f_dist.sf(F, df1, df2)) if F > 0 else 1.0
        return F, df1, df2, p

    reference_model = getattr(cfg, "reference_model", "sequential_specific_s7")
    nested_ftest_models = set(getattr(cfg, "nested_ftest_models", []) or [])
    ftest_alpha = float(getattr(cfg, "ftest_alpha", 0.05))
    output_unit = getattr(cfg, "output_unit", "")

    def _kd_symbol(param_name):
        """Readable Kd-side symbol for the aggregate parameter table."""
        if param_name.startswith("Ks_"):
            return "K_d" + param_name.split("_", 1)[1]
        if param_name in ("Kn", "beta"):
            return "K_d,n"
        if param_name == "Ks":
            return "K_d"
        return ""

    def _ci95(value, se_log_or_value, log_scale):
        if value is None or not np.isfinite(value):
            return np.nan, np.nan
        if se_log_or_value is None or not np.isfinite(se_log_or_value):
            return np.nan, np.nan
        half = 1.96 * float(se_log_or_value)
        if log_scale:
            if value <= 0:
                return np.nan, np.nan
            log_value = np.log(value)
            lo_log = log_value - half
            hi_log = log_value + half
            lo = 0.0 if lo_log < -745.0 else float(np.exp(lo_log))
            hi = np.inf if hi_log > 709.0 else float(np.exp(hi_log))
            return lo, hi
        return float(value - half), float(value + half)

    model_names = list(per_model_results.keys())
    if len(model_names) < 2:
        print("[compare_models] Need >= 2 models to compare.")
        return

    stems = [d["stem"] for d in per_model_results[model_names[0]]]
    csv_rows = []

    print("\n" + "=" * 70)
    print(f"CROSS-MODEL METRICS (no winner declared — analyst judges): "
          f"{' | '.join(model_names)}")
    print("=" * 70)

    for file_idx, stem in enumerate(stems):
        print(f"\n=== {stem} ===")
        header = (f"{'Model':<30} {'k':>3} {'SSR':>11} {'R2':>8} {'RMSE':>10} "
                  f"{'BIC':>10} {'AIC':>10} {'AICc':>10}")
        print(header)
        print("-" * len(header))
        for name in model_names:
            d = per_model_results[name][file_idx]
            bic, aic, aicc, r2, rmse = _metrics(d["SSR"], d.get("SST"),
                                                d["n_obs"], d["n_params"])
            print(f"{name:<30} {d['n_params']:>3d} {d['SSR']:>11.5f} {r2:>8.4f} "
                  f"{rmse:>10.3e} {bic:>10.2f} {aic:>10.2f} {aicc:>10.2f}")
            csv_rows.append({
                "File": stem,
                "Model": name,
                "Role": model_role(name),
                "S_eff": d.get("S_eff", np.nan),
                "N_eff": d.get("N_eff", np.nan),
                "n_params": d["n_params"],
                "n_obs": d["n_obs"],
                "dof": d["n_obs"] - d["n_params"],
                "SSR": d["SSR"],
                "RMSE": rmse,
                "R2": r2,
                "BIC": bic,
                "AIC": aic,
                "AICc": aicc,
                "fit_success": d.get("fit_success", np.nan),
                "fit_status": d.get("fit_status", np.nan),
                "fit_message": d.get("fit_message", ""),
                "fit_cost": d.get("fit_cost", np.nan),
                "fit_optimality": d.get("fit_optimality", np.nan),
            })

    df_all = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, "model_comparison.csv")
    df_all.to_csv(csv_path, index=False, float_format="%.6e")

    param_rows = []
    for name in model_names:
        for d in per_model_results[name]:
            labels = d.get("param_names", [])
            values = d.get("param_values", [])
            ka_vals = d.get("Ka_M_inv", [])
            kd_vals = d.get("Kd_out", [])
            std_param = d.get("std_param", None)
            std_ka = d.get("std_Ka_M", None)
            std_kd = d.get("std_Kd_out", None)
            for idx, label in enumerate(labels):
                is_dimless = is_dimensionless_param(label)
                value = values[idx] if idx < len(values) else np.nan
                ka_m = ka_vals[idx] if idx < len(ka_vals) else np.nan
                kd_out = kd_vals[idx] if idx < len(kd_vals) else np.nan
                se_raw = (
                    std_param[idx]
                    if std_param is not None and idx < len(std_param) else np.nan
                )
                se_value = np.nan
                value_ci_low = value_ci_high = np.nan
                kd_ci_low = kd_ci_high = np.nan
                if is_dimless:
                    se_value = se_raw
                    value_ci_low, value_ci_high = _ci95(value, se_raw, log_scale=False)
                    value_unit = "unitless"
                    ci_method = "jacobian_normal_approx"
                else:
                    se_value = (
                        std_ka[idx]
                        if std_ka is not None and idx < len(std_ka) else np.nan
                    )
                    value_ci_low, value_ci_high = _ci95(ka_m, se_raw, log_scale=True)
                    kd_ci_low, kd_ci_high = _ci95(kd_out, se_raw, log_scale=True)
                    value_unit = "M^-1"
                    ci_method = "jacobian_log_scale_approx"
                se_kd = (
                    std_kd[idx] if std_kd is not None and idx < len(std_kd) else np.nan
                )
                param_rows.append({
                    "File": d["stem"],
                    "Model": name,
                    "Role": model_role(name),
                    "S_eff": d.get("S_eff", np.nan),
                    "N_eff": d.get("N_eff", np.nan),
                    "n_obs": d["n_obs"],
                    "n_params": d["n_params"],
                    "dof": d["n_obs"] - d["n_params"],
                    "fit_success": d.get("fit_success", np.nan),
                    "Param": label,
                    "Kd_symbol": _kd_symbol(label),
                    "Value": value,
                    "Value_unit": value_unit,
                    "SE_Value": se_value,
                    "CI95_low_Value": value_ci_low,
                    "CI95_high_Value": value_ci_high,
                    "Ka_M_inv": ka_m,
                    "Kd": kd_out,
                    "Kd_unit": output_unit,
                    "SE_Kd": se_kd,
                    "CI95_low_Kd": kd_ci_low,
                    "CI95_high_Kd": kd_ci_high,
                    "CI_method": ci_method,
                    "Std_Kd": se_kd,
                })
    params_path = os.path.join(out_dir, "model_fit_parameters.csv")
    pd.DataFrame(param_rows).to_csv(params_path, index=False, float_format="%.6e")

    # --- replicate-averaged summary (one row per model) ---
    mean_k = {nm: int(round(np.mean([d["n_params"] for d in per_model_results[nm]])))
              for nm in model_names}
    mean_ssr = {nm: float(np.mean([d["SSR"] for d in per_model_results[nm]]))
                for nm in model_names}
    n_obs_common = per_model_results[model_names[0]][0]["n_obs"]

    summ_rows = []
    for name in model_names:
        sub = df_all[df_all["Model"] == name]
        row = {
            "Model": name,
            "Role": model_role(name),
            "n_params": mean_k[name],
            "n_reps": len(sub),
            "n_success": int(pd.Series(sub["fit_success"]).fillna(False).astype(bool).sum()),
            "S_eff_values": ",".join(str(int(x)) for x in sorted(sub["S_eff"].dropna().unique())),
            "N_eff_values": ",".join(str(int(x)) for x in sorted(sub["N_eff"].dropna().unique())),
        }
        for col in ["SSR", "RMSE", "R2", "BIC", "AIC", "AICc"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std(ddof=1)
        summ_rows.append(row)
    df_summ = pd.DataFrame(summ_rows)
    summ_path = os.path.join(out_dir, "model_comparison_summary.csv")
    df_summ.to_csv(summ_path, index=False, float_format="%.6e")

    # --- focused nested F-tests: canonical reference vs allowed NSB candidates ---
    ft_rows = []
    excluded_rows = []
    if reference_model in per_model_results:
        ref_by_file = {d["stem"]: d for d in per_model_results[reference_model]}
        for full_model in model_names:
            if full_model == reference_model:
                continue
            if full_model not in nested_ftest_models:
                excluded_rows.append({
                    "Model": full_model,
                    "Reference": reference_model,
                    "Reason": "not_configured_as_nested_reduction_to_reference",
                })
                continue
            for full in per_model_results[full_model]:
                red = ref_by_file.get(full["stem"])
                if red is None:
                    excluded_rows.append({
                        "Model": full_model,
                        "Reference": reference_model,
                        "File": full["stem"],
                        "Reason": "matching_reference_replicate_not_found",
                    })
                    continue
                n_used = min(red["n_obs"], full["n_obs"])
                F, df1, df2, p = _ftest(
                    red["SSR"], red["n_params"], full["SSR"], full["n_params"], n_used
                )
                ft_rows.append({
                    "File": full["stem"],
                    "reduced_model": reference_model,
                    "full_model": full_model,
                    "question": "does_explicit_NSB_improve_fit_relative_to_S7_SB",
                    "S_reduced": red.get("S_eff", np.nan),
                    "N_reduced": red.get("N_eff", np.nan),
                    "S_full": full.get("S_eff", np.nan),
                    "N_full": full.get("N_eff", np.nan),
                    "k_reduced": red["n_params"],
                    "k_full": full["n_params"],
                    "n_obs_reduced": red["n_obs"],
                    "n_obs_full": full["n_obs"],
                    "n_obs_used": n_used,
                    "SSR_reduced": red["SSR"],
                    "SSR_full": full["SSR"],
                    "dSSR": red["SSR"] - full["SSR"],
                    "F": F,
                    "df1": df1,
                    "df2": df2,
                    "p_value": p,
                    "alpha": ftest_alpha,
                    "significant_NSB_improvement": bool(np.isfinite(p) and p < ftest_alpha),
                })
    else:
        excluded_rows.append({
            "Model": "*",
            "Reference": reference_model,
            "Reason": "reference_model_not_fit",
        })

    ft_path = os.path.join(out_dir, "reference_nsb_ftest.csv")
    pd.DataFrame(ft_rows).to_csv(ft_path, index=False, float_format="%.6e")

    ft_summary_rows = []
    if ft_rows:
        df_ft = pd.DataFrame(ft_rows)
        for model_name_i, sub in df_ft.groupby("full_model", sort=False):
            sig = sub["significant_NSB_improvement"].astype(bool)
            ft_summary_rows.append({
                "full_model": model_name_i,
                "reference_model": reference_model,
                "tested": True,
                "exclude_reason": "",
                "n_reps": len(sub),
                "n_significant_reps": int(sig.sum()),
                "all_reps_significant": bool(sig.all()),
                "any_rep_significant": bool(sig.any()),
                "p_min": sub["p_value"].min(),
                "p_median": sub["p_value"].median(),
                "p_max": sub["p_value"].max(),
                "note": "p-values are replicate-level; do not average p-values for inference",
            })
    tested_models = {row["full_model"] for row in ft_summary_rows}
    for excluded in excluded_rows:
        excluded_model = excluded.get("Model")
        if excluded_model in tested_models:
            continue
        if model_role(excluded_model) != "canonical_s7_nsb_candidate":
            continue
        ft_summary_rows.append({
            "full_model": excluded_model,
            "reference_model": reference_model,
            "tested": False,
            "exclude_reason": excluded.get("Reason", ""),
            "n_reps": len(per_model_results.get(excluded_model, [])),
            "n_significant_reps": np.nan,
            "all_reps_significant": np.nan,
            "any_rep_significant": np.nan,
            "p_min": np.nan,
            "p_median": np.nan,
            "p_max": np.nan,
            "note": "not a configured nested reduction to the S=7 sequential reference; compare by AICc/BIC",
        })
    ft_summary_path = os.path.join(out_dir, "reference_nsb_ftest_summary.csv")
    pd.DataFrame(ft_summary_rows).to_csv(ft_summary_path, index=False, float_format="%.6e")

    excluded_path = os.path.join(out_dir, "reference_nsb_ftest_excluded.csv")
    pd.DataFrame(excluded_rows).to_csv(excluded_path, index=False)

    old_matrix_path = os.path.join(out_dir, "model_ftest_matrix.csv")
    if os.path.exists(old_matrix_path):
        os.remove(old_matrix_path)

    print("\n" + "-" * 70)
    print("REPLICATE-AVERAGED METRICS (no ranking applied)")
    print("-" * 70)
    for _, r in df_summ.iterrows():
        print(f"  {r['Model']:<30} k={r['n_params']:>2}  R2={r['R2_mean']:.4f}  "
              f"RMSE={r['RMSE_mean']:.3e}  BIC={r['BIC_mean']:.2f}  AIC={r['AIC_mean']:.2f}")
    print(f"\nREFERENCE NESTED F-TEST ({reference_model} vs configured NSB candidates; "
          "replicate-level p-values):")
    for r in ft_rows:
        if np.isnan(r["F"]):
            continue
        print(f"  {r['File']:<14} {r['full_model']:<28} "
              f"F={r['F']:7.3f}  p={r['p_value']:.2e}")
    print(f"\nFiles analysed: {len(stems)}  (same-k pairs: compare by SSR/BIC directly)")
    print(f"\nSaved per-replicate metrics to: {csv_path}")
    print(f"Saved averaged metrics to:      {summ_path}")
    print(f"Saved fit parameter table to:   {params_path}")
    print(f"Saved reference F-tests to:     {ft_path}")
    print(f"Saved F-test summary to:        {ft_summary_path}")
