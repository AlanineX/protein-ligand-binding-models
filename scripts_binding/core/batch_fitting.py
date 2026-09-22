"""Reusable batch fitting helpers for Kd scans and showcase workflows."""

import os

import numpy as np
import pandas as pd

from scripts_binding.core.bootstrap import bootstrap_fit, format_kd_ci
from scripts_binding.core.config import RunConfig
from scripts_binding.core.fitting import _least_squares_fit, load_binding_csv
from scripts_binding.models import REGISTRY
from scripts_binding.models.metadata import (
    base_model_name,
    display_model_name,
    normalize_model_name,
    parameter_values_from_optimizer,
)


def _load_titration(path):
    """Load one titration CSV via the canonical core loader."""
    cfg = RunConfig(input_unit="M", output_unit="uM")
    _df, L_totals_M, _I_cols, F_exps = load_binding_csv(path, cfg)
    return L_totals_M, [F_exps[i, :] for i in range(F_exps.shape[0])]


def pad_F(F_exp, target_len):
    """Pad F_exp to target_len with NaN, preserving literal zero measurements."""
    if len(F_exp) >= target_len:
        return F_exp[:target_len]
    return np.concatenate([F_exp, np.full(target_len - len(F_exp), np.nan)])


def _make_nan_safe_residual(model):
    """Wrap model.residual_vector so NaN positions in F_exp are excluded."""
    base_fn = model.residual_vector

    def wrapped(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history):
        local_hist = []
        res = base_fn(ln_params, L_totals_M, P_tot_M, F_exps, S, N, local_hist)
        res = np.where(np.isnan(res), 0.0, res)
        ssr_history.append(float(np.dot(res, res)))
        return res

    return wrapped


def _count_valid_obs(F_padded):
    """n_obs = number of non-NaN cells across all (point, species)."""
    return int(sum(np.sum(~np.isnan(F)) for F in F_padded))


def bic(ssr, n_obs, n_par):
    if ssr <= 0 or n_obs <= n_par:
        return np.nan
    return n_obs * np.log(ssr / n_obs) + n_par * np.log(n_obs)


def fit_one(model, L_totals, F_exps, P_tot, S, N, max_nfev=5000, cfg=None):
    """Fit one model to one titration, with NaN-padded missing species ignored."""
    if cfg is None:
        cfg = RunConfig(input_unit="M", output_unit="uM")
    target_len = S + N + 1
    F_padded = np.asarray([pad_F(F, target_len) for F in F_exps], dtype=float)
    lnK0 = model.initial_lnK(S)
    res, lnK_opt, _history, _transform, _n_starts = _least_squares_fit(
        model, model.MODEL_NAME, lnK0, L_totals, P_tot, F_padded, S, N,
        cfg, verbose=0, max_nfev=max_nfev,
    )
    res.x = lnK_opt
    n_obs = _count_valid_obs(F_padded)
    ssr = float(np.dot(res.fun, res.fun))
    try:
        sv = np.linalg.svd(res.jac, compute_uv=False)
        rank_eff = int(np.sum(sv > sv[0] * 1e-8)) if sv[0] > 0 else 0
        rank_full = len(sv)
    except (np.linalg.LinAlgError, IndexError):
        rank_eff, rank_full = -1, len(res.x)
    return res, ssr, bic(ssr, n_obs, model.n_params(S)), rank_eff, rank_full


def _fit_job(args):
    """One (buffer, temp, rep, model) fit; safe to run in a worker process."""
    (buffer_name, csv_path, S, N, T, R, model_name,
     p_tot, n_boot, boot_seed) = args
    model_name = normalize_model_name(model_name)
    try:
        L_totals, F_exps = _load_titration(csv_path)
        model = REGISTRY[base_model_name(model_name)]
        res, ssr, bic_v, rank_eff, rank_full = fit_one(
            model, L_totals, F_exps, p_tot, S, N
        )
        boot = None
        if n_boot > 0:
            boot = bootstrap_fit(model, L_totals, F_exps, p_tot, S, N,
                                 n_boot=n_boot, seed=boot_seed)
        labels = list(model.param_labels(S))
        values = parameter_values_from_optimizer(labels, res.x)
        return {
            "ok": True, "buffer": buffer_name, "temp": T, "rep": R,
            "model_name": model_name, "S": S, "N": N,
            "n_params": model.n_params(S),
            "ssr": ssr, "bic": bic_v,
            "converged": bool(res.success),
            "jac_rank_eff": rank_eff, "jac_rank_full": rank_full,
            "labels": labels, "values": values,
            "boot": boot,
        }
    except Exception as exc:  # noqa: BLE001 - return a failed job without stopping the batch
        return {
            "ok": False, "buffer": buffer_name, "temp": T, "rep": R,
            "model_name": model_name, "csv_path": csv_path,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _g4(x):
    """Format to 4 significant figures (%.4g). Auto scientific notation."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return f"{x:.4g}"


def _build_rows_from_result(r, use_display_model=True):
    """Expand one fit result into long-format parameter rows."""
    model_name = display_model_name(r["model_name"]) if use_display_model else r["model_name"]
    fit_meta = {
        "buffer": r["buffer"], "temp": r["temp"], "rep": r["rep"],
        "model": model_name, "S": r["S"], "N": r["N"],
        "n_params": r["n_params"],
        "ssr": f"{r['ssr']:.4e}",
        "bic": f"{r['bic']:.1f}",
        "converged": r["converged"],
        "jac_rank_eff": r["jac_rank_eff"],
        "jac_rank_full": r["jac_rank_full"],
        "identifiable": r["jac_rank_eff"] == r["jac_rank_full"],
    }
    boot = r.get("boot")
    if boot and "__meta__" in boot:
        fit_meta["n_boot_success"] = boot["__meta__"]["n_success"]
        fit_meta["n_boot_total"] = boot["__meta__"]["n_boot"]

    rows = []
    for lbl, val in zip(r["labels"], r["values"]):
        row = {**fit_meta, "param": lbl, "value": val}
        if lbl == "gamma":
            row["Kd_uM"] = np.nan
        else:
            row["Kd_uM"] = (1e6 / val) if val and val > 0 else np.nan
        if boot and lbl in boot:
            if lbl == "gamma":
                row["p50"] = _g4(boot[lbl]["p50"])
                row["lo95"] = _g4(boot[lbl]["lo95"])
                row["hi95"] = _g4(boot[lbl]["hi95"])
            else:
                ci = format_kd_ci(boot, lbl)
                row["Kd_uM_p50"] = _g4(ci["p50_Kd_uM"])
                row["Kd_uM_lo95"] = _g4(ci["lo95_Kd_uM"])
                row["Kd_uM_hi95"] = _g4(ci["hi95_Kd_uM"])
                row["n_boot_param_success"] = ci["n_success"]
        rows.append(row)
    return rows


def _build_wide_row(r, use_display_model=True):
    """Build one row per fit with Ka/Kd columns for applicable parameters."""
    model_name = display_model_name(r["model_name"]) if use_display_model else r["model_name"]
    S = r["S"]
    row = {
        "buffer": r["buffer"], "temp": r["temp"], "rep": r["rep"],
        "model": model_name, "S": S, "N": r["N"],
        "n_params": r["n_params"], "ssr": r["ssr"], "bic": r["bic"],
        "converged": r["converged"],
    }
    wide_cols = ["Kn_Ka", "Kn_Kd_uM"]
    for k in range(1, S + 1):
        wide_cols += [f"Ks_{k}_Ka", f"Ks_{k}_Kd_uM"]
    wide_cols += ["Ks_Ka", "Ks_Kd_uM", "beta_Ka", "beta_Kd_uM", "gamma"]
    for c in wide_cols:
        row[c] = ""

    for lbl, val in zip(r["labels"], r["values"]):
        if lbl == "gamma":
            row["gamma"] = val
        elif lbl == "beta":
            row["beta_Ka"] = val
            row["beta_Kd_uM"] = (1e6 / val) if val and val > 0 else ""
        else:
            row[f"{lbl}_Ka"] = val
            row[f"{lbl}_Kd_uM"] = (1e6 / val) if val and val > 0 else ""
    return row


def write_markdown(out_dir, df_long, models_to_run, n_nsb):
    """Generate a summary Markdown report from a long-format fit DataFrame."""
    df = df_long.copy()
    for col in ["Kd_uM", "bic", "ssr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    display_models = [display_model_name(m) for m in models_to_run]
    lines = ["# NSB Model Comparison\n",
             "Mean Kd values across replicates per temperature, per buffer, per model.\n",
             "Models compared:"]
    for m in display_models:
        lines.append(f"- **{m}**")
    lines.append(f"\nN_NSB (max NSB stoichiometry beyond S) = {n_nsb}.\n")

    for buf, dfb in df.groupby("buffer"):
        lines.append(f"\n## {buf}\n")

        nsb_rows = dfb[dfb["param"].isin(["Kn", "beta"])]
        if nsb_rows["Kd_uM"].notna().any():
            mean_kd_n = nsb_rows.groupby(["temp", "model"])["Kd_uM"].mean().unstack("model")
            lines.append("### Mean Kd_NSB (µM) by temperature\n")
            lines.append("(for power_law_nonspecific, reported value is 1/beta = Kd at first NSB step)\n")
            lines.append(mean_kd_n.round(1).to_markdown())
            lines.append("")

        bic_rows = dfb.drop_duplicates(subset=["temp", "rep", "model"])
        mean_bic = bic_rows.groupby(["temp", "model"])["bic"].mean().unstack("model")
        lines.append("### Mean BIC by temperature (lower = better)\n")
        lines.append(mean_bic.round(1).to_markdown())
        lines.append("")

        d25 = dfb[(dfb["temp"] == 25) & (
            dfb["param"].str.startswith("Ks_") | (dfb["param"] == "Ks")
        )]
        if len(d25) > 0:
            ks_summary = (d25.groupby(["model", "param"])["Kd_uM"]
                             .mean().unstack("param"))
            ordered_cols = sorted([c for c in ks_summary.columns if c.startswith("Ks_")],
                                  key=lambda c: int(c.split("_")[1])) \
                           + (["Ks"] if "Ks" in ks_summary.columns else [])
            ks_summary = ks_summary[ordered_cols]
            ks_summary.columns = [f"{c}_Kd_uM" for c in ks_summary.columns]
            lines.append("### Mean Ks_i Kd (µM) at 25 °C\n")
            lines.append(ks_summary.to_markdown())
            lines.append("")

        winners = bic_rows.loc[bic_rows.groupby(["temp", "rep"])["bic"].idxmin(),
                               ["temp", "rep", "model", "bic"]].reset_index(drop=True)
        win_counts = winners["model"].value_counts()
        lines.append("### BIC winner counts (across temp × rep)\n")
        for m, c in win_counts.items():
            lines.append(f"- **{m}**: {c} / {len(winners)}")
        lines.append("")

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
