"""Results tables + CSV output."""
import numpy as np
import pandas as pd

from ..models.metadata import is_dimensionless_param


def _format_number(value, spec=".3e"):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:{spec}}"


def print_results_table(param_names, raw_params, Ka_opt_M, Kd_opt_M, has_errors, cfg,
                        std_param=None, std_Ka_M=None, std_Kd_M=None):
    """Print fitted parameters without assigning Ka/Kd units to gamma."""
    ka_unit = f"{cfg.output_unit}^-1"
    kd_unit = cfg.output_unit
    show_uncertainty = bool(getattr(cfg, "report_uncertainty", True))
    if show_uncertainty:
        header = (f"{'Parameter':>10} | {'Fitted value':^18} | {'Unit':^12} | "
                  f"{'95% half-width':^16} | {f'Kd ({kd_unit})':^14} | {'95% half-width':^16}")
    else:
        header = f"{'Parameter':>10} | {'Fitted value':^18} | {'Unit':^12} | {f'Kd ({kd_unit})':^14}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(param_names):
        if is_dimensionless_param(name):
            value = raw_params[i]
            value_unit = "unitless"
            value_unc = 1.96 * std_param[i] if has_errors and std_param is not None else np.nan
        else:
            value = Ka_opt_M[i] / cfg.scale_m_to_out
            value_unit = ka_unit
            value_unc = (
                1.96 * std_Ka_M[i] / cfg.scale_m_to_out
                if has_errors and std_Ka_M is not None else np.nan
            )

        Kd_out = Kd_opt_M[i] * cfg.scale_m_to_out
        kd_unc = (
            1.96 * std_Kd_M[i] * cfg.scale_m_to_out
            if has_errors and std_Kd_M is not None else np.nan
        )
        if show_uncertainty:
            print(f"{name:>10} | {_format_number(value):^18} | {value_unit:^12} | "
                  f"{_format_number(value_unc, '.2e'):^16} | {_format_number(Kd_out):^14} | {_format_number(kd_unc, '.2e'):^16}")
        else:
            print(f"{name:>10} | {_format_number(value):^18} | {value_unit:^12} | {_format_number(Kd_out):^14}")
    print("-" * len(header), "\n")


def save_kd_csv(param_names, raw_params, Ka_opt_M, Kd_opt_M, has_errors, std_param,
                std_Ka_M, std_Kd_M, kd_csv, cfg):
    """Write fitted parameter values plus Ka/Kd when the parameter is a Ka."""
    ka_unit = f"{cfg.output_unit}^-1"
    kd_unit = cfg.output_unit
    kd_records = []
    for i, name in enumerate(param_names):
        if is_dimensionless_param(name):
            value = raw_params[i]
            value_unit = "unitless"
            se_value = std_param[i] if has_errors and std_param is not None else np.nan
            Ka_out = std_Ka_out = Kd_out = std_Kd_out = np.nan
        else:
            value = Ka_opt_M[i] / cfg.scale_m_to_out
            value_unit = ka_unit
            se_value = std_Ka_M[i] / cfg.scale_m_to_out if has_errors and std_Ka_M is not None else np.nan
            Ka_out = value
            std_Ka_out = se_value
            Kd_out = Kd_opt_M[i] * cfg.scale_m_to_out
            std_Kd_out = std_Kd_M[i] * cfg.scale_m_to_out if has_errors and std_Kd_M is not None else np.nan
        if getattr(cfg, "report_uncertainty", True):
            kd_records.append((name, value, se_value, value_unit, Ka_out, std_Ka_out, Kd_out, std_Kd_out))
        else:
            kd_records.append((name, value, value_unit, Ka_out, Kd_out))
    if getattr(cfg, "report_uncertainty", True):
        columns = [
            "Param", "Value", "SE_Value", "Value_unit",
            f"Ka({ka_unit})", f"SE_Ka({ka_unit})", f"Kd({kd_unit})", f"SE_Kd({kd_unit})",
        ]
    else:
        columns = ["Param", "Value", "Value_unit", f"Ka({ka_unit})", f"Kd({kd_unit})"]
    pd.DataFrame(kd_records, columns=columns).to_csv(kd_csv, index=False, float_format='%.6e')


def print_per_point_summary(df, L_totals_M, F_exps, L_free_list, F_calcs, num_species, cfg):
    """Print per-point F_exp vs F_calc table."""
    rows = []
    for idx, (entry, L_tot_M, F_exp) in enumerate(zip(df['Entry'], L_totals_M, F_exps)):
        L_free_M = L_free_list[idx]
        F_calc = F_calcs[idx]
        ssr_i = np.nansum((F_calc - F_exp) ** 2)
        row = {
            'Entry': entry,
            f'[L]tot({cfg.output_unit})': f"{L_tot_M * cfg.scale_m_to_out:.2f}",
            f'[L]free({cfg.output_unit})': f"{L_free_M * cfg.scale_m_to_out:.3f}",
            'SSR_i': f"{ssr_i:.2e}"
        }
        for j in range(num_species):
            row[f"Fexp_{j}"] = f"{F_exp[j]:.3f}"
            row[f"Fcalc_{j}"] = f"{F_calc[j]:.3f}"
        rows.append(row)

    df_report = pd.DataFrame(rows)
    print("--- Per-point Summary ---")
    print(df_report.to_string(index=False, max_colwidth=10))
    print()
