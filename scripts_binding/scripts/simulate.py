"""Forward-simulate bound-state curves from a parameter CSV."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scripts_binding.core.csv_io import read_csv

from scripts_binding.models import REGISTRY
from scripts_binding.core.config import RunConfig, UNIT_MAP
from scripts_binding.core.plotting import setup_matplotlib, plot_species_curves
from scripts_binding.models.metadata import base_model_name, normalize_model_name


def _to_ln(param, value, unit, is_kd):
    """One parameter spec -> the scalar stored in the model's ln_params array."""
    if str(param).strip().lower() == "gamma":
        return float(value)                       # gamma is stored raw, not logged
    scale = UNIT_MAP[str(unit).strip()] if unit and str(unit).strip() not in ("", "nan") else 1.0
    if int(is_kd):
        ka = 1.0 / (float(value) * scale)         # Kd (in unit) -> Ka (M^-1)
    else:
        ka = float(value)                         # already Ka in M^-1
    return float(np.log(ka))


def _build_lnK(rows, model):
    """Assemble (lnK, S, N) for one label's rows in the model's param order."""
    by_param = {str(r["param"]).strip(): r for _, r in rows.iterrows()}
    if "S" not in by_param or "N" not in by_param:
        raise ValueError("each label needs structural rows param=S and param=N")
    S = int(float(by_param["S"]["value"]))
    N = int(float(by_param["N"]["value"]))
    lnK = []
    for p in model.param_labels(S):
        if p not in by_param:
            raise ValueError(f"missing parameter '{p}' (model {model.MODEL_NAME}, S={S})")
        r = by_param[p]
        lnK.append(_to_ln(p, r["value"], r.get("unit"), r.get("is_kd", 1)))
    return np.asarray(lnK), S, N


def simulate_one(model_name, lnK, S, N, cfg, L_grid_M, tick_L_M, out_dir, label,
                 write_csv=False, x_tick_rotation=None, ladder_markers=False):
    """Simulate + plot one parameter set; reuses the fit-curve plotter.

    The plot already *is* the curve; the per-point CSV (L_tot, L_free, I0..In)
    is the same data, so it is only written when `write_csv` is set.
    """
    model_name = normalize_model_name(model_name)
    model = REGISTRY[base_model_name(model_name)]
    num_species = S + N + 1

    L_free = np.array([model.free_ligand(L, cfg.p_total_m, lnK, S, N) for L in L_grid_M])
    F_grid = np.vstack([model.mole_fractions(Lf, lnK, S, N) for Lf in L_free])
    if F_grid.shape[1] < num_species:             # specific model emits S+1 cols
        F_grid = np.pad(F_grid, ((0, 0), (0, num_species - F_grid.shape[1])))

    # data_F=None -> only the model lines; tick concentrations (data_L_M) set the
    # x-gridlines.
    n_specific = None if model_name == "sequential_specific" else S
    svg = os.path.join(out_dir, f"{label}_sim.svg")
    plot_species_curves(L_grid_M, F_grid, num_species, cfg, output_svg=svg,
                        title=None, n_specific=n_specific, data_L_M=tick_L_M,
                        x_tick_rotation=x_tick_rotation,
                        show_ladder_markers=ladder_markers)

    csv = None
    if write_csv:
        csv = os.path.join(out_dir, f"{label}_sim.csv")
        out = {f"L_tot_{cfg.output_unit}": L_grid_M * cfg.scale_m_to_out,
               f"L_free_{cfg.output_unit}": L_free * cfg.scale_m_to_out}
        for i in range(num_species):
            out[f"I{i}"] = F_grid[:, i]
        pd.DataFrame(out).to_csv(csv, index=False)
    return svg, csv


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", required=True, help="parameter CSV (see sim_params_template.csv)")
    p.add_argument("--out-dir", required=True, help="output directory for the SVG curves")
    p.add_argument("--csv", action="store_true",
                   help="also write per-point curve CSVs (L_tot, L_free, I0..In); "
                        "off by default since the data is already the plotted curve")
    p.add_argument("--p-tot", type=float, default=1.0, help="total protein conc (default 1.0)")
    p.add_argument("--p-tot-unit", default="uM", help="protein conc unit (default uM)")
    p.add_argument("--ligand-min", type=float, default=0.0, help="min total ligand (default 0)")
    p.add_argument("--ligand-max", type=float, default=300.0, help="max total ligand (default 300)")
    p.add_argument("--ligand-unit", default="uM", help="ligand conc unit (default uM)")
    p.add_argument("--n-grid", type=int, default=300, help="points on the curve (default 300)")
    p.add_argument("--ligand-ticks", default=None,
                   help="comma list of x-tick concentrations (ligand-unit); "
                        "default = 6 evenly spaced over the range")
    p.add_argument("--x-tick-rotation", type=float, default=None,
                   help="rotate x-tick labels by this many degrees")
    p.add_argument("--ladder-markers", action="store_true",
                   help="draw visible markers where the model curves intersect ligand ticks")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = RunConfig(input_unit="M", output_unit=args.ligand_unit,
                    p_total_val=args.p_tot, p_total_unit=args.p_tot_unit,
                    save_plots=True, show_plots=False)
    setup_matplotlib(cfg)

    scale_lig = UNIT_MAP[args.ligand_unit]
    L_grid_M = np.linspace(args.ligand_min, args.ligand_max, args.n_grid) * scale_lig
    if args.ligand_ticks:
        tick_L_M = np.array([float(t) for t in args.ligand_ticks.split(",")]) * scale_lig
    else:
        tick_L_M = np.linspace(args.ligand_min, args.ligand_max, 6) * scale_lig

    df = read_csv(args.params, comment="#")
    df["label"] = df["label"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    failed = []
    for label in dict.fromkeys(df["label"]):          # preserve first-seen order
        if label in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
            failed.append(label)
            print(f"[simulate] invalid output label: {label!r}")
            continue
        rows = df[df["label"] == label]
        model_name = normalize_model_name(rows["model"].iloc[0])
        if base_model_name(model_name) not in REGISTRY:
            print(f"[simulate] {label}: unknown model '{model_name}' — skipped")
            failed.append(label)
            continue
        try:
            lnK, S, N = _build_lnK(rows, REGISTRY[base_model_name(model_name)])
            svg, csv = simulate_one(model_name, lnK, S, N, cfg,
                                    L_grid_M, tick_L_M, args.out_dir, label,
                                    write_csv=args.csv,
                                    x_tick_rotation=args.x_tick_rotation,
                                    ladder_markers=args.ladder_markers)
            print(f"[simulate] {label} ({model_name}, S={S}, N={N}) -> {svg}")
        except Exception as exc:
            print(f"[simulate] {label}: {type(exc).__name__}: {exc} — skipped")
            failed.append(label)
    if failed:
        raise SystemExit(f"Simulation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
