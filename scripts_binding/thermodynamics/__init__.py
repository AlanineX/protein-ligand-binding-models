"""Thermodynamic analysis via van't Hoff / non-linear van't Hoff fitting.

Takes per-replicate Kd(T) data, applies Wyman-Gill statistical correction,
fits NLVH (ΔH, ΔS, ΔCp) or LVH (ΔH, ΔS; ΔCp ≡ 0), reports ΔG/ΔH/−TΔS with
error propagation and per-replicate spread.

Public API:
    load_kd_csv       — parse a Kd CSV file
    statistical_correction, nlvh_equation, lvh_equation, fit_nlvh, fit_lvh
    run_analysis      — unified pipeline (single N, single method, all sites)
    run_grid          — grid scan over N × method × dataset
"""
from .loader import load_kd_csv
from .nlvh import (
    statistical_correction,
    nlvh_equation, lvh_equation,
    fit_nlvh, fit_lvh,
    R_KJ,
)
from .pipeline import run_analysis, run_grid
