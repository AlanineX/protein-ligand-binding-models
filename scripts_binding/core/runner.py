"""CLI entry: python -m scripts_binding <config.yaml>."""
import os
import shutil
import sys
import argparse
from pathlib import Path
from contextlib import contextmanager
from glob import glob

import numpy as np

from .config import load_configs
from .plotting import setup_matplotlib
from .fitting import fit_file, plot_fit_results, auto_select_S
from .summary import build_summary, build_deconv_summary, compare_models_bic_aic
from ..models import REGISTRY
from ..models.metadata import base_model_name, is_sequential_specific_model, output_model_name


class TeeLogger:
    """Context manager that duplicates stdout to a log file."""
    def __init__(self, log_path, mode="w"):
        self.path = log_path
        self.mode = mode
        self.terminal = None
        self.log = None

    def __enter__(self):
        self.terminal = sys.stdout
        self.log = open(self.path, self.mode, encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@contextmanager
def tee_to(log_path, mode="w"):
    with TeeLogger(log_path, mode) as logger:
        yield logger


def gather_data_paths(cfg):
    if cfg.data_path:
        return [cfg.data_path]
    pattern = os.path.join(cfg.base_dir, cfg.csv_name_wildcard)
    return sorted(glob(pattern))


def _resolve_mode_S(data_paths, cfg, model_name):
    """Modal S across replicates (auto mode); else None."""
    if cfg.s_mode.lower() != "auto":
        return None
    per_rep_S = [auto_select_S(path, cfg, model_name) for path in data_paths]
    mode_S = max(set(per_rep_S), key=per_rep_S.count)
    print(f"\n[Auto-S] Per-replicate best S: {per_rep_S}")
    print(f"[Auto-S] Mode S = {mode_S} — forcing all replicates to use S = {mode_S}\n")
    return mode_S


def _prepare_compact_output_dir(cfg):
    """Remove old verbose all-model artifacts and create compact output dirs."""
    out = Path(cfg.out_dir).resolve()
    marker = out / ".protein_ligand_binding_models_output"
    if out == out.parent or out == Path.home().resolve():
        raise ValueError(f"Unsafe output directory: {out}")
    if out.exists() and any(out.iterdir()) and not marker.is_file():
        raise ValueError(f"Output directory contains unmanaged files: {out}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    marker.touch(exist_ok=True)

    # Compact all-model output keeps only figures/ and logs/ as subdirectories.
    for name in os.listdir(cfg.out_dir):
        path = os.path.join(cfg.out_dir, name)
        if os.path.isdir(path) and name not in {"figures", "logs"}:
            shutil.rmtree(path)

    for subdir in ("figures", "logs"):
        path = os.path.join(cfg.out_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Remove old root-level CSV-heavy outputs. Optional CSV export will recreate
    # only the canonical sheet-level CSVs if requested.
    for name in os.listdir(cfg.out_dir):
        path = os.path.join(cfg.out_dir, name)
        if os.path.isfile(path) and (
            name.endswith(".csv")
            or name.endswith(".xlsx")
            or name in {"summary_report.md", "run_manifest.json", "model_comparison_log.txt"}
        ):
            os.remove(path)


def _run_model(model_name, data_paths, cfg):
    """Fit all files with one model; returns per-file info dicts."""
    model_out_name = output_model_name(model_name, getattr(cfg, "model_output_names", {}))
    if getattr(cfg, "compact_outputs", False):
        model_dir = os.path.join(cfg.out_dir, "figures", model_out_name)
        log_dir = os.path.join(cfg.out_dir, "logs")
    else:
        model_dir = os.path.join(cfg.out_dir, model_out_name)
        log_dir = model_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{model_out_name}_log.txt")
    results = []

    with tee_to(log_path, mode="w"):
        mode_S = _resolve_mode_S(data_paths, cfg, model_name) if not is_sequential_specific_model(model_name) else None
        for path in data_paths:
            info = fit_file(path, model_dir, cfg, model_name, S_override=mode_S)
            if not cfg.brief:
                plot_fit_results(info, cfg)
            results.append(info)

        if cfg.summary_enable and results:
            ref_L, F_exp_mean, F_exp_std, _, mean_Kd_out, num_species = build_summary(
                all_L_tot=[r["L_totals_M"] for r in results],
                all_F_exp=[r["F_exps"] for r in results],
                all_Kd=[r["Kd_out"] for r in results],
                all_stems=[r["stem"] for r in results],
                num_species_list=[r["num_species"] for r in results],
                all_lnK=[r["lnK_opt"] for r in results],
                all_S=[r["S_eff"] for r in results],
                all_N=[r["N_eff"] for r in results],
                model_name=model_name,
                out_dir=model_dir,
                label=model_name,
                cfg=cfg,
            )
            if cfg.deconv_enable and hasattr(REGISTRY[base_model_name(model_name)], "partition_terms"):
                all_S = [r["S_eff"] for r in results]
                summary_S = max(set(all_S), key=all_S.count)
                mean_lnK = np.mean([r["lnK_opt"] for r in results], axis=0)
                build_deconv_summary(
                    model_name, ref_L, F_exp_mean, F_exp_std, mean_lnK,
                    summary_S, num_species, model_dir, cfg,
                )

    print(f"[{model_name}] Log saved to {log_path}")
    return results


def run_single(cfg):
    """One (system, temperature) run across every model in cfg.models."""
    setup_matplotlib(cfg)

    data_paths = gather_data_paths(cfg)
    if not data_paths:
        raise FileNotFoundError(
            f"No CSV files found for pattern: {os.path.join(cfg.base_dir, cfg.csv_name_wildcard)}"
        )

    if not cfg.models:
        raise ValueError("cfg.models is empty — specify at least one model name.")

    if getattr(cfg, "compact_outputs", False):
        _prepare_compact_output_dir(cfg)

    per_model = {name: _run_model(name, data_paths, cfg) for name in cfg.models}

    if len(cfg.models) >= 2:
        cmp_log = os.path.join(
            cfg.out_dir, "logs" if getattr(cfg, "compact_outputs", False) else "",
            "model_comparison_log.txt",
        )
        os.makedirs(os.path.dirname(cmp_log), exist_ok=True)
        with tee_to(cmp_log, mode="w"):
            compare_models_bic_aic(
                {name: [{**r,
                         "SST": float(np.nansum((r["F_exps"] - np.nanmean(r["F_exps"])) ** 2))}
                        for r in results]
                 for name, results in per_model.items()},
                cfg.out_dir,
                cfg,
            )
        print(f"[Model Comparison] Log saved to {cmp_log}")


def run_all(yaml_path):
    configs = load_configs(yaml_path)
    print(f"Loaded {len(configs)} run configuration(s) from {yaml_path}")
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{len(configs)}: {cfg.csv_name_wildcard} | S={cfg.s} | S_MODE={cfg.s_mode}")
        print(f"  Models: {cfg.models}")
        print(f"  Base: {cfg.base_dir}")
        print(f"  Out:  {cfg.out_dir}")
        print(f"{'='*60}")
        run_single(cfg)


def main():
    parser = argparse.ArgumentParser(description="Fit protein-ligand binding distributions")
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()
    run_all(args.config)


if __name__ == "__main__":
    main()
