"""RunConfig dataclass + YAML loader."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import os
from pathlib import Path

from scripts_binding.models.metadata import (
    canonicalize_model_list,
    canonicalize_override_map,
    configured_name_map,
    normalize_model_name,
)

UNIT_MAP = {'M': 1.0, 'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9, 'pM': 1e-12}


@dataclass
class RunConfig:
    # --- Paths ---
    base_dir: str = ""
    out_dir: str = ""
    csv_name_wildcard: str = ""
    data_path: Optional[str] = None

    # --- Units ---
    input_unit: str = "M"
    output_unit: str = "uM"
    p_total_val: float = 1.0
    p_total_unit: str = "uM"

    # --- Model parameters ---
    s: int = 4
    s_mode: str = "auto"
    n_override: Optional[int] = None
    model_s_overrides: Dict[str, int] = field(default_factory=dict)
    model_n_overrides: Dict[str, int] = field(default_factory=dict)
    auto_adjust_s: bool = True
    min_species_frac: float = 0.01
    trim_specific_low_pop: bool = True
    constrain_nsb_weaker_than_specific: bool = True
    nsb_constraint_multistart_n: int = 24
    nsb_constraint_max_nfev: int = 2000
    nsb_constraint_fallback_to_unconstrained: bool = True
    nsb_constraint_log_margin: float = 1e-6
    s_iteration_nsb_constraint_multistart_n: Optional[int] = None
    s_iteration_nsb_constraint_max_nfev: Optional[int] = None

    # --- Models to run (names from models.REGISTRY) ---
    models: List[str] = field(default_factory=lambda: ["sequential_specific", "sequential_adduct"])
    reference_model: str = "sequential_specific_s7"
    nested_ftest_models: List[str] = field(default_factory=lambda: [
        "sequential_specific_s9",
        "sequential_adduct",
        "competing_adduct",
        "stochastic_adduct",
        "occupancy_decay",
    ])
    ftest_alpha: float = 0.05
    model_display_names: Dict[str, str] = field(default_factory=dict)
    model_output_names: Dict[str, str] = field(default_factory=dict)

    # --- Deconvolution ---
    deconv_enable: bool = True
    deconv_source: str = "calc"
    deconv_use_grid: bool = False
    deconv_grid_points: int = 40
    deconv_csv_path: Optional[str] = None
    report_ligand_conc: List[float] = field(default_factory=lambda: [30])

    # --- Plot/Debug ---
    save_plots: bool = True
    show_plots: bool = False
    plot_format: str = "svg"          # per-figure output format: "svg" or "png"
    debug_validate: bool = True
    debug_index: int = 0
    debug_ligand_conc: Optional[float] = 30
    debug_i_index: int = 4
    deconv_legend_loc: str = "best"
    show_kd_in_legend: bool = False

    # --- Summary ---
    summary_enable: bool = True
    summary_show_calc_shade: bool = False
    # Brief mode skips per-replicate plots and per-replicate *_kd.csv files.
    # In compact mode, summary figures/logs are still written but summary CSVs
    # are suppressed unless export_csv is enabled for workbook-sheet exports.
    brief: bool = False
    # Compact all-model output: write one workbook, one report, one manifest,
    # figures/, and logs/. Legacy per-model CSVs are suppressed unless
    # export_csv is explicitly enabled.
    compact_outputs: bool = False
    export_csv: bool = False
    report_uncertainty: bool = True
    s_iteration_workbook_metrics: List[str] = field(default_factory=lambda: ["R2", "AICc", "p_value"])
    s_iteration_workbook_p_value_mode: str = "threshold"

    # --- Run identity ---
    system_name: str = ""
    temperature_C: Optional[int] = None
    # --- Rendering ---
    max_image_dim: int = 1800
    base_fontsize: int = 20
    colormap: str = "PRGn"
    specific_colormap: Optional[str] = None
    nonspecific_colormap: Optional[str] = None

    # --- Derived (computed in __post_init__) ---
    scale_l_in_to_m: float = field(init=False, default=0.0)
    scale_p_in_to_m: float = field(init=False, default=0.0)
    scale_m_to_out: float = field(init=False, default=0.0)
    p_total_m: float = field(init=False, default=0.0)

    def __post_init__(self):
        from matplotlib import colormaps
        for key in ("colormap", "specific_colormap", "nonspecific_colormap"):
            name = getattr(self, key)
            if name is not None and name not in colormaps:
                raise ValueError(f"Unknown Matplotlib colormap for {key}: {name}")
        raw_s_overrides = dict(self.model_s_overrides or {})
        self.models = canonicalize_model_list(self.models, raw_s_overrides)
        self.reference_model = normalize_model_name(
            self.reference_model,
            raw_s_overrides.get(self.reference_model),
        )
        self.nested_ftest_models = canonicalize_model_list(
            self.nested_ftest_models,
            raw_s_overrides,
        )
        self.model_s_overrides = canonicalize_override_map(self.model_s_overrides)
        self.model_n_overrides = canonicalize_override_map(self.model_n_overrides)
        self.model_display_names = configured_name_map(self.model_display_names)
        self.model_output_names = configured_name_map(self.model_output_names)
        self.scale_l_in_to_m = UNIT_MAP[self.input_unit]
        self.scale_p_in_to_m = UNIT_MAP[self.p_total_unit]
        self.scale_m_to_out = 1.0 / UNIT_MAP[self.output_unit]
        self.p_total_m = self.p_total_val * self.scale_p_in_to_m


def load_configs(yaml_path: str) -> List[RunConfig]:
    """One RunConfig per (system, temperature) pair from a YAML with 'defaults' + 'systems'."""
    config_dir = Path(yaml_path).resolve().parent
    with open(yaml_path, encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    systems = raw.get("systems", [])
    configs = []

    for system in systems:
        sys_cfg = dict(system)
        temps = sys_cfg.pop("temperatures", [25])
        name = sys_cfg.pop("name", "unknown")
        base_dir = Path(sys_cfg.pop("base_dir", "."))
        if not base_dir.is_absolute():
            base_dir = config_dir / base_dir
        base_dir = str(base_dir.resolve())
        wildcard_fmt = sys_cfg.pop("wildcard_fmt", "*.csv")
        out_fmt = sys_cfg.pop("out_fmt", "output_qS_{t}C")

        # Merge: defaults < system-level overrides
        merged = {**defaults, **sys_cfg}
        valid_keys = {f.name for f in RunConfig.__dataclass_fields__.values() if f.init}
        unknown = set(merged) - valid_keys
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")

        for t in temps:
            wildcard = wildcard_fmt.format(t=t)
            out_dir = os.path.join(base_dir, out_fmt.format(t=t))

            cfg_dict = {
                **merged,
                "base_dir": base_dir,
                "out_dir": out_dir,
                "csv_name_wildcard": wildcard,
                "system_name": name,
                "temperature_C": t,
            }
            if cfg_dict.get("data_path") and not Path(cfg_dict["data_path"]).is_absolute():
                cfg_dict["data_path"] = str((config_dir / cfg_dict["data_path"]).resolve())

            # Only pass keys that RunConfig accepts
            filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            configs.append(RunConfig(**filtered))

    return configs
