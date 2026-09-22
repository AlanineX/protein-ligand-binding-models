"""RunConfig dataclass + YAML loader."""
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from scripts_binding.models import REGISTRY
from scripts_binding.models.metadata import (
    DEFAULT_NSB_MODELS,
    base_model_name,
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
    data_path: str | None = None

    # --- Units ---
    input_unit: str = "M"
    output_unit: str = "uM"
    p_total_val: float = 1.0
    p_total_unit: str = "uM"

    # --- Model parameters ---
    s: int = 4
    s_mode: str = "auto"
    n_override: int | None = None
    model_s_overrides: dict[str, int] = field(default_factory=dict)
    model_n_overrides: dict[str, int] = field(default_factory=dict)
    auto_adjust_s: bool = True
    min_species_frac: float = 0.01
    trim_specific_low_pop: bool = True
    constrain_nsb_weaker_than_specific: bool = True
    nsb_constraint_multistart_n: int = 24
    nsb_constraint_max_nfev: int = 2000
    nsb_constraint_fallback_to_unconstrained: bool = True
    nsb_constraint_log_margin: float = 1e-6

    # --- Models to run (names from models.REGISTRY) ---
    models: list[str] = field(default_factory=lambda: ["sequential_specific", "sequential_adduct"])
    reference_model: str = "sequential_specific"
    nested_ftest_models: list[str] = field(default_factory=lambda: [
        "sequential_adduct",
        "competing_adduct",
        "stochastic_adduct",
        "occupancy_decay",
    ])
    ftest_alpha: float = 0.05
    model_display_names: dict[str, str] = field(default_factory=dict)
    model_output_names: dict[str, str] = field(default_factory=dict)

    # --- Deconvolution ---
    deconv_enable: bool = True
    deconv_source: str = "calc"
    deconv_use_grid: bool = False
    deconv_grid_points: int = 40
    deconv_csv_path: str | None = None
    report_ligand_conc: list[float] = field(default_factory=lambda: [30])

    # --- Plot/Debug ---
    save_plots: bool = True
    show_plots: bool = False
    plot_format: str = "svg"          # per-figure output format: "svg" or "png"
    debug_validate: bool = True
    debug_index: int = 0
    debug_ligand_conc: float | None = 30
    debug_i_index: int = 4
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

    # --- Run identity ---
    system_name: str = ""
    temperature_C: int | None = None
    # --- Rendering ---
    max_image_dim: int = 1800
    base_fontsize: int = 20
    colormap: str = "PRGn"
    specific_colormap: str | None = None
    nonspecific_colormap: str | None = None

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
        self.models = canonicalize_model_list(self.models)
        if not self.models:
            raise ValueError("models must contain at least one model ID")
        unknown_models = [name for name in self.models if base_model_name(name) not in REGISTRY]
        if unknown_models:
            raise ValueError(f"Unknown model ID(s): {', '.join(unknown_models)}")
        self.reference_model = normalize_model_name(self.reference_model)
        self.nested_ftest_models = canonicalize_model_list(self.nested_ftest_models)
        if base_model_name(self.reference_model) not in REGISTRY:
            raise ValueError(f"Unknown reference_model: {self.reference_model}")
        invalid_ftests = [name for name in self.nested_ftest_models if name not in DEFAULT_NSB_MODELS]
        if invalid_ftests:
            raise ValueError(f"Models not eligible for nested F-tests: {', '.join(invalid_ftests)}")
        if not isinstance(self.s_mode, str) or self.s_mode.lower() not in {"auto", "manual", "fixed"}:
            raise ValueError("s_mode must be auto, manual, or fixed (legacy)")
        if not isinstance(self.deconv_source, str) or self.deconv_source.lower() not in {"calc", "exp"}:
            raise ValueError("deconv_source must be calc or exp")
        self.deconv_source = self.deconv_source.lower()
        if self.plot_format not in {"svg", "png"}:
            raise ValueError("plot_format must be svg or png")
        for key in ("input_unit", "output_unit", "p_total_unit"):
            if getattr(self, key) not in UNIT_MAP:
                raise ValueError(f"{key} must be one of: {', '.join(UNIT_MAP)}")
        self.model_s_overrides = canonicalize_override_map(self.model_s_overrides)
        self.model_n_overrides = canonicalize_override_map(self.model_n_overrides)
        self.model_display_names = configured_name_map(self.model_display_names)
        self.model_output_names = configured_name_map(self.model_output_names)
        self.scale_l_in_to_m = UNIT_MAP[self.input_unit]
        self.scale_p_in_to_m = UNIT_MAP[self.p_total_unit]
        self.scale_m_to_out = 1.0 / UNIT_MAP[self.output_unit]
        self.p_total_m = self.p_total_val * self.scale_p_in_to_m


def load_configs(yaml_path: str) -> list[RunConfig]:
    """One RunConfig per (system, temperature) pair from a YAML with 'defaults' + 'systems'."""
    config_dir = Path(yaml_path).resolve().parent
    with open(yaml_path, encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise TypeError("YAML root must be a mapping with a systems list")

    defaults = raw.get("defaults", {})
    systems = raw.get("systems", [])
    if not isinstance(defaults, dict) or not isinstance(systems, list):
        raise TypeError("YAML needs a defaults mapping and a systems list")
    if not systems:
        raise ValueError("YAML needs at least one system")
    derived_keys = {"base_dir", "out_dir", "csv_name_wildcard", "system_name", "temperature_C"}
    if derived_keys & defaults.keys():
        raise ValueError(f"Set derived paths and labels in systems: {', '.join(sorted(derived_keys & defaults.keys()))}")
    configs = []

    for system in systems:
        if not isinstance(system, dict):
            raise TypeError("Each system must be a YAML mapping")
        sys_cfg = dict(system)
        temps = sys_cfg.pop("temperatures", [25])
        name = sys_cfg.pop("name", "unknown")
        base_dir = Path(sys_cfg.pop("base_dir", "."))
        if not base_dir.is_absolute():
            base_dir = config_dir / base_dir
        base_dir = str(base_dir.resolve())
        csv_pattern = sys_cfg.pop("csv_pattern", None)
        old_csv_pattern = sys_cfg.pop("wildcard_fmt", None)
        if csv_pattern is not None and old_csv_pattern is not None:
            raise ValueError("Use csv_pattern or wildcard_fmt, not both")
        csv_pattern = csv_pattern if csv_pattern is not None else old_csv_pattern
        if csv_pattern is None:
            csv_pattern = "*.csv"

        output_folder = sys_cfg.pop("output_folder", None)
        old_output_folder = sys_cfg.pop("out_fmt", None)
        if output_folder is not None and old_output_folder is not None:
            raise ValueError("Use output_folder or out_fmt, not both")
        output_folder = output_folder if output_folder is not None else old_output_folder
        if output_folder is None:
            output_folder = "output_qS_{t}C"

        # Merge: defaults < system-level overrides
        merged = {**defaults, **sys_cfg}
        valid_keys = {f.name for f in fields(RunConfig) if f.init}
        unknown = set(merged) - valid_keys
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        if derived_keys & merged.keys():
            raise ValueError(f"System contains derived keys: {', '.join(sorted(derived_keys & merged.keys()))}")

        for t in temps:
            input_csv_pattern = csv_pattern.format(t=t)
            out_dir = os.path.join(base_dir, output_folder.format(t=t))

            cfg_dict = {
                **merged,
                "base_dir": base_dir,
                "out_dir": out_dir,
                "csv_name_wildcard": input_csv_pattern,
                "system_name": name,
                "temperature_C": t,
            }
            for key in ("data_path", "deconv_csv_path"):
                if cfg_dict.get(key) and not Path(cfg_dict[key]).is_absolute():
                    cfg_dict[key] = str((config_dir / cfg_dict[key]).resolve())

            # Only pass keys that RunConfig accepts
            filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            configs.append(RunConfig(**filtered))

    return configs
