"""Shared model metadata used by scripts, reports, and workflows."""

import re

import numpy as np

DISPLAY_MODEL_NAMES = {
    "sequential_specific": "Sequential-specific",
    "sequential_adduct": "Sequential-adduct",
    "competing_adduct": "Competing-adduct",
    "stochastic_adduct": "Stochastic-adduct",
    "occupancy_decay": "Occupancy-decay",
    "shared_site": "Shared-site",
}

DIMENSIONLESS_PARAMS = {"gamma"}

DEFAULT_NSB_MODELS = [
    "sequential_adduct",
    "competing_adduct",
    "stochastic_adduct",
    "occupancy_decay",
    "shared_site",
]

def normalize_model_name(model_name):
    """Return the canonical identifier for a configured model name."""
    if model_name is None:
        return model_name
    return str(model_name)


def canonicalize_model_list(model_names):
    """Canonicalize a model-name list, preserving order and removing duplicates."""
    out = []
    for name in model_names or []:
        canonical = normalize_model_name(name)
        if canonical not in out:
            out.append(canonical)
    return out


def canonicalize_override_map(mapping):
    """Canonicalize model-name keys in a per-model override dictionary."""
    out = {}
    for key, value in (mapping or {}).items():
        canonical = normalize_model_name(key)
        out[canonical] = value
    return out


def configured_name_map(mapping):
    """Canonicalize a config mapping keyed by model name."""
    out = {}
    for key, value in (mapping or {}).items():
        out[normalize_model_name(key)] = value
    return out


def base_model_name(model_name):
    """Return the implemented model name for a configured model identifier."""
    canonical = normalize_model_name(model_name)
    if is_sequential_specific_model(canonical):
        return "sequential_specific"
    return canonical


def is_sequential_specific_model(model_name):
    """True for the base model or any site-count identifier."""
    canonical = normalize_model_name(model_name)
    return canonical == "sequential_specific" or bool(re.fullmatch(r"sequential_specific_s\d+", canonical))


def model_role(model_name):
    """Human-readable role used in comparison outputs."""
    canonical = normalize_model_name(model_name)
    if is_sequential_specific_model(canonical):
        return "sequential_specific_baseline"
    if canonical in DEFAULT_NSB_MODELS:
        return "adduct_candidate"
    return "candidate"


def display_model_name(model_name, overrides=None):
    """Return the preferred human-readable display name for a model."""
    canonical = normalize_model_name(model_name)
    overrides = configured_name_map(overrides)
    if canonical in overrides:
        return str(overrides[canonical])
    m = re.fullmatch(r"sequential_specific_s(\d+)", canonical)
    if m:
        if int(m.group(1)) == 0:
            return "Sequential-specific null (S=0)"
        return f"Sequential-specific (S={int(m.group(1))})"
    return DISPLAY_MODEL_NAMES.get(canonical, canonical)


def output_model_name(model_name, overrides=None):
    """Return the configured filesystem-safe output name for a model."""
    canonical = normalize_model_name(model_name)
    overrides = configured_name_map(overrides)
    name = str(overrides.get(canonical, canonical))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._-") or canonical


def output_parameter_name(parameter_name):
    """Display-only parameter label for reports and exported tables."""
    label = str(parameter_name)
    m = re.match(r"^Ks_(\d+)$", label)
    if m:
        return f"Kd,{m.group(1)}"
    if label == "Ks":
        return "Kd,avg"
    if label in {"Kn", "beta"}:
        return "Kd,n"
    return label


def is_dimensionless_param(param_name):
    """True for fitted parameters that are not log(Ka)-encoded constants."""
    return param_name in DIMENSIONLESS_PARAMS


def parameter_values_from_optimizer(param_names, raw_params):
    """Convert optimizer coordinates to reported parameter values.

    Most model coordinates are log association constants and report as Ka in
    M^-1. Dimensionless coordinates such as gamma are already in natural units.
    """
    return np.array([
        float(raw) if is_dimensionless_param(name) else float(np.exp(raw))
        for name, raw in zip(param_names, raw_params)
    ], dtype=float)


def ka_kd_from_optimizer(param_names, raw_params):
    """Return (reported_values, Ka_M_inv, Kd_M) for model parameters.

    Dimensionless parameters have finite reported_values but NaN Ka/Kd fields.
    """
    values = parameter_values_from_optimizer(param_names, raw_params)
    Ka_M_inv = np.full(len(values), np.nan, dtype=float)
    Kd_M = np.full(len(values), np.nan, dtype=float)
    for i, (name, value) in enumerate(zip(param_names, values)):
        if is_dimensionless_param(name):
            continue
        Ka_M_inv[i] = value
        Kd_M[i] = 1.0 / value if value > 0 else np.nan
    return values, Ka_M_inv, Kd_M
