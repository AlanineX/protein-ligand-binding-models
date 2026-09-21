"""Shared model metadata used by scripts, reports, and workflows."""

import re

import numpy as np


MODEL_SPECS = {
    "sequential_specific": {
        "display_name": "Sequential-specific",
        "output_name": "sequential_specific",
    },
    "sequential_specific_s0": {
        "display_name": "Sequential-specific null (S=0)",
        "output_name": "sequential_specific_s0",
    },
    "sequential_specific_s7": {
        "display_name": "Sequential-specific (S=7)",
        "output_name": "sequential_specific_s7",
    },
    "sequential_specific_s9": {
        "display_name": "Sequential-specific (S=9)",
        "output_name": "sequential_specific_s9",
    },
    "sequential_specific_s10": {
        "display_name": "Sequential-specific (S=10)",
        "output_name": "sequential_specific_s10",
    },
    "sequential_adduct": {
        "display_name": "Sequential-adduct",
        "output_name": "sequential_adduct",
    },
    "competing_adduct": {
        "display_name": "Competing-adduct",
        "output_name": "competing_adduct",
    },
    "stochastic_adduct": {
        "display_name": "Stochastic-adduct",
        "output_name": "stochastic_adduct",
    },
    "occupancy_decay": {
        "display_name": "Occupancy-decay",
        "output_name": "occupancy_decay",
    },
    "shared_site": {
        "display_name": "Shared-site",
        "output_name": "shared_site",
    },
}

CANONICAL_MODEL_NAMES = set(MODEL_SPECS)
DISPLAY_MODEL_NAMES = {name: spec["display_name"] for name, spec in MODEL_SPECS.items()}
DEFAULT_OUTPUT_MODEL_NAMES = {name: spec["output_name"] for name, spec in MODEL_SPECS.items()}

DIMENSIONLESS_PARAMS = {"gamma"}

SEQUENTIAL_SPECIFIC_IDENTIFIERS = {
    "sequential_specific",
    "sequential_specific_s7",
    "sequential_specific_s9",
    "sequential_specific_s10",
}

DEFAULT_NSB_MODELS = [
    "sequential_adduct",
    "competing_adduct",
    "stochastic_adduct",
    "occupancy_decay",
    "shared_site",
]

DEFAULT_BATCH_MODELS = list(DEFAULT_NSB_MODELS)

DEFAULT_COMPARISON_MODELS = [
    "sequential_specific_s7",
    "sequential_specific_s9",
    *DEFAULT_NSB_MODELS,
]

DEFAULT_NESTED_FTEST_MODELS = [
    "sequential_specific_s9",
    "sequential_specific_s10",
    "sequential_adduct",
    "competing_adduct",
    "stochastic_adduct",
    "occupancy_decay",
]


def normalize_model_name(model_name, s_override=None):
    """Return the canonical identifier for a configured model name."""
    if model_name is None:
        return model_name
    return str(model_name)


def canonicalize_model_list(model_names, s_overrides=None):
    """Canonicalize a model-name list, preserving order and removing duplicates."""
    s_overrides = s_overrides or {}
    out = []
    for name in model_names or []:
        canonical = normalize_model_name(name, s_overrides.get(name))
        if canonical not in out:
            out.append(canonical)
    return out


def canonicalize_override_map(mapping):
    """Canonicalize model-name keys in a per-model override dictionary."""
    out = {}
    for key, value in (mapping or {}).items():
        canonical = normalize_model_name(key, value)
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
    """True for the sequential-specific implementation and S-specific identifiers."""
    canonical = normalize_model_name(model_name)
    return (
        canonical in SEQUENTIAL_SPECIFIC_IDENTIFIERS
        or bool(re.fullmatch(r"sequential_specific_s\d+", canonical))
    )


def model_role(model_name):
    """Human-readable role used in comparison outputs."""
    canonical = normalize_model_name(model_name)
    if canonical == "sequential_specific_s0":
        return "null_sequential_specific_baseline"
    if canonical in {"sequential_specific_s9", "sequential_specific_s10"}:
        return "apparent_sequential_specific_baseline"
    if canonical == "sequential_specific_s7":
        return "canonical_s7_reference"
    if canonical in DEFAULT_NSB_MODELS:
        return "canonical_s7_adduct_candidate"
    if canonical == "sequential_specific":
        return "sequential_specific"
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
    default_name = DEFAULT_OUTPUT_MODEL_NAMES.get(canonical, canonical)
    name = str(overrides.get(canonical, default_name))
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


def internal_parameter_name(parameter_name, model_name=None):
    """Convert exported Kd-style labels back to internal fit labels."""
    label = str(parameter_name)
    m = re.match(r"^K_?d[,_](\d+)$", label, flags=re.IGNORECASE)
    if m:
        return f"Ks_{m.group(1)}"
    if re.match(r"^K_?d[,_]?avg$", label, flags=re.IGNORECASE):
        return "Ks"
    if re.match(r"^K_?d[,_]?n$", label, flags=re.IGNORECASE):
        if normalize_model_name(model_name) in {"competing_adduct", "occupancy_decay"}:
            return "beta"
        return "Kn"
    return label


def raw_model_name(model_name):
    """Return the canonical registry name for a configured model identifier."""
    return normalize_model_name(model_name)


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


def model_param_map(model_name, S):
    """Map table parameter columns to raw parameter labels for a model.

    Returned keys are ``Kn``, ``Ks_1`` ... ``Ks_S``. Values are parameter labels
    in fit CSVs, or ``None`` when the model has no parameter for that column.
    """
    model_name = raw_model_name(model_name)
    cols = ["Kn"] + [f"Ks_{k}" for k in range(1, S + 1)]
    if model_name in SEQUENTIAL_SPECIFIC_IDENTIFIERS:
        return {c: (None if c == "Kn" else c) for c in cols}
    if model_name in ("sequential_adduct", "stochastic_adduct"):
        return {c: c for c in cols}
    if model_name == "shared_site":
        return {c: ("Kn" if c == "Kn" else "Ks" if c == "Ks_1" else None) for c in cols}
    if model_name in ("occupancy_decay", "competing_adduct"):
        return {c: ("beta" if c == "Kn" else c) for c in cols}
    return {c: c for c in cols}
