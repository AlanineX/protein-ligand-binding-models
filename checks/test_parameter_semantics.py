"""Regression checks for parameter units and NaN-aware fitting helpers."""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.core.fitting import _trim_low_pop_species
from scripts_binding.core.reporting import save_kd_csv
from scripts_binding.models.metadata import ka_kd_from_optimizer


def test_gamma_is_not_reported_as_kd():
    labels = ["beta", "gamma", "Ks_1"]
    raw = np.array([np.log(2.0e3), 0.5, np.log(1.0e5)])
    values, ka, kd = ka_kd_from_optimizer(labels, raw)

    assert np.allclose(values[[0, 2]], [2.0e3, 1.0e5])
    assert values[1] == 0.5
    assert np.isnan(ka[1])
    assert np.isnan(kd[1])
    assert np.allclose(kd[[0, 2]], [5.0e-4, 1.0e-5])


def test_trim_low_pop_species_ignores_missing_tail():
    F = np.array([
        [1.0, 0.0, np.nan],
        [0.8, 0.2, np.nan],
    ])
    trimmed, cols = _trim_low_pop_species(F, ["I0", "I1", "I2"], min_frac=0.01)

    assert cols == ["I0", "I1"]
    assert trimmed.shape == (2, 2)
    assert np.all(np.isfinite(trimmed))
    assert np.allclose(trimmed.sum(axis=1), 1.0)


def test_gamma_kd_csv_fields_are_blank():
    labels = ["beta", "gamma", "Ks_1"]
    raw = np.array([np.log(2.0e3), 0.5, np.log(1.0e5)])
    _values, ka, kd = ka_kd_from_optimizer(labels, raw)

    class Cfg:
        output_unit = "uM"
        scale_m_to_out = 1e6

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        path = fh.name
    try:
        save_kd_csv(labels, raw, ka, kd, False, None, None, None, path, Cfg)
        df = pd.read_csv(path)
        gamma = df[df["Param"] == "gamma"].iloc[0]
        assert gamma["Value"] == 0.5
        assert gamma["Value_unit"] == "unitless"
        assert np.isnan(gamma["Ka(uM^-1)"])
        assert np.isnan(gamma["Kd(uM)"])
    finally:
        os.unlink(path)


def main():
    test_gamma_is_not_reported_as_kd()
    test_trim_low_pop_species_ignores_missing_tail()
    test_gamma_kd_csv_fields_are_blank()
    print("PASS: parameter semantics and NaN-aware trimming")


if __name__ == "__main__":
    main()
