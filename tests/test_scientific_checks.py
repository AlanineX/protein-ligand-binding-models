"""Run the existing numerical recovery and paper regression checks."""

from pathlib import Path
import subprocess
import sys

import pytest


CHECKS = [
    "sequential_specific", "sequential_adduct", "competing_adduct",
    "stochastic_adduct", "occupancy_decay", "shared_site",
    "paper_daubenfeld", "paper_guan", "paper_shimon",
    "nsb_constraint", "parameter_semantics",
]


@pytest.mark.parametrize("name", CHECKS)
def test_scientific_check(name):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "checks" / f"test_{name}.py")],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
