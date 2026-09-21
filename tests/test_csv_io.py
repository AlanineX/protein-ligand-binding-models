from pathlib import Path

import pytest

from scripts_binding.core.csv_io import read_csv
from scripts_binding.core.config import RunConfig
from scripts_binding.core.fitting import load_binding_csv


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "cp1252"])
def test_binding_csv_encodings(tmp_path: Path, encoding: str):
    path = tmp_path / "titration.csv"
    path.write_bytes("Entry,I0,I1,note\n0,1,0,café\n5,0.5,0.5,café\n".encode(encoding))
    frame, ligand, columns, fractions = load_binding_csv(
        path, RunConfig(input_unit="uM", p_total_val=1, p_total_unit="uM")
    )
    assert frame["note"].iloc[0] == "café"
    assert columns == ["I0", "I1"]
    assert ligand.tolist() == pytest.approx([0.0, 5e-6])
    assert fractions[1].tolist() == [0.5, 0.5]


def test_csv_invalid_bytes_raise(tmp_path: Path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"Entry,I0\n0,1\n1,\x81\n")
    with pytest.raises(UnicodeError):
        read_csv(path)
