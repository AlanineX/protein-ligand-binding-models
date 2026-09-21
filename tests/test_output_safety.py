from pathlib import Path

import pytest

from scripts_binding.core.config import RunConfig
from scripts_binding.core.runner import _prepare_compact_output_dir


def test_unmanaged_compact_directory_is_preserved(tmp_path: Path):
    out = tmp_path / "existing"
    out.mkdir()
    keep = out / "important.txt"
    keep.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unmanaged files"):
        _prepare_compact_output_dir(RunConfig(out_dir=str(out)))
    assert keep.read_text(encoding="utf-8") == "keep"
