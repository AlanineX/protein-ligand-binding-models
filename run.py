"""Run a binding-fit YAML from VS Code or a terminal."""

import sys
from pathlib import Path

from scripts_binding.core.runner import run_all

DEFAULT_CONFIG = Path(__file__).resolve().parent / "examples" / "synthetic" / "fit.yaml"


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python run.py [path/to/fit.yaml]")
    config = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_CONFIG
    if not config.is_absolute():
        config = Path(__file__).resolve().parent / config
    run_all(config)
