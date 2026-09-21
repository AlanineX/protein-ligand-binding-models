"""Project path helpers for script and workflow defaults."""

from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def project_path(*parts):
    """Return a path under the repository/workspace root."""
    return PROJECT_ROOT.joinpath(*parts)


def package_path(*parts):
    """Return a path under the scripts_binding package root."""
    return PACKAGE_ROOT.joinpath(*parts)


def env_path(env_name, default):
    """Resolve an environment override or a project-relative default path."""
    raw = os.environ.get(env_name)
    value = Path(raw).expanduser() if raw else Path(default)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def resolve_path(value, base=PROJECT_ROOT):
    """Resolve a CLI path, treating relative paths as relative to ``base``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path
