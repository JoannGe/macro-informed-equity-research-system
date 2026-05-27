"""Small file-system helpers used across the research platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:
    """Return the repository root for commands run from anywhere below it."""

    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dictionary for blank files."""

    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the main application configuration."""

    return load_yaml(path)


def ensure_directories(config: dict[str, Any]) -> None:
    """Create the configured local data directories if they do not exist."""

    data_config = config.get("data", {})
    for key in ("raw_dir", "demo_dir", "processed_dir"):
        directory = data_config.get(key)
        if directory:
            resolve_path(directory).mkdir(parents=True, exist_ok=True)


def read_dataframe(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV or parquet file based on extension."""

    full_path = resolve_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Required data file was not found: {full_path}")
    if full_path.suffix.lower() == ".parquet":
        return pd.read_parquet(full_path)
    return pd.read_csv(full_path, parse_dates=parse_dates)


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a dataframe as CSV or parquet based on extension."""

    full_path = resolve_path(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    if full_path.suffix.lower() == ".parquet":
        df.to_parquet(full_path, index=index)
    else:
        df.to_csv(full_path, index=index)
    return full_path


def processed_path(filename: str, config: dict[str, Any]) -> Path:
    """Return an absolute path under the processed data directory."""

    return resolve_path(config.get("data", {}).get("processed_dir", "data/processed")) / filename


def demo_path(filename: str, config: dict[str, Any]) -> Path:
    """Return an absolute path under the demo data directory."""

    return resolve_path(config.get("data", {}).get("demo_dir", "data/demo")) / filename
