"""Centralised filesystem paths for the docstring builder tooling."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / ".cache" / "docstring_builder.json"
DOCFACTS_PATH = REPO_ROOT / "docs" / "_build" / "docfacts.json"
MANIFEST_PATH = REPO_ROOT / "docs" / "_build" / "docstrings_manifest.json"
OBSERVABILITY_PATH = REPO_ROOT / "docs" / "_build" / "observability_docstrings.json"
DRIFT_DIR = REPO_ROOT / "docs" / "_build" / "drift"
DOCFACTS_DIFF_PATH = DRIFT_DIR / "docfacts.html"
NAVMAP_DIFF_PATH = DRIFT_DIR / "navmap.html"
SCHEMA_DIFF_PATH = DRIFT_DIR / "schema.html"
DOCSTRINGS_DIFF_PATH = DRIFT_DIR / "docstrings.html"

OBSERVABILITY_MAX_ERRORS = 20
REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 13

__all__ = [
    "CACHE_PATH",
    "DOCFACTS_DIFF_PATH",
    "DOCFACTS_PATH",
    "DOCSTRINGS_DIFF_PATH",
    "DRIFT_DIR",
    "MANIFEST_PATH",
    "NAVMAP_DIFF_PATH",
    "OBSERVABILITY_MAX_ERRORS",
    "OBSERVABILITY_PATH",
    "REPO_ROOT",
    "REQUIRED_PYTHON_MAJOR",
    "REQUIRED_PYTHON_MINOR",
    "SCHEMA_DIFF_PATH",
]
