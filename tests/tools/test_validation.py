from __future__ import annotations

from pathlib import Path

import pytest
from tools import ValidationError, require_directory, require_file


def test_require_file_returns_resolved_path(tmp_path: Path) -> None:
    file_path = tmp_path / "config.yaml"
    file_path.write_text("{}", encoding="utf-8")

    result = require_file(file_path)

    assert result == file_path.resolve()


def test_require_file_raises_for_missing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        require_file(tmp_path / "missing.json")


def test_require_directory_raises_for_file(tmp_path: Path) -> None:
    file_path = tmp_path / "config.yaml"
    file_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError):
        require_directory(file_path)
