from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.auto_docstrings as auto_docstrings


@pytest.fixture()
def repo_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    src = repo / "src"
    docs_scripts = repo / "docs" / "_scripts"
    src.mkdir(parents=True)
    docs_scripts.mkdir(parents=True)

    monkeypatch.setattr(auto_docstrings, "REPO_ROOT", repo)
    monkeypatch.setattr(auto_docstrings, "SRC_ROOT", src)
    return repo


def test_module_name_for_src_package(repo_layout: Path) -> None:
    file_path = repo_layout / "src" / "kgfoundry_common" / "config.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")

    module = auto_docstrings.module_name_for(file_path)

    assert module == "kgfoundry_common.config"


def test_module_name_for_src_dunder_init(repo_layout: Path) -> None:
    file_path = repo_layout / "src" / "kgfoundry_common" / "__init__.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")

    module = auto_docstrings.module_name_for(file_path)

    assert module == "kgfoundry_common"


def test_module_name_for_non_src_files(repo_layout: Path) -> None:
    file_path = repo_layout / "docs" / "_scripts" / "render.py"
    file_path.write_text("", encoding="utf-8")

    module = auto_docstrings.module_name_for(file_path)

    assert module == "docs._scripts.render"
