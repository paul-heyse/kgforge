from __future__ import annotations

from pathlib import Path

import tools.generate_docstrings as docstrings


def test_default_protected_paths() -> None:
    """Ensure the default skip list protects the orchestration scripts."""

    repo = Path(__file__).resolve().parents[2]
    expected = {
        (repo / "tools" / "auto_docstrings.py").resolve(),
        (repo / "tools" / "generate_docstrings.py").resolve(),
        (repo / "tools" / "add_module_docstrings.py").resolve(),
        (repo / "tools" / "check_docstrings.py").resolve(),
    }
    assert docstrings.DEFAULT_PROTECTED == expected


def test_generate_docstrings_skips_protected(tmp_path: Path, monkeypatch) -> None:
    """Protected files remain unchanged while other targets are updated."""

    target = tmp_path / "pkg"
    protected = target / "tools" / "auto_docstrings.py"
    unprotected = target / "tools" / "module.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("print('stay safe')\n", encoding="utf-8")
    unprotected.write_text("print('touch me')\n", encoding="utf-8")

    seen_skips: list[set[Path]] = []

    def fake_run_doq(directory: Path) -> None:
        for file_path in sorted(directory.rglob("*.py")):
            content = file_path.read_text(encoding="utf-8")
            file_path.write_text(content + "# doq\n", encoding="utf-8")

    def fake_run_fallback(directory: Path, skip: list[Path] | None = None) -> None:
        skip_set = {Path(item).resolve() for item in (skip or [])}
        seen_skips.append(skip_set)
        for file_path in sorted(directory.rglob("*.py")):
            if file_path.resolve() in skip_set:
                continue
            content = file_path.read_text(encoding="utf-8")
            file_path.write_text(content + "# fallback\n", encoding="utf-8")

    monkeypatch.setattr(docstrings, "run_doq", fake_run_doq)
    monkeypatch.setattr(docstrings, "run_fallback", fake_run_fallback)

    docstrings.generate_docstrings([target], {protected})

    assert protected.read_text(encoding="utf-8") == "print('stay safe')\n"
    updated = unprotected.read_text(encoding="utf-8")
    assert "# doq" in updated
    assert "# fallback" in updated
    assert any(protected.resolve() in skip_entry for skip_entry in seen_skips)
