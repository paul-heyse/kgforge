#!/usr/bin/env python
"""Coordinate the docstring generation pipeline for the repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC_TEMPLATES = REPO / "tools" / "doq_templates" / "numpy"
TARGETS = [
    REPO / "src",
    REPO / "tools",
    REPO / "docs" / "_scripts",
]
LOG_DIR = REPO / "site" / "_build" / "docstrings"
LOG_FILE = LOG_DIR / "fallback.log"
ENV_INCLUDE_PROTECTED = "KGFOUNDRY_DOCSTRINGS_INCLUDE_PROTECTED"
ENV_ADDITIONAL_PROTECTED = "KGFOUNDRY_DOCSTRINGS_PROTECTED"
DEFAULT_PROTECTED = {
    REPO / "tools" / "auto_docstrings.py",
    REPO / "tools" / "generate_docstrings.py",
    REPO / "tools" / "add_module_docstrings.py",
    REPO / "tools" / "check_docstrings.py",
}


def _normalize_path(path: Path) -> Path:
    """Return an absolute path anchored to the repository root."""

    return path if path.is_absolute() else (REPO / path)


def _resolve_paths(paths: Iterable[Path]) -> set[Path]:
    """Resolve ``paths`` to absolute equivalents for reliable comparisons."""

    return {candidate.resolve() for candidate in map(_normalize_path, paths)}


def _truthy(value: str | None) -> bool:
    """Interpret environment flag strings such as ``"1"`` or ``"true"``."""

    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for docstring generation orchestration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="*",
        type=Path,
        help="Optional directories to scan instead of the default src/tools/docs set.",
    )
    parser.add_argument(
        "--include-protected",
        action="store_true",
        help="Process orchestration scripts that are skipped by default.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        type=Path,
        help="Additional file paths to exclude from automated docstring generation.",
    )
    return parser.parse_args(argv)


def iter_protected(target: Path, protected: Iterable[Path]) -> Iterator[Path]:
    """Yield protected files that live under ``target``."""

    target = target.resolve()
    for path in protected:
        try:
            path.resolve().relative_to(target)
        except ValueError:
            continue
        if path.exists():
            yield path


@contextmanager
def hidden(paths: Iterable[Path]) -> Iterator[None]:
    """Temporarily hide ``paths`` so external tools cannot mutate them."""

    renamed: list[tuple[Path, Path]] = []
    try:
        for original in paths:
            temporary = original.with_name(original.name + ".protected")
            counter = 0
            while temporary.exists():
                counter += 1
                temporary = original.with_name(f"{original.name}.protected{counter}")
            original.rename(temporary)
            renamed.append((original, temporary))
        yield
    finally:
        for original, temporary in reversed(renamed):
            if not temporary.exists():
                continue
            if original.exists():
                original.unlink()
            temporary.rename(original)


def has_python_files(path: Path) -> bool:
    """Return ``True`` when ``path`` contains at least one Python source file."""

    return any(path.rglob("*.py"))


def run_doq(target: Path) -> None:
    """Run ``doq`` using the repository's custom NumPy templates.

    Parameters
    ----------
    target : Path
        Directory whose Python files should receive template docstrings.
    """

    cmd = [
        sys.executable,
        "-m",
        "doq.cli",
        "--formatter",
        "numpy",
        "-t",
        str(DOC_TEMPLATES),
        "-w",
        "-r",
        "-d",
        str(target),
    ]
    subprocess.run(cmd, check=True)


def run_fallback(target: Path, skip: Iterable[Path] | None = None) -> None:
    """Invoke the auto-docstring fallback generator for ``target``.

    Parameters
    ----------
    target : Path
        Directory that should be processed by :mod:`tools.auto_docstrings`.
    skip : Iterable[pathlib.Path], optional
        Explicit file paths that the fallback generator should ignore.
    """

    cmd = [
        sys.executable,
        "tools/auto_docstrings.py",
        "--target",
        str(target),
        "--log",
        str(LOG_FILE),
    ]
    if skip:
        for path in skip:
            cmd.extend(["--skip", str(path)])
    subprocess.run(cmd, check=True)


def generate_docstrings(targets: Iterable[Path], protected: Iterable[Path]) -> None:
    """Run the docstring pipeline while leaving ``protected`` files untouched.

    Parameters
    ----------
    targets : Iterable[pathlib.Path]
        Directories that should be processed.
    protected : Iterable[pathlib.Path]
        Absolute or repository-relative file paths that must be skipped.
    """

    protected_paths = list(_resolve_paths(protected))
    for target in targets:
        target = _normalize_path(target).resolve()
        if not target.exists() or not target.is_dir():
            continue
        if not has_python_files(target):
            continue
        try:
            relative = target.relative_to(REPO)
        except ValueError:
            relative = target
        print(f"[docstrings] Updating {relative}")
        protected_for_target = list(iter_protected(target, protected_paths))
        with hidden(protected_for_target):
            run_doq(target)
        run_fallback(target, protected_for_target)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse configuration sources and execute docstring generation."""

    args = parse_args(argv)
    include_protected = args.include_protected or _truthy(
        os.environ.get(ENV_INCLUDE_PROTECTED),
    )
    additional = list(args.skip)
    env_extra = os.environ.get(ENV_ADDITIONAL_PROTECTED)
    if env_extra:
        additional.extend(Path(item) for item in env_extra.split(os.pathsep) if item)
    protected: set[Path] = set()
    if not include_protected:
        protected.update(DEFAULT_PROTECTED)
    protected.update(additional)

    targets = args.targets or TARGETS

    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    generate_docstrings(targets, protected)


if __name__ == "__main__":
    main()
