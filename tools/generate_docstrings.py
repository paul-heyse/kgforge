#!/usr/bin/env python
"""Overview of generate docstrings.

This module bundles generate docstrings logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
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


def has_python_files(path: Path) -> bool:
    """Compute has python files.

    Carry out the has python files operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.generate_docstrings import has_python_files
    >>> result = has_python_files(...)
    >>> result  # doctest: +ELLIPSIS
    """
    return any(path.rglob("*.py"))


def _record_doq_failure(
    target: Path, cmd: Sequence[str], result: subprocess.CompletedProcess[str]
) -> None:
    """Persist information about a failed DoQ invocation for post-run triage."""
    rel_target = target.relative_to(REPO) if target.is_relative_to(REPO) else target
    header = (
        "[docstrings] doq failed for "
        f"{rel_target} (exit {result.returncode}) running: {' '.join(cmd)}"
    )
    print(header, file=sys.stderr)
    if result.stdout:
        print(result.stdout, file=sys.stderr)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(header + "\n")
        if result.stdout:
            log.write("stdout:\n" + result.stdout + "\n")
        if result.stderr:
            log.write("stderr:\n" + result.stderr + "\n")


def run_doq(target: Path) -> bool:
    """Compute run doq.

    Carry out the run doq operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    target : Path
    target : Path
        Description for ``target``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.generate_docstrings import run_doq
    >>> result = run_doq(...)
    >>> result  # doctest: +ELLIPSIS
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
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        _record_doq_failure(target, cmd, result)
        return False
    return True


def run_fallback(target: Path) -> None:
    """Compute run fallback.

    Carry out the run fallback operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    target : Path
    target : Path
        Description for ``target``.

    Examples
    --------
    >>> from tools.generate_docstrings import run_fallback
    >>> run_fallback(...)  # doctest: +ELLIPSIS
    """
    cmd = [
        sys.executable,
        "tools/auto_docstrings.py",
        "--target",
        str(target),
        "--log",
        str(LOG_FILE),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Examples
    --------
    >>> from tools.generate_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    """
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        if not target.exists() or not target.is_dir():
            continue
        if not has_python_files(target):
            continue
        print(f"[docstrings] Updating {target.relative_to(REPO)}")
        run_doq(target)
        run_fallback(target)


if __name__ == "__main__":
    main()
