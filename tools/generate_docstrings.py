#!/usr/bin/env python
"""Generate Docstrings utilities."""

from __future__ import annotations

import subprocess
import sys
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

    Carry out the has python files operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.
    """
    
    
    
    
    
    
    
    return any(path.rglob("*.py"))


def run_doq(target: Path) -> None:
    """Compute run doq.

    Carry out the run doq operation.

    Parameters
    ----------
    target : Path
        Description for ``target``.
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


def run_fallback(target: Path) -> None:
    """Compute run fallback.

    Carry out the run fallback operation.

    Parameters
    ----------
    target : Path
        Description for ``target``.
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

    Carry out the main operation.
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
