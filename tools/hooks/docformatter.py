#!/usr/bin/env python
"""Docformatter utilities."""

from __future__ import annotations

import subprocess
import sys


def git_diff_names() -> set[str]:
    """Compute git diff names.

    Carry out the git diff names operation.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















    result = subprocess.run(
        ["git", "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in result.stdout.splitlines() if line.strip()}


def main() -> int:
    """Compute main.

    Carry out the main operation.

    Returns
    -------
    int
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




















    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    before = git_diff_names()

    cmd = [
        sys.executable,
        "-m",
        "docformatter",
        "--in-place",
        "--wrap-summaries=100",
        "--wrap-descriptions=100",
        "-r",
        "src",
    ]
    result = subprocess.run(cmd, cwd=repo)
    if result.returncode not in {0, 3}:
        return result.returncode

    after = git_diff_names()
    changed = sorted(after - before)
    if changed:
        sys.stderr.write("docformatter updated:\n")
        for path in changed:
            sys.stderr.write(f"  {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
