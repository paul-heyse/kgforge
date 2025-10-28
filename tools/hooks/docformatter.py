#!/usr/bin/env python
"""Provide utilities for module.

Auto-generated API documentation for the ``tools.hooks.docformatter`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.hooks.docformatter
"""


from __future__ import annotations

import subprocess
import sys


def git_diff_names() -> set[str]:
    """Return git diff names.

    Auto-generated reference for the ``git_diff_names`` callable defined in ``tools.hooks.docformatter``.
    
    Returns
    -------
    Set[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.hooks.docformatter import git_diff_names
    >>> result = git_diff_names()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.hooks.docformatter
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in result.stdout.splitlines() if line.strip()}


def main() -> int:
    """Return main.

    Auto-generated reference for the ``main`` callable defined in ``tools.hooks.docformatter``.
    
    Returns
    -------
    int
        Description of return value.
    
    Examples
    --------
    >>> from tools.hooks.docformatter import main
    >>> result = main()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.hooks.docformatter
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
