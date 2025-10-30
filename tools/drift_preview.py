"""Helpers for generating HTML drift previews."""

from __future__ import annotations

import difflib
from pathlib import Path

_DIFF = difflib.HtmlDiff(wrapcolumn=80)


def write_html_diff(before: str, after: str, output: Path, title: str) -> None:
    """Render an HTML diff between ``before`` and ``after`` to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    html = _DIFF.make_file(
        before.splitlines(),
        after.splitlines(),
        fromdesc=f"{title} (previous)",
        todesc=f"{title} (current)",
        context=True,
        numlines=20,
    )
    output.write_text(html, encoding="utf-8")


__all__ = ["write_html_diff"]
