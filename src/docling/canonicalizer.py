"""Module for docling.canonicalizer.

NavMap:
- canonicalize_text: Apply NFC, normalize whitespace and bullets, preserve….
"""

from __future__ import annotations

import re
import unicodedata


def canonicalize_text(blocks: list[str]) -> str:
    """Apply NFC, normalize whitespace and bullets, preserve single newlines between blocks."""

    def norm(s: str) -> str:
        """Norm.

        Args:
            s (str): TODO.

        Returns:
            str: TODO.
        """
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[\u2022\u25E6\u2013]", "-", s)  # bullets/dashes
        s = re.sub(r"[\x00-\x1F]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    normed = [norm(b) for b in blocks if b.strip()]
    return "\n".join(normed)
