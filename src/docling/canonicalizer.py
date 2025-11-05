"""Overview of canonicalizer.

This module bundles canonicalizer logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from kgfoundry_common.navmap_types import NavMap

__all__ = ["canonicalize_text"]

__navmap__: Final[NavMap] = {
    "title": "docling.canonicalizer",
    "synopsis": "String canonicalisation utilities for docling preprocessing",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@docling",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        "canonicalize_text": {
            "owner": "@docling",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor canonicalize_text]
def canonicalize_text(blocks: list[str]) -> str:
    """Canonicalize text blocks by normalizing Unicode and whitespace.

    Normalizes Unicode characters (NFC), standardizes line endings,
    replaces bullet points and dashes with hyphens, removes control
    characters, and collapses whitespace. Returns a single joined string.

    Parameters
    ----------
    blocks : list[str]
        List of text blocks to canonicalize.

    Returns
    -------
    str
        Canonicalized text joined with newlines, with empty blocks filtered.
    """

    def norm(s: str) -> str:
        """Compute norm.

        Carry out the norm operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        s : str
            Description for ``s``.

        Returns
        -------
        str
            Description of return value.

        Examples
        --------
        >>> from docling.canonicalizer import norm
        >>> result = norm(...)
        >>> result  # doctest: +ELLIPSIS
        """
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[\u2022\u25E6\u2013]", "-", s)  # bullets/dashes
        s = re.sub(r"[\x00-\x1F]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    normed = [norm(b) for b in blocks if b.strip()]
    return "\n".join(normed)
