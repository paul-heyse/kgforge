"""Module for kgforge_common.ids."""

from __future__ import annotations

import base64
import hashlib


def urn_doc_from_text(text: str) -> str:
    """Urn doc from text.

    Args:
        text (str): TODO.

    Returns:
        str: TODO.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()[:16]
    b32 = base64.b32encode(h).decode("ascii").strip("=").lower()
    return f"urn:doc:sha256:{b32}"


def urn_chunk(doc_hash: str, start: int, end: int) -> str:
    """Urn chunk.

    Args:
        doc_hash (str): TODO.
        start (int): TODO.
        end (int): TODO.

    Returns:
        str: TODO.
    """
    return f"urn:chunk:{doc_hash.split(':')[-1]}:{start}-{end}"
