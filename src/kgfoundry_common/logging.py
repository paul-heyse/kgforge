"""Overview of logging.

This module bundles logging logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["JsonFormatter", "setup_logging"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.logging",
    "synopsis": "Structured logging helpers shared across kgfoundry",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        "JsonFormatter": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
        "setup_logging": {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor JsonFormatter]
class JsonFormatter(logging.Formatter):
    """Describe JsonFormatter.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    fmt : inspect._empty, optional
        Describe ``fmt``.
        Defaults to ``None``.
    datefmt : inspect._empty, optional
        Describe ``datefmt``.
        Defaults to ``None``.
    style : inspect._empty, optional
        Describe ``style``.
        Defaults to ``'%'``.
    validate : inspect._empty, optional
        Describe ``validate``.
        Defaults to ``True``.
    defaults : inspect._empty, optional
        Describe ``defaults``.
        Defaults to ``None``.
        

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def format(self, record: logging.LogRecord) -> str:
        """Describe format.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        record : logging.LogRecord
            Describe ``record``.
            

        Returns
        -------
        str
            Describe return value.
"""
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        for k in ("run_id", "doc_id", "chunk_id"):
            v = getattr(record, k, None)
            if v:
                data[k] = v
        return json.dumps(data)


# [nav:anchor setup_logging]
def setup_logging(level: int = logging.INFO) -> None:
    """Describe setup logging.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    level : int, optional
        Describe ``level``.
        Defaults to ``20``.
"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
