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
    """Model the JsonFormatter.
<!-- auto:docstring-builder v1 -->

Represent the jsonformatter data structure used throughout the project. The class encapsulates
behaviour behind a well-defined interface for collaborating components. Instances are typically
created by factories or runtime orchestrators documented nearby.

Parameters
----------
fmt : Any, optional
    Describe ``fmt``.
    Defaults to ``None``.
    Defaults to ``None``.
datefmt : Any, optional
    Describe ``datefmt``.
    Defaults to ``None``.
    Defaults to ``None``.
style : Any, optional
    Describe ``style``.
    Defaults to ``'%'``.
    Defaults to ``'%'``.
validate : Any, optional
    Describe ``validate``.
    Defaults to ``True``.
    Defaults to ``True``.
defaults : Any, optional
    Describe ``defaults``.
    Defaults to ``None``.
    Defaults to ``None``.
"""

    def format(self, record: logging.LogRecord) -> str:
        """Compute format.
<!-- auto:docstring-builder v1 -->

Carry out the format operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
record : logging.LogRecord
    Description for ``record``.
    
    
    

Returns
-------
str
    Description of return value.
    
    
    

Examples
--------
>>> from kgfoundry_common.logging import format
>>> result = format(...)
>>> result  # doctest: +ELLIPSIS
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
    """Compute setup logging.
<!-- auto:docstring-builder v1 -->

Carry out the setup logging operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
level : int, optional
    Optional parameter default ``logging.INFO``. Description for ``level``.
    
    
    Defaults to ``20``.
    
    Defaults to ``20``.

Examples
--------
>>> from kgfoundry_common.logging import setup_logging
>>> setup_logging()  # doctest: +ELLIPSIS
"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
