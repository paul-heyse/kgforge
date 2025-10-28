"""Logging utilities."""

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
            "symbols": ["JsonFormatter", "setup_logging"],
        },
    ],
}


# [nav:anchor JsonFormatter]
class JsonFormatter(logging.Formatter):
    """Describe JsonFormatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Compute format.

        Carry out the format operation.

        Parameters
        ----------
        record : logging.LogRecord
            Description for ``record``.

        Returns
        -------
        str
            Description of return value.
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

    Carry out the setup logging operation.

    Parameters
    ----------
    level : int | None
        Description for ``level``.
    """
    
    
    
    
    
    
    
    
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
