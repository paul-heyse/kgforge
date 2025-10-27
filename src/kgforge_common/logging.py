"""Module for kgforge_common.logging.

NavMap:
- JsonFormatter: Jsonformatter.
- setup_logging: Setup logging.
"""

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """Jsonformatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format.

        Args:
            record (logging.LogRecord): TODO.

        Returns:
            str: TODO.
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


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging.

    Args:
        level (int): TODO.

    Returns:
        None: TODO.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
