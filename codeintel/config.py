"""Server configuration and resource limits for CodeIntel MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerLimits:
    """Immutable server resource limits and feature flags.

    All limits are configurable via environment variables with sensible defaults.
    """

    max_ast_bytes: int = int(os.environ.get("CODEINTEL_MAX_AST_BYTES", "1048576"))  # 1 MiB
    max_outline_items: int = int(os.environ.get("CODEINTEL_MAX_OUTLINE_ITEMS", "2000"))
    list_limit_default: int = int(os.environ.get("CODEINTEL_LIMIT_DEFAULT", "100"))
    list_limit_max: int = int(os.environ.get("CODEINTEL_LIMIT_MAX", "1000"))
    tool_timeout_s: float = float(os.environ.get("CODEINTEL_TOOL_TIMEOUT_S", "10.0"))
    rate_limit_qps: float = float(os.environ.get("CODEINTEL_RATE_LIMIT_QPS", "5.0"))
    rate_limit_burst: int = int(os.environ.get("CODEINTEL_RATE_LIMIT_BURST", "10"))
    enable_ts_query: bool = os.environ.get("CODEINTEL_ENABLE_TS_QUERY", "0") == "1"


LIMITS = ServerLimits()
