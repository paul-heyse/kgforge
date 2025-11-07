"""Server configuration and resource limits for CodeIntel MCP server.

This module provides dependency-injection friendly configuration for CodeIntel
services. Configuration can be loaded from environment variables or explicitly
provided for testing and multi-instance deployments.

Aligns with AGENTS.md principle 6 (Configuration via environment variables) and
principle 7 (explicit dependency injection, no global state).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ServerLimits:
    """Immutable server resource limits and feature flags.

    All limits are configurable via environment variables with sensible defaults.
    Use ServerLimits.from_env() to load from environment, or construct directly
    for testing with custom limits.

    Attributes
    ----------
    max_ast_bytes : int
        Maximum file size in bytes for AST generation. Files larger than this
        are rejected to prevent memory exhaustion. Default: 1 MiB.
    max_outline_items : int
        Maximum number of outline items to return. Prevents unbounded responses
        for files with thousands of symbols. Default: 2000.
    list_limit_default : int
        Default limit for list operations when not specified. Default: 100.
    list_limit_max : int
        Maximum allowed limit for list operations. Prevents clients from
        requesting unbounded results. Default: 1000.
    search_limit_max : int
        Maximum number of index results that discovery tools may return. This
        caps `code.searchSymbols` and `code.findReferences` fan-out. Default: 10 000.
    tool_timeout_s : float
        Maximum execution time for tool handlers in seconds. Operations
        exceeding this are cancelled. Default: 10.0.
    rate_limit_qps : float
        Sustained queries per second allowed. Default: 5.0.
    rate_limit_burst : int
        Burst capacity for rate limiter. Allows short bursts above QPS.
        Default: 10.
    enable_ts_query : bool
        Whether to enable arbitrary Tree-sitter queries via ts.query tool.
        Disabled by default for security (prevents resource-intensive queries).
        Default: False.

    Examples
    --------
    Load from environment (production):

    >>> limits = ServerLimits.from_env()
    >>> limits.tool_timeout_s
    10.0

    Create with custom values (testing):

    >>> limits = ServerLimits.defaults()
    >>> limits.rate_limit_qps
    5.0

    Override specific values:

    >>> limits = ServerLimits(
    ...     max_ast_bytes=2097152,  # 2 MiB
    ...     max_outline_items=5000,
    ...     list_limit_default=100,
    ...     list_limit_max=1000,
    ...     tool_timeout_s=30.0,
    ...     rate_limit_qps=10.0,
    ...     rate_limit_burst=20,
    ...     enable_ts_query=True,
    ... )
    """

    max_ast_bytes: int
    max_outline_items: int
    list_limit_default: int
    list_limit_max: int
    search_limit_max: int
    tool_timeout_s: float
    rate_limit_qps: float
    rate_limit_burst: int
    enable_ts_query: bool

    @classmethod
    def from_env(cls) -> ServerLimits:
        """Load limits from environment variables with defaults.

        Environment Variables
        ---------------------
        CODEINTEL_MAX_AST_BYTES : int
            Maximum file size for AST generation (bytes). Default: 1048576 (1 MiB).
        CODEINTEL_MAX_OUTLINE_ITEMS : int
            Maximum outline items to return. Default: 2000.
        CODEINTEL_LIMIT_DEFAULT : int
            Default limit for list operations. Default: 100.
        CODEINTEL_LIMIT_MAX : int
            Maximum allowed limit for lists. Default: 1000.
        CODEINTEL_TOOL_TIMEOUT_S : float
            Tool execution timeout in seconds. Default: 10.0.
        CODEINTEL_RATE_LIMIT_QPS : float
            Sustained queries per second. Default: 5.0.
        CODEINTEL_RATE_LIMIT_BURST : int
            Rate limiter burst capacity. Default: 10.
        CODEINTEL_SEARCH_LIMIT_MAX : int
            Maximum allowed results for index-backed queries. Default: 10000.
        CODEINTEL_ENABLE_TS_QUERY : str
            Enable arbitrary TS queries ("1" to enable). Default: "0".

        Returns
        -------
        ServerLimits
            Configuration loaded from environment.

        Examples
        --------
        >>> import os
        >>> os.environ["CODEINTEL_TOOL_TIMEOUT_S"] = "30"
        >>> limits = ServerLimits.from_env()
        >>> limits.tool_timeout_s
        30.0
        """
        return cls(
            max_ast_bytes=int(os.environ.get("CODEINTEL_MAX_AST_BYTES", "1048576")),
            max_outline_items=int(os.environ.get("CODEINTEL_MAX_OUTLINE_ITEMS", "2000")),
            list_limit_default=int(os.environ.get("CODEINTEL_LIMIT_DEFAULT", "100")),
            list_limit_max=int(os.environ.get("CODEINTEL_LIMIT_MAX", "1000")),
            search_limit_max=int(os.environ.get("CODEINTEL_SEARCH_LIMIT_MAX", "10000")),
            tool_timeout_s=float(os.environ.get("CODEINTEL_TOOL_TIMEOUT_S", "10.0")),
            rate_limit_qps=float(os.environ.get("CODEINTEL_RATE_LIMIT_QPS", "5.0")),
            rate_limit_burst=int(os.environ.get("CODEINTEL_RATE_LIMIT_BURST", "10")),
            enable_ts_query=os.environ.get("CODEINTEL_ENABLE_TS_QUERY", "0") == "1",
        )

    @classmethod
    def defaults(cls) -> ServerLimits:
        """Return default limits for testing and development.

        Returns
        -------
        ServerLimits
            Default configuration with standard values.

        Examples
        --------
        >>> limits = ServerLimits.defaults()
        >>> limits.max_ast_bytes
        1048576
        """
        return cls(
            max_ast_bytes=1048576,  # 1 MiB
            max_outline_items=2000,
            list_limit_default=100,
            list_limit_max=1000,
            search_limit_max=10000,
            tool_timeout_s=10.0,
            rate_limit_qps=5.0,
            rate_limit_burst=10,
            enable_ts_query=False,
        )

    @classmethod
    def permissive(cls) -> ServerLimits:
        """Return permissive limits for testing (high limits, all features enabled).

        Returns
        -------
        ServerLimits
            Permissive configuration with relaxed limits.

        Examples
        --------
        >>> limits = ServerLimits.permissive()
        >>> limits.enable_ts_query
        True
        >>> limits.tool_timeout_s
        60.0
        """
        return cls(
            max_ast_bytes=10485760,  # 10 MiB
            max_outline_items=10000,
            list_limit_default=1000,
            list_limit_max=10000,
            search_limit_max=100000,
            tool_timeout_s=60.0,
            rate_limit_qps=100.0,
            rate_limit_burst=200,
            enable_ts_query=True,
        )


@lru_cache(maxsize=1)
def _cached_limits() -> ServerLimits:
    return ServerLimits.from_env()


def get_limits(*, refresh: bool = False) -> ServerLimits:
    """Return the cached server limits, optionally reloading from the environment.

    Parameters
    ----------
    refresh : bool, optional
        When ``True`` the cached limits are cleared and reloaded from the
        environment. Use this in long-running processes that need to pick up
        configuration changes.

    Returns
    -------
    ServerLimits
        Singleton limits instance loaded from environment.

    Examples
    --------
    >>> limits = get_limits()
    >>> limits.max_ast_bytes > 0
    True

    >>> refreshed = get_limits(refresh=True)
    >>> refreshed is limits
    False
    """
    if refresh:
        _cached_limits.cache_clear()
    return _cached_limits()


def reset_limits_cache() -> None:
    """Clear the cached limits. Convenience wrapper for tests."""
    _cached_limits.cache_clear()


@dataclass(frozen=True)
class ServerContext:
    """Server runtime context for dependency injection.

    This bundles commonly needed dependencies (limits, repo root) for passing
    to services. This enables clean testing and multi-instance deployments.

    Attributes
    ----------
    limits : ServerLimits
        Resource limits and feature flags.
    repo_root : Path
        Repository root path for file sandboxing.

    Examples
    --------
    Create context from environment:

    >>> ctx = ServerContext.from_env()
    >>> ctx.repo_root.is_dir()
    True

    Create context with custom values:

    >>> from pathlib import Path
    >>> ctx = ServerContext(
    ...     limits=ServerLimits.permissive(),
    ...     repo_root=Path("/workspace/myrepo"),
    ... )
    """

    limits: ServerLimits
    repo_root: Path

    @classmethod
    def from_env(cls) -> ServerContext:
        """Create context from environment variables.

        Environment Variables
        ---------------------
        KGF_REPO_ROOT : str
            Repository root path. Default: current working directory.

        Returns
        -------
        ServerContext
            Context loaded from environment.
        """
        return cls(
            limits=get_limits(refresh=True),
            repo_root=Path(os.environ.get("KGF_REPO_ROOT", Path.cwd())).resolve(),
        )

    @classmethod
    def for_testing(cls, repo_root: Path | None = None) -> ServerContext:
        """Create context with permissive limits for testing.

        Parameters
        ----------
        repo_root : Path | None
            Repository root for testing. If None, uses current directory.

        Returns
        -------
        ServerContext
            Testing context with permissive limits.
        """
        return cls(
            limits=ServerLimits.permissive(),
            repo_root=repo_root.resolve() if repo_root else Path.cwd().resolve(),
        )


# Backward compatibility: expose LIMITS as module-level for existing code
# New code should use get_limits() or pass ServerLimits explicitly
LIMITS = get_limits()


__all__ = [
    "LIMITS",
    "ServerContext",
    "ServerLimits",
    "get_limits",
    "reset_limits_cache",
]
