I'll help you develop a comprehensive plan to address the configuration and context management deficiencies in the CodeIntel MCP project. Let me start by setting up the environment and thoroughly reviewing the codebase.

[1 tool called]

Now let me read all the files in the codeintel_rev directory to understand the full implementation:

[12 tools called]

Now let me read a few more files to complete the context:

[3 tools called]

Perfect! I now have a complete understanding of the codebase. Let me create a comprehensive, exhaustive plan to address Section 1 of the design review: **Centralize Configuration and Context Management**.

## Exhaustive Implementation Plan: Section 1 - Centralize Configuration and Context Management

### Executive Summary

The current CodeIntel MCP implementation exhibits three critical configuration management deficiencies that violate AGENTS.md design principles:

1. **Repeated Settings Loading**: Adapters call `load_settings()` multiple times per request, causing redundant environment variable parsing
2. **Inconsistent Context Access**: Mixed use of `load_settings()` (adapters) vs `get_service_context().settings` (semantic adapter)
3. **Deferred FAISS Initialization**: Lazy loading causes first-request latency and lacks proper health checks

This plan delivers a **structural, holistic, best-in-class solution** that:
- Eliminates all redundant configuration loading
- Centralizes configuration lifecycle management in FastAPI lifespan
- Implements proper startup health checks with early failure detection
- Provides extensibility for future multi-repository support
- Achieves 100% compliance with AGENTS.md principles (explicit dependency injection, no global state, RFC 9457 error handling)

---

### Phase 1: Configuration Lifecycle Architecture (Foundation)

#### 1.1 Create Configuration Context Manager

**Objective**: Centralize configuration loading and validation in FastAPI application startup, eliminating all adapter-level `load_settings()` calls.

**Implementation**:

```python
# codeintel_rev/app/config_context.py
"""Application-level configuration context manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel_rev.config.settings import Settings, load_settings
from codeintel_rev.io.duckdb_catalog import DuckDBCatalog
from codeintel_rev.io.faiss_manager import FAISSManager
from codeintel_rev.io.vllm_client import VLLMClient
from kgfoundry_common.errors import ConfigurationError, KgFoundryError
from kgfoundry_common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import contextmanager

LOGGER = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ResolvedPaths:
    """Canonicalized filesystem paths for runtime operations.
    
    All paths are absolute and resolved relative to repo_root. This eliminates
    ambiguity and ensures consistent path handling throughout the application.
    """
    
    repo_root: Path
    data_dir: Path
    vectors_dir: Path
    faiss_index: Path
    duckdb_path: Path
    scip_index: Path


def resolve_application_paths(settings: Settings) -> ResolvedPaths:
    """Resolve all configured paths to absolute paths.
    
    Parameters
    ----------
    settings : Settings
        Application settings containing path configuration.
    
    Returns
    -------
    ResolvedPaths
        Fully resolved absolute paths for all resources.
    
    Raises
    ------
    ConfigurationError
        If repo_root is not a valid directory or paths cannot be resolved.
    """
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    
    if not repo_root.exists():
        raise ConfigurationError(
            f"Repository root does not exist: {repo_root}",
            context={"repo_root": str(repo_root), "source": "REPO_ROOT env var"}
        )
    
    if not repo_root.is_dir():
        raise ConfigurationError(
            f"Repository root is not a directory: {repo_root}",
            context={"repo_root": str(repo_root)}
        )
    
    def _resolve(path_str: str) -> Path:
        """Resolve a path string relative to repo_root."""
        path = Path(path_str)
        if path.is_absolute():
            return path.expanduser().resolve()
        return (repo_root / path).resolve()
    
    return ResolvedPaths(
        repo_root=repo_root,
        data_dir=_resolve(settings.paths.data_dir),
        vectors_dir=_resolve(settings.paths.vectors_dir),
        faiss_index=_resolve(settings.paths.faiss_index),
        duckdb_path=_resolve(settings.paths.duckdb_path),
        scip_index=_resolve(settings.paths.scip_index),
    )


@dataclass(slots=True)
class ApplicationContext:
    """Application-wide context holding all configuration and long-lived clients.
    
    This is the single source of truth for configuration throughout the application.
    It's initialized once during FastAPI startup and injected into request handlers.
    
    Attributes
    ----------
    settings : Settings
        Immutable application settings loaded from environment variables.
    paths : ResolvedPaths
        Canonicalized filesystem paths for all resources.
    vllm_client : VLLMClient
        vLLM embedding service client (persistent HTTP connection pool).
    faiss_manager : FAISSManager
        FAISS index manager (lazy-loads GPU resources on first search).
    """
    
    settings: Settings
    paths: ResolvedPaths
    vllm_client: VLLMClient
    faiss_manager: FAISSManager
    
    @classmethod
    def create(cls) -> ApplicationContext:
        """Create application context from environment variables.
        
        Returns
        -------
        ApplicationContext
            Initialized context with all clients and configuration.
        
        Raises
        ------
        ConfigurationError
            If configuration is invalid or required resources are missing.
        """
        LOGGER.info("Loading application configuration from environment")
        settings = load_settings()
        paths = resolve_application_paths(settings)
        
        vllm_client = VLLMClient(settings.vllm)
        faiss_manager = FAISSManager(
            index_path=paths.faiss_index,
            vec_dim=settings.index.vec_dim,
            nlist=settings.index.faiss_nlist,
            use_cuvs=settings.index.use_cuvs,
        )
        
        LOGGER.info(
            "Application context created",
            extra={
                "repo_root": str(paths.repo_root),
                "faiss_index": str(paths.faiss_index),
                "vllm_url": settings.vllm.base_url,
            }
        )
        
        return cls(
            settings=settings,
            paths=paths,
            vllm_client=vllm_client,
            faiss_manager=faiss_manager,
        )
    
    @contextmanager
    def open_catalog(self) -> Iterator[DuckDBCatalog]:
        """Open a DuckDB catalog connection.
        
        Yields
        ------
        DuckDBCatalog
            Catalog instance for querying chunk metadata.
        
        Notes
        -----
        The catalog connection is automatically closed when the context exits.
        """
        catalog = DuckDBCatalog(self.paths.duckdb_path, self.paths.vectors_dir)
        try:
            catalog.open()
            yield catalog
        finally:
            catalog.close()
```

**Rationale**:
- **Eliminates Redundancy**: Settings loaded once at startup, not per-request
- **Explicit Lifecycle**: Configuration bound to FastAPI lifespan, not global state
- **Type-Safe Paths**: `ResolvedPaths` eliminates path resolution logic duplication
- **AGENTS.MD Compliance**: Follows dependency injection pattern, structured error handling with RFC 9457

---

#### 1.2 Integrate Configuration Context into FastAPI Lifespan

**Objective**: Move configuration initialization from lazy/on-demand to explicit startup phase with proper error handling and health checks.

**Implementation**:

```python
# codeintel_rev/app/main.py (updated lifespan function)
"""FastAPI application with MCP server mount.

Provides health/readiness endpoints, CORS, and streaming support.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from codeintel_rev.app.config_context import ApplicationContext
from codeintel_rev.app.readiness import ReadinessProbe
from codeintel_rev.mcp_server.server import asgi_app as mcp_asgi
from kgfoundry_common.errors import ConfigurationError
from kgfoundry_common.logging import get_logger

LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager with explicit configuration initialization.
    
    This function runs during FastAPI startup and shutdown, managing the
    configuration lifecycle explicitly rather than relying on lazy loading.
    
    Startup sequence:
    1. Load configuration from environment (fail fast if invalid)
    2. Initialize long-lived clients (vLLM, FAISS manager)
    3. Run readiness checks (verify indexes exist, vLLM reachable)
    4. Optionally pre-load FAISS index (controlled by FAISS_PRELOAD env var)
    
    Shutdown sequence:
    1. Clear readiness state
    2. Explicitly close any open resources
    
    Yields
    ------
    None
        Control to the FastAPI application after successful initialization.
    
    Raises
    ------
    ConfigurationError
        If configuration is invalid or required resources are missing.
        FastAPI will fail to start, preventing broken deployment.
    """
    LOGGER.info("Starting application initialization")
    
    try:
        # Phase 1: Load configuration
        context = ApplicationContext.create()
        app.state.context = context
        
        # Phase 2: Initialize readiness probe
        readiness = ReadinessProbe(context)
        await readiness.initialize()
        app.state.readiness = readiness
        
        # Phase 3: Optional FAISS preloading (controlled by env var)
        if context.settings.index.get("faiss_preload", False):
            LOGGER.info("Pre-loading FAISS index during startup")
            preload_success = await asyncio.to_thread(_preload_faiss_index, context)
            if not preload_success:
                LOGGER.warning(
                    "FAISS index pre-load failed; will lazy-load on first search"
                )
        
        LOGGER.info("Application initialization complete")
        
    except ConfigurationError as exc:
        # Log structured error and re-raise to prevent FastAPI from starting
        LOGGER.error(
            "Application startup failed due to configuration error",
            exc_info=True,
            extra={"error_code": exc.code.value, "context": exc.context}
        )
        raise
    except Exception as exc:
        LOGGER.error("Unexpected error during application startup", exc_info=True)
        raise
    
    try:
        yield
    finally:
        LOGGER.info("Starting application shutdown")
        await readiness.shutdown()
        LOGGER.info("Application shutdown complete")


def _preload_faiss_index(context: ApplicationContext) -> bool:
    """Pre-load FAISS index during startup to avoid first-request latency.
    
    Parameters
    ----------
    context : ApplicationContext
        Application context containing FAISS manager.
    
    Returns
    -------
    bool
        True if index loaded successfully, False otherwise.
    """
    try:
        context.faiss_manager.load_cpu_index()
        LOGGER.info("FAISS CPU index loaded successfully")
        
        # Attempt GPU clone if available
        gpu_enabled = context.faiss_manager.clone_to_gpu()
        if gpu_enabled:
            LOGGER.info("FAISS GPU acceleration enabled")
        else:
            reason = context.faiss_manager.gpu_disabled_reason or "Unknown"
            LOGGER.warning("FAISS GPU acceleration unavailable: %s", reason)
        
        return True
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning("FAISS index pre-load failed: %s", exc)
        return False


app = FastAPI(
    title="CodeIntel MCP Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

# ... (rest of middleware and endpoints remain the same)
```

**Key Changes**:
1. **Explicit Initialization**: Configuration loaded in lifespan, not lazily
2. **Fail-Fast Behavior**: ConfigurationError prevents app startup with clear diagnostics
3. **Optional Pre-loading**: `FAISS_PRELOAD` env var controls eager vs lazy index loading
4. **Structured Logging**: All startup phases logged with context for observability

---

#### 1.3 Refactor Readiness Probe to Use ApplicationContext

**Objective**: Update `ReadinessProbe` to consume `ApplicationContext` instead of re-loading settings, eliminating redundant configuration access.

**Implementation**:

```python
# codeintel_rev/app/readiness.py (new file)
"""Application readiness checks for Kubernetes health probes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from kgfoundry_common.logging import get_logger

if TYPE_CHECKING:
    from codeintel_rev.app.config_context import ApplicationContext

LOGGER = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class CheckResult:
    """Outcome of a single readiness check.
    
    Attributes
    ----------
    healthy : bool
        Whether the resource is ready for use.
    detail : str | None
        Diagnostic detail when unhealthy or degraded.
    """
    
    healthy: bool
    detail: str | None = None
    
    def as_payload(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        payload: dict[str, Any] = {"healthy": self.healthy}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ReadinessProbe:
    """Manages readiness checks across core dependencies.
    
    Parameters
    ----------
    context : ApplicationContext
        Application context containing configuration and clients.
    """
    
    def __init__(self, context: ApplicationContext) -> None:
        self._context = context
        self._lock = asyncio.Lock()
        self._last_checks: dict[str, CheckResult] = {}
    
    async def initialize(self) -> None:
        """Prime readiness state on application startup."""
        await self.refresh()
    
    async def refresh(self) -> Mapping[str, CheckResult]:
        """Recompute readiness checks asynchronously.
        
        Returns
        -------
        Mapping[str, CheckResult]
            Latest readiness results keyed by resource name.
        """
        checks = await asyncio.to_thread(self._run_checks)
        async with self._lock:
            self._last_checks = checks
            return dict(self._last_checks)
    
    async def shutdown(self) -> None:
        """Clear readiness state on shutdown."""
        async with self._lock:
            self._last_checks.clear()
    
    def snapshot(self) -> Mapping[str, CheckResult]:
        """Return the latest readiness snapshot.
        
        Returns
        -------
        Mapping[str, CheckResult]
            Most recent readiness results.
        
        Raises
        ------
        RuntimeError
            If the probe has not been initialized yet.
        """
        if not self._last_checks:
            msg = "Readiness probe not initialized"
            raise RuntimeError(msg)
        return dict(self._last_checks)
    
    def _run_checks(self) -> dict[str, CheckResult]:
        """Execute all readiness checks synchronously."""
        results: dict[str, CheckResult] = {}
        paths = self._context.paths
        
        # Check 1: Repository root exists
        results["repo_root"] = self._check_directory(paths.repo_root)
        
        # Check 2: Data directories exist (create if missing)
        results["data_dir"] = self._check_directory(paths.data_dir, create=True)
        results["vectors_dir"] = self._check_directory(paths.vectors_dir, create=True)
        
        # Check 3: FAISS index exists
        results["faiss_index"] = self._check_file(
            paths.faiss_index,
            description="FAISS index",
            optional=False
        )
        
        # Check 4: DuckDB catalog exists
        results["duckdb_catalog"] = self._check_file(
            paths.duckdb_path,
            description="DuckDB catalog",
            optional=False
        )
        
        # Check 5: SCIP index exists (optional - may be regenerated)
        results["scip_index"] = self._check_file(
            paths.scip_index,
            description="SCIP index",
            optional=True
        )
        
        # Check 6: vLLM service reachable
        results["vllm_service"] = self._check_vllm_connection()
        
        return results
    
    @staticmethod
    def _check_directory(path: Path, *, create: bool = False) -> CheckResult:
        """Ensure a directory exists (creating it if requested)."""
        try:
            if create:
                path.mkdir(parents=True, exist_ok=True)
            exists = path.is_dir()
        except OSError as exc:
            return CheckResult(healthy=False, detail=f"Cannot access directory {path}: {exc}")
        
        if not exists:
            return CheckResult(healthy=False, detail=f"Directory missing: {path}")
        return CheckResult(healthy=True)
    
    @staticmethod
    def _check_file(path: Path, *, description: str, optional: bool = False) -> CheckResult:
        """Validate existence of a filesystem resource."""
        try:
            exists = path.is_file()
        except OSError as exc:
            return CheckResult(
                healthy=False,
                detail=f"Cannot access {description} at {path}: {exc}"
            )
        
        if exists:
            return CheckResult(healthy=True)
        
        detail = f"{description} not found at {path}"
        if optional:
            return CheckResult(healthy=True, detail=detail)
        return CheckResult(healthy=False, detail=detail)
    
    def _check_vllm_connection(self) -> CheckResult:
        """Verify vLLM service is reachable with a lightweight health check."""
        vllm_url = self._context.settings.vllm.base_url
        parsed = urlparse(vllm_url)
        
        # Phase 1: URL validation
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return CheckResult(
                healthy=False,
                detail=f"Invalid vLLM endpoint URL: {vllm_url}"
            )
        
        # Phase 2: HTTP reachability check (non-blocking, short timeout)
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{vllm_url}/health", follow_redirects=True)
                if response.is_success:
                    return CheckResult(healthy=True)
                else:
                    return CheckResult(
                        healthy=False,
                        detail=f"vLLM health check failed: HTTP {response.status_code}"
                    )
        except httpx.HTTPError as exc:
            return CheckResult(
                healthy=False,
                detail=f"vLLM service unreachable: {exc}"
            )
```

**Rationale**:
- **No Redundant Loading**: `ReadinessProbe` receives `ApplicationContext`, doesn't call `load_settings()`
- **Comprehensive Checks**: Validates all critical resources (indexes, directories, services)
- **vLLM Health Check**: Actually verifies vLLM is reachable, not just URL format
- **Fail-Fast Startup**: Missing FAISS index prevents app from starting (configurable)

---

### Phase 2: Adapter Refactoring (Eliminate Redundant `load_settings()`)

#### 2.1 Update Adapter Interfaces to Accept ApplicationContext

**Objective**: Refactor all adapter functions to accept `ApplicationContext` as a parameter instead of calling `load_settings()` internally.

**Implementation Pattern** (applied to all 4 adapters):

```python
# codeintel_rev/mcp_server/adapters/files.py (UPDATED)
"""File and scope management adapter.

Provides file listing, reading, and scope configuration.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel_rev.io.path_utils import (
    PathOutsideRepositoryError,
    resolve_within_repo,
)
from codeintel_rev.mcp_server.schemas import ScopeIn

if TYPE_CHECKING:
    from codeintel_rev.app.config_context import ApplicationContext


def set_scope(context: ApplicationContext, scope: ScopeIn) -> dict:
    """Set query scope for subsequent operations.
    
    Parameters
    ----------
    context : ApplicationContext
        Application context (currently unused, reserved for future multi-repo support).
    scope : ScopeIn
        Scope configuration with repos, branches, paths, languages.
    
    Returns
    -------
    dict
        Effective scope configuration.
    """
    # Future enhancement: Store scope in context for multi-repo support
    return {
        "effective_scope": scope,
        "status": "ok",
    }


def list_paths(
    context: ApplicationContext,
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    """List files in repository.
    
    Parameters
    ----------
    context : ApplicationContext
        Application context containing repo root and settings.
    path : str | None
        Starting path relative to repo root (None = root).
    include_globs : list[str] | None
        Glob patterns to include (e.g., ["*.py"]).
    exclude_globs : list[str] | None
        Glob patterns to exclude (e.g., ["__pycache__", "*.pyc"]).
    max_results : int
        Maximum number of files to return.
    
    Returns
    -------
    dict
        File listing with paths and metadata.
    """
    repo_root = context.paths.repo_root  # NO load_settings() call!
    search_root, error = _resolve_search_root(repo_root, path)
    if search_root is None:
        return {"items": [], "total": 0, "error": error or "Path not found or not a directory"}
    
    # ... rest of function remains unchanged ...
```

**Apply This Pattern To**:
1. `files.py` - ✅ Shown above
2. `text_search.py` - Pass context, use `context.paths.repo_root`
3. `history.py` - Pass context, use `context.paths.repo_root`
4. `semantic.py` - Already uses `get_service_context()`, will be updated in Phase 2.2

---

#### 2.2 Consolidate ServiceContext into ApplicationContext

**Objective**: Eliminate `ServiceContext` class and move its responsibilities into `ApplicationContext`, removing the `@lru_cache` singleton pattern in favor of explicit FastAPI state management.

**Implementation**:

```python
# codeintel_rev/app/config_context.py (EXTENDED ApplicationContext)

from threading import Lock

@dataclass(slots=True)
class ApplicationContext:
    """Application-wide context holding all configuration and long-lived clients."""
    
    settings: Settings
    paths: ResolvedPaths
    vllm_client: VLLMClient
    faiss_manager: FAISSManager
    _faiss_lock: Lock = field(default_factory=Lock, init=False)
    _faiss_loaded: bool = field(default=False, init=False)
    _faiss_gpu_attempted: bool = field(default=False, init=False)
    
    def ensure_faiss_ready(self) -> tuple[bool, list[str], str | None]:
        """Load FAISS index (once) and attempt GPU clone.
        
        This method is thread-safe and idempotent. It will:
        1. Load CPU index on first call
        2. Attempt GPU clone on first call (if not preloaded)
        3. Return cached state on subsequent calls
        
        Returns
        -------
        tuple[bool, list[str], str | None]
            Tuple of (ready, limits, error). ``ready`` is True if FAISS is available,
            ``limits`` contains warnings about GPU availability, and ``error`` is
            None on success or an error message on failure.
        """
        limits: list[str] = []
        
        if not self.paths.faiss_index.exists():
            return False, limits, f"FAISS index not found at {self.paths.faiss_index}"
        
        with self._faiss_lock:
            if not self._faiss_loaded:
                try:
                    self.faiss_manager.load_cpu_index()
                except (FileNotFoundError, RuntimeError) as exc:
                    return False, limits, f"FAISS index load failed: {exc}"
                self._faiss_loaded = True
            
            if self.faiss_manager.gpu_index is None and not self._faiss_gpu_attempted:
                self._faiss_gpu_attempted = True
                try:
                    gpu_enabled = self.faiss_manager.clone_to_gpu()
                except RuntimeError as exc:
                    limits.append(str(exc))
                    gpu_enabled = False
                if not gpu_enabled and self.faiss_manager.gpu_disabled_reason:
                    limits.append(self.faiss_manager.gpu_disabled_reason)
            elif (
                self.faiss_manager.gpu_disabled_reason
                and limits.count(self.faiss_manager.gpu_disabled_reason) == 0
            ):
                limits.append(self.faiss_manager.gpu_disabled_reason)
        
        return True, limits, None
    
    @contextmanager
    def open_catalog(self) -> Iterator[DuckDBCatalog]:
        """Yield a DuckDB catalog context manager."""
        catalog = DuckDBCatalog(self.paths.duckdb_path, self.paths.vectors_dir)
        try:
            catalog.open()
            yield catalog
        finally:
            catalog.close()
```

**Delete Files**:
- `codeintel_rev/mcp_server/service_context.py` (functionality moved to `ApplicationContext`)

---

#### 2.3 Update Semantic Adapter to Use ApplicationContext

**Objective**: Refactor `semantic.py` to receive `ApplicationContext` from FastAPI state instead of calling `get_service_context()`.

**Implementation**:

```python
# codeintel_rev/mcp_server/adapters/semantic.py (UPDATED)

async def semantic_search(context: ApplicationContext, query: str, limit: int = 20) -> AnswerEnvelope:
    """Perform semantic search using embeddings.
    
    Parameters
    ----------
    context : ApplicationContext
        Application context containing FAISS manager, vLLM client, and settings.
    query : str
        Search query text.
    limit : int, optional
        Maximum number of results to return. Defaults to 20.
    
    Returns
    -------
    AnswerEnvelope
        Semantic search response payload.
    """
    return await asyncio.to_thread(_semantic_search_sync, context, query, limit)


def _semantic_search_sync(context: ApplicationContext, query: str, limit: int) -> AnswerEnvelope:
    """Synchronous semantic search implementation."""
    start_time = perf_counter()
    with _observe("semantic_search") as observation:
        # Use context directly - NO get_service_context() call
        requested_limit = limit
        max_results = max(1, context.settings.limits.max_results)
        effective_limit = max(1, min(requested_limit, max_results))
        
        # ... rest of function uses context.ensure_faiss_ready(), context.vllm_client, etc. ...
```

---

### Phase 3: MCP Server Integration (Dependency Injection)

#### 3.1 Update MCP Tool Wrappers to Extract and Pass ApplicationContext

**Objective**: Modify FastMCP tool decorators to extract `ApplicationContext` from FastAPI state and pass it to adapter functions.

**Implementation**:

```python
# codeintel_rev/mcp_server/server.py (UPDATED with context injection)

from fastapi import Request
from fastmcp import FastMCP

from codeintel_rev.app.config_context import ApplicationContext
from codeintel_rev.mcp_server.adapters import files as files_adapter
from codeintel_rev.mcp_server.adapters import history as history_adapter
from codeintel_rev.mcp_server.adapters import semantic as semantic_adapter
from codeintel_rev.mcp_server.adapters import text_search as text_search_adapter
from codeintel_rev.mcp_server.schemas import AnswerEnvelope, ScopeIn

mcp = FastMCP("CodeIntel MCP")


def _get_context(request: Request) -> ApplicationContext:
    """Extract ApplicationContext from FastAPI state.
    
    Parameters
    ----------
    request : Request
        FastAPI request object containing app state.
    
    Returns
    -------
    ApplicationContext
        Application context for the current request.
    
    Raises
    ------
    RuntimeError
        If context is not initialized (should never happen after startup).
    """
    context: ApplicationContext | None = request.app.state.get("context")
    if context is None:
        msg = "ApplicationContext not initialized in app state"
        raise RuntimeError(msg)
    return context


# ==================== Scope & Navigation ====================

@mcp.tool()
def set_scope(request: Request, scope: ScopeIn) -> dict:
    """Set query scope for subsequent operations."""
    context = _get_context(request)
    return files_adapter.set_scope(context, scope)


@mcp.tool()
def list_paths(
    request: Request,
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    """List files in scope."""
    context = _get_context(request)
    return files_adapter.list_paths(
        context,
        path=path,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_results=max_results,
    )


@mcp.tool()
def open_file(
    request: Request,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Read file content."""
    context = _get_context(request)
    return files_adapter.open_file(context, path, start_line, end_line)


# ==================== Search ====================

@mcp.tool()
def search_text(
    request: Request,
    query: str,
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    paths: list[str] | None = None,
    max_results: int = 50,
) -> dict:
    """Fast text search (ripgrep-like)."""
    context = _get_context(request)
    return text_search_adapter.search_text(
        context,
        query,
        regex=regex,
        case_sensitive=case_sensitive,
        paths=paths,
        max_results=max_results,
    )


@mcp.tool()
async def semantic_search(
    request: Request,
    query: str,
    limit: int = 20,
) -> AnswerEnvelope:
    """Semantic code search using embeddings."""
    context = _get_context(request)
    return await semantic_adapter.semantic_search(context, query, limit)


# ==================== Git History ====================

@mcp.tool()
def blame_range(
    request: Request,
    path: str,
    start_line: int,
    end_line: int,
) -> dict:
    """Git blame for line range."""
    context = _get_context(request)
    return history_adapter.blame_range(context, path, start_line, end_line)


@mcp.tool()
def file_history(
    request: Request,
    path: str,
    limit: int = 50,
) -> dict:
    """Get file commit history."""
    context = _get_context(request)
    return history_adapter.file_history(context, path, limit)
```

**Rationale**:
- **Explicit Dependency Injection**: Context extracted from FastAPI state, passed explicitly
- **No Global State**: No `@lru_cache` or module-level singletons
- **Testable**: Easy to inject mock context for unit tests
- **FastMCP Compatible**: Uses FastAPI `Request` object available in tool handlers

---

### Phase 4: Testing and Validation

#### 4.1 Unit Tests for Configuration Context

**Create**: `tests/codeintel_rev/test_config_context.py`

```python
"""Unit tests for ApplicationContext and configuration management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeintel_rev.app.config_context import (
    ApplicationContext,
    resolve_application_paths,
)
from codeintel_rev.config.settings import load_settings
from kgfoundry_common.errors import ConfigurationError


def test_resolve_application_paths_success(tmp_path: Path) -> None:
    """Test successful path resolution with valid repo root."""
    # Arrange
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "data").mkdir()
    
    os.environ["REPO_ROOT"] = str(repo_root)
    settings = load_settings()
    
    # Act
    paths = resolve_application_paths(settings)
    
    # Assert
    assert paths.repo_root == repo_root.resolve()
    assert paths.data_dir == (repo_root / "data").resolve()
    assert paths.vectors_dir.parent == paths.data_dir


def test_resolve_application_paths_missing_repo_root() -> None:
    """Test that missing repo root raises ConfigurationError."""
    # Arrange
    os.environ["REPO_ROOT"] = "/nonexistent/path"
    settings = load_settings()
    
    # Act & Assert
    with pytest.raises(ConfigurationError, match="Repository root does not exist"):
        resolve_application_paths(settings)


def test_application_context_create(tmp_path: Path, monkeypatch) -> None:
    """Test ApplicationContext.create() initializes all clients."""
    # Arrange
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "data").mkdir()
    (repo_root / "data" / "vectors").mkdir()
    (repo_root / "data" / "faiss").mkdir()
    (repo_root / "data" / "faiss" / "code.ivfpq.faiss").touch()
    (repo_root / "data" / "catalog.duckdb").touch()
    
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    monkeypatch.setenv("VLLM_URL", "http://localhost:8001/v1")
    
    # Act
    context = ApplicationContext.create()
    
    # Assert
    assert context.settings is not None
    assert context.paths.repo_root == repo_root.resolve()
    assert context.vllm_client is not None
    assert context.faiss_manager is not None
```

#### 4.2 Integration Tests for FastAPI Lifespan

**Create**: `tests/codeintel_rev/test_app_lifespan.py`

```python
"""Integration tests for FastAPI application lifespan."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeintel_rev.app.main import app


@pytest.fixture
def test_repo(tmp_path: Path, monkeypatch):
    """Set up a minimal test repository environment."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    
    # Create required directory structure
    data_dir = repo_root / "data"
    data_dir.mkdir()
    (data_dir / "vectors").mkdir()
    (data_dir / "faiss").mkdir()
    
    # Create empty index files
    (data_dir / "faiss" / "code.ivfpq.faiss").touch()
    (data_dir / "catalog.duckdb").touch()
    
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    monkeypatch.setenv("VLLM_URL", "http://localhost:8001/v1")
    
    return repo_root


def test_app_startup_with_valid_config(test_repo: Path) -> None:
    """Test that FastAPI app starts successfully with valid configuration."""
    with TestClient(app) as client:
        # App should start without errors
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_app_startup_fails_with_missing_faiss_index(tmp_path: Path, monkeypatch) -> None:
    """Test that FastAPI app fails to start when FAISS index is missing."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "data").mkdir()
    
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    
    # App startup should fail with clear error
    with pytest.raises(RuntimeError, match="FAISS index not found"):
        with TestClient(app):
            pass


def test_readiness_probe_reflects_context_state(test_repo: Path) -> None:
    """Test that /readyz endpoint reflects ApplicationContext state."""
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        
        data = response.json()
        assert data["ready"] is True
        assert "checks" in data
        assert "faiss_index" in data["checks"]
        assert data["checks"]["faiss_index"]["healthy"] is True
```

#### 4.3 Adapter Tests with Mocked Context

**Create**: `tests/codeintel_rev/adapters/test_files_adapter.py`

```python
"""Unit tests for files adapter with ApplicationContext injection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from codeintel_rev.app.config_context import ApplicationContext, ResolvedPaths
from codeintel_rev.config.settings import Settings, load_settings
from codeintel_rev.mcp_server.adapters.files import list_paths, open_file


@pytest.fixture
def mock_context(tmp_path: Path) -> ApplicationContext:
    """Create a mock ApplicationContext for testing."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "test.py").write_text("print('hello')")
    
    paths = ResolvedPaths(
        repo_root=repo_root,
        data_dir=repo_root / "data",
        vectors_dir=repo_root / "data" / "vectors",
        faiss_index=repo_root / "data" / "faiss" / "code.ivfpq.faiss",
        duckdb_path=repo_root / "data" / "catalog.duckdb",
        scip_index=repo_root / "index.scip",
    )
    
    # Create minimal mock context
    context = Mock(spec=ApplicationContext)
    context.paths = paths
    context.settings = load_settings()
    
    return context


def test_list_paths_with_context(mock_context: ApplicationContext) -> None:
    """Test list_paths adapter receives and uses ApplicationContext."""
    # Act
    result = list_paths(mock_context, path=None, max_results=100)
    
    # Assert
    assert "items" in result
    assert len(result["items"]) == 1
    assert result["items"][0]["path"] == "test.py"


def test_open_file_with_context(mock_context: ApplicationContext) -> None:
    """Test open_file adapter receives and uses ApplicationContext."""
    # Act
    result = open_file(mock_context, "test.py")
    
    # Assert
    assert "content" in result
    assert result["content"] == "print('hello')"
    assert "error" not in result
```

---

### Phase 5: Multi-Repository Support (Future-Proofing)

#### 5.1 Design Repository Context Abstraction

**Objective**: Prepare architecture for future multi-repository support without breaking existing single-repo functionality.

**Implementation** (placeholder structure):

```python
# codeintel_rev/app/repository_context.py (NEW FILE - future enhancement)
"""Multi-repository context management (future enhancement).

This module provides the architecture for supporting multiple repositories
in a single CodeIntel MCP instance. Currently, the system is single-repo,
but this design allows incremental migration to multi-repo support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel_rev.app.config_context import ApplicationContext
    from codeintel_rev.config.settings import Settings
    from codeintel_rev.io.faiss_manager import FAISSManager
    from codeintel_rev.io.vllm_client import VLLMClient


@dataclass(slots=True, frozen=True)
class RepositoryContext:
    """Per-repository context holding repo-specific resources.
    
    In multi-repo mode, each repository has its own FAISS index, DuckDB catalog,
    and SCIP index. The vLLM client is shared across repositories.
    
    Attributes
    ----------
    repo_id : str
        Unique repository identifier (e.g., "github.com/owner/repo").
    repo_root : Path
        Repository root directory.
    faiss_manager : FAISSManager
        FAISS index manager for this repository.
    duckdb_path : Path
        DuckDB catalog path for this repository.
    scip_index : Path
        SCIP index path for this repository.
    """
    
    repo_id: str
    repo_root: Path
    faiss_manager: FAISSManager
    duckdb_path: Path
    scip_index: Path


class RepositoryRegistry:
    """Registry for managing multiple repository contexts.
    
    This class will be used in the future to support multi-repository queries.
    For now, it provides a single-repo facade over ApplicationContext.
    """
    
    def __init__(self, app_context: ApplicationContext) -> None:
        """Initialize registry with application context.
        
        Parameters
        ----------
        app_context : ApplicationContext
            Global application context.
        """
        self._app_context = app_context
        self._repositories: dict[str, RepositoryContext] = {}
    
    def register_repository(self, repo_context: RepositoryContext) -> None:
        """Register a repository context.
        
        Parameters
        ----------
        repo_context : RepositoryContext
            Repository context to register.
        """
        self._repositories[repo_context.repo_id] = repo_context
    
    def get_repository(self, repo_id: str) -> RepositoryContext | None:
        """Get repository context by ID.
        
        Parameters
        ----------
        repo_id : str
            Repository identifier.
        
        Returns
        -------
        RepositoryContext | None
            Repository context if registered, otherwise None.
        """
        return self._repositories.get(repo_id)
    
    def get_default_repository(self) -> RepositoryContext:
        """Get the default (single) repository context.
        
        Returns
        -------
        RepositoryContext
            Default repository context derived from ApplicationContext.
        """
        # For now, create a single-repo context from app_context
        return RepositoryContext(
            repo_id="default",
            repo_root=self._app_context.paths.repo_root,
            faiss_manager=self._app_context.faiss_manager,
            duckdb_path=self._app_context.paths.duckdb_path,
            scip_index=self._app_context.paths.scip_index,
        )
```

#### 5.2 Update ScopeIn Handling for Future Multi-Repo Support

**Objective**: Document how `ScopeIn` will be used in multi-repo mode.

**Implementation** (docstring update):

```python
# codeintel_rev/mcp_server/schemas.py (UPDATED ScopeIn docstring)

class ScopeIn(TypedDict, total=False):
    """Query scope parameters for filtering search results.
    
    Defines the scope of a code intelligence query, allowing filtering by
    repository, branch, commit, file patterns, and languages. All fields
    are optional (total=False) - unspecified fields don't filter results.
    
    **Single-Repository Mode** (current):
    The `repos` field is ignored since only one repository is indexed. Other
    filters (languages, include_globs, etc.) are applied to the single repo.
    
    **Multi-Repository Mode** (future):
    The `repos` field will select which repository indexes to query. If empty
    or omitted, all repositories are queried. The RepositoryRegistry will
    dispatch queries to the appropriate per-repo FAISS/DuckDB instances.
    
    Attributes
    ----------
    repos : list[str]
        List of repository identifiers to query (e.g., ["github.com/owner/repo1"]).
        In single-repo mode, this field is currently ignored.
    branches : list[str]
        Branch filter (future enhancement - requires per-branch indexing).
    commit : str
        Commit SHA filter (future enhancement - requires commit-aware indexing).
    include_globs : list[str]
        File path patterns to include (e.g., ["**/*.py"]).
    exclude_globs : list[str]
        File path patterns to exclude (e.g., ["**/test_*.py"]).
    languages : list[str]
        Programming languages to include (e.g., ["python", "typescript"]).
    """
    
    repos: list[str]
    branches: list[str]
    commit: str
    include_globs: list[str]
    exclude_globs: list[str]
    languages: list[str]
```

---

### Phase 6: Configuration Enhancement (Optional Pre-loading)

#### 6.1 Add FAISS Pre-loading Environment Variable

**Objective**: Allow users to control whether FAISS index is loaded during startup (eager) or on first query (lazy).

**Implementation**:

```python
# codeintel_rev/config/settings.py (ADD to IndexConfig)

class IndexConfig(msgspec.Struct, frozen=True):
    """Indexing and search configuration."""
    
    vec_dim: int = 2560
    chunk_budget: int = 2200
    faiss_nlist: int = 8192
    faiss_nprobe: int = 128
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    rrf_k: int = 60
    use_cuvs: bool = True
    faiss_preload: bool = False  # NEW: Control eager vs lazy FAISS loading


def load_settings() -> Settings:
    """Load settings from environment variables."""
    # ... existing code ...
    
    index = IndexConfig(
        vec_dim=int(os.environ.get("VEC_DIM", "2560")),
        chunk_budget=int(os.environ.get("CHUNK_BUDGET", "2200")),
        faiss_nlist=int(os.environ.get("FAISS_NLIST", "8192")),
        faiss_nprobe=int(os.environ.get("FAISS_NPROBE", "128")),
        bm25_k1=float(os.environ.get("BM25_K1", "0.9")),
        bm25_b=float(os.environ.get("BM25_B", "0.4")),
        rrf_k=int(os.environ.get("RRF_K", "60")),
        use_cuvs=os.environ.get("USE_CUVS", "1").lower() in {"1", "true", "yes"},
        faiss_preload=os.environ.get("FAISS_PRELOAD", "0").lower() in {"1", "true", "yes"},  # NEW
    )
    
    # ... rest of function ...
```

**Usage**:
```bash
# Option 1: Lazy loading (default - first query loads index)
export FAISS_PRELOAD=0

# Option 2: Eager loading (startup loads index - faster first query)
export FAISS_PRELOAD=1
```

---

### Phase 7: Documentation and Migration Guide

#### 7.1 Update README with Configuration Best Practices

**Create**: `codeintel_rev/docs/CONFIGURATION.md`

```markdown
# Configuration Management

## Overview

CodeIntel MCP uses a **centralized configuration lifecycle** where all settings are loaded once during FastAPI application startup. This eliminates redundant environment variable parsing and ensures consistent configuration across all components.

## Configuration Loading Sequence

1. **Startup**: FastAPI `lifespan()` function runs
2. **Load Settings**: `ApplicationContext.create()` loads environment variables
3. **Validate Paths**: All paths resolved and validated
4. **Initialize Clients**: vLLM and FAISS manager created
5. **Readiness Checks**: Health probes verify all resources exist
6. **Optional Pre-loading**: FAISS index loaded if `FAISS_PRELOAD=1`

## Key Environment Variables

### Required
- `REPO_ROOT`: Absolute path to repository root (must exist)
- `FAISS_INDEX`: Path to FAISS index file (must exist for startup)
- `DUCKDB_PATH`: Path to DuckDB catalog (must exist for startup)

### Optional
- `FAISS_PRELOAD`: Pre-load FAISS index during startup (`0` = lazy, `1` = eager)
- `VLLM_URL`: vLLM service endpoint (default: `http://127.0.0.1:8001/v1`)

## Configuration Best Practices

### Development
```bash
export REPO_ROOT=/path/to/repo
export FAISS_PRELOAD=0  # Lazy loading for faster startup
```

### Production
```bash
export REPO_ROOT=/var/lib/codeintel/repo
export FAISS_PRELOAD=1  # Eager loading for consistent response times
export FAISS_INDEX=/var/lib/codeintel/faiss/code.ivfpq.faiss
export DUCKDB_PATH=/var/lib/codeintel/catalog.duckdb
```

### Kubernetes
```yaml
env:
  - name: REPO_ROOT
    value: /data/repo
  - name: FAISS_PRELOAD
    value: "1"
readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 10  # Allow time for FAISS pre-loading
```

## Configuration Lifecycle

### Startup Phase
- Configuration loaded from environment
- Fail-fast if required resources missing
- Clear error messages via RFC 9457 Problem Details

### Runtime Phase
- Configuration is immutable (frozen dataclasses)
- No re-loading or hot-reloading
- Changes require application restart

### Shutdown Phase
- Resources explicitly closed
- No dangling connections or locks

## Troubleshooting

### "FAISS index not found"
Ensure `FAISS_INDEX` points to a valid file created by `bin/index_all.py`.

### "Repository root does not exist"
Verify `REPO_ROOT` is an absolute path to an existing directory.

### Slow First Query
Set `FAISS_PRELOAD=1` to load index during startup instead of first query.
```

---

### Implementation Timeline

| Phase | Duration | Dependencies | Risk Level |
|-------|----------|--------------|------------|
| Phase 1: Configuration Lifecycle | 2 days | None | Low |
| Phase 2: Adapter Refactoring | 3 days | Phase 1 complete | Medium |
| Phase 3: MCP Server Integration | 2 days | Phase 2 complete | Medium |
| Phase 4: Testing & Validation | 3 days | Phase 3 complete | Low |
| Phase 5: Multi-Repo Future-Proofing | 1 day | Phase 1 complete | Low |
| Phase 6: Configuration Enhancement | 1 day | Phase 1 complete | Low |
| Phase 7: Documentation | 1 day | All phases complete | Low |

**Total Estimated Duration**: 13 days (with 2-3 days buffer for integration testing)

---

### Success Criteria

1. ✅ **Zero redundant `load_settings()` calls** - All adapters receive `ApplicationContext`
2. ✅ **Explicit dependency injection** - No global singletons or `@lru_cache` patterns
3. ✅ **Fail-fast startup** - Missing FAISS index prevents app from starting
4. ✅ **Comprehensive health checks** - `/readyz` verifies all resources
5. ✅ **Optional FAISS pre-loading** - Controlled via `FAISS_PRELOAD` env var
6. ✅ **100% test coverage** - Unit and integration tests for all new code
7. ✅ **Multi-repo extensibility** - Architecture supports future multi-repo mode
8. ✅ **AGENTS.MD compliance** - RFC 9457 errors, structured logging, zero suppressions

---

### Risk Mitigation

**Risk**: Breaking changes to adapter signatures
**Mitigation**: Incremental refactoring with backward-compatible intermediate phase

**Risk**: FastMCP compatibility issues with Request injection
**Mitigation**: Early prototype and FastMCP documentation review

**Risk**: Performance regression from context passing
**Mitigation**: Benchmark before/after with `pytest-benchmark`

This plan delivers a **production-grade, best-in-class configuration management system** fully aligned with AGENTS.MD principles, eliminates all noted deficiencies, and provides a solid foundation for future multi-repository support.