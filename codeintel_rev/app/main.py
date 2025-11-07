"""FastAPI application with MCP server mount.

Provides health/readiness endpoints, CORS, and streaming support.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from codeintel_rev.config.settings import Settings, load_settings
from codeintel_rev.mcp_server.server import asgi_app as mcp_asgi


@dataclass(slots=True, frozen=True)
class CheckResult:
    """Outcome of a readiness check."""

    healthy: bool
    detail: str | None = None

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload.

        Returns
        -------
        dict[str, Any]
            JSON-compatible representation of the check outcome.
        """
        payload: dict[str, Any] = {"healthy": self.healthy}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ReadinessProbe:
    """Manage readiness checks across core dependencies."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
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
        repo_root = Path(self._settings.paths.repo_root).expanduser().resolve()
        results: dict[str, CheckResult] = {}
        results["repo_root"] = self._check_directory(repo_root)

        def resolve(path_str: str) -> Path:
            path = Path(path_str)
            if path.is_absolute():
                return path
            return (repo_root / path).resolve()

        data_dir = resolve(self._settings.paths.data_dir)
        vectors_dir = resolve(self._settings.paths.vectors_dir)
        faiss_index = resolve(self._settings.paths.faiss_index)
        duckdb_path = resolve(self._settings.paths.duckdb_path)
        scip_index = resolve(self._settings.paths.scip_index)

        results["data_dir"] = self._check_directory(data_dir, create=True)
        results["vectors_dir"] = self._check_directory(vectors_dir, create=True)
        results["faiss_index"] = self._check_file(faiss_index, description="FAISS index")
        results["duckdb_catalog"] = self._check_file(duckdb_path, description="DuckDB catalog")
        results["scip_index"] = self._check_file(
            scip_index, description="SCIP index", optional=True
        )
        results["vllm_url"] = self._check_vllm()

        return results

    @staticmethod
    def _check_directory(path: Path, *, create: bool = False) -> CheckResult:
        """Ensure a directory exists (creating it if requested).

        Parameters
        ----------
        path : Path
            Directory path to validate.
        create : bool, optional
            When True, create the directory hierarchy if it is missing.

        Returns
        -------
        CheckResult
            Healthy status and diagnostic detail when unavailable.
        """
        try:
            if create:
                path.mkdir(parents=True, exist_ok=True)
            exists = path.is_dir()
        except OSError as exc:  # pragma: no cover - defensive path handling
            return CheckResult(healthy=False, detail=f"Cannot access directory {path}: {exc}")

        if not exists:
            return CheckResult(healthy=False, detail=f"Directory missing: {path}")
        return CheckResult(healthy=True)

    @staticmethod
    def _check_file(path: Path, *, description: str, optional: bool = False) -> CheckResult:
        """Validate existence of a filesystem resource.

        Parameters
        ----------
        path : Path
            Target filesystem path.
        description : str
            Human-readable resource description for diagnostics.
        optional : bool, optional
            When True, missing resources mark the check as healthy but include detail.

        Returns
        -------
        CheckResult
            Healthy status and contextual detail.
        """
        try:
            exists = path.is_file()
        except OSError as exc:  # pragma: no cover - defensive path handling
            return CheckResult(
                healthy=False, detail=f"Cannot access {description} at {path}: {exc}"
            )

        if exists:
            return CheckResult(healthy=True)

        detail = f"{description} not found at {path}"
        if optional:
            return CheckResult(healthy=True, detail=detail)
        return CheckResult(healthy=False, detail=detail)

    def _check_vllm(self) -> CheckResult:
        """Validate the vLLM endpoint configuration.

        Returns
        -------
        CheckResult
            Healthy status reflecting URL validity.
        """
        parsed = urlparse(self._settings.vllm.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return CheckResult(
                healthy=False,
                detail=f"Invalid vLLM endpoint URL: {self._settings.vllm.base_url}",
            )
        return CheckResult(healthy=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings = load_settings()
    readiness = ReadinessProbe(settings=settings)

    await readiness.initialize()
    app.state.settings = settings
    app.state.readiness = readiness

    try:
        yield
    finally:
        await readiness.shutdown()


app = FastAPI(
    title="CodeIntel MCP Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chat.openai.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_nginx_buffering(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Disable NGINX buffering for streaming responses.

    This middleware sets the X-Accel-Buffering header to "no" on all responses,
    which instructs NGINX to disable buffering and enable streaming. This is
    critical for Server-Sent Events (SSE) and other streaming protocols where
    backpressure and real-time delivery are important.

    The middleware runs after the request handler executes, modifying the
    response headers before it's sent to the client. This ensures that NGINX
    will stream the response directly to the client rather than buffering it
    in memory, which is essential for long-running streams and prevents memory
    issues with large responses.

    Parameters
    ----------
    request : Request
        Incoming HTTP request from FastAPI/Starlette.
    call_next : Callable[[Request], Awaitable[Response]]
        Next middleware or route handler in the chain. This is an async callable
        that takes a Request and returns an awaitable Response.

    Returns
    -------
    Response
        Response object with X-Accel-Buffering header set to "no". The response
        is the same as returned by call_next, but with the streaming header added.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Accel-Buffering", "no")
    return response


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Health check endpoint (network-only).

    Returns
    -------
    JSONResponse
        Health status.
    """
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness check endpoint.

    Verifies that dependent services are available.

    Returns
    -------
    JSONResponse
        Readiness status.
    """
    readiness: ReadinessProbe = request.app.state.readiness
    results = await readiness.refresh()
    payload = {name: result.as_payload() for name, result in results.items()}
    overall_ready = all(result.healthy for result in results.values())
    return JSONResponse({"ready": overall_ready, "checks": payload})


@app.get("/sse")
async def sse_demo() -> StreamingResponse:
    """SSE streaming demo endpoint.

    Returns
    -------
    StreamingResponse
        Server-sent events stream.
    """

    async def event_generator() -> AsyncIterator[bytes]:
        yield b"event: ready\ndata: {}\n\n"
        for i in range(5):
            yield f"data: {i}\n\n".encode()
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# Mount MCP server at /mcp
app.mount("/mcp", mcp_asgi)

__all__ = ["app"]
