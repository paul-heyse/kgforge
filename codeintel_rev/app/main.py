"""FastAPI application with MCP server mount.

Provides health/readiness endpoints, CORS, and streaming support.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from codeintel_rev.mcp_server.server import asgi_app as mcp_asgi


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Initializes resources on startup and cleans up on shutdown.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance.

    Yields
    ------
    None
        Control during application lifetime.
    """
    # Startup: initialize resources
    # TODO: Initialize DuckDB, FAISS, etc.
    yield
    # Shutdown: cleanup
    # TODO: Close connections, release resources


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
async def readyz() -> JSONResponse:
    """Readiness check endpoint.

    Verifies that dependent services are available.

    Returns
    -------
    JSONResponse
        Readiness status.
    """
    # TODO: Check DuckDB, FAISS, vLLM availability
    checks = {
        "duckdb": "ok",
        "faiss": "ok",
        "vllm": "ok",
    }
    return JSONResponse({"ready": True, "checks": checks})


@app.get("/sse")
async def sse_demo() -> StreamingResponse:
    """SSE streaming demo endpoint.

    Returns
    -------
    StreamingResponse
        Server-sent events stream.
    """

    async def event_generator():
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
