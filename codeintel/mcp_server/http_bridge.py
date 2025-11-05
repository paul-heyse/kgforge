"""FastAPI bridge exposing Tree-sitter tools over HTTP for ChatGPT Actions."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from codeintel.mcp_server import tools

app = FastAPI(
    title="KGFoundry Tree-sitter Bridge",
    version="0.1.0",
    description="HTTP wrapper around the MCP Tree-sitter utilities.",
)


class QueryPayload(BaseModel):
    """Request schema for the ``/ts/query`` endpoint."""

    path: str = Field(..., description="Absolute or repo-relative file path")
    language: str = Field("python", description="Tree-sitter language identifier")
    query: str = Field(..., description="Tree-sitter S-expression to execute")


class DirectoryPayload(BaseModel):
    """Request schema for endpoints that operate on directories."""

    directory: str = Field(..., description="Directory to analyse")


class CallsPayload(BaseModel):
    """Request schema for the ``/ts/calls`` endpoint."""

    directory: str = Field(..., description="Directory to analyse")
    language: str = Field("python", description="Tree-sitter language identifier")
    callee: str | None = Field(
        default=None,
        description="Optional callee name filter",
    )


class ErrorsPayload(BaseModel):
    """Request schema for the ``/ts/errors`` endpoint."""

    path: str = Field(..., description="File to analyse")
    language: str = Field("python", description="Tree-sitter language identifier")


@app.post("/ts/query")
def ts_query(payload: QueryPayload) -> dict[str, object]:
    """Run a Tree-sitter query and return structured captures.

    Parameters
    ----------
    payload : QueryPayload
        Request payload containing file path, language, and query string.

    Returns
    -------
    dict[str, object]
        JSON-serialisable capture payload.

    Raises
    ------
    HTTPException
        If the file cannot be found or the language is unsupported.
    """
    try:
        result = tools.run_ts_query(payload.path, language=payload.language, query=payload.query)
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"captures": result.captures}


@app.post("/ts/symbols")
def ts_symbols(payload: DirectoryPayload) -> dict[str, object]:
    """Summarise Python definitions for the supplied directory.

    Parameters
    ----------
    payload : DirectoryPayload
        Request payload containing directory path to analyze.

    Returns
    -------
    dict[str, object]
        Collection of files and their symbol definitions.

    Raises
    ------
    HTTPException
        If the directory cannot be resolved.
    """
    try:
        files = tools.list_python_symbols(payload.directory)
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"files": files}


@app.post("/ts/calls")
def ts_calls(payload: CallsPayload) -> dict[str, object]:
    """Enumerate call expressions detected within ``directory``.

    Parameters
    ----------
    payload : CallsPayload
        Request payload containing directory path, language, and optional callee filter.

    Returns
    -------
    dict[str, object]
        Call edges grouped by file.

    Raises
    ------
    HTTPException
        If the directory cannot be resolved or the language is unsupported.
    """
    try:
        edges = tools.list_calls(
            payload.directory, language=payload.language, callee=payload.callee
        )
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"edges": edges}


@app.post("/ts/errors")
def ts_errors(payload: ErrorsPayload) -> dict[str, object]:
    """Detect syntax errors using Tree-sitter.

    Parameters
    ----------
    payload : ErrorsPayload
        Request payload containing file path and language identifier.

    Returns
    -------
    dict[str, object]
        Syntax error captures for the requested file.

    Raises
    ------
    HTTPException
        If the file cannot be located.
    """
    try:
        errors = tools.list_errors(payload.path, language=payload.language)
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"errors": errors}
