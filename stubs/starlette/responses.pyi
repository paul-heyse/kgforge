"""Type stubs for Starlette Response."""

from __future__ import annotations

import os
from collections.abc import AsyncIterable, Iterable, Mapping, MutableMapping
from pathlib import Path

__all__ = ["FileResponse", "JSONResponse", "Response", "StreamingResponse"]

class Response:
    """Starlette Response object with precise type annotations."""

    headers: MutableMapping[str, str]

    def __init__(
        self,
        content: bytes | bytearray | memoryview | str | None = None,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: object | None = None,
    ) -> None:
        """Initialize response."""
        ...

class JSONResponse(Response):
    """Starlette JSONResponse with precise type annotations."""

    body: bytes

    def __init__(
        self,
        content: object,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize JSON response.

        Parameters
        ----------
        content : Any
            JSON-serializable content.
        status_code : int, optional
            HTTP status code. Defaults to 200.
        headers : dict[str, str] | None, optional
            Response headers. Defaults to None.
        """
        ...

class FileResponse(Response):
    """Starlette FileResponse for serving files."""

    def __init__(
        self,
        path: str | Path,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        filename: str | None = None,
        stat_result: os.stat_result | None = None,
        method: str | None = None,
        content_disposition_type: str = "attachment",
    ) -> None:
        """Initialize file response.

        Parameters
        ----------
        path : str | Path
            Path to file to serve.
        status_code : int, optional
            HTTP status code. Defaults to 200.
        headers : dict[str, str] | None, optional
            Response headers. Defaults to None.
        media_type : str | None, optional
            Media type for Content-Type header. Defaults to None.
        filename : str | None, optional
            Filename for Content-Disposition header. Defaults to None.
        stat_result : Any | None, optional
            File stat result. Defaults to None.
        method : str | None, optional
            HTTP method. Defaults to None.
        content_disposition_type : str, optional
            Content-Disposition type. Defaults to "attachment".
        """
        ...

class StreamingResponse(Response):
    """Starlette streaming response for iterables or async iterables of bytes."""

    def __init__(
        self,
        content: Iterable[bytes] | AsyncIterable[bytes],
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: object | None = None,
    ) -> None:
        """Initialise streaming response."""
        ...
