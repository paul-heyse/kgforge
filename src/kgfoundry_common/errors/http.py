"""HTTP adapters for Problem Details exception handling.

This module provides FastAPI exception handlers and helpers for converting
KgFoundryError exceptions to RFC 9457 Problem Details responses.

Examples
--------
>>> from fastapi import FastAPI
>>> from kgfoundry_common.errors.http import register_problem_details_handler
>>> app = FastAPI()
>>> register_problem_details_handler(app)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from kgfoundry_common.errors import KgFoundryError

__all__ = ["problem_details_response", "register_problem_details_handler"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def problem_details_response(
    error: KgFoundryError,
    request: Request | None = None,
) -> JSONResponse:
    """Convert KgFoundryError to RFC 9457 Problem Details JSONResponse.

    Parameters
    ----------
    error : KgFoundryError
        Exception to convert.
    request : Request | None, optional
        FastAPI request object for instance URI. Defaults to None.

    Returns
    -------
    JSONResponse
        Problem Details JSON response with appropriate status code.

    Examples
    --------
    >>> from kgfoundry_common.errors import DownloadError
    >>> from kgfoundry_common.errors.http import problem_details_response
    >>> error = DownloadError("Download failed")
    >>> response = problem_details_response(error)
    >>> assert response.status_code == 503
    >>> assert "type" in response.body.decode()
    """
    instance = None
    if request:
        instance = str(request.url.path)
        if request.url.query:
            instance += f"?{request.url.query}"

    details = error.to_problem_details(instance=instance)

    # Log the error at appropriate level
    logger.log(error.log_level, "Error: %s", error.message, exc_info=error.__cause__)

    return JSONResponse(
        status_code=error.http_status,
        content=details,
        headers={
            "Content-Type": "application/problem+json",
        },
    )


def register_problem_details_handler(app: FastAPI) -> None:
    """Register FastAPI exception handler for KgFoundryError.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance.

    Examples
    --------
    >>> from fastapi import FastAPI
    >>> from kgfoundry_common.errors.http import register_problem_details_handler
    >>> app = FastAPI()
    >>> register_problem_details_handler(app)
    """

    @app.exception_handler(KgFoundryError)
    async def kgfoundry_error_handler(request: Request, exc: KgFoundryError) -> JSONResponse:
        """Handle KgFoundryError exceptions with Problem Details."""
        return problem_details_response(exc, request=request)
