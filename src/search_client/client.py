"""Thin HTTP client for interacting with the kgfoundry Search API."""

from __future__ import annotations

from typing import Any, Final, Protocol, cast

import requests

from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "KGFoundryClient",
    "RequestsHttp",
    "SupportsHttp",
    "SupportsResponse",
]

__navmap__: Final[NavMap] = {
    "title": "search_client.client",
    "synopsis": "Lightweight client wrapper around the kgfoundry Search API",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}


class SupportsResponse(Protocol):
    """Protocol describing the minimal HTTP response surface used by the client.

    Notes
    -----
    Implementations are expected to mirror :class:`requests.Response` for the provided
    methods so callers can work with a small shared interface.
    """

    def raise_for_status(self) -> None:
        """Raise an HTTP error if the response indicates failure."""

    def json(self) -> dict[str, Any]:
        """Return the response payload as JSON.

        Returns
        -------
        dict[str, Any]
            Decoded JSON body returned by the HTTP service.
        """
        ...


class SupportsHttp(Protocol):
    """Protocol describing the HTTP verbs required by :class:`KGFoundryClient`.

    Notes
    -----
    Implementations only need to provide ``get`` and ``post`` methods that mirror the
    behaviour of :mod:`requests`.
    """

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Issue an HTTP ``GET`` request.

        Parameters
        ----------
        url : str
            Absolute or relative request URL.
        *args : object
            Positional arguments forwarded to the HTTP implementation.
        **kwargs : object
            Keyword arguments forwarded to the HTTP implementation.

        Returns
        -------
        SupportsResponse
            Response wrapper produced by the HTTP implementation.
        """
        ...

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Issue an HTTP ``POST`` request.

        Parameters
        ----------
        url : str
            Absolute or relative request URL.
        *args : object
            Positional arguments forwarded to the HTTP implementation.
        **kwargs : object
            Keyword arguments forwarded to the HTTP implementation.

        Returns
        -------
        SupportsResponse
            Response wrapper produced by the HTTP implementation.
        """
        ...


class RequestsHttp(SupportsHttp):
    """HTTP adapter that delegates HTTP verbs to :mod:`requests`.

    Notes
    -----
    This thin wrapper exists to make the high-level client easy to test by swapping
    in alternative transports.
    """

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Send a ``GET`` request using :func:`requests.get`.

        Parameters
        ----------
        url : str
            Absolute or relative request URL.
        *args : object
            Positional arguments forwarded to :func:`requests.get`.
        **kwargs : object
            Keyword arguments forwarded to :func:`requests.get`.

        Returns
        -------
        SupportsResponse
            Response returned by :mod:`requests`.
        """
        return cast(
            SupportsResponse,
            requests.get(
                url,
                *cast(tuple[Any, ...], args),
                **cast(dict[str, Any], kwargs),
            ),
        )

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Send a ``POST`` request using :func:`requests.post`.

        Parameters
        ----------
        url : str
            Absolute or relative request URL.
        *args : object
            Positional arguments forwarded to :func:`requests.post`.
        **kwargs : object
            Keyword arguments forwarded to :func:`requests.post`.

        Returns
        -------
        SupportsResponse
            Response returned by :mod:`requests`.
        """
        return cast(
            SupportsResponse,
            requests.post(
                url,
                *cast(tuple[Any, ...], args),
                **cast(dict[str, Any], kwargs),
            ),
        )


_DEFAULT_HTTP: Final[SupportsHttp] = RequestsHttp()


class KGFoundryClient:
    """High-level client for the kgfoundry Search API.

    Parameters
    ----------
    base_url : str, optional
        Base URL for the API, by default ``"http://localhost:8080"``.
    api_key : str | None, optional
        Optional API key used for bearer authentication.
    timeout : float, optional
        Timeout applied to HTTP requests in seconds, by default ``30.0``.
    http : SupportsHttp | None, optional
        Custom HTTP transport. The default transport uses :mod:`requests`.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        http: SupportsHttp | None = None,
    ) -> None:
        """Instantiate the client with connection details.

        Parameters
        ----------
        base_url : str, optional
            Base URL for the API, by default ``"http://localhost:8080"``.
        api_key : str | None, optional
            Optional API key used for bearer authentication.
        timeout : float, optional
            Timeout applied to HTTP requests in seconds, by default ``30.0``.
        http : SupportsHttp | None, optional
            Custom HTTP transport. The default transport uses :mod:`requests`.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: SupportsHttp = http or _DEFAULT_HTTP

    def _headers(self) -> dict[str, str]:
        """Build default headers for authenticated requests.

        Returns
        -------
        dict[str, str]
            Header dictionary including ``Authorization`` when an API key is configured.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def healthz(self) -> dict[str, Any]:
        """Fetch the service health endpoint.

        Returns
        -------
        dict[str, Any]
            JSON payload describing service health.

        Raises
        ------
        requests.HTTPError
            Raised when the API responds with a non-success status code.
        """
        response = self._http.get(f"{self.base_url}/healthz", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Execute a semantic search request.

        Parameters
        ----------
        query : str
            Natural language query string.
        k : int, optional
            Number of results to return, by default ``10``.
        filters : dict[str, Any] | None, optional
            Structured filters applied to the query, by default ``None``.
        explain : bool, optional
            When ``True`` return model explanations, by default ``False``.

        Returns
        -------
        dict[str, Any]
            JSON response containing the ranked search results.

        Raises
        ------
        requests.HTTPError
            Raised when the API responds with a non-success status code.
        """
        payload = {"query": query, "k": k, "filters": filters or {}, "explain": explain}
        response = self._http.post(
            f"{self.base_url}/search",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """Retrieve graph concepts that match the provided query string.

        Parameters
        ----------
        q : str
            Concept search query.
        limit : int, optional
            Maximum number of results to return, by default ``50``.

        Returns
        -------
        dict[str, Any]
            JSON response containing matching concepts.

        Raises
        ------
        requests.HTTPError
            Raised when the API responds with a non-success status code.
        """
        response = self._http.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
