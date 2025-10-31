"""Thin HTTP client for interacting with the kgfoundry Search API."""

from __future__ import annotations

from typing import Final, Protocol, cast

import requests

from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import JsonValue

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

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.


    Notes
    -----
    Implementations are expected to mirror :class:`requests.Response` for the provided
    methods so callers can work with a small shared interface.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def raise_for_status(self) -> None:
        """Raise an HTTP error if the response indicates failure.

        <!-- auto:docstring-builder v1 -->
        """

    def json(self) -> JsonValue:
        """Return the response payload as JSON.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        JsonValue
            Decoded JSON body returned by the HTTP service. Can be a dict, list, str, int, float, bool, or None.
        """
        ...


class SupportsHttp(Protocol):
    """Protocol describing the HTTP verbs required by :class:`KGFoundryClient`.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.


    Notes
    -----
    Implementations only need to provide ``get`` and ``post`` methods that mirror the
    behaviour of :mod:`requests`.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Issue an HTTP ``GET`` request.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        *args : object
            Describe ``args``.
        **kwargs : object
            Describe ``kwargs``.


        Returns
        -------
        SupportsResponse
            Response wrapper produced by the HTTP implementation.
        """
        ...

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Issue an HTTP ``POST`` request.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        *args : object
            Describe ``args``.
        **kwargs : object
            Describe ``kwargs``.


        Returns
        -------
        SupportsResponse
            Response wrapper produced by the HTTP implementation.
        """
        ...


class RequestsHttp(SupportsHttp):
    """HTTP adapter that delegates HTTP verbs to :mod:`requests`.

    <!-- auto:docstring-builder v1 -->

    Notes
    -----
    This thin wrapper exists to make the high-level client easy to test by swapping
    in alternative transports.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Send a ``GET`` request using :func:`requests.get`.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        *args : object
            Describe ``args``.
        **kwargs : object
            Describe ``kwargs``.


        Returns
        -------
        SupportsResponse
            Response returned by :mod:`requests`.
        """
        # Cast args/kwargs for requests API compatibility
        # requests.get accepts *args and **kwargs with Any types
        # mypy cannot infer the complex overloads, so we use type: ignore
        return cast(SupportsResponse, requests.get(url, *args, **kwargs))  # type: ignore[arg-type]

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Send a ``POST`` request using :func:`requests.post`.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        *args : object
            Describe ``args``.
        **kwargs : object
            Describe ``kwargs``.


        Returns
        -------
        SupportsResponse
            Response returned by :mod:`requests`.
        """
        # Cast args/kwargs for requests API compatibility
        # requests.post accepts *args and **kwargs with Any types
        # mypy cannot infer the complex overloads, so we use type: ignore
        return cast(SupportsResponse, requests.post(url, *args, **kwargs))  # type: ignore[arg-type]


_DEFAULT_HTTP: Final[SupportsHttp] = RequestsHttp()


class KGFoundryClient:
    """High-level client for the kgfoundry Search API.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    base_url : str, optional
        Describe ``base_url``.
        Defaults to ``'http://localhost:8080'``.
    api_key : str | None, optional
        Describe ``api_key``.
        Defaults to ``None``.
    timeout : float, optional
        Describe ``timeout``.
        Defaults to ``30.0``.
    http : SupportsHttp | None, optional
        Describe ``http``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        http: SupportsHttp | None = None,
    ) -> None:
        """Instantiate the client with connection details.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        base_url : str, optional
            Describe ``base_url``.
            Defaults to ``'http://localhost:8080'``.
        api_key : str | None, optional
            Describe ``api_key``.
            Defaults to ``None``.
        timeout : float, optional
            Describe ``timeout``.
            Defaults to ``30.0``.
        http : SupportsHttp | None, optional
            Describe ``http``.
            Defaults to ``None``.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: SupportsHttp = http or _DEFAULT_HTTP

    def _headers(self) -> dict[str, str]:
        """Build default headers for authenticated requests.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        dict[str, str]
            Header dictionary including ``Authorization`` when an API key is configured.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def healthz(self) -> JsonValue:
        """Fetch the service health endpoint.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        JsonValue
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
        filters: dict[str, JsonValue] | None = None,
        explain: bool = False,
    ) -> JsonValue:
        """Execute a semantic search request.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int, optional
            Describe ``k``.
            Defaults to ``10``.
        filters : dict[str, JsonValue] | None, optional
            Describe ``filters``.
            Defaults to ``None``.
        explain : bool, optional
            Describe ``explain``.
            Defaults to ``False``.


        Returns
        -------
        JsonValue
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

    def concepts(self, q: str, limit: int = 50) -> JsonValue:
        """Retrieve graph concepts that match the provided query string.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        q : str
            Describe ``q``.
        limit : int, optional
            Describe ``limit``.
            Defaults to ``50``.


        Returns
        -------
        JsonValue
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
