"""Thin HTTP client for interacting with the kgfoundry Search API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import requests

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        object
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

    def get(
        self,
        url: str,
        *,
        timeout: float | tuple[float | None, float | None] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> SupportsResponse:
        """Issue an HTTP ``GET`` request.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        timeout : float | tuple[float | None, float | None] | None, optional
            Request timeout (seconds or connect/read tuple). Defaults to ``None``.
        headers : Mapping[str, str] | None, optional
            HTTP headers to include with the request. Defaults to ``None``.

        Returns
        -------
        SupportsResponse
            Response wrapper produced by the HTTP implementation.
        """
        ...

    def post(
        self,
        url: str,
        *,
        json: JsonValue | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | tuple[float | None, float | None] | None = None,
    ) -> SupportsResponse:
        """Issue an HTTP ``POST`` request.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        json : JsonValue | None, optional
            JSON payload to include in the request body. Defaults to ``None``.
        headers : Mapping[str, str] | None, optional
            HTTP headers to include with the request. Defaults to ``None``.
        timeout : float | tuple[float | None, float | None] | None, optional
            Request timeout (seconds or connect/read tuple). Defaults to ``None``.

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

    def __init__(self, session: requests.Session | None = None) -> None:
        """Initialize search client.

        Parameters
        ----------
        session : requests.Session | None, optional
            HTTP session to use. Creates new session if None.
            Defaults to None.
        """
        self._session = session or requests.Session()

    def get(
        self,
        url: str,
        *,
        timeout: float | tuple[float | None, float | None] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> SupportsResponse:
        """Send a ``GET`` request using :func:`requests.get`.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        timeout : float | tuple[float | None, float | None] | None, optional
            Request timeout in seconds or connect/read tuple. Defaults to ``None``.
        headers : Mapping[str, str] | None, optional
            Headers to include with the request. Defaults to ``None``.

        Returns
        -------
        SupportsResponse
            Response returned by :mod:`requests`.
        """
        response = self._session.get(url, timeout=timeout, headers=headers)
        return cast("SupportsResponse", response)

    def post(
        self,
        url: str,
        *,
        json: JsonValue | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | tuple[float | None, float | None] | None = None,
    ) -> SupportsResponse:
        """Send a ``POST`` request using :func:`requests.post`.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        url : str
            Describe ``url``.
        json : JsonValue | None, optional
            JSON body to send with the request. Defaults to ``None``.
        headers : Mapping[str, str] | None, optional
            Headers to include with the request. Defaults to ``None``.
        timeout : float | tuple[float | None, float | None] | None, optional
            Request timeout in seconds or connect/read tuple. Defaults to ``None``.

        Returns
        -------
        SupportsResponse
            Response returned by :mod:`requests`.
        """
        response = self._session.post(url, json=json, headers=headers, timeout=timeout)
        return cast("SupportsResponse", response)


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
        api_key : str | NoneType, optional
            Describe ``api_key``.
            Defaults to ``None``.
        timeout : float, optional
            Describe ``timeout``.
            Defaults to ``30.0``.
        http : SupportsHttp | NoneType, optional
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
        object
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
        *,
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
        filters : dict[str, object] | NoneType, optional
            Describe ``filters``.
            Defaults to ``None``.
        explain : bool, optional
            Describe ``explain``.
            Defaults to ``False``.

        Returns
        -------
        object
            JSON response containing the ranked search results.

        Raises
        ------
        requests.HTTPError
        Raised when the API responds with a non-success status code.
        """
        filters_payload: dict[str, JsonValue] = filters.copy() if filters is not None else {}
        payload: dict[str, JsonValue] = {
            "query": query,
            "k": k,
            "filters": filters_payload,
            "explain": explain,
        }
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
        object
            JSON response containing matching concepts.

        Raises
        ------
        requests.HTTPError
        Raised when the API responds with a non-success status code.
        """
        body: dict[str, JsonValue] = {"q": q, "limit": limit}
        response = self._http.post(
            f"{self.base_url}/graph/concepts",
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
