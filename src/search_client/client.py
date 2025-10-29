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
    """Describe SupportsResponse.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.






    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def raise_for_status(self) -> None:
        """Describe raise for status.

        <!-- auto:docstring-builder v1 -->

        Python's object protocol for this class. Use it to integrate with built-in operators,
        protocols, or runtime behaviours that expect instances to participate in the language's data
        model.
        """

    def json(self) -> dict[str, Any]:
        """Describe json.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Returns
        -------
        dict[str, Any]
            Describe return value.
        """
        ...


class SupportsHttp(Protocol):
    """Describe SupportsHttp.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.






    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Describe get.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

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
            Describe return value.
        """
        ...

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Describe post.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

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
            Describe return value.
        """
        ...


class RequestsHttp(SupportsHttp):
    """Describe RequestsHttp.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Describe get.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

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
            Describe return value.
        """
        return requests.get(
            url,
            *cast(tuple[Any, ...], args),
            **cast(dict[str, Any], kwargs),
        )

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        """Describe post.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

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
            Describe return value.
        """
        return requests.post(
            url,
            *cast(tuple[Any, ...], args),
            **cast(dict[str, Any], kwargs),
        )


_DEFAULT_HTTP: Final[SupportsHttp] = RequestsHttp()


class KGFoundryClient:
    """Describe KGFoundryClient.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

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
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

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
        """Describe  headers.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Returns
        -------
        dict[str, str]
            Describe return value.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def healthz(self) -> dict[str, Any]:
        """Describe healthz.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Returns
        -------
        dict[str, Any]
            Describe return value.
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
        """Describe search.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int, optional
            Describe ``k``.
            Defaults to ``10``.
        filters : dict[str, Any] | None, optional
            Describe ``filters``.
            Defaults to ``None``.
        explain : bool, optional
            Describe ``explain``.
            Defaults to ``False``.






        Returns
        -------
        dict[str, Any]
            Describe return value.
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
        """Describe concepts.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        q : str
            Describe ``q``.
        limit : int, optional
            Describe ``limit``.
            Defaults to ``50``.






        Returns
        -------
        dict[str, Any]
            Describe return value.
        """
        response = self._http.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
