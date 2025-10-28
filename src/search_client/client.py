"""Overview of client.

This module bundles client logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Any, Final, Protocol

import requests

from kgfoundry_common.navmap_types import NavMap

__all__ = ["KGFoundryClient"]

__navmap__: Final[NavMap] = {
    "title": "search_client.client",
    "synopsis": "Lightweight HTTP client for the kgfoundry Search API",
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
        "KGFoundryClient": {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        },
    },
}


class _SupportsResponse(Protocol):
    """Describe SupportsResponse."""

    def raise_for_status(self) -> None:
        """Compute raise for status.

        Carry out the raise for status operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Examples
        --------
        >>> from search_client.client import raise_for_status
        >>> raise_for_status()  # doctest: +ELLIPSIS
        """

    def json(self) -> dict[str, Any]:
        """Compute json.

        Serialise the instance to JSON text. It respects include and exclude options so APIs can shape their payloads precisely. Pydantic populates this attribute during model construction, so applications should treat it as read-only metadata.

        Returns
        -------
        collections.abc.Mapping
            Description of return value.

        Examples
        --------
        >>> from search_client.client import json
        >>> result = json()
        >>> result  # doctest: +ELLIPSIS
        """


class _SupportsHttp(Protocol):
    """Describe SupportsHttp."""

    def get(self, url: str, *, timeout: float) -> _SupportsResponse:
        """Compute get.

        Carry out the get operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        url : str
        url : str
            Description for ``url``.
        timeout : float
        timeout : float
            Description for ``timeout``.

        Returns
        -------
        src.search_client.client._SupportsResponse
            Description of return value.

        Examples
        --------
        >>> from search_client.client import get
        >>> result = get(..., ...)
        >>> result  # doctest: +ELLIPSIS
        """

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _SupportsResponse:
        """Compute post.

        Carry out the post operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        url : str
        url : str
            Description for ``url``.
        json : collections.abc.Mapping
        json : collections.abc.Mapping
            Description for ``json``.
        headers : collections.abc.Mapping
        headers : collections.abc.Mapping
            Description for ``headers``.
        timeout : float
        timeout : float
            Description for ``timeout``.

        Returns
        -------
        src.search_client.client._SupportsResponse
            Description of return value.

        Examples
        --------
        >>> from search_client.client import post
        >>> result = post(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        """


# [nav:anchor KGFoundryClient]
class KGFoundryClient:
    """Model the KGFoundryClient.

    Represent the kgfoundryclient data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        http: _SupportsHttp | None = None,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        base_url : str | None
        base_url : str | None, optional, default='http://localhost:8080'
            Description for ``base_url``.
        api_key : str | None
        api_key : str | None, optional, default=None
            Description for ``api_key``.
        timeout : float | None
        timeout : float | None, optional, default=30.0
            Description for ``timeout``.
        http : _SupportsHttp | None
        http : _SupportsHttp | None, optional, default=None
            Description for ``http``.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: _SupportsHttp = http or requests

    def _headers(self) -> dict[str, str]:
        """Compute headers.

        Carry out the headers operation.

        Returns
        -------
        Mapping[str, str]
            Description of return value.
        """
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def healthz(self) -> dict[str, Any]:
        """Compute healthz.

        Carry out the healthz operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Returns
        -------
        collections.abc.Mapping
            Description of return value.

        Examples
        --------
        >>> from search_client.client import healthz
        >>> result = healthz()
        >>> result  # doctest: +ELLIPSIS
        """
        r = self._http.get(f"{self.base_url}/healthz", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Compute search.

        Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        query : str
        query : str
            Description for ``query``.
        k : int | None
        k : int | None, optional, default=10
            Description for ``k``.
        filters : Mapping[str, Any] | None
        filters : Mapping[str, Any] | None, optional, default=None
            Description for ``filters``.
        explain : bool | None
        explain : bool | None, optional, default=False
            Description for ``explain``.

        Returns
        -------
        collections.abc.Mapping
            Description of return value.

        Examples
        --------
        >>> from search_client.client import search
        >>> result = search(...)
        >>> result  # doctest: +ELLIPSIS
        """
        payload = {"query": query, "k": k, "filters": filters or {}, "explain": explain}
        r = self._http.post(
            f"{self.base_url}/search", json=payload, headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """Compute concepts.

        Carry out the concepts operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        q : str
        q : str
            Description for ``q``.
        limit : int | None
        limit : int | None, optional, default=50
            Description for ``limit``.

        Returns
        -------
        collections.abc.Mapping
            Description of return value.

        Examples
        --------
        >>> from search_client.client import concepts
        >>> result = concepts(...)
        >>> result  # doctest: +ELLIPSIS
        """
        r = self._http.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
