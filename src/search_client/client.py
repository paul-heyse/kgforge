"""Client utilities."""

from __future__ import annotations

from typing import Any, Final, Protocol

import requests

from kgfoundry_common.navmap_types import NavMap

__all__ = ["KGFoundryClient"]

__navmap__: Final[NavMap] = {
    "title": "search_client.client",
    "synopsis": "Client SDK for interacting with the kgfoundry search API",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["KGFoundryClient"],
        },
    ],
}


class _SupportsResponse(Protocol):
    """Describe SupportsResponse."""

    def raise_for_status(self) -> None:
        """Return raise for status."""

    def json(self) -> dict[str, Any]:
        """Return json.

        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        """


class _SupportsHttp(Protocol):
    """Describe SupportsHttp."""

    def get(self, url: str, *, timeout: float) -> _SupportsResponse:
        """Return get.

        Parameters
        ----------
        url : str
            Description for ``url``.
        timeout : float
            Description for ``timeout``.

        Returns
        -------
        src.search_client.client._SupportsResponse
            Description of return value.
        """

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _SupportsResponse:
        """Return post.

        Parameters
        ----------
        url : str
            Description for ``url``.
        json : Mapping[str, Any]
            Description for ``json``.
        headers : Mapping[str, str]
            Description for ``headers``.
        timeout : float
            Description for ``timeout``.

        Returns
        -------
        src.search_client.client._SupportsResponse
            Description of return value.
        """


# [nav:anchor KGFoundryClient]
class KGFoundryClient:
    """Describe KGFoundryClient."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        http: _SupportsHttp | None = None,
    ) -> None:
        """Return init.

        Parameters
        ----------
        base_url : str | None
            Description for ``base_url``.
        api_key : str | None
            Description for ``api_key``.
        timeout : float | None
            Description for ``timeout``.
        http : _SupportsHttp | None
            Description for ``http``.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: _SupportsHttp = http or requests

    def _headers(self) -> dict[str, str]:
        """Return headers.

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
        """Return healthz.

        Returns
        -------
        Mapping[str, Any]
            Description of return value.
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
        """Return search.

        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int | None
            Description for ``k``.
        filters : Mapping[str, Any] | None
            Description for ``filters``.
        explain : bool | None
            Description for ``explain``.

        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        """
        payload = {"query": query, "k": k, "filters": filters or {}, "explain": explain}
        r = self._http.post(
            f"{self.base_url}/search", json=payload, headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """Return concepts.

        Parameters
        ----------
        q : str
            Description for ``q``.
        limit : int | None
            Description for ``limit``.

        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        """
        r = self._http.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
