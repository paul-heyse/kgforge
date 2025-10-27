"""Client SDK for interacting with the kgfoundry search API."""

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
    """Minimal response protocol shared by httpx and requests objects."""

    def raise_for_status(self) -> None:
        """Raise For Status."""

    def json(self) -> dict[str, Any]:
        """Json."""


class _SupportsHttp(Protocol):
    """Subset of the client interface that the wrapper needs."""

    def get(self, url: str, *, timeout: float) -> _SupportsResponse:
        """Get."""

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _SupportsResponse:
        """Post."""


# [nav:anchor KGFoundryClient]
class KGFoundryClient:
    """Minimal synchronous client for the kgfoundry search API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        http: _SupportsHttp | None = None,
    ) -> None:
        """Configure the client."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: _SupportsHttp = http or requests

    def _headers(self) -> dict[str, str]:
        """Build request headers with an optional bearer token."""
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def healthz(self) -> dict[str, Any]:
        """Call the `/healthz` endpoint and return its JSON body."""
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
        """Execute a search query and return the API response body."""
        payload = {"query": query, "k": k, "filters": filters or {}, "explain": explain}
        r = self._http.post(
            f"{self.base_url}/search", json=payload, headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """Lookup related concepts via `/graph/concepts`."""
        r = self._http.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
