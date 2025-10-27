"""Module for search_client.client.

NavMap:
- KGForgeClient: Kgforgeclient.
"""

from __future__ import annotations

from typing import Any

import requests


class KGForgeClient:
    """Kgforgeclient."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Init.

        Args:
            base_url (str): TODO.
            api_key (Optional[str]): TODO.
            timeout (float): TODO.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        """Headers.

        Returns:
            Dict[str, str]: TODO.
        """
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def healthz(self) -> dict[str, Any]:
        """Healthz.

        Returns:
            Dict[str, Any]: TODO.
        """
        r = requests.get(f"{self.base_url}/healthz", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Search.

        Args:
            query (str): TODO.
            k (int): TODO.
            filters (Optional[Dict[str, Any]]): TODO.
            explain (bool): TODO.

        Returns:
            Dict[str, Any]: TODO.
        """
        payload = {"query": query, "k": k, "filters": filters or {}, "explain": explain}
        r = requests.post(
            f"{self.base_url}/search", json=payload, headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """Concepts.

        Args:
            q (str): TODO.
            limit (int): TODO.

        Returns:
            Dict[str, Any]: TODO.
        """
        r = requests.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
