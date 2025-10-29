"""Overview of client.

This module bundles client logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, cast

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


# [nav:anchor SupportsResponse]
class SupportsResponse(Protocol):
    """Describe SupportsResponse."""

    def raise_for_status(self) -> None:
        """Compute raise for status.

        Carry out the raise for status operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Examples
        --------
        >>> from search_client.client import raise_for_status
        >>> raise_for_status()  # doctest: +ELLIPSIS
        """
        ...

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
        ...


# [nav:anchor SupportsHttp]
class SupportsHttp(Protocol):
    """Describe SupportsHttp."""

    def get(
        self,
        url: Any,
        *,
        params: Any | None = ...,
        headers: Any | None = ...,
        cookies: Any | None = ...,
        auth: Any | None = ...,
        follow_redirects: Any | None = ...,
        timeout: Any | None = ...,
        extensions: Any | None = ...,
        **kwargs: Any,
    ) -> Any:
        """Compute get.

        Retrieve a value for ``key`` while falling back to a default when absent. The convenience wrapper mirrors ``dict.get`` so configuration objects remain ergonomic. Use it to express optional access without raising ``KeyError``.

        Parameters
        ----------
        url : str
            Description for ``url``.
        timeout : float
            Description for ``timeout``.

        Returns
        -------
        src.search_client.client.SupportsResponse
            Description of return value.

        Examples
        --------
        >>> from search_client.client import get
        >>> result = get(..., ...)
        >>> result  # doctest: +ELLIPSIS
        """
        ...

    def post(
        self,
        url: Any,
        *,
        content: Any | None = ...,
        data: Any | None = ...,
        files: Any | None = ...,
        json: Any | None = ...,
        headers: Any | None = ...,
        cookies: Any | None = ...,
        auth: Any | None = ...,
        follow_redirects: Any | None = ...,
        timeout: Any | None = ...,
        extensions: Any | None = ...,
        **kwargs: Any,
    ) -> Any:
        """Compute post.

        Carry out the post operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        url : str
            Description for ``url``.
        json : collections.abc.Mapping
            Description for ``json``.
        headers : collections.abc.Mapping
            Description for ``headers``.
        timeout : float
            Description for ``timeout``.

        Returns
        -------
        src.search_client.client.SupportsResponse
            Description of return value.

        Examples
        --------
        >>> from search_client.client import post
        >>> result = post(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        """
        ...


class RequestsHttp(SupportsHttp):
    """Adapter that fulfils :class:`SupportsHttp` using ``requests``."""

    def get(
        self,
        url: Any,
        *,
        params: Any | None = None,
        headers: Any | None = None,
        cookies: Any | None = None,
        auth: Any | None = None,
        follow_redirects: Any | None = None,
        timeout: Any | None = None,
        extensions: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Issue a GET request via :mod:`requests`."""
        if extensions is not None:
            kwargs.setdefault("extensions", extensions)
        if follow_redirects is not None:
            kwargs.setdefault("allow_redirects", follow_redirects)
        response = requests.get(
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            timeout=timeout,
            **kwargs,
        )
        return cast(SupportsResponse, response)

    def post(
        self,
        url: Any,
        *,
        content: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        json: Any | None = None,
        headers: Any | None = None,
        cookies: Any | None = None,
        auth: Any | None = None,
        follow_redirects: Any | None = None,
        timeout: Any | None = None,
        extensions: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Issue a POST request via :mod:`requests`."""
        if content is not None and data is None:
            data = content
        if extensions is not None:
            kwargs.setdefault("extensions", extensions)
        if follow_redirects is not None:
            kwargs.setdefault("allow_redirects", follow_redirects)
        response = requests.post(
            url,
            data=data,
            json=json,
            files=files,
            headers=headers,
            cookies=cookies,
            auth=auth,
            timeout=timeout,
            **kwargs,
        )
        return cast(SupportsResponse, response)


_DEFAULT_HTTP: Final[SupportsHttp] = RequestsHttp()


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
        http: SupportsHttp | None = None,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        base_url : str | None
            Optional parameter default ``'http://localhost:8080'``. Description for ``base_url``.
        api_key : str | None
            Optional parameter default ``None``. Description for ``api_key``.
        timeout : float | None
            Optional parameter default ``30.0``. Description for ``timeout``.
        http : SupportsHttp | None
            Optional parameter default ``None``. Description for ``http``.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: SupportsHttp = http or _DEFAULT_HTTP

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
        return cast(dict[str, Any], r.json())

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
            Description for ``query``.
        k : int | None
            Optional parameter default ``10``. Description for ``k``.
        filters : Mapping[str, Any] | None
            Optional parameter default ``None``. Description for ``filters``.
        explain : bool | None
            Optional parameter default ``False``. Description for ``explain``.

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
            f"{self.base_url}/search",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return cast(dict[str, Any], r.json())

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """Compute concepts.

        Carry out the concepts operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        q : str
            Description for ``q``.
        limit : int | None
            Optional parameter default ``50``. Description for ``limit``.

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
        return cast(dict[str, Any], r.json())
