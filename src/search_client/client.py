"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
search_client.client
"""


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
    """
    Represent SupportsResponse.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    raise_for_status()
        Method description.
    json()
        Method description.
    
    Examples
    --------
    >>> from search_client.client import _SupportsResponse
    >>> result = _SupportsResponse()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_client.client
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def raise_for_status(self) -> None:
        """
        Return raise for status.
        
        Examples
        --------
        >>> from search_client.client import raise_for_status
        >>> raise_for_status()  # doctest: +ELLIPSIS
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        

    def json(self) -> dict[str, Any]:
        """
        Return json.
        
        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import json
        >>> result = json()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        


class _SupportsHttp(Protocol):
    """
    Represent SupportsHttp.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    get()
        Method description.
    post()
        Method description.
    
    Examples
    --------
    >>> from search_client.client import _SupportsHttp
    >>> result = _SupportsHttp()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_client.client
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def get(self, url: str, *, timeout: float) -> _SupportsResponse:
        """
        Return get.
        
        Parameters
        ----------
        url : str
            Description for ``url``.
        timeout : float
            Description for ``timeout``.
        
        Returns
        -------
        _SupportsResponse
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import get
        >>> result = get(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _SupportsResponse:
        """
        Return post.
        
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
        _SupportsResponse
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import post
        >>> result = post(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        


# [nav:anchor KGFoundryClient]
class KGFoundryClient:
    """
    Represent KGFoundryClient.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    __init__()
        Method description.
    _headers()
        Method description.
    healthz()
        Method description.
    search()
        Method description.
    concepts()
        Method description.
    
    Examples
    --------
    >>> from search_client.client import KGFoundryClient
    >>> result = KGFoundryClient()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_client.client
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        http: _SupportsHttp | None = None,
    ) -> None:
        """
        Return init.
        
        Parameters
        ----------
        base_url : str, optional
            Description for ``base_url``.
        api_key : str | None, optional
            Description for ``api_key``.
        timeout : float, optional
            Description for ``timeout``.
        http : _SupportsHttp | None, optional
            Description for ``http``.
        
        Examples
        --------
        >>> from search_client.client import __init__
        >>> __init__(..., ..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._http: _SupportsHttp = http or requests

    def _headers(self) -> dict[str, str]:
        """
        Return headers.
        
        Returns
        -------
        Mapping[str, str]
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import _headers
        >>> result = _headers()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def healthz(self) -> dict[str, Any]:
        """
        Return healthz.
        
        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import healthz
        >>> result = healthz()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
        """
        Return search.
        
        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int, optional
            Description for ``k``.
        filters : Mapping[str, Any] | None, optional
            Description for ``filters``.
        explain : bool, optional
            Description for ``explain``.
        
        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import search
        >>> result = search(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        payload = {"query": query, "k": k, "filters": filters or {}, "explain": explain}
        r = self._http.post(
            f"{self.base_url}/search", json=payload, headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def concepts(self, q: str, limit: int = 50) -> dict[str, Any]:
        """
        Return concepts.
        
        Parameters
        ----------
        q : str
            Description for ``q``.
        limit : int, optional
            Description for ``limit``.
        
        Returns
        -------
        Mapping[str, Any]
            Description of return value.
        
        Examples
        --------
        >>> from search_client.client import concepts
        >>> result = concepts(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_client.client
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        r = self._http.post(
            f"{self.base_url}/graph/concepts",
            json={"q": q, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
