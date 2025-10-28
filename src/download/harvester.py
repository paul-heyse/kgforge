"""Provide utilities for module.

Auto-generated API documentation for the ``src.download.harvester`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
src.download.harvester
"""


from __future__ import annotations

import os
import time
from typing import Any, Final

import requests
from kgfoundry.kgfoundry_common.models import Doc

from kgfoundry_common.errors import DownloadError, UnsupportedMIMEError
from kgfoundry_common.navmap_types import NavMap

__all__ = ["OpenAccessHarvester"]

__navmap__: Final[NavMap] = {
    "title": "download.harvester",
    "synopsis": "Utilities for harvesting open-access PDFs from OpenAlex.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["OpenAccessHarvester"],
        },
    ],
}

HTTP_OK = 200


# [nav:anchor OpenAccessHarvester]
class OpenAccessHarvester:
    """Represent OpenAccessHarvester.

    Attributes
    ----------
    None
        No public attributes documented.

    Methods
    -------
    __init__()
        Method description.
    search()
        Method description.
    resolve_pdf()
        Method description.
    download_pdf()
        Method description.
    run()
        Method description.

    Examples
    --------
    >>> from download.harvester import OpenAccessHarvester
    >>> result = OpenAccessHarvester()
    >>> result  # doctest: +ELLIPSIS

    See Also
    --------
    download.harvester

    Notes
    -----
    Document class invariants and lifecycle details here.
    """

    def __init__(  # noqa: PLR0913 - parameters mirror external API options
        self,
        user_agent: str,
        contact_email: str,
        openalex_base: str = "https://api.openalex.org",
        unpaywall_base: str = "https://api.unpaywall.org",
        pdf_host_base: str | None = None,
        out_dir: str = "/data/pdfs",
    ) -> None:
        """Return init.

        Parameters
        ----------
        user_agent : str
            Description for ``user_agent``.
        contact_email : str
            Description for ``contact_email``.
        openalex_base : str, optional
            Description for ``openalex_base``.
        unpaywall_base : str, optional
            Description for ``unpaywall_base``.
        pdf_host_base : str | None, optional
            Description for ``pdf_host_base``.
        out_dir : str, optional
            Description for ``out_dir``.

        Examples
        --------
        >>> from download.harvester import __init__
        >>> __init__(..., ..., ..., ..., ..., ...)  # doctest: +ELLIPSIS

        See Also
        --------
        download.harvester

        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        self.ua = user_agent
        self.email = contact_email
        self.openalex = openalex_base.rstrip("/")
        self.unpaywall = unpaywall_base.rstrip("/")
        self.pdf_host = (pdf_host_base or "").rstrip("/")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{self.ua} ({self.email})"})

    def search(self, topic: str, years: str, max_works: int) -> list[dict[str, Any]]:
        """Return search.

        Parameters
        ----------
        topic : str
            Description for ``topic``.
        years : str
            Description for ``years``.
        max_works : int
            Description for ``max_works``.

        Returns
        -------
        List[dict[str, Any]]
            Description of return value.

        Examples
        --------
        >>> from download.harvester import search
        >>> result = search(..., ..., ...)
        >>> result  # doctest: +ELLIPSIS

        See Also
        --------
        download.harvester

        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        url = f"{self.openalex}/works"
        params: dict[str, str | int] = {
            "topic": topic,
            "per_page": min(200, max_works),
            "cursor": "*",
        }
        if years:
            params["filter"] = years
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])[:max_works]

    def resolve_pdf(self, work: dict[str, Any]) -> str | None:
        """Return resolve pdf.

        Parameters
        ----------
        work : Mapping[str, Any]
            Description for ``work``.

        Returns
        -------
        str | None
            Description of return value.

        Examples
        --------
        >>> from download.harvester import resolve_pdf
        >>> result = resolve_pdf(...)
        >>> result  # doctest: +ELLIPSIS

        See Also
        --------
        download.harvester

        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        best = work.get("best_oa_location") or {}
        if best and best.get("pdf_url"):
            return best["pdf_url"]
        for location in work.get("locations", []):
            if location.get("pdf_url"):
                return location["pdf_url"]
        doi = work.get("doi")
        if doi:
            response = self.session.get(
                f"{self.unpaywall}/v2/{doi}", params={"email": self.email}, timeout=15
            )
            if response.ok:
                payload = response.json()
                url = (payload.get("best_oa_location") or {}).get("url_for_pdf")
                if url:
                    return url
        if self.pdf_host and doi:
            return f"{self.pdf_host}/pdf/{doi.replace('/', '_')}.pdf"
        return None

    def download_pdf(self, url: str, target_path: str) -> str:
        """Return download pdf.

        Parameters
        ----------
        url : str
            Description for ``url``.
        target_path : str
            Description for ``target_path``.

        Returns
        -------
        str
            Description of return value.

        Raises
        ------
        DownloadError
            Raised when validation fails.
        UnsupportedMIMEError
            Raised when validation fails.

        Examples
        --------
        >>> from download.harvester import download_pdf
        >>> result = download_pdf(..., ...)
        >>> result  # doctest: +ELLIPSIS

        See Also
        --------
        download.harvester

        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        response = self.session.get(url, timeout=60)
        if response.status_code != HTTP_OK:
            message = f"Bad status {response.status_code} for {url}"
            raise DownloadError(message)
        content_type = response.headers.get("Content-Type", "application/pdf")
        if not content_type.startswith("application/"):
            message = f"Not a PDF-like content type: {content_type}"
            raise UnsupportedMIMEError(message)
        with open(target_path, "wb") as file_handle:
            file_handle.write(response.content)
        return target_path

    def run(self, topic: str, years: str, max_works: int) -> list[Doc]:
        """Return run.

        Parameters
        ----------
        topic : str
            Description for ``topic``.
        years : str
            Description for ``years``.
        max_works : int
            Description for ``max_works``.

        Returns
        -------
        List[Doc]
            Description of return value.

        Examples
        --------
        >>> from download.harvester import run
        >>> result = run(..., ..., ...)
        >>> result  # doctest: +ELLIPSIS

        See Also
        --------
        download.harvester

        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        docs: list[Doc] = []
        works = self.search(topic, years, max_works)
        for work in works:
            pdf_url = self.resolve_pdf(work)
            if not pdf_url:
                continue
            filename = (work.get("doi") or work.get("id") or str(int(time.time() * 1000))).replace(
                "/", "_"
            ) + ".pdf"
            destination = os.path.join(self.out_dir, filename)
            self.download_pdf(pdf_url, destination)
            doc = Doc(
                id=f"urn:doc:source:openalex:{work.get('id', 'unknown')}",
                openalex_id=work.get("id"),
                doi=work.get("doi"),
                title=work.get("title", ""),
                authors=[],
                pub_date=None,
                license=None,
                language="en",
                pdf_uri=destination,
                source="openalex",
                content_hash=None,
            )
            docs.append(doc)
        return docs
