"""Overview of harvester.

This module bundles harvester logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import os
import time
from typing import Any, Final

import requests

from kgfoundry_common.errors import DownloadError, UnsupportedMIMEError
from kgfoundry_common.models import Doc
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
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@download",
        "stability": "beta",
        "since": "0.2.0",
    },
    "symbols": {
        "OpenAccessHarvester": {
            "owner": "@download",
            "stability": "beta",
            "since": "0.2.0",
        },
    },
}

HTTP_OK = 200


# [nav:anchor OpenAccessHarvester]
class OpenAccessHarvester:
    """Model the OpenAccessHarvester.

    Represent the openaccessharvester data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """

    def __init__(
        self,
        user_agent: str,
        contact_email: str,
        openalex_base: str = "https://api.openalex.org",
        unpaywall_base: str = "https://api.unpaywall.org",
        pdf_host_base: str | None = None,
        out_dir: str = "/data/pdfs",
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        user_agent : str
        user_agent : str
            Description for ``user_agent``.
        contact_email : str
        contact_email : str
            Description for ``contact_email``.
        openalex_base : str | None
        openalex_base : str | None, optional, default='https://api.openalex.org'
            Description for ``openalex_base``.
        unpaywall_base : str | None
        unpaywall_base : str | None, optional, default='https://api.unpaywall.org'
            Description for ``unpaywall_base``.
        pdf_host_base : str | None
        pdf_host_base : str | None, optional, default=None
            Description for ``pdf_host_base``.
        out_dir : str | None
        out_dir : str | None, optional, default='/data/pdfs'
            Description for ``out_dir``.
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
        """Compute search.

        Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
        
        Parameters
        ----------
        topic : str
        topic : str
            Description for ``topic``.
        years : str
        years : str
            Description for ``years``.
        max_works : int
        max_works : int
            Description for ``max_works``.
        
        Returns
        -------
        List[dict[str, typing.Any]]
            Description of return value.
        
        Examples
        --------
        >>> from download.harvester import search
        >>> result = search(..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
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
        """Compute resolve pdf.

        Carry out the resolve pdf operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
        
        Parameters
        ----------
        work : collections.abc.Mapping
        work : collections.abc.Mapping
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
        ...
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
        """Compute download pdf.

        Carry out the download pdf operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
        
        Parameters
        ----------
        url : str
        url : str
            Description for ``url``.
        target_path : str
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
        ...
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
        """Compute run.

        Carry out the run operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
        
        Parameters
        ----------
        topic : str
        topic : str
            Description for ``topic``.
        years : str
        years : str
            Description for ``years``.
        max_works : int
        max_works : int
            Description for ``max_works``.
        
        Returns
        -------
        List[src.kgfoundry_common.models.Doc]
            Description of return value.
        
        Examples
        --------
        >>> from download.harvester import run
        >>> result = run(..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
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
