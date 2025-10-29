"""Overview of harvester.

This module bundles harvester logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

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


@dataclass(frozen=True)
class HarvesterConfig:
    """Immutable configuration bundle for harvester endpoints."""

    openalex_base: str = "https://api.openalex.org"
    unpaywall_base: str = "https://api.unpaywall.org"
    pdf_host_base: str | None = None
    out_dir: str = "/data/pdfs"


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
        config: HarvesterConfig | None = None,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        user_agent : str
            Description for ``user_agent``.
        contact_email : str
            Description for ``contact_email``.
        config : HarvesterConfig | None
            Optional parameter default ``None``. Description for ``config``.
        """
        cfg = config or HarvesterConfig()
        self.ua = user_agent
        self.email = contact_email
        self.openalex = cfg.openalex_base.rstrip("/")
        self.unpaywall = cfg.unpaywall_base.rstrip("/")
        self.pdf_host = (cfg.pdf_host_base or "").rstrip("/")
        self.out_dir = cfg.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"{self.ua} ({self.email})"})

    def search(self, topic: str, years: str, max_works: int) -> list[dict[str, object]]:
        """Compute search.

        Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
        List[dict[str, object]]
            Description of return value.

        Raises
        ------
        TypeError
            Raised when validation fails.

        Examples
        --------
        >>> from download.harvester import search
        >>> result = search(..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
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
        payload = response.json()
        if not isinstance(payload, dict):
            message = "OpenAlex response payload must be a mapping"
            raise TypeError(message)
        results = payload.get("results", [])
        if not isinstance(results, list):
            message = "OpenAlex response must contain a list of results"
            raise TypeError(message)
        typed_results: list[dict[str, object]] = []
        for item in results:
            if isinstance(item, dict):
                typed_results.append(item)
        return typed_results[:max_works]

    def resolve_pdf(self, work: Mapping[str, object]) -> str | None:
        """Compute resolve pdf.

        Carry out the resolve pdf operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
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
        """
        direct = self._lookup_direct_pdf(work)
        if direct:
            return direct

        from_locations = self._lookup_locations_pdf(work.get("locations"))
        if from_locations:
            return from_locations

        doi_obj = work.get("doi")
        if isinstance(doi_obj, str) and doi_obj:
            unpaywall_url = self._resolve_unpaywall_pdf(doi_obj)
            if unpaywall_url:
                return unpaywall_url
            host_url = self._host_pdf_url(doi_obj)
            if host_url:
                return host_url
        return None

    def _lookup_direct_pdf(self, work: Mapping[str, object]) -> str | None:
        """Return the primary PDF URL if embedded directly in the payload."""
        best = work.get("best_oa_location")
        if isinstance(best, Mapping):
            pdf_url = best.get("pdf_url")
            if isinstance(pdf_url, str) and pdf_url:
                return pdf_url
        return None

    def _lookup_locations_pdf(self, locations: object) -> str | None:
        """Scan secondary locations for a resolvable PDF link."""
        if isinstance(locations, (str, bytes)):
            return None
        if not isinstance(locations, Sequence):
            return None
        for location in locations:
            if isinstance(location, Mapping):
                candidate = location.get("pdf_url")
                if isinstance(candidate, str) and candidate:
                    return candidate
        return None

    def _resolve_unpaywall_pdf(self, doi: str) -> str | None:
        """Resolve a DOI via Unpaywall when available."""
        response = self.session.get(
            f"{self.unpaywall}/v2/{doi}", params={"email": self.email}, timeout=15
        )
        if not response.ok:
            return None
        payload = response.json()
        if isinstance(payload, Mapping):
            best_location = payload.get("best_oa_location")
            if isinstance(best_location, Mapping):
                url = best_location.get("url_for_pdf")
                if isinstance(url, str) and url:
                    return url
        return None

    def _host_pdf_url(self, doi: str) -> str | None:
        """Return a hosted PDF URL when a distribution host is configured."""
        if not self.pdf_host:
            return None
        return f"{self.pdf_host}/pdf/{doi.replace('/', '_')}.pdf"

    def download_pdf(self, url: str, target_path: str) -> str:
        """Compute download pdf.

        Carry out the download pdf operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
            Description for ``topic``.
        years : str
            Description for ``years``.
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
        """
        docs: list[Doc] = []
        works = self.search(topic, years, max_works)
        for work in works:
            pdf_url = self.resolve_pdf(work)
            if not pdf_url:
                continue
            raw_name = work.get("doi") or work.get("id") or str(int(time.time() * 1000))
            filename = f"{str(raw_name).replace('/', '_')}.pdf"
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
