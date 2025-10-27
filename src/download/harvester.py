"""Harvest open-access documents from OpenAlex and Unpaywall.

NavMap:
- OpenAccessHarvester: Coordinate OpenAlex lookups, Unpaywall fallbacks, and
  local persistence.
"""

from __future__ import annotations

import os
import time
from typing import Any, Final

import requests
from kgfoundry.kgfoundry_common.exceptions import DownloadError, UnsupportedMIMEError
from kgfoundry.kgfoundry_common.models import Doc

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
    """Coordinate OpenAlex and Unpaywall lookups to persist open-access PDFs."""

    def __init__(  # noqa: PLR0913 - parameters mirror external API options
        self,
        user_agent: str,
        contact_email: str,
        openalex_base: str = "https://api.openalex.org",
        unpaywall_base: str = "https://api.unpaywall.org",
        pdf_host_base: str | None = None,
        out_dir: str = "/data/pdfs",
    ) -> None:
        """Initialize the harvester with API endpoints and storage options.

        Parameters
        ----------
        user_agent : str
            User agent string advertised to OpenAlex and downstream APIs.
        contact_email : str
            Contact address required by Unpaywall for polite usage.
        openalex_base : str, optional
            Base URL for OpenAlex API requests.
        unpaywall_base : str, optional
            Base URL for Unpaywall API lookups.
        pdf_host_base : str | None, optional
            Optional fallback host that mirrors PDFs by DOI.
        out_dir : str, optional
            Directory where downloaded PDFs will be stored.
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
        """Search OpenAlex for works matching the provided topic and year window.

        Parameters
        ----------
        topic : str
            OpenAlex topic identifier or free-form search term.
        years : str
            Publication-year filter expressed in the OpenAlex filter syntax.
        max_works : int
            Maximum number of works to retrieve.

        Returns
        -------
        list[dict[str, Any]]
            Work payloads returned by the OpenAlex API.
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
        """Resolve a PDF URL for an OpenAlex work using API metadata and fallbacks.

        Parameters
        ----------
        work : dict[str, Any]
            Work record returned from :meth:`search`.

        Returns
        -------
        str | None
            Direct PDF URL if one can be resolved; otherwise ``None``.
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
        """Download a PDF to the given path and validate the content type.

        Parameters
        ----------
        url : str
            Direct URL to the PDF.
        target_path : str
            Absolute path on disk where the PDF should be written.

        Returns
        -------
        str
            Path to the stored PDF.

        Raises
        ------
        DownloadError
            Raised when the response code indicates a failure.
        UnsupportedMIMEError
            Raised when the response is not a PDF-like MIME type.
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
        """Harvest PDFs for matching works and build :class:`Doc` records.

        Parameters
        ----------
        topic : str
            Topic identifier or search term forwarded to :meth:`search`.
        years : str
            Publication-year filter string forwarded to :meth:`search`.
        max_works : int
            Maximum number of works to process.

        Returns
        -------
        list[Doc]
            Documents referencing the downloaded PDFs.
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
