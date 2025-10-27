"""Module for download.harvester.

NavMap:
- OpenAccessHarvester: Openaccessharvester.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from kgforge.kgforge_common.exceptions import DownloadError, UnsupportedMIMEError
from kgforge.kgforge_common.models import Doc

HTTP_OK = 200


class OpenAccessHarvester:
    """Openaccessharvester."""

    def __init__(  # noqa: PLR0913 - parameters mirror external API options
        self,
        user_agent: str,
        contact_email: str,
        openalex_base: str = "https://api.openalex.org",
        unpaywall_base: str = "https://api.unpaywall.org",
        pdf_host_base: str | None = None,
        out_dir: str = "/data/pdfs",
    ) -> None:
        """Init.

        Parameters
        ----------
        user_agent : str
            TODO.
        contact_email : str
            TODO.
        openalex_base : str
            TODO.
        unpaywall_base : str
            TODO.
        pdf_host_base : Optional[str]
            TODO.
        out_dir : str
            TODO.
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
        """Search.

        Parameters
        ----------
        topic : str
            TODO.
        years : str
            TODO.
        max_works : int
            TODO.

        Returns
        -------
        list[dict]
            TODO.
        """
        url = f"{self.openalex}/works"
        params = {"topic": topic, "per_page": min(200, max_works), "cursor": "*"}
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])[:max_works]

    def resolve_pdf(self, work: dict[str, Any]) -> str | None:
        """Resolve pdf.

        Parameters
        ----------
        work : dict
            TODO.

        Returns
        -------
        Optional[str]
            TODO.
        """
        best = work.get("best_oa_location") or {}
        if best and best.get("pdf_url"):
            return best["pdf_url"]
        for loc in work.get("locations", []):
            if loc.get("pdf_url"):
                return loc["pdf_url"]
        doi = work.get("doi")
        if doi:
            r = self.session.get(
                f"{self.unpaywall}/v2/{doi}", params={"email": self.email}, timeout=15
            )
            if r.ok:
                j = r.json()
                url = (j.get("best_oa_location") or {}).get("url_for_pdf")
                if url:
                    return url
        if self.pdf_host and doi:
            return f"{self.pdf_host}/pdf/{doi.replace('/', '_')}.pdf"
        return None

    def download_pdf(self, url: str, target_path: str) -> str:
        """Download pdf.

        Parameters
        ----------
        url : str
            TODO.
        target_path : str
            TODO.

        Returns
        -------
        str
            TODO.
        """
        r = self.session.get(url, timeout=60)
        if r.status_code != HTTP_OK:
            message = f"Bad status {r.status_code} for {url}"
            raise DownloadError(message)
        ctype = r.headers.get("Content-Type", "application/pdf")
        if not ctype.startswith("application/"):
            message = f"Not a PDF-like content-type: {ctype}"
            raise UnsupportedMIMEError(message)
        with open(target_path, "wb") as f:
            f.write(r.content)
        return target_path

    def run(self, topic: str, years: str, max_works: int) -> list[Doc]:
        """Run.

        Parameters
        ----------
        topic : str
            TODO.
        years : str
            TODO.
        max_works : int
            TODO.

        Returns
        -------
        List[Doc]
            TODO.
        """
        docs: list[Doc] = []
        works = self.search(topic, years, max_works)
        for w in works:
            pdf_url = self.resolve_pdf(w)
            if not pdf_url:
                continue
            filename = (w.get("doi") or w.get("id") or str(int(time.time() * 1000))).replace(
                "/", "_"
            ) + ".pdf"
            dest = os.path.join(self.out_dir, filename)
            self.download_pdf(pdf_url, dest)
            doc = Doc(
                id=f"urn:doc:source:openalex:{w.get('id', 'unknown')}",
                openalex_id=w.get("id"),
                doi=w.get("doi"),
                title=w.get("title", ""),
                authors=[],
                pub_date=None,
                license=None,
                language="en",
                pdf_uri=dest,
                source="openalex",
                content_hash=None,
            )
            docs.append(doc)
        return docs
