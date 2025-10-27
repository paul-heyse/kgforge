"""Convenience layer for inserting artefacts into the DuckDB registry.

NavMap:
- DuckDBRegistryHelper: Handy helper for opening short-lived DuckDB connections.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final

import duckdb
from kgfoundry.kgfoundry_common.models import Doc, DoctagsAsset

from kgfoundry_common.navmap_types import NavMap

__all__ = ["DuckDBRegistryHelper"]

__navmap__: Final[NavMap] = {
    "title": "registry.helper",
    "synopsis": "Helper utilities that simplify writing records into the DuckDB registry.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["DuckDBRegistryHelper"],
        },
    ],
}


# [nav:anchor DuckDBRegistryHelper]
class DuckDBRegistryHelper:
    """Open short-lived DuckDB connections for registry operations."""

    def __init__(self, db_path: str) -> None:
        """Store the DuckDB catalog location."""
        self.db_path = db_path

    def _con(self) -> duckdb.DuckDBPyConnection:
        """Return a fresh DuckDB connection."""
        return duckdb.connect(self.db_path)

    def new_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Create a new run record and return its identifier."""
        run_id = str(uuid.uuid4())
        con = self._con()
        con.execute(
            (
                "INSERT INTO runs "
                "(run_id,purpose,model_id,revision,started_at,config) "
                "VALUES (?,?,?,?,CURRENT_TIMESTAMP,?)"
            ),
            [run_id, purpose, model_id, revision, json.dumps(config)],
        )
        con.close()
        return run_id

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Mark a run as finished and emit a pipeline event."""
        con = self._con()
        con.execute("UPDATE runs SET finished_at=CURRENT_TIMESTAMP WHERE run_id=?", [run_id])
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [
                str(uuid.uuid4()),
                "RunClosed",
                run_id,
                json.dumps({"success": success, "notes": notes or ""}),
            ],
        )
        con.close()

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Create a dataset placeholder associated with ``run_id``."""
        dataset_id = str(uuid.uuid4())
        con = self._con()
        con.execute(
            (
                "INSERT INTO datasets "
                "(dataset_id,kind,parquet_root,run_id,created_at) "
                "VALUES (?,?,?,?,CURRENT_TIMESTAMP)"
            ),
            [dataset_id, kind, "", run_id],
        )
        con.close()
        return dataset_id

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Persist parquet location metadata and emit a commit event."""
        con = self._con()
        con.execute(
            "UPDATE datasets SET parquet_root=? WHERE dataset_id=?", [parquet_root, dataset_id]
        )
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [
                str(uuid.uuid4()),
                "DatasetCommitted",
                dataset_id,
                json.dumps({"rows": rows, "root": parquet_root}),
            ],
        )
        con.close()

    def rollback_dataset(self, dataset_id: str) -> None:
        """Delete a dataset and record a rollback event."""
        con = self._con()
        con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [str(uuid.uuid4()), "DatasetRolledBack", dataset_id, "{}"],
        )
        con.close()

    def register_documents(self, docs: list[Doc]) -> None:
        """Insert or update document metadata rows."""
        con = self._con()
        for doc in docs:
            con.execute(
                """INSERT OR REPLACE INTO documents
                (doc_id, openalex_id, doi, arxiv_id, pmcid, title, authors, pub_date, license,
                 language, pdf_uri, source, content_hash, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
                [
                    doc.id,
                    doc.openalex_id,
                    doc.doi,
                    doc.arxiv_id,
                    doc.pmcid,
                    doc.title,
                    json.dumps(doc.authors),
                    doc.pub_date,
                    doc.license,
                    doc.language,
                    doc.pdf_uri,
                    doc.source,
                    doc.content_hash or "",
                ],
            )
        con.close()

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Insert doctag metadata rows for VLM artefacts."""
        con = self._con()
        for asset in assets:
            con.execute(
                "INSERT OR REPLACE INTO doctags VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                [
                    asset.doc_id,
                    asset.doctags_uri,
                    asset.pages,
                    asset.vlm_model,
                    asset.vlm_revision,
                    asset.avg_logprob,
                ],
            )
        con.close()

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Emit an ad-hoc event into the pipeline events table."""
        con = self._con()
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [str(uuid.uuid4()), event_name, subject_id, json.dumps(payload)],
        )
        con.close()
