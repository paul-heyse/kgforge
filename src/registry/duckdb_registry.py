"""DuckDB-backed registry implementation used by the fixture pipeline.

NavMap:
- DuckDBRegistry: Minimal registry wrapper for DuckDB-backed artefacts.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final

import duckdb
from kgfoundry.kgfoundry_common.models import Doc, DoctagsAsset

from kgfoundry_common.navmap_types import NavMap

__all__ = ["DuckDBRegistry"]

__navmap__: Final[NavMap] = {
    "title": "registry.duckdb_registry",
    "synopsis": "Minimal registry wrapper storing pipeline artefacts in DuckDB.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["DuckDBRegistry"],
        },
    ],
}


# [nav:anchor DuckDBRegistry]
class DuckDBRegistry:
    """Persist pipeline artefacts and events inside a DuckDB catalog."""

    def __init__(self, db_path: str) -> None:
        """Open a DuckDB connection, creating the database if required.

        Parameters
        ----------
        db_path : str
            Filesystem path to the DuckDB catalog.
        """
        self.db_path = db_path
        self.con = duckdb.connect(db_path, read_only=False)
        self.con.execute("PRAGMA threads=14")

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Create a dataset placeholder linked to an in-flight run."""
        dataset_id = str(uuid.uuid4())
        self.con.execute(
            (
                "INSERT INTO datasets("
                "dataset_id, kind, parquet_root, run_id, created_at"
                ") VALUES (?, ?, '', ?, now())"
            ),
            [dataset_id, kind, run_id],
        )
        return dataset_id

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Finalize the dataset by recording its parquet root."""
        self.con.execute(
            "UPDATE datasets SET parquet_root=? WHERE dataset_id=?", [parquet_root, dataset_id]
        )

    def rollback_dataset(self, dataset_id: str) -> None:
        """Discard a dataset that failed to materialise."""
        self.con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Record a new pipeline run and return its identifier."""
        run_id = str(uuid.uuid4())
        self.con.execute(
            (
                "INSERT INTO runs("
                "run_id, purpose, model_id, revision, started_at, config"
                ") VALUES (?, ?, ?, ?, now(), ?)"
            ),
            [run_id, purpose, model_id, revision, json.dumps(config)],
        )
        return run_id

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Mark a run as finished, optionally storing completion notes."""
        _ = success  # placeholder until success flag/notes are persisted
        _ = notes
        self.con.execute("UPDATE runs SET finished_at=now() WHERE run_id=?", [run_id])

    def register_documents(self, docs: list[Doc]) -> None:
        """Insert or update document metadata rows."""
        for doc in docs:
            self.con.execute(
                (
                    "INSERT OR REPLACE INTO documents("
                    "doc_id, openalex_id, doi, arxiv_id, pmcid, title, authors, "
                    "pub_date, license, language, pdf_uri, source, content_hash, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())"
                ),
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
                    doc.content_hash,
                ],
            )

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Persist doctag asset metadata for downstream retrieval."""
        for asset in assets:
            self.con.execute(
                (
                    "INSERT OR REPLACE INTO doctags("
                    "doc_id, doctags_uri, pages, vlm_model, vlm_revision, avg_logprob, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, now())"
                ),
                [
                    asset.doc_id,
                    asset.doctags_uri,
                    asset.pages,
                    asset.vlm_model,
                    asset.vlm_revision,
                    asset.avg_logprob,
                ],
            )

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Append a pipeline event row for observability."""
        self.con.execute(
            (
                "INSERT INTO pipeline_events("
                "event_id, event_name, subject_id, payload, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, now())"
            ),
            [event_name, subject_id, json.dumps(payload)],
        )

    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None:
        """Record an incident with lightweight context for debugging."""
        self.con.execute(
            (
                "INSERT INTO incidents("
                "id, event, subject_id, error_class, message, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, ?, now())"
            ),
            [event, subject_id, error_class, message],
        )
