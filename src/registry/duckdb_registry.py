"""Overview of duckdb registry.

This module bundles duckdb registry logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final

from kgfoundry_common.models import Doc, DoctagsAsset
from kgfoundry_common.navmap_types import NavMap
from registry import duckdb_helpers

__all__ = ["DuckDBRegistry"]

__navmap__: Final[NavMap] = {
    "title": "registry.duckdb_registry",
    "synopsis": "Minimal registry wrapper storing pipeline artefacts in DuckDB.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@registry",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        "DuckDBRegistry": {
            "owner": "@registry",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor DuckDBRegistry]
class DuckDBRegistry:
    """DuckDB-backed implementation of the registry protocol."""

    def __init__(self, db_path: str) -> None:
        """Initialize the registry with a DuckDB database path."""
        self.db_path = db_path
        self.con = duckdb_helpers.connect(db_path, read_only=False)

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Insert a dataset placeholder row and return its identifier."""
        dataset_id = str(uuid.uuid4())
        duckdb_helpers.execute(
            self.con,
            (
                "INSERT INTO datasets("
                "dataset_id, kind, parquet_root, run_id, created_at"
                ") VALUES (?, ?, '', ?, now())"
            ),
            [dataset_id, kind, run_id],
            operation="registry.duckdb.begin_dataset",
        )
        return dataset_id

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Update dataset metadata once Parquet artifacts are materialized."""
        del rows
        duckdb_helpers.execute(
            self.con,
            "UPDATE datasets SET parquet_root=? WHERE dataset_id=?",
            [parquet_root, dataset_id],
            operation="registry.duckdb.commit_dataset",
        )

    def rollback_dataset(self, dataset_id: str) -> None:
        """Delete a dataset placeholder if the build fails."""
        duckdb_helpers.execute(
            self.con,
            "DELETE FROM datasets WHERE dataset_id=?",
            [dataset_id],
            operation="registry.duckdb.rollback_dataset",
        )

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Create a run record and return the generated identifier."""
        run_id = str(uuid.uuid4())
        duckdb_helpers.execute(
            self.con,
            (
                "INSERT INTO runs("
                "run_id, purpose, model_id, revision, started_at, config"
                ") VALUES (?, ?, ?, ?, now(), ?)"
            ),
            [run_id, purpose, model_id, revision, json.dumps(config)],
            operation="registry.duckdb.insert_run",
        )
        return run_id

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Mark a run as finished and record the completion timestamp."""
        _ = success  # placeholder until success flag/notes are persisted
        _ = notes
        duckdb_helpers.execute(
            self.con,
            "UPDATE runs SET finished_at=now() WHERE run_id=?",
            [run_id],
            operation="registry.duckdb.close_run",
        )

    def register_documents(self, docs: list[Doc]) -> None:
        """Insert or update document metadata rows."""
        for doc in docs:
            duckdb_helpers.execute(
                self.con,
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
                operation="registry.duckdb.register_documents",
            )

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Insert or update doctags asset records."""
        for asset in assets:
            duckdb_helpers.execute(
                self.con,
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
                operation="registry.duckdb.register_doctags",
            )

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Persist an arbitrary pipeline event with structured payload."""
        duckdb_helpers.execute(
            self.con,
            (
                "INSERT INTO pipeline_events("
                "event_id, event_name, subject_id, payload, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, now())"
            ),
            [event_name, subject_id, json.dumps(payload)],
            operation="registry.duckdb.emit_event",
        )

    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None:
        """Record an incident emitted by registry clients."""
        duckdb_helpers.execute(
            self.con,
            (
                "INSERT INTO incidents("
                "id, event, subject_id, error_class, message, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, ?, now())"
            ),
            [event, subject_id, error_class, message],
            operation="registry.duckdb.incident",
        )
