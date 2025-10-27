"""Module for registry.duckdb_registry.

NavMap:
- DuckDBRegistry: Minimal DuckDB-backed registry (skeleton).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

import duckdb
from kgforge.kgforge_common.models import Doc, DoctagsAsset


class DuckDBRegistry:
    """Minimal DuckDB-backed registry (skeleton)."""

    def __init__(self, db_path: str) -> None:
        """Init.

        Parameters
        ----------
        db_path : str
            TODO.
        """
        self.db_path = db_path
        self.con = duckdb.connect(db_path, read_only=False)
        self.con.execute("PRAGMA threads=14")

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Begin dataset.

        Parameters
        ----------
        kind : str
            TODO.
        run_id : str
            TODO.

        Returns
        -------
        str
            TODO.
        """
        dsid = str(uuid.uuid4())
        self.con.execute(
            (
                "INSERT INTO datasets("
                "dataset_id, kind, parquet_root, run_id, created_at"
                ") VALUES (?, ?, '', ?, now())"
            ),
            [dsid, kind, run_id],
        )
        return dsid

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Commit dataset.

        Parameters
        ----------
        dataset_id : str
            TODO.
        parquet_root : str
            TODO.
        rows : int
            TODO.

        Returns
        -------
        None
            TODO.
        """
        self.con.execute(
            "UPDATE datasets SET parquet_root=? WHERE dataset_id=?", [parquet_root, dataset_id]
        )

    def rollback_dataset(self, dataset_id: str) -> None:
        """Rollback dataset.

        Parameters
        ----------
        dataset_id : str
            TODO.

        Returns
        -------
        None
            TODO.
        """
        self.con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Insert run.

        Parameters
        ----------
        purpose : str
            TODO.
        model_id : str | None
            TODO.
        revision : str | None
            TODO.
        config : dict
            TODO.

        Returns
        -------
        str
            TODO.
        """
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
        """Close run.

        Parameters
        ----------
        run_id : str
            TODO.
        success : bool
            TODO.
        notes : str | None
            TODO.

        Returns
        -------
        None
            TODO.
        """
        self.con.execute("UPDATE runs SET finished_at=now() WHERE run_id=?", [run_id])

    def register_documents(self, docs: list[Doc]) -> None:
        """Register documents.

        Parameters
        ----------
        docs : List[Doc]
            TODO.

        Returns
        -------
        None
            TODO.
        """
        for d in docs:
            self.con.execute(
                (
                    "INSERT OR REPLACE INTO documents("
                    "doc_id, openalex_id, doi, arxiv_id, pmcid, title, authors, "
                    "pub_date, license, language, pdf_uri, source, content_hash, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())"
                ),
                [
                    d.id,
                    d.openalex_id,
                    d.doi,
                    d.arxiv_id,
                    d.pmcid,
                    d.title,
                    json.dumps(d.authors),
                    d.pub_date,
                    d.license,
                    d.language,
                    d.pdf_uri,
                    d.source,
                    d.content_hash,
                ],
            )

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Register doctags.

        Parameters
        ----------
        assets : List[DoctagsAsset]
            TODO.

        Returns
        -------
        None
            TODO.
        """
        for a in assets:
            self.con.execute(
                (
                    "INSERT OR REPLACE INTO doctags("
                    "doc_id, doctags_uri, pages, vlm_model, vlm_revision, avg_logprob, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, now())"
                ),
                [a.doc_id, a.doctags_uri, a.pages, a.vlm_model, a.vlm_revision, a.avg_logprob],
            )

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Emit event.

        Parameters
        ----------
        event_name : str
            TODO.
        subject_id : str
            TODO.
        payload : Dict
            TODO.

        Returns
        -------
        None
            TODO.
        """
        self.con.execute(
            (
                "INSERT INTO pipeline_events("
                "event_id, event_name, subject_id, payload, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, now())"
            ),
            [event_name, subject_id, json.dumps(payload)],
        )

    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None:
        """Incident.

        Parameters
        ----------
        event : str
            TODO.
        subject_id : str
            TODO.
        error_class : str
            TODO.
        message : str
            TODO.

        Returns
        -------
        None
            TODO.
        """
        self.con.execute(
            (
                "INSERT INTO incidents("
                "id, event, subject_id, error_class, message, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, ?, now())"
            ),
            [event, subject_id, error_class, message],
        )
