"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
registry.duckdb_registry
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
    """
    Represent DuckDBRegistry.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    __init__()
        Method description.
    begin_dataset()
        Method description.
    commit_dataset()
        Method description.
    rollback_dataset()
        Method description.
    insert_run()
        Method description.
    close_run()
        Method description.
    register_documents()
        Method description.
    register_doctags()
        Method description.
    emit_event()
        Method description.
    incident()
        Method description.
    
    Examples
    --------
    >>> from registry.duckdb_registry import DuckDBRegistry
    >>> result = DuckDBRegistry()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    registry.duckdb_registry
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def __init__(self, db_path: str) -> None:
        """
        Return init.
        
        Parameters
        ----------
        db_path : str
            Description for ``db_path``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import __init__
        >>> __init__(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.db_path = db_path
        self.con = duckdb.connect(db_path, read_only=False)
        self.con.execute("PRAGMA threads=14")

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """
        Return begin dataset.
        
        Parameters
        ----------
        kind : str
            Description for ``kind``.
        run_id : str
            Description for ``run_id``.
        
        Returns
        -------
        str
            Description of return value.
        
        Examples
        --------
        >>> from registry.duckdb_registry import begin_dataset
        >>> result = begin_dataset(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
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
        """
        Return commit dataset.
        
        Parameters
        ----------
        dataset_id : str
            Description for ``dataset_id``.
        parquet_root : str
            Description for ``parquet_root``.
        rows : int
            Description for ``rows``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import commit_dataset
        >>> commit_dataset(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.con.execute(
            "UPDATE datasets SET parquet_root=? WHERE dataset_id=?", [parquet_root, dataset_id]
        )

    def rollback_dataset(self, dataset_id: str) -> None:
        """
        Return rollback dataset.
        
        Parameters
        ----------
        dataset_id : str
            Description for ``dataset_id``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import rollback_dataset
        >>> rollback_dataset(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """
        Return insert run.
        
        Parameters
        ----------
        purpose : str
            Description for ``purpose``.
        model_id : str | None
            Description for ``model_id``.
        revision : str | None
            Description for ``revision``.
        config : Mapping[str, object]
            Description for ``config``.
        
        Returns
        -------
        str
            Description of return value.
        
        Examples
        --------
        >>> from registry.duckdb_registry import insert_run
        >>> result = insert_run(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
        """
        Return close run.
        
        Parameters
        ----------
        run_id : str
            Description for ``run_id``.
        success : bool
            Description for ``success``.
        notes : str | None, optional
            Description for ``notes``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import close_run
        >>> close_run(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        _ = success  # placeholder until success flag/notes are persisted
        _ = notes
        self.con.execute("UPDATE runs SET finished_at=now() WHERE run_id=?", [run_id])

    def register_documents(self, docs: list[Doc]) -> None:
        """
        Return register documents.
        
        Parameters
        ----------
        docs : List[Doc]
            Description for ``docs``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import register_documents
        >>> register_documents(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
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
        """
        Return register doctags.
        
        Parameters
        ----------
        assets : List[DoctagsAsset]
            Description for ``assets``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import register_doctags
        >>> register_doctags(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
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
        """
        Return emit event.
        
        Parameters
        ----------
        event_name : str
            Description for ``event_name``.
        subject_id : str
            Description for ``subject_id``.
        payload : Mapping[str, object]
            Description for ``payload``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import emit_event
        >>> emit_event(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
        """
        Return incident.
        
        Parameters
        ----------
        event : str
            Description for ``event``.
        subject_id : str
            Description for ``subject_id``.
        error_class : str
            Description for ``error_class``.
        message : str
            Description for ``message``.
        
        Examples
        --------
        >>> from registry.duckdb_registry import incident
        >>> incident(..., ..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.duckdb_registry
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.con.execute(
            (
                "INSERT INTO incidents("
                "id, event, subject_id, error_class, message, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, ?, now())"
            ),
            [event, subject_id, error_class, message],
        )
