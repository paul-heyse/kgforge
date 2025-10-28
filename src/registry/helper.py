"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
registry.helper
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
    """Represent DuckDBRegistryHelper.

    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    __init__()
        Method description.
    _con()
        Method description.
    new_run()
        Method description.
    close_run()
        Method description.
    begin_dataset()
        Method description.
    commit_dataset()
        Method description.
    rollback_dataset()
        Method description.
    register_documents()
        Method description.
    register_doctags()
        Method description.
    emit_event()
        Method description.
    
    Examples
    --------
    >>> from registry.helper import DuckDBRegistryHelper
    >>> result = DuckDBRegistryHelper()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    registry.helper
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def __init__(self, db_path: str) -> None:
        """Return init.

        Parameters
        ----------
        db_path : str
            Description for ``db_path``.
        
        Examples
        --------
        >>> from registry.helper import __init__
        >>> __init__(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        self.db_path = db_path

    def _con(self) -> duckdb.DuckDBPyConnection:
        """Return con.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Description of return value.
        
        Examples
        --------
        >>> from registry.helper import _con
        >>> result = _con()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        return duckdb.connect(self.db_path)

    def new_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Return new run.

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
        >>> from registry.helper import new_run
        >>> result = new_run(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
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
        """Return close run.

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
        >>> from registry.helper import close_run
        >>> close_run(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
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
        """Return begin dataset.

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
        >>> from registry.helper import begin_dataset
        >>> result = begin_dataset(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
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
        """Return commit dataset.

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
        >>> from registry.helper import commit_dataset
        >>> commit_dataset(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
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
        """Return rollback dataset.

        Parameters
        ----------
        dataset_id : str
            Description for ``dataset_id``.
        
        Examples
        --------
        >>> from registry.helper import rollback_dataset
        >>> rollback_dataset(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        con = self._con()
        con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [str(uuid.uuid4()), "DatasetRolledBack", dataset_id, "{}"],
        )
        con.close()

    def register_documents(self, docs: list[Doc]) -> None:
        """Return register documents.

        Parameters
        ----------
        docs : List[Doc]
            Description for ``docs``.
        
        Examples
        --------
        >>> from registry.helper import register_documents
        >>> register_documents(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
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
        """Return register doctags.

        Parameters
        ----------
        assets : List[DoctagsAsset]
            Description for ``assets``.
        
        Examples
        --------
        >>> from registry.helper import register_doctags
        >>> register_doctags(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
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
        """Return emit event.

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
        >>> from registry.helper import emit_event
        >>> emit_event(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.helper
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        con = self._con()
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [str(uuid.uuid4()), event_name, subject_id, json.dumps(payload)],
        )
        con.close()
