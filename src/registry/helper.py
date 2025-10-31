"""Overview of helper.

This module bundles helper logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from contextlib import closing
from typing import Final

from duckdb import DuckDBPyConnection

from kgfoundry_common.models import Doc, DoctagsAsset
from kgfoundry_common.navmap_types import NavMap
from registry import duckdb_helpers

__all__ = ["DuckDBRegistryHelper"]

__navmap__: Final[NavMap] = {
    "title": "registry.helper",
    "synopsis": "Helper utilities that simplify writing records into the DuckDB registry.",
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
        "DuckDBRegistryHelper": {
            "owner": "@registry",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor DuckDBRegistryHelper]
class DuckDBRegistryHelper:
    """Describe DuckDBRegistryHelper.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    db_path : str
        Describe ``db_path``.
"""

    def __init__(self, db_path: str) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        db_path : str
            Describe ``db_path``.
"""
        self.db_path = db_path

    def _connect(self) -> DuckDBPyConnection:
        """Create a DuckDB connection for registry operations.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        DuckDBPyConnection
            Describe return value.
"""
        return duckdb_helpers.connect(self.db_path)

    def new_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Describe new run.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        purpose : str
            Describe ``purpose``.
        model_id : str | NoneType
            Describe ``model_id``.
        revision : str | NoneType
            Describe ``revision``.
        config : str | object
            Describe ``config``.

        Returns
        -------
        str
            Describe return value.
"""
        run_id = str(uuid.uuid4())
        with closing(self._connect()) as con:
            duckdb_helpers.execute(
                con,
                (
                    "INSERT INTO runs "
                    "(run_id,purpose,model_id,revision,started_at,config) "
                    "VALUES (?,?,?,?,CURRENT_TIMESTAMP,?)"
                ),
                [run_id, purpose, model_id, revision, json.dumps(config)],
                operation="registry.helper.new_run",
            )
        return run_id

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Describe close run.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        run_id : str
            Describe ``run_id``.
        success : bool
            Describe ``success``.
        notes : str | NoneType, optional
            Describe ``notes``.
            Defaults to ``None``.
"""
        with closing(self._connect()) as con:
            duckdb_helpers.execute(
                con,
                "UPDATE runs SET finished_at=CURRENT_TIMESTAMP WHERE run_id=?",
                [run_id],
                operation="registry.helper.close_run.finish",
            )
            payload: dict[str, object] = {"success": success, "notes": notes or ""}
            duckdb_helpers.execute(
                con,
                "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                [
                    str(uuid.uuid4()),
                    "RunClosed",
                    run_id,
                    json.dumps(payload),
                ],
                operation="registry.helper.close_run.event",
            )

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Describe begin dataset.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        kind : str
            Describe ``kind``.
        run_id : str
            Describe ``run_id``.

        Returns
        -------
        str
            Describe return value.
"""
        dataset_id = str(uuid.uuid4())
        with closing(self._connect()) as con:
            duckdb_helpers.execute(
                con,
                (
                    "INSERT INTO datasets "
                    "(dataset_id,kind,parquet_root,run_id,created_at) "
                    "VALUES (?,?,?,?,CURRENT_TIMESTAMP)"
                ),
                [dataset_id, kind, "", run_id],
                operation="registry.helper.begin_dataset",
            )
        return dataset_id

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Describe commit dataset.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        dataset_id : str
            Describe ``dataset_id``.
        parquet_root : str
            Describe ``parquet_root``.
        rows : int
            Describe ``rows``.
"""
        with closing(self._connect()) as con:
            duckdb_helpers.execute(
                con,
                "UPDATE datasets SET parquet_root=? WHERE dataset_id=?",
                [parquet_root, dataset_id],
                operation="registry.helper.commit_dataset.update",
            )
            payload: dict[str, object] = {"rows": rows, "root": parquet_root}
            duckdb_helpers.execute(
                con,
                "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                [
                    str(uuid.uuid4()),
                    "DatasetCommitted",
                    dataset_id,
                    json.dumps(payload),
                ],
                operation="registry.helper.commit_dataset.event",
            )

    def rollback_dataset(self, dataset_id: str) -> None:
        """Describe rollback dataset.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        dataset_id : str
            Describe ``dataset_id``.
"""
        with closing(self._connect()) as con:
            duckdb_helpers.execute(
                con,
                "DELETE FROM datasets WHERE dataset_id=?",
                [dataset_id],
                operation="registry.helper.rollback_dataset.delete",
            )
            duckdb_helpers.execute(
                con,
                "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                [str(uuid.uuid4()), "DatasetRolledBack", dataset_id, "{}"],
                operation="registry.helper.rollback_dataset.event",
            )

    def register_documents(self, docs: list[Doc]) -> None:
        """Describe register documents.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        docs : list[Doc]
            Describe ``docs``.
"""
        with closing(self._connect()) as con:
            for doc in docs:
                authors_list = list(doc.authors) if doc.authors is not None else []
                authors_json = json.dumps(authors_list)
                duckdb_helpers.execute(
                    con,
                    """INSERT OR REPLACE INTO documents
                (doc_id, openalex_id, doi, arxiv_id, pmcid, title, authors,
                 pub_date, license, language, pdf_uri, source, content_hash, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
                    [
                        doc.id,
                        doc.openalex_id,
                        doc.doi,
                        doc.arxiv_id,
                        doc.pmcid,
                        doc.title,
                        authors_json,
                        doc.pub_date,
                        doc.license,
                        doc.language,
                        doc.pdf_uri,
                        doc.source,
                        doc.content_hash or "",
                    ],
                    operation="registry.helper.register_documents",
                )

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Describe register doctags.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        assets : list[DoctagsAsset]
            Describe ``assets``.
"""
        with closing(self._connect()) as con:
            for asset in assets:
                duckdb_helpers.execute(
                    con,
                    "INSERT OR REPLACE INTO doctags VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                    [
                        asset.doc_id,
                        asset.doctags_uri,
                        asset.pages,
                        asset.vlm_model,
                        asset.vlm_revision,
                        asset.avg_logprob,
                    ],
                    operation="registry.helper.register_doctags",
                )

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Describe emit event.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        event_name : str
            Describe ``event_name``.
        subject_id : str
            Describe ``subject_id``.
        payload : str | object
            Describe ``payload``.
"""
        with closing(self._connect()) as con:
            payload_dict: dict[str, object] = dict(payload)
            duckdb_helpers.execute(
                con,
                "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                [str(uuid.uuid4()), event_name, subject_id, json.dumps(payload_dict)],
                operation="registry.helper.emit_event",
            )
