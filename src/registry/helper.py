"""Overview of helper.

This module bundles helper logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final

import duckdb

from kgfoundry_common.models import Doc, DoctagsAsset
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
    """Helper for writing registry artefacts into DuckDB tables.
<!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    db_path : str
        File path pointing at the DuckDB registry database.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _con(self) -> duckdb.DuckDBPyConnection:
        """Open a DuckDB connection to the configured database file.
<!-- auto:docstring-builder v1 -->

        Returns
        -------
        duckdb.DuckDBPyConnection
            Mutable connection handle bound to ``self.db_path``.
        """
        return duckdb.connect(self.db_path)

    def new_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Compute new run.
<!-- auto:docstring-builder v1 -->

Carry out the new run operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
        """Compute close run.
<!-- auto:docstring-builder v1 -->

Carry out the close run operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
run_id : str
    Description for ``run_id``.
success : bool
    Description for ``success``.
notes : str | None, optional
    Defaults to ``None``.
    Description for ``notes``.
    
    
    
    Defaults to ``None``.

Examples
--------
>>> from registry.helper import close_run
>>> close_run(..., ...)  # doctest: +ELLIPSIS
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
        """Compute begin dataset.
<!-- auto:docstring-builder v1 -->

Carry out the begin dataset operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
        """Compute commit dataset.
<!-- auto:docstring-builder v1 -->

Carry out the commit dataset operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
        """Compute rollback dataset.
<!-- auto:docstring-builder v1 -->

Carry out the rollback dataset operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
dataset_id : str
    Description for ``dataset_id``.
    
    
    

Examples
--------
>>> from registry.helper import rollback_dataset
>>> rollback_dataset(...)  # doctest: +ELLIPSIS
"""
        con = self._con()
        con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [str(uuid.uuid4()), "DatasetRolledBack", dataset_id, "{}"],
        )
        con.close()

    def register_documents(self, docs: list[Doc]) -> None:
        """Compute register documents.
<!-- auto:docstring-builder v1 -->

Carry out the register documents operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
docs : list[Doc]
    Description for ``docs``.
    
    
    

Examples
--------
>>> from registry.helper import register_documents
>>> register_documents(...)  # doctest: +ELLIPSIS
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
        """Compute register doctags.
<!-- auto:docstring-builder v1 -->

Carry out the register doctags operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
assets : list[DoctagsAsset]
    Description for ``assets``.
    
    
    

Examples
--------
>>> from registry.helper import register_doctags
>>> register_doctags(...)  # doctest: +ELLIPSIS
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
        """Compute emit event.
<!-- auto:docstring-builder v1 -->

Carry out the emit event operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
"""
        con = self._con()
        con.execute(
            "INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [str(uuid.uuid4()), event_name, subject_id, json.dumps(payload)],
        )
        con.close()
