"""Overview of duckdb registry.

This module bundles duckdb registry logic for the kgfoundry stack. It
groups related helpers so downstream packages can import a single
cohesive namespace. Refer to the functions and classes below for
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
    """Describe DuckDBRegistry.

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
        self.con = duckdb.connect(db_path, read_only=False)
        self.con.execute("PRAGMA threads=14")

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
        self.con.execute(
            "UPDATE datasets SET parquet_root=? WHERE dataset_id=?", [parquet_root, dataset_id]
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
        self.con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Describe insert run.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
purpose : str
    Describe ``purpose``.
model_id : str | None
    Describe ``model_id``.
revision : str | None
    Describe ``revision``.
config : Mapping[str, object]
    Describe ``config``.

Returns
-------
str
    Describe return value.
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
        """Describe close run.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
run_id : str
    Describe ``run_id``.
success : bool
    Describe ``success``.
notes : str | None, optional
    Describe ``notes``.
    Defaults to ``None``.
"""
        _ = success  # placeholder until success flag/notes are persisted
        _ = notes
        self.con.execute("UPDATE runs SET finished_at=now() WHERE run_id=?", [run_id])

    def register_documents(self, docs: list[Doc]) -> None:
        """Describe register documents.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
docs : list[Doc]
    Describe ``docs``.
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
        """Describe register doctags.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
assets : list[DoctagsAsset]
    Describe ``assets``.
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
        """Describe emit event.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
event_name : str
    Describe ``event_name``.
subject_id : str
    Describe ``subject_id``.
payload : Mapping[str, object]
    Describe ``payload``.
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
        """Describe incident.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
event : str
    Describe ``event``.
subject_id : str
    Describe ``subject_id``.
error_class : str
    Describe ``error_class``.
message : str
    Describe ``message``.
"""
        self.con.execute(
            (
                "INSERT INTO incidents("
                "id, event, subject_id, error_class, message, created_at"
                ") VALUES (gen_random_uuid(), ?, ?, ?, ?, now())"
            ),
            [event, subject_id, error_class, message],
        )
