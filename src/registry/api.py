"""Api utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

from kgfoundry.kgfoundry_common.models import Doc, DoctagsAsset

from kgfoundry_common.navmap_types import NavMap

__all__ = ["Registry"]

__navmap__: Final[NavMap] = {
    "title": "registry.api",
    "synopsis": "Module for registry.api",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["Registry"],
        },
    ],
}


# [nav:anchor Registry]
class Registry(Protocol):
    """Describe Registry."""

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
        """
        ...

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
        """
        ...

    def rollback_dataset(self, dataset_id: str) -> None:
        """Return rollback dataset.

        Parameters
        ----------
        dataset_id : str
            Description for ``dataset_id``.
        """
        ...

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Return insert run.

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
        """
        ...

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Return close run.

        Parameters
        ----------
        run_id : str
            Description for ``run_id``.
        success : bool
            Description for ``success``.
        notes : str | None
            Description for ``notes``.
        """
        ...

    def register_documents(self, docs: list[Doc]) -> None:
        """Return register documents.

        Parameters
        ----------
        docs : List[Doc]
            Description for ``docs``.
        """
        ...

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Return register doctags.

        Parameters
        ----------
        assets : List[DoctagsAsset]
            Description for ``assets``.
        """
        ...

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
        """
        ...

    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None:
        """Return incident.

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
        """
        ...
