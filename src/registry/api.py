"""Api utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

from kgfoundry_common.models import Doc, DoctagsAsset
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
        """Compute begin dataset.

        Carry out the begin dataset operation.

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
        """Compute commit dataset.

        Carry out the commit dataset operation.

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
        """Compute rollback dataset.

        Carry out the rollback dataset operation.

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
        """Compute insert run.

        Carry out the insert run operation.

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
        """Compute close run.

        Carry out the close run operation.

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
        """Compute register documents.

        Carry out the register documents operation.

        Parameters
        ----------
        docs : List[Doc]
            Description for ``docs``.
        """
        
        
        
        
        
        ...

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Compute register doctags.

        Carry out the register doctags operation.

        Parameters
        ----------
        assets : List[DoctagsAsset]
            Description for ``assets``.
        """
        
        
        
        
        
        ...

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Compute emit event.

        Carry out the emit event operation.

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
        """Compute incident.

        Carry out the incident operation.

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
