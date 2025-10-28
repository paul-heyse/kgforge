"""Overview of api.

This module bundles api logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol

from kgfoundry_common.models import Doc, DoctagsAsset
from kgfoundry_common.navmap_types import NavMap

__all__ = ["Registry"]

__navmap__: Final[NavMap] = {
    "title": "registry.api",
    "synopsis": "Protocol defining the registry interface",
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
        "Registry": {
            "owner": "@registry",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor Registry]
class Registry(Protocol):
    """Model the Registry.

    Represent the registry data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Compute begin dataset.

        Carry out the begin dataset operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        kind : str
        kind : str
            Description for ``kind``.
        run_id : str
        run_id : str
            Description for ``run_id``.

        Returns
        -------
        str
            Description of return value.

        Examples
        --------
        >>> from registry.api import begin_dataset
        >>> result = begin_dataset(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        """
        ...

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Compute commit dataset.

        Carry out the commit dataset operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        dataset_id : str
        dataset_id : str
            Description for ``dataset_id``.
        parquet_root : str
        parquet_root : str
            Description for ``parquet_root``.
        rows : int
        rows : int
            Description for ``rows``.

        Examples
        --------
        >>> from registry.api import commit_dataset
        >>> commit_dataset(..., ..., ...)  # doctest: +ELLIPSIS
        """
        ...

    def rollback_dataset(self, dataset_id: str) -> None:
        """Compute rollback dataset.

        Carry out the rollback dataset operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        dataset_id : str
        dataset_id : str
            Description for ``dataset_id``.

        Examples
        --------
        >>> from registry.api import rollback_dataset
        >>> rollback_dataset(...)  # doctest: +ELLIPSIS
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

        Carry out the insert run operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        purpose : str
        purpose : str
            Description for ``purpose``.
        model_id : str | None
        model_id : str | None
            Description for ``model_id``.
        revision : str | None
        revision : str | None
            Description for ``revision``.
        config : collections.abc.Mapping
        config : collections.abc.Mapping
            Description for ``config``.

        Returns
        -------
        str
            Description of return value.

        Examples
        --------
        >>> from registry.api import insert_run
        >>> result = insert_run(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        """
        ...

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Compute close run.

        Carry out the close run operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        run_id : str
        run_id : str
            Description for ``run_id``.
        success : bool
        success : bool
            Description for ``success``.
        notes : str | None
        notes : str | None, optional, default=None
            Description for ``notes``.

        Examples
        --------
        >>> from registry.api import close_run
        >>> close_run(..., ...)  # doctest: +ELLIPSIS
        """
        ...

    def register_documents(self, docs: list[Doc]) -> None:
        """Compute register documents.

        Carry out the register documents operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        docs : List[src.kgfoundry_common.models.Doc]
        docs : List[src.kgfoundry_common.models.Doc]
            Description for ``docs``.

        Examples
        --------
        >>> from registry.api import register_documents
        >>> register_documents(...)  # doctest: +ELLIPSIS
        """
        ...

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Compute register doctags.

        Carry out the register doctags operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        assets : List[src.kgfoundry_common.models.DoctagsAsset]
        assets : List[src.kgfoundry_common.models.DoctagsAsset]
            Description for ``assets``.

        Examples
        --------
        >>> from registry.api import register_doctags
        >>> register_doctags(...)  # doctest: +ELLIPSIS
        """
        ...

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Compute emit event.

        Carry out the emit event operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        event_name : str
        event_name : str
            Description for ``event_name``.
        subject_id : str
        subject_id : str
            Description for ``subject_id``.
        payload : collections.abc.Mapping
        payload : collections.abc.Mapping
            Description for ``payload``.

        Examples
        --------
        >>> from registry.api import emit_event
        >>> emit_event(..., ..., ...)  # doctest: +ELLIPSIS
        """
        ...

    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None:
        """Compute incident.

        Carry out the incident operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        event : str
        event : str
            Description for ``event``.
        subject_id : str
        subject_id : str
            Description for ``subject_id``.
        error_class : str
        error_class : str
            Description for ``error_class``.
        message : str
        message : str
            Description for ``message``.

        Examples
        --------
        >>> from registry.api import incident
        >>> incident(..., ..., ..., ...)  # doctest: +ELLIPSIS
        """
        ...
