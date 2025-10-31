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
    """Describe Registry.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

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
        ...

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
        ...

    def rollback_dataset(self, dataset_id: str) -> None:
        """Describe rollback dataset.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        dataset_id : str
            Describe ``dataset_id``.
        """
        ...

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
        ...

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
        ...

    def register_documents(self, docs: list[Doc]) -> None:
        """Describe register documents.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        docs : list[Doc]
            Describe ``docs``.
        """
        ...

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Describe register doctags.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        assets : list[DoctagsAsset]
            Describe ``assets``.
        """
        ...

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
        ...

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
        ...
