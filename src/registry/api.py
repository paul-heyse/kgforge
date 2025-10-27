"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
registry.api
"""


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
    """
    Represent Registry.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
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
    >>> from registry.api import Registry
    >>> result = Registry()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    registry.api
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

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
        >>> from registry.api import begin_dataset
        >>> result = begin_dataset(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...

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
        >>> from registry.api import commit_dataset
        >>> commit_dataset(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...

    def rollback_dataset(self, dataset_id: str) -> None:
        """
        Return rollback dataset.
        
        Parameters
        ----------
        dataset_id : str
            Description for ``dataset_id``.
        
        Examples
        --------
        >>> from registry.api import rollback_dataset
        >>> rollback_dataset(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...

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
        >>> from registry.api import insert_run
        >>> result = insert_run(..., ..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...

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
        >>> from registry.api import close_run
        >>> close_run(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...

    def register_documents(self, docs: list[Doc]) -> None:
        """
        Return register documents.
        
        Parameters
        ----------
        docs : List[Doc]
            Description for ``docs``.
        
        Examples
        --------
        >>> from registry.api import register_documents
        >>> register_documents(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        ...

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """
        Return register doctags.
        
        Parameters
        ----------
        assets : List[DoctagsAsset]
            Description for ``assets``.
        
        Examples
        --------
        >>> from registry.api import register_doctags
        >>> register_doctags(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        ...

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
        >>> from registry.api import emit_event
        >>> emit_event(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...

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
        >>> from registry.api import incident
        >>> incident(..., ..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        registry.api
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        ...
