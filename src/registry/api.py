"""Module for registry.api.

NavMap:
- Registry: Registry protocol describing persistence operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from kgfoundry.kgfoundry_common.models import Doc, DoctagsAsset


class Registry(Protocol):
    """Registry protocol describing persistence operations."""

    def begin_dataset(self, kind: str, run_id: str) -> str:
        """Begin tracking a dataset build for the given run."""
        ...

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        """Finalize a dataset build."""
        ...

    def rollback_dataset(self, dataset_id: str) -> None:
        """Rollback a dataset build after failure."""
        ...

    def insert_run(
        self,
        purpose: str,
        model_id: str | None,
        revision: str | None,
        config: Mapping[str, object],
    ) -> str:
        """Register a new processing run."""
        ...

    def close_run(self, run_id: str, success: bool, notes: str | None = None) -> None:
        """Mark a run as complete."""
        ...

    def register_documents(self, docs: list[Doc]) -> None:
        """Register document metadata with the registry."""
        ...

    def register_doctags(self, assets: list[DoctagsAsset]) -> None:
        """Register DocTags assets with the registry."""
        ...

    def emit_event(self, event_name: str, subject_id: str, payload: Mapping[str, object]) -> None:
        """Emit an audit event for monitoring."""
        ...

    def incident(self, event: str, subject_id: str, error_class: str, message: str) -> None:
        """Record an incident for visibility."""
        ...
