"""Public helpers for generating CLI diagrams used in documentation tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tools.mkdocs_suite.docs._scripts import gen_cli_diagram as _impl

OperationEntry = _impl.OperationEntry

__all__ = ["OperationEntry", "collect_operations", "write_diagram"]


def collect_operations(spec: Mapping[str, object]) -> list[OperationEntry]:
    """Return CLI operations extracted from the OpenAPI specification mapping.

    Parameters
    ----------
    spec : Mapping[str, object]
        OpenAPI specification dictionary containing paths and operations.

    Returns
    -------
    list[OperationEntry]
        List of operation tuples, each containing ``(method, path, operation_id,
        summary, tags)`` where ``operation_id`` and ``summary`` may be ``None``.
    """
    return _impl.collect_operations(spec)


def write_diagram(operations: Sequence[OperationEntry]) -> None:
    """Write the CLI diagram to the MkDocs virtual filesystem."""
    _impl.write_diagram(operations)
