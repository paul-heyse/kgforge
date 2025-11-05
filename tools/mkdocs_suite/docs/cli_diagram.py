"""Public helpers for generating CLI diagrams used in documentation tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tools.mkdocs_suite.docs._scripts import gen_cli_diagram as _impl

OperationEntry = _impl.OperationEntry

__all__ = ["OperationEntry", "collect_operations", "write_diagram"]


def collect_operations(spec: Mapping[str, object]) -> list[OperationEntry]:
    """Return CLI operations extracted from the OpenAPI specification mapping."""
    return _impl.collect_operations(spec)


def write_diagram(operations: Sequence[OperationEntry]) -> None:
    """Write the CLI diagram to the MkDocs virtual filesystem."""
    _impl.write_diagram(operations)
