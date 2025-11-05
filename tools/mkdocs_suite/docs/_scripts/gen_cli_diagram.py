"""Generate CLI diagrams from the Typer OpenAPI specification.

Reads ``docs/openapi/openapi-cli.yaml`` and emits a D2 diagram grouped by tags.
Nodes link directly into the rendered ReDoc page for each CLI operation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import mkdocs_gen_files
import yaml

OperationEntry = tuple[str, str, str | None, str, tuple[str, ...]]

__all__ = ["OperationEntry", "collect_operations", "main", "write_diagram"]

LOGGER = logging.getLogger(__name__)

DOCS_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = DOCS_ROOT / "openapi" / "openapi-cli.yaml"
REDOC_PAGE = "api/openapi-cli.md"


def _load_cli_spec() -> dict[str, object]:
    if not SPEC_PATH.exists():
        return {}
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _operation_anchor(operation_id: str) -> str:
    return f"../{REDOC_PAGE}#operation/{operation_id}"


HTTP_METHODS: Final[set[str]] = {
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "trace",
}


def _collect_operations(
    spec: Mapping[str, object],
) -> list[OperationEntry]:
    operations: list[OperationEntry] = []
    paths_section = spec.get("paths")
    if not isinstance(paths_section, dict):
        if paths_section not in (None, {}):
            LOGGER.warning(
                "CLI specification contains non-mapping 'paths' section",
                extra={"paths_type": type(paths_section).__name__},
            )
        return operations

    for path, path_item in paths_section.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            operation_id_obj = op.get("operationId")
            operation_id: str | None = None
            if operation_id_obj is not None:
                operation_id = str(operation_id_obj).strip() or None
            tag_values = op.get("tags") or ["cli"]
            tags = tuple(dict.fromkeys(str(tag) for tag in tag_values))
            summary = (op.get("summary") or "").strip()
            operations.append((method.upper(), path, operation_id, summary, tags))
    return operations


def _write_diagram(
    operations: list[OperationEntry],
) -> None:
    diagram_path = "diagrams/cli_by_tag.d2"
    with mkdocs_gen_files.open(diagram_path, "w") as handle:
        handle.write('direction: right\nCLI: "CLI" {\n')
        unique_tags = sorted({tag for *_, tags in operations for tag in tags})
        handle.write("\n".join(f'  "{tag}": "{tag}" {{}}' for tag in unique_tags))
        if unique_tags:
            handle.write("\n")
        written_nodes: set[str] = set()
        for method, path, operation_id, summary, tags in operations:
            node_id = f"{method} {path}"
            label = f"{method} {path}\\n{summary}" if summary else node_id
            if node_id not in written_nodes:
                link_attr = (
                    f' {{ link: "{_operation_anchor(operation_id)}" }}' if operation_id else ""
                )
                handle.write(f'  "{node_id}": "{label}"{link_attr}\n')
                written_nodes.add(node_id)
            for tag in tags:
                handle.write(f'  "{tag}" -> "{node_id}"\n')
        handle.write("}\n")


def collect_operations(spec: Mapping[str, object]) -> list[OperationEntry]:
    """Return CLI operations extracted from the OpenAPI specification."""
    return _collect_operations(spec)


def write_diagram(operations: Sequence[OperationEntry]) -> None:
    """Emit a D2 diagram linking CLI tags to operations."""
    _write_diagram(list(operations))


def main() -> None:
    spec = _load_cli_spec()
    operations = collect_operations(spec)
    if not operations:
        LOGGER.info("No CLI operations discovered in specification")
        return
    write_diagram(operations)
    with mkdocs_gen_files.open("diagrams/index.md", "a") as handle:
        handle.write("- [CLI by Tag](./cli_by_tag.d2)\n")


if __name__ == "__main__":  # pragma: no cover - executed by mkdocs
    main()
