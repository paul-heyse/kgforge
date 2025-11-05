"""Generate CLI diagrams from the Typer OpenAPI specification.

Reads ``docs/openapi/openapi-cli.yaml`` and emits a D2 diagram grouped by tags.
Nodes link directly into the rendered ReDoc page for each CLI operation.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files
import yaml

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


def main() -> None:
    spec = _load_cli_spec()
    operations: list[tuple[str, str, str, str, str]] = []
    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() != "post":
                continue
            if not isinstance(op, dict):
                continue
            operation_id = op.get("operationId") or ""
            tags = op.get("tags") or ["cli"]
            summary = (op.get("summary") or "").strip()
            for tag in tags:
                operations.append((str(tag), method.upper(), path, operation_id, summary))

    diagram_path = "diagrams/cli_by_tag.d2"
    with mkdocs_gen_files.open(diagram_path, "w") as handle:
        handle.write('direction: right\nCLI: "CLI" {\n')
        for tag in sorted({tag for tag, *_ in operations}):
            handle.write(f'  "{tag}": "{tag}" {{}}\n')
        for tag, method, path, operation_id, summary in operations:
            node_id = f"{method} {path}"
            label = f"{method} {path}\\n{summary}" if summary else node_id
            handle.write(
                f'  "{node_id}": "{label}" {{ link: "{_operation_anchor(operation_id)}" }}\n'
            )
            handle.write(f'  "{tag}" -> "{node_id}"\n')
        handle.write("}\n")

    with mkdocs_gen_files.open("diagrams/index.md", "a") as handle:
        handle.write("- [CLI by Tag](./cli_by_tag.d2)\n")


if __name__ == "__main__":  # pragma: no cover - executed by mkdocs
    main()
