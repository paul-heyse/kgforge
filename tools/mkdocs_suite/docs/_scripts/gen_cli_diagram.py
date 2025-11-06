"""Generate CLI diagrams using shared CLI tooling context.

This script leverages :mod:`tools._shared.cli_tooling` to load augment metadata
and registry configuration, ensuring diagrams reflect the same operation
context as the OpenAPI generator.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

import mkdocs_gen_files

from tools._shared.cli_tooling import (
    CLIConfigError,
    CLIToolingContext,
    CLIToolSettings,
    load_cli_tooling_context,
)
from tools._shared.problem_details import (
    ProblemDetailsParams,
    build_problem_details,
    render_problem,
)
from tools.typer_to_openapi_cli import import_object, to_click_command, walk_commands

OperationEntry = tuple[str, str, str | None, str, tuple[str, ...]]

__all__ = ["OperationEntry", "collect_operations", "main", "write_diagram"]

LOGGER = logging.getLogger(__name__)

DOCS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOCS_ROOT.parents[2]
AUGMENT_PATH = REPO_ROOT / "openapi" / "_augment_cli.yaml"
REGISTRY_PATH = REPO_ROOT / "tools" / "mkdocs_suite" / "api_registry.yaml"
REDOC_PAGE = "api/openapi-cli.md"

DEFAULT_INTERFACE_ID = "orchestration-cli"
DEFAULT_BIN_NAME = "kgf"
DEFAULT_TITLE = "KGFoundry CLI"
DEFAULT_VERSION = "0.0.0"


def _operation_anchor(operation_id: str | None) -> str:
    anchor = (operation_id or "").strip()
    return f"../{REDOC_PAGE}#operation/{anchor}"


def _default_cli_settings(interface_id: str | None) -> CLIToolSettings:
    return CLIToolSettings(
        bin_name=DEFAULT_BIN_NAME,
        title=DEFAULT_TITLE,
        version=DEFAULT_VERSION,
        augment_path=AUGMENT_PATH,
        registry_path=REGISTRY_PATH,
        interface_id=interface_id or DEFAULT_INTERFACE_ID,
    )


def _load_click_command(
    context: CLIToolingContext,
    target_interface: str | None,
) -> object:
    cli_config = cast("CLIConfig", context.cli_config)
    interface_id = target_interface or cli_config.interface_id
    if interface_id is None:
        _raise_diagram_error(
            "CLI interface identifier is not configured.",
            instance="urn:cli-diagram:missing-interface",
        )
    interface_meta = context.registry.get_interface(interface_id)
    if interface_meta is None:
        interface_meta = cli_config.interface_meta
    if interface_meta is None:
        _raise_diagram_error(
            f"Interface '{interface_id}' metadata is unavailable.",
            instance="urn:cli-diagram:missing-metadata",
            extras={"interface_id": interface_id},
        )
    entrypoint = interface_meta.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        _raise_diagram_error(
            f"Interface '{interface_id}' is missing an 'entrypoint' attribute.",
            instance="urn:cli-diagram:missing-entrypoint",
            extras={"interface_id": interface_id},
        )
    app_obj = import_object(entrypoint)
    return to_click_command(app_obj)


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


def collect_operations(
    context: CLIToolingContext | None = None,
    *,
    interface_id: str | None = None,
    click_cmd: object | None = None,
) -> list[OperationEntry]:
    """Return CLI operations derived from the shared CLI tooling context.

    Parameters
    ----------
    context : CLIToolingContext | None, optional
        Pre-loaded tooling context. When ``None``, the default repository
        configuration is loaded automatically.
    interface_id : str | None, optional
        Interface identifier to resolve. Defaults to the repository's CLI
        interface when omitted.
    click_cmd : object | None, optional
        Pre-resolved click command tree. When omitted the function imports the
        entrypoint declared in the registry metadata.

    Returns
    -------
    list[OperationEntry]
        List of operation tuples containing method, path, operation identifier,
        summary, and canonical tags. The method is always ``POST`` because CLI
        invocations map to action-style HTTP operations in the documentation.

    Raises
    ------
    CLIConfigError
        Raised when tooling metadata cannot be loaded or when the CLI entrypoint
        cannot be resolved.
    """
    if context is None:
        settings = _default_cli_settings(interface_id)
        context = load_cli_tooling_context(settings)
        interface_id = settings.interface_id

    cli_context = cast("CLIToolingContext", context)
    cli_config = cast("CLIConfig", cli_context.cli_config)
    command = click_cmd or _load_click_command(cli_context, interface_id or cli_config.interface_id)
    operations: list[OperationEntry] = []
    operation_context = cli_config.operation_context

    for tokens, command_obj in walk_commands(command, []):
        path, operation, tags = operation_context.build_operation(tokens, command_obj)
        method = "POST"
        op_id_raw = operation.get("operationId")
        operation_id = str(op_id_raw).strip() if op_id_raw is not None else None
        summary = str(operation.get("summary") or "").strip()
        tag_values = operation.get("tags") or ["cli"]
        tag_tuple = tuple(dict.fromkeys(str(tag) for tag in tag_values))
        operations.append((method, path, operation_id, summary, tag_tuple))

    return operations


def write_diagram(operations: Sequence[OperationEntry]) -> None:
    """Emit a D2 diagram linking CLI tags to operations."""
    _write_diagram(list(operations))


def main() -> None:
    try:
        operations = collect_operations()
    except CLIConfigError as exc:
        LOGGER.error(
            "Failed to collect CLI operations for diagram generation",
            extra={"status": "error"},
        )
        LOGGER.debug("CLI diagram problem details: %s", render_problem(exc.problem))
        return
    if not operations:
        LOGGER.info("No CLI operations discovered in configuration")
        return
    write_diagram(operations)
    with mkdocs_gen_files.open("diagrams/index.md", "a") as handle:
        handle.write("- [CLI by Tag](./cli_by_tag.d2)\n")


if __name__ == "__main__":  # pragma: no cover - executed by mkdocs
    main()


def _raise_diagram_error(
    detail: str,
    *,
    instance: str,
    status: int = 500,
    extras: Mapping[str, object] | None = None,
) -> NoReturn:
    problem = build_problem_details(
        ProblemDetailsParams(
            type="https://kgfoundry.dev/problems/cli-diagram",
            title="CLI diagram generation error",
            status=status,
            detail=detail,
            instance=instance,
            extensions=extras,
        )
    )
    LOGGER.error(detail, extra={**({} if extras is None else dict(extras)), "status": "error"})
    raise CLIConfigError(problem)
