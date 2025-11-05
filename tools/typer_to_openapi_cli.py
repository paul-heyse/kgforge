"""Generate an OpenAPI 3.1 document describing a Typer/Click CLI.

The resulting YAML includes rich ``x-cli`` metadata (examples, handlers, env vars)
and groups operations using tag metadata supplied via ``_augment_cli.yaml``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml

try:  # Typer is optional at runtime; degrade gracefully when absent.
    from typer.main import get_command as typer_get_command
except (ImportError, AttributeError):  # pragma: no cover - Typer may be unavailable in slim envs
    typer_get_command = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

AugmentPayload = dict[str, object]
JSONMapping = Mapping[str, object]

SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = SCRIPT_ROOT / "mkdocs_suite" / "api_registry.yaml"
HTTP_METHODS: set[str] = {
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "trace",
}


def import_object(path: str) -> object:
    """Import ``pkg.mod:attr`` strings and return the referenced object.

    Parameters
    ----------
    path : str
        Module path or ``module:attribute`` string to import.

    Returns
    -------
    object
        The imported object (module or attribute).
    """
    if ":" in path:
        module_name, attribute = path.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    return importlib.import_module(path)


def to_click_command(obj: object) -> click.core.Command:
    """Adapt Typer applications or raw click commands to ``click.Command``.

    Parameters
    ----------
    obj : object
        Typer application or click.Command instance.

    Returns
    -------
    click.core.Command
        Click command instance.

    Raises
    ------
    RuntimeError
        If Typer is not installed and obj is not already a click.Command.
    """
    if isinstance(obj, click.core.Command):
        return obj
    if typer_get_command is None:
        message = "Typer is not installed; install typer or pass a click.Command."
        raise RuntimeError(message)
    return typer_get_command(obj)


def snake_to_kebab(value: str) -> str:
    """Convert snake_case to kebab-case.

    Parameters
    ----------
    value : str
        String in snake_case format.

    Returns
    -------
    str
        String converted to kebab-case.
    """
    return value.replace("_", "-")


def param_schema(param: click.Parameter) -> tuple[dict[str, object], bool, str]:
    """Return (schema, required, example-name) triple for a click parameter.

    Parameters
    ----------
    param : click.Parameter
        Click parameter to analyze.

    Returns
    -------
    tuple[dict[str, object], bool, str]
        Schema dictionary, required flag, and example parameter name.
    """
    schema: dict[str, object] = {"type": "string"}
    required = bool(getattr(param, "required", False))
    example_name = param.name

    param_type = getattr(param, "type", None)
    type_name = getattr(param_type, "name", "").lower()
    if hasattr(param_type, "choices") and param_type.choices:
        schema["enum"] = list(param_type.choices)
    elif type_name in {"int", "integer"}:
        schema["type"] = "integer"
    elif type_name in {"float"}:
        schema["type"] = "number"
    elif type_name in {"bool", "boolean"}:
        schema["type"] = "boolean"
    elif type_name in {"path", "file", "filename"}:
        schema["format"] = "path"

    if getattr(param, "multiple", False) or getattr(param, "nargs", 1) not in {1, None}:
        schema = {"type": "array", "items": schema}

    help_text = getattr(param, "help", None)
    if help_text:
        schema["description"] = help_text

    return schema, required, example_name


def build_example(bin_name: str, tokens: Sequence[str], params: Sequence[click.Parameter]) -> str:
    """Construct a CLI usage example for documentation.

    Parameters
    ----------
    bin_name : str
        Binary/command name.
    tokens : Sequence[str]
        Command tokens (subcommands).
    params : Sequence[click.Parameter]
        Command parameters to include in example.

    Returns
    -------
    str
        CLI usage example string.
    """
    parts = [bin_name, *tokens]
    for param in params:
        if param.param_type_name == "argument":
            parts.append(f"<{param.name}>")
            continue
        option = f"--{snake_to_kebab(param.name)}"
        if getattr(param, "is_flag", False):
            parts.append(option)
        else:
            parts.extend([option, f"<{param.name}>"])
    return " ".join(parts)


def walk_commands(
    cmd: click.core.Command, tokens: list[str]
) -> list[tuple[list[str], click.core.Command]]:
    """Traverse nested click group returning runnable commands.

    Parameters
    ----------
    cmd : click.core.Command
        Click command to traverse.
    tokens : list[str]
        Current command path tokens.

    Returns
    -------
    list[tuple[list[str], click.core.Command]]
        List of (token path, command) tuples for all runnable commands.
    """
    results: list[tuple[list[str], click.core.Command]] = []
    if isinstance(cmd, click.core.Group) and getattr(cmd, "commands", {}):
        if getattr(cmd, "callback", None):
            results.append((tokens, cmd))
        for name, subcommand in cmd.commands.items():  # type: ignore[attr-defined]
            results.extend(walk_commands(subcommand, [*tokens, name]))
        return results
    results.append((tokens, cmd))
    return results


def _augment_lookup(augment: AugmentPayload, op_id: str, tokens: Sequence[str]) -> JSONMapping:
    operations = augment.get("operations") or {}
    override = operations.get(op_id)
    if override:
        return override
    token_key = " ".join(tokens)
    return operations.get(token_key, {})


def _load_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    interfaces = payload.get("interfaces") if isinstance(payload, dict) else None
    return interfaces if isinstance(interfaces, dict) else {}


def _ensure_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _operation_metadata(
    interface_meta: JSONMapping | None,
    tokens: Sequence[str],
    operation_id: str,
) -> dict[str, object]:
    if not interface_meta:
        return {}
    operations = interface_meta.get("operations")
    if not isinstance(operations, Mapping):
        return {}
    slug = "-".join(snake_to_kebab(token) for token in tokens)
    candidates = (operation_id, slug, " ".join(tokens))
    for key in candidates:
        meta = operations.get(key)
        if isinstance(meta, Mapping):
            return dict(meta)
    return {}


def _interface_metadata(
    interface_id: str | None,
    interface_meta: JSONMapping | None,
) -> dict[str, object] | None:
    if not interface_id and not interface_meta:
        return None
    meta = dict(interface_meta) if isinstance(interface_meta, Mapping) else {}
    identifier = meta.get("id", interface_id)
    return {
        "id": identifier,
        "module": meta.get("module"),
        "owner": meta.get("owner"),
        "stability": meta.get("stability"),
        "binary": meta.get("binary"),
        "protocol": meta.get("protocol"),
    }


def _initial_document(
    title: str,
    version: str,
    augment: JSONMapping,
    interface_id: str | None,
    interface_meta: JSONMapping | None,
) -> dict[str, object]:
    tag_entries = [
        dict(raw) for raw in augment.get("tags", []) if isinstance(raw, Mapping) and "name" in raw
    ]
    document: dict[str, object] = {
        "openapi": "3.1.0",
        "info": {"title": title, "version": version},
        "tags": tag_entries,
        "paths": {},
    }
    if interface_id and interface_meta:
        extension = _interface_metadata(interface_id, interface_meta)
        if extension:
            document.setdefault("info", {}).setdefault("x-kgf-interface", extension)
    tag_groups = augment.get("x-tagGroups")
    if isinstance(tag_groups, list):
        document["x-tagGroups"] = tag_groups
    return document


def _request_schema_for_command(
    params: Sequence[click.Parameter],
) -> tuple[dict[str, object], bool]:
    properties: dict[str, object] = {}
    required: list[str] = []
    for param in params:
        schema, is_required, example_name = param_schema(param)
        properties[example_name] = schema
        is_flag = bool(getattr(param, "is_flag", False))
        has_default = getattr(param, "default", None) is not None
        if is_required and not is_flag and not has_default:
            required.append(example_name)
    schema: dict[str, object] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = sorted(required)
    return schema, bool(properties)


def _apply_override_to_x_cli(block: dict[str, object], override: JSONMapping) -> None:
    updates = {
        key: value
        for key, value in override.items()
        if key.startswith("x-") or key in {"env", "handler", "x-codeSamples"}
    }
    if updates:
        block.update(updates)


def _collect_problem_details(
    op_meta: JSONMapping,
    interface_meta: JSONMapping | None,
    override: JSONMapping,
) -> list[str]:
    details: list[str] = []
    details.extend(_ensure_str_list(op_meta.get("problem_details")))
    if interface_meta:
        details.extend(_ensure_str_list(interface_meta.get("problem_details")))
    details.extend(_ensure_str_list(override.get("x-problemDetails")))
    return details


def _select_operation_tags(
    override: JSONMapping,
    op_meta: JSONMapping,
    interface_meta: JSONMapping | None,
    default_tag: str,
) -> list[str]:
    for candidate in (
        _ensure_str_list(override.get("tags")),
        _ensure_str_list(op_meta.get("tags")),
        _ensure_str_list(interface_meta.get("tags")) if interface_meta else [],
    ):
        if candidate:
            return candidate
    return [default_tag]


def _operation_payload(
    operation_id: str,
    summary: str,
    description: str,
    request_schema: dict[str, object],
    has_properties: bool,
    tags: Sequence[str],
    x_cli_block: dict[str, object],
) -> dict[str, object]:
    return {
        "operationId": operation_id,
        "tags": list(tags),
        "summary": summary,
        "description": description,
        "x-cli": x_cli_block,
        "requestBody": {
            "required": has_properties,
            "content": {"application/json": {"schema": request_schema}},
        },
        "responses": {
            "200": {
                "description": "CLI result",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "stdout": {"type": "string"},
                                "stderr": {"type": "string"},
                                "exitCode": {"type": "integer"},
                                "artifacts": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["exitCode"],
                        }
                    }
                },
            }
        },
    }


@dataclass(frozen=True, slots=True)
class _OperationDescriptor:
    """Immutable descriptor capturing command metadata for an operation."""

    raw_tokens: tuple[str, ...]
    command: click.core.Command
    tokens: tuple[str, ...] = field(init=False)
    params: tuple[click.Parameter, ...] = field(init=False)
    help_text: str = field(init=False)
    short_help: str = field(init=False)

    def __post_init__(self) -> None:
        tokens = list(self.raw_tokens) or [self.command.name or "run"]
        object.__setattr__(self, "tokens", tuple(tokens))
        params = tuple(getattr(self.command, "params", []))
        object.__setattr__(self, "params", params)
        help_text = (getattr(self.command, "help", "") or "").strip()
        object.__setattr__(self, "help_text", help_text)
        short_help = str(getattr(self.command, "short_help", "") or "").strip()
        object.__setattr__(self, "short_help", short_help)

    @property
    def kebab_tokens(self) -> tuple[str, ...]:
        return tuple(snake_to_kebab(token) for token in self.tokens)

    @property
    def path(self) -> str:
        return "/cli/" + "/".join(snake_to_kebab(token) for token in self.tokens)

    @property
    def operation_id(self) -> str:
        return "cli." + ".".join(self.tokens)

    @property
    def default_tag(self) -> str:
        return self.tokens[0]

    def summary(self, op_meta: JSONMapping) -> str:
        summary_value = op_meta.get("summary") or self.short_help
        if summary_value:
            return str(summary_value).strip()
        if self.help_text:
            return self.help_text.split("\n", 1)[0]
        return "Run CLI command"

    def description(self, op_meta: JSONMapping) -> str:
        return str(op_meta.get("description") or self.help_text)


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Context bundle shared across OpenAPI operation builders."""

    bin_name: str
    augment: JSONMapping
    interface_id: str | None
    interface_meta: JSONMapping | None

    def build_operation(
        self, tokens: Sequence[str], command: click.core.Command
    ) -> tuple[str, dict[str, object], list[str]]:
        descriptor = _OperationDescriptor(tuple(tokens), command)
        override = _augment_lookup(self.augment, descriptor.operation_id, descriptor.tokens)
        op_meta = _operation_metadata(
            self.interface_meta, descriptor.tokens, descriptor.operation_id
        )
        tags = _select_operation_tags(
            override, op_meta, self.interface_meta, descriptor.default_tag
        )
        params = list(descriptor.params)
        request_schema, has_properties = _request_schema_for_command(params)
        x_cli_block = _build_x_cli_block(descriptor, params, op_meta, self)
        _apply_override_to_x_cli(x_cli_block, override)
        operation = _operation_payload(
            descriptor.operation_id,
            descriptor.summary(op_meta),
            descriptor.description(op_meta),
            request_schema,
            has_properties,
            tags,
            x_cli_block,
        )
        problem_details = _collect_problem_details(op_meta, self.interface_meta, override)
        if problem_details:
            operation["x-problemDetails"] = problem_details
        interface_extension = _interface_metadata(self.interface_id, self.interface_meta)
        if interface_extension:
            operation["x-kgf-interface"] = interface_extension
        return descriptor.path, operation, tags


@dataclass(frozen=True, slots=True)
class CLIConfig:
    """Configuration describing the CLI and metadata sources."""

    bin_name: str
    title: str
    version: str
    augment: JSONMapping
    interface_id: str | None = None
    interface_meta: JSONMapping | None = None

    @property
    def operation_context(self) -> OperationContext:
        return OperationContext(
            bin_name=self.bin_name,
            augment=self.augment,
            interface_id=self.interface_id,
            interface_meta=self.interface_meta,
        )


def _build_x_cli_block(
    descriptor: _OperationDescriptor,
    params: Sequence[click.Parameter],
    op_meta: JSONMapping,
    context: OperationContext,
) -> dict[str, object]:
    kebab_tokens = list(descriptor.kebab_tokens)
    example = build_example(context.bin_name, kebab_tokens, params)
    block: dict[str, object] = {
        "bin": context.bin_name,
        "command": " ".join(kebab_tokens),
        "examples": [example],
        "exitCodes": [{"code": 0, "meaning": "success"}],
    }
    if context.interface_meta:
        block.setdefault(
            "x-interface",
            context.interface_meta.get("id", context.interface_id),
        )
    if op_meta.get("handler"):
        block["x-handler"] = op_meta["handler"]
    if op_meta.get("env"):
        block["x-env"] = list(op_meta["env"])
    if op_meta.get("code_samples"):
        block.setdefault("x-codeSamples", []).extend(op_meta["code_samples"])
    examples = _ensure_str_list(op_meta.get("examples"))
    if examples:
        block["examples"].extend(examples)
    problem_details = _ensure_str_list(op_meta.get("problem_details"))
    if problem_details:
        block.setdefault("x-problemDetails", []).extend(problem_details)
    return block


def _augment_document_tags(document: dict[str, object], referenced_tags: set[str]) -> None:
    existing = {
        entry.get("name") for entry in document.get("tags", []) if isinstance(entry, Mapping)
    }
    for tag in sorted(referenced_tags):
        if tag not in existing:
            document.setdefault("tags", []).append(
                {"name": tag, "description": f"Commands for {tag}."}
            )


def make_openapi(click_cmd: click.core.Command, config: CLIConfig) -> dict[str, object]:
    """Produce an OpenAPI 3.1 document representing the CLI.

    Parameters
    ----------
    click_cmd : click.core.Command
        Root click command or group to traverse.
    config : CLIConfig
        Configuration object containing title, version, augment metadata,
        interface settings, and operation context.

    Returns
    -------
    dict[str, object]
        Complete OpenAPI 3.1 document dictionary with paths, operations,
        tags, and extensions.
    """
    document = _initial_document(
        config.title,
        config.version,
        config.augment,
        config.interface_id,
        config.interface_meta,
    )
    context = config.operation_context
    referenced_tags: set[str] = set()

    for tokens, command in walk_commands(click_cmd, []):
        path, operation, tags = context.build_operation(tokens, command)
        document.setdefault("paths", {}).setdefault(path, {})["post"] = operation
        referenced_tags.update(tags)

    _augment_document_tags(document, referenced_tags)
    return document


def load_augment(path: str) -> AugmentPayload:
    """Load optional augmentation metadata from a YAML file.

    Parameters
    ----------
    path : str
        Filesystem path to the YAML augmentation file.

    Returns
    -------
    AugmentPayload
        Mapping describing additional tag metadata and operation overrides.
        Returns an empty dictionary if the file does not exist.

    Raises
    ------
    TypeError
        If the YAML document does not decode to a mapping.
    """
    augment_path = Path(path)
    if not augment_path.exists():
        return {}
    with augment_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            message = f"Augment file {augment_path} must contain a YAML mapping."
            raise TypeError(message)
        return data


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments for the OpenAPI generator.

    Parameters
    ----------
    argv : Sequence[str]
        Command-line arguments to parse.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace containing app, bin, title, version,
        augment, out, interface-id, and registry attributes.
    """
    parser = argparse.ArgumentParser(description="Generate OpenAPI spec for a Typer/Click CLI.")
    parser.add_argument(
        "--app", required=True, help="Import path to Typer or Click app (pkg.mod:attr)"
    )
    parser.add_argument("--bin", default="kgf", help="Binary name displayed in examples")
    parser.add_argument("--title", default="KGFoundry CLI", help="OpenAPI info.title value")
    parser.add_argument("--version", default="0.0.0", help="OpenAPI info.version value")
    parser.add_argument(
        "--augment",
        default="openapi/_augment_cli.yaml",
        help="YAML file with tag metadata and per-operation overrides",
    )
    parser.add_argument(
        "--out",
        default="openapi/openapi-cli.yaml",
        help="Destination path of the generated OpenAPI YAML",
    )
    parser.add_argument(
        "--interface-id",
        default=None,
        help="Interface identifier from api_registry.yaml for metadata enrichment",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to api_registry.yaml containing interface metadata",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for generating the CLI OpenAPI specification."""
    args = parse_args(argv if argv is not None else sys.argv[1:])

    app_obj = import_object(args.app)
    click_cmd = to_click_command(app_obj)
    augment = load_augment(args.augment)
    registry = _load_registry(Path(args.registry))
    interface_meta = registry.get(args.interface_id) if args.interface_id else None
    if interface_meta and isinstance(interface_meta, dict):
        interface_meta.setdefault("id", args.interface_id)
    else:
        interface_meta = None

    config = CLIConfig(
        bin_name=args.bin,
        title=args.title,
        version=args.version,
        augment=augment,
        interface_id=args.interface_id,
        interface_meta=interface_meta,
    )
    spec = make_openapi(click_cmd, config)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    click.echo(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    sys.exit(main())
