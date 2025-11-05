#!/usr/bin/env python3
"""Generate an OpenAPI 3.1 document describing a Typer/Click CLI.

The resulting YAML includes rich ``x-cli`` metadata (examples, handlers, env vars)
and groups operations using tag metadata supplied via ``_augment_cli.yaml``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import click
import yaml

try:  # Typer is optional at runtime; degrade gracefully when absent.
    from typer.main import get_command as typer_get_command
except Exception:  # pragma: no cover - Typer may be unavailable in slim envs
    typer_get_command = None  # type: ignore[assignment]


AugmentPayload = dict[str, Any]


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = SCRIPT_ROOT / "mkdocs_suite" / "api_registry.yaml"


def import_object(path: str) -> Any:
    """Import ``pkg.mod:attr`` strings and return the referenced object."""
    if ":" in path:
        module_name, attribute = path.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    return importlib.import_module(path)


def to_click_command(obj: Any) -> click.core.Command:
    """Adapt Typer applications or raw click commands to ``click.Command``."""
    if isinstance(obj, click.core.Command):
        return obj
    if typer_get_command is None:
        message = "Typer is not installed; install typer or pass a click.Command."
        raise RuntimeError(message)
    return typer_get_command(obj)


def snake_to_kebab(value: str) -> str:
    return value.replace("_", "-")


def param_schema(param: click.Parameter) -> tuple[dict[str, Any], bool, str]:
    """Return (schema, required, example-name) triple for a click parameter."""
    schema: dict[str, Any] = {"type": "string"}
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


def build_example(bin_name: str, tokens: Iterable[str], params: Iterable[click.Parameter]) -> str:
    """Construct a CLI usage example for documentation."""
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
    """Traverse nested click groups returning runnable commands."""
    results: list[tuple[list[str], click.core.Command]] = []
    if isinstance(cmd, click.core.Group) and getattr(cmd, "commands", {}):
        if getattr(cmd, "callback", None):
            results.append((tokens, cmd))
        for name, subcommand in cmd.commands.items():  # type: ignore[attr-defined]
            results.extend(walk_commands(subcommand, tokens + [name]))
        return results
    results.append((tokens, cmd))
    return results


def _augment_lookup(augment: AugmentPayload, op_id: str, tokens: list[str]) -> Mapping[str, Any]:
    operations = augment.get("operations") or {}
    override = operations.get(op_id)
    if override:
        return override
    token_key = " ".join(tokens)
    return operations.get(token_key, {})


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    interfaces = payload.get("interfaces") if isinstance(payload, dict) else None
    return interfaces if isinstance(interfaces, dict) else {}


def _ensure_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _operation_metadata(
    interface_meta: Mapping[str, Any] | None,
    tokens: list[str],
    operation_id: str,
) -> dict[str, Any]:
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


def make_openapi(
    click_cmd: click.core.Command,
    bin_name: str,
    title: str,
    version: str,
    augment: AugmentPayload | None = None,
    *,
    interface_id: str | None = None,
    interface_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce an OpenAPI 3.1 document representing the CLI."""
    augment = augment or {}
    tag_defs = {tag["name"]: tag for tag in augment.get("tags", [])}
    tag_groups = augment.get("x-tagGroups")

    document: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {"title": title, "version": version},
        "tags": list(tag_defs.values()),
        "paths": {},
    }
    if interface_id and interface_meta:
        document.setdefault("info", {}).setdefault(
            "x-kgf-interface",
            {
                "id": interface_id,
                "module": interface_meta.get("module"),
                "owner": interface_meta.get("owner"),
                "stability": interface_meta.get("stability"),
                "binary": interface_meta.get("binary"),
                "protocol": interface_meta.get("protocol"),
            },
        )
    if tag_groups:
        document["x-tagGroups"] = tag_groups

    operations = walk_commands(click_cmd, [])
    referenced_tags: set[str] = set()

    for tokens, command in operations:
        if not tokens:
            tokens = [command.name or "run"]

        path = "/cli/" + "/".join(snake_to_kebab(token) for token in tokens)
        operation_id = "cli." + ".".join(tokens)
        override = _augment_lookup(augment, operation_id, tokens)
        op_meta = _operation_metadata(interface_meta, tokens, operation_id)

        meta_tags = _ensure_str_list(op_meta.get("tags"))
        override_tags = _ensure_str_list(override.get("tags"))
        interface_tags = _ensure_str_list(interface_meta.get("tags")) if interface_meta else []
        tags = override_tags or meta_tags or interface_tags or [tokens[0]]
        referenced_tags.update(tags)

        props: dict[str, Any] = {}
        required: list[str] = []
        params = list(getattr(command, "params", []))
        for param in params:
            schema, is_required, name_for_example = param_schema(param)
            props[name_for_example] = schema
            is_flag = bool(getattr(param, "is_flag", False))
            has_default = getattr(param, "default", None) is not None
            if is_required and not is_flag and not has_default:
                required.append(name_for_example)

        request_schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            request_schema["required"] = sorted(required)

        help_text = (getattr(command, "help", "") or "").strip()
        summary = str(
            op_meta.get("summary")
            or getattr(command, "short_help", "")
            or (help_text.split("\n", 1)[0] if help_text else "Run CLI command")
        ).strip()
        description = str(op_meta.get("description") or help_text)

        example = build_example(bin_name, [snake_to_kebab(t) for t in tokens], params)
        x_cli_block: dict[str, Any] = {
            "bin": bin_name,
            "command": " ".join(snake_to_kebab(t) for t in tokens),
            "examples": [example],
            "exitCodes": [{"code": 0, "meaning": "success"}],
        }
        if interface_meta:
            x_cli_block.setdefault("x-interface", interface_meta.get("id", interface_id))
        if op_meta.get("handler"):
            x_cli_block["x-handler"] = op_meta["handler"]
        if op_meta.get("env"):
            x_cli_block["x-env"] = list(op_meta["env"])
        if op_meta.get("code_samples"):
            x_cli_block.setdefault("x-codeSamples", []).extend(op_meta["code_samples"])
        if op_meta.get("examples"):
            x_cli_block["examples"].extend(_ensure_str_list(op_meta["examples"]))
        if op_meta.get("problem_details"):
            x_cli_block.setdefault("x-problemDetails", []).extend(
                _ensure_str_list(op_meta["problem_details"])
            )
        for key, value in override.items():
            if key.startswith("x-") or key in {"env", "handler", "x-codeSamples"}:
                x_cli_block[key] = value

        operation = {
            "operationId": operation_id,
            "tags": tags,
            "summary": summary,
            "description": description,
            "x-cli": x_cli_block,
            "requestBody": {
                "required": bool(props),
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

        problem_details = _ensure_str_list(op_meta.get("problem_details"))
        if interface_meta:
            problem_details.extend(_ensure_str_list(interface_meta.get("problem_details")))
        problem_details.extend(_ensure_str_list(override.get("x-problemDetails")))
        if problem_details:
            operation["x-problemDetails"] = problem_details
        if interface_id or interface_meta:
            operation["x-kgf-interface"] = {
                "id": interface_id,
                "owner": interface_meta.get("owner") if interface_meta else None,
                "module": interface_meta.get("module") if interface_meta else None,
                "stability": interface_meta.get("stability") if interface_meta else None,
                "binary": interface_meta.get("binary") if interface_meta else None,
            }

        document.setdefault("paths", {}).setdefault(path, {})["post"] = operation

    existing_tags = {tag["name"] for tag in document.get("tags", [])}
    for tag in sorted(referenced_tags):
        if tag not in existing_tags:
            document["tags"].append({"name": tag, "description": f"Commands for {tag}."})

    return document


def load_augment(path: str) -> AugmentPayload:
    augment_path = Path(path)
    if not augment_path.exists():
        return {}
    with augment_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            message = f"Augment file {augment_path} must contain a YAML mapping."
            raise ValueError(message)
        return data


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
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


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    app_obj = import_object(args.app)
    click_cmd = to_click_command(app_obj)
    augment = load_augment(args.augment)
    registry = _load_registry(Path(args.registry))
    interface_meta = registry.get(args.interface_id) if args.interface_id else None
    if interface_meta and isinstance(interface_meta, dict):
        interface_meta.setdefault("id", args.interface_id)
    else:
        interface_meta = None

    spec = make_openapi(
        click_cmd,
        args.bin,
        args.title,
        args.version,
        augment,
        interface_id=args.interface_id,
        interface_meta=interface_meta,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    sys.exit(main())
