"""Render an interface catalog sourced from nav sidecars and registry metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import mkdocs_gen_files
import yaml

SUITE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = SUITE_ROOT / "api_registry.yaml"


def _load_registry() -> dict[str, dict[str, object]]:
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    interfaces = payload.get("interfaces", {}) if isinstance(payload, dict) else {}
    if isinstance(interfaces, dict):
        return {str(key): value for key, value in interfaces.items() if isinstance(value, dict)}
    return {}


def _ensure_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _collect_nav_interfaces() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for nav_path in REPO_ROOT.glob("src/**/_nav.json"):
        module_name = nav_path.parent.name
        with nav_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        interfaces = data.get("interfaces") or []
        if not isinstance(interfaces, list):
            continue
        for entry in interfaces:
            if not isinstance(entry, dict):
                continue
            record = dict(entry)
            record.setdefault("module", module_name)
            records.append(record)
    return records


def _spec_link(record: dict[str, object]) -> str:
    spec_path = record.get("spec")
    if not isinstance(spec_path, str) or not spec_path:
        return "—"
    if spec_path.endswith("openapi-cli.yaml"):
        return "[CLI Spec](../api/openapi-cli.md)"
    if spec_path.endswith("openapi.yaml"):
        return "[HTTP API](../api/index.md)"
    docs_path = SUITE_ROOT / "docs" / spec_path
    if docs_path.exists():
        rel = Path("..") / spec_path
        return f"[{spec_path}]({rel.as_posix()})"
    return spec_path


def _problem_details(record: dict[str, object]) -> str:
    details = record.get("problem_details")
    if isinstance(details, list):
        return ", ".join(str(item) for item in details)
    if isinstance(details, str):
        return details
    return "—"


def _write_catalog_intro(handle: mkdocs_gen_files.files) -> None:
    """Emit the static intro blurb for the interface catalog page."""
    handle.write("# Interface Catalog\n\n")
    handle.write(
        "This catalog is generated from `_nav.json` sidecars and the shared interface registry.\n\n"
    )


def _write_interface_table(
    handle: mkdocs_gen_files.files,
    interfaces: list[dict[str, object]],
    registry: Mapping[str, dict[str, object]],
) -> set[str]:
    """Render the overview table and return the set of discovered interface IDs.

    Returns
    -------
    set[str]
        Normalized identifiers sourced from nav sidecars that will have detail
        sections rendered later in the document.
    """
    handle.write(
        "| Interface | Type | Module | Owner | Stability | Spec | Description | Problem Details |\n"
    )
    handle.write(
        "| --------- | ---- | ------ | ----- | -------- | ---- | ----------- | ---------------- |\n"
    )

    interface_ids: set[str] = set()
    for record in interfaces:
        identifier = str(record.get("id") or record.get("entrypoint") or "—")
        reg_entry = registry.get(identifier, {})
        description = record.get("description") or reg_entry.get("description") or "—"
        spec_cell = _spec_link(record)
        owner = record.get("owner") or reg_entry.get("owner") or "—"
        row = "| {id} | {type} | {module} | {owner} | {stability} | {spec} | {desc} | {problems} |".format(
            id=identifier,
            type=record.get("type", "—"),
            module=record.get("module", "—"),
            owner=owner,
            stability=record.get("stability", "—"),
            spec=spec_cell,
            desc=description,
            problems=_problem_details(record),
        )
        handle.write(row + "\n")
        if record.get("id"):
            interface_ids.add(str(record["id"]))
    return interface_ids


def _write_optional_list(handle: mkdocs_gen_files.files, label: str, values: list[str]) -> None:
    """Write a bullet list line when values are present."""
    if values:
        handle.write(f"  - {label}: {', '.join(values)}\n")


def _write_optional_line(handle: mkdocs_gen_files.files, label: str, value: object) -> None:
    """Write a bullet list line when ``value`` is truthy."""
    if value:
        handle.write(f"  - {label}: `{value}`\n")


def _write_code_samples(handle: mkdocs_gen_files.files, samples: object) -> None:
    """Emit code samples for an operation when provided in the registry."""
    if not isinstance(samples, list) or not samples:
        return
    handle.write("  - Code Samples:\n")
    for sample in samples:
        if isinstance(sample, Mapping):
            lang = sample.get("lang", "")
            source = sample.get("source", "")
            handle.write(f"    * ({lang}) `{source}`\n")


def _operations_block(handle: mkdocs_gen_files.files, operations: object) -> None:
    """Render operation metadata if the registry entry exposes operations."""
    if not isinstance(operations, Mapping) or not operations:
        return
    handle.write("\n### Operations\n\n")
    for op_key, op_meta in operations.items():
        if not isinstance(op_meta, Mapping):
            continue
        op_id = str(op_meta.get("operation_id") or f"cli.{op_key.replace('-', '_')}")
        summary = str(op_meta.get("summary") or "")
        handle.write(f"- [`{op_id}`](../api/openapi-cli.md) — {summary}\n")
        _write_optional_list(handle, "Tags", _ensure_str_list(op_meta.get("tags")))
        _write_optional_line(handle, "Handler", op_meta.get("handler"))
        _write_optional_list(handle, "Env", _ensure_str_list(op_meta.get("env")))
        _write_optional_list(
            handle, "Problem Details", _ensure_str_list(op_meta.get("problem_details"))
        )
        _write_code_samples(handle, op_meta.get("code_samples"))


def _write_interface_details(
    handle: mkdocs_gen_files.files,
    interface_ids: set[str],
    interfaces: list[dict[str, object]],
    registry: Mapping[str, dict[str, object]],
) -> None:
    """Write detailed sections per interface identifier."""
    lookup = {str(record["id"]): record for record in interfaces if record.get("id")}
    for identifier in sorted(interface_ids):
        nav_meta = lookup.get(identifier, {})
        reg_entry: Mapping[str, object] = registry.get(identifier, {})
        if not isinstance(reg_entry, Mapping):
            reg_entry = {}

        handle.write(f"## {identifier}\n\n")
        handle.write(
            "* **Type:** {type}\n".format(type=nav_meta.get("type") or reg_entry.get("type") or "—")
        )
        handle.write(
            "* **Module:** {module}\n".format(
                module=nav_meta.get("module") or reg_entry.get("module") or "—"
            )
        )
        handle.write(
            "* **Owner:** {owner}\n".format(
                owner=nav_meta.get("owner") or reg_entry.get("owner") or "—"
            )
        )
        handle.write(
            "* **Stability:** {stability}\n".format(
                stability=nav_meta.get("stability") or reg_entry.get("stability") or "—"
            )
        )
        if reg_entry.get("description") and reg_entry.get("description") != nav_meta.get(
            "description"
        ):
            handle.write(f"* **Description:** {reg_entry['description']}\n")

        _operations_block(handle, reg_entry.get("operations"))
        handle.write("\n")


def render_interface_catalog() -> None:
    registry = _load_registry()
    interfaces = _collect_nav_interfaces()
    interfaces.sort(key=lambda item: (str(item.get("type")), str(item.get("id"))))

    with mkdocs_gen_files.open("api/interfaces.md", "w") as handle:
        _write_catalog_intro(handle)
        interface_ids = _write_interface_table(handle, interfaces, registry)
        interface_ids.update(str(key) for key in registry)
        handle.write("\n")
        _write_interface_details(handle, interface_ids, interfaces, registry)


render_interface_catalog()
