"""Utilities for harmonising existing docstrings with runtime signatures."""

from __future__ import annotations

import inspect
import importlib
import textwrap
import types
from dataclasses import dataclass
from types import NoneType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from tools.docstring_builder.harvest import SymbolHarvest


def _resolve_object(symbol: SymbolHarvest) -> object | None:
    """Import the runtime object referenced by a harvested symbol."""

    try:
        module = importlib.import_module(symbol.module)
    except Exception:  # pragma: no cover - runtime import guard
        return None

    module_parts = symbol.module.split(".")
    qname_parts = symbol.qname.split(".")
    attr_parts = qname_parts[len(module_parts) :]
    try:
        obj: object = module
        for part in attr_parts:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return None


def _format_default(value: object) -> str:
    if value is inspect._empty:
        return ""
    if isinstance(value, str):
        return repr(value)
    if value is Ellipsis:
        return "..."
    return repr(value)


def _alias_for_type(annotation: Any, module_globals: dict[str, Any] | None) -> str | None:
    if not module_globals:
        return None
    for name, value in module_globals.items():
        if value is annotation:
            return name
    return None


def _module_alias(module_name: str, module_globals: dict[str, Any] | None) -> str | None:
    if not module_globals:
        return None
    for name, value in module_globals.items():
        if getattr(value, "__name__", None) == module_name:
            return name
    return None


def _format_type(annotation: Any, module_globals: dict[str, Any] | None = None) -> str | None:
    if annotation is inspect._empty:
        return None
    if isinstance(annotation, str):
        return annotation
    if annotation is Any:
        return "Any"
    if annotation in {None, NoneType}:  # pragma: no branch - identical behaviour
        return "None"

    if isinstance(annotation, types.UnionType):
        parts = [_format_type(arg, module_globals) or "Any" for arg in annotation.__args__]
        return " | ".join(parts)

    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        parts = [_format_type(arg, module_globals) or "Any" for arg in args]
        return " | ".join(parts)
    if origin is Annotated:
        base, *_ = get_args(annotation)
        return _format_type(base, module_globals)
    if origin is Literal:
        values = ", ".join(repr(arg) for arg in get_args(annotation))
        return f"Literal[{values}]"

    if isinstance(annotation, type):
        alias = _alias_for_type(annotation, module_globals)
        if alias:
            return alias
        module = getattr(annotation, "__module__", "")
        qualname = getattr(annotation, "__qualname__", str(annotation))
        if module == "builtins":
            return qualname
        module_alias = _module_alias(module, module_globals)
        if module_alias:
            return f"{module_alias}.{qualname}"
        if module in {"typing", "collections.abc"}:
            return qualname
        if module == "numpy":
            alias = _module_alias("numpy", module_globals)
            if alias:
                return f"{alias}.{qualname}"
        return f"{module}.{qualname}"

    if origin in {list, set, tuple, dict}:
        args = get_args(annotation)
        name = origin.__name__
        if not args:
            return name
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            inner = _format_type(args[0], module_globals) or "Any"
            return f"tuple[{inner}, ...]"
        inner = ", ".join(_format_type(arg, module_globals) or "Any" for arg in args)
        return f"{name}[{inner}]"

    if origin is not None:
        module = getattr(origin, "__module__", "")
        qualname = getattr(origin, "__qualname__", str(origin))
        if module == "numpy" and qualname == "ndarray":
            args = get_args(annotation)
            dtype_text = "Any"
            if len(args) >= 2:
                dtype_arg = args[1]
                dtype_args = get_args(dtype_arg)
                if dtype_args:
                    dtype_obj = dtype_args[0]
                else:
                    dtype_obj = getattr(dtype_arg, "type", dtype_arg)
                dtype_text = _format_type(dtype_obj, module_globals) or "Any"
            base = "NDArray" if module_globals and "NDArray" in module_globals else "numpy.ndarray"
            return f"{base}[{dtype_text}]"
        alias = _alias_for_type(origin, module_globals)
        module_alias = _module_alias(module, module_globals)
        prefix = ""
        if alias:
            prefix = alias
        elif module_alias:
            prefix = f"{module_alias}.{qualname}"
        elif module in {"builtins", "typing"}:
            prefix = qualname
        else:
            prefix = f"{module}.{qualname}"
        args = get_args(annotation)
        if args:
            inner = ", ".join(_format_type(arg, module_globals) or "Any" for arg in args)
            return f"{prefix}[{inner}]"
        return prefix

    if hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
        module = getattr(annotation, "__module__", "")
        qualname = getattr(annotation, "__qualname__")
        if module in {"builtins", "typing", "collections.abc"}:
            return qualname
        module_alias = _module_alias(module, module_globals)
        if module_alias:
            return f"{module_alias}.{qualname}"
        return f"{module}.{qualname}"

    return repr(annotation)


@dataclass(slots=True)
class _Section:
    title: str | None
    content: list[str]


def _parse_sections(docstring: str) -> list[_Section]:
    lines = docstring.splitlines()
    sections: list[_Section] = []
    current = _Section(title=None, content=[])
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        underline = next_line.strip()
        if (
            line.strip()
            and not line.startswith(" ")
            and underline
            and set(underline) == {"-"}
            and len(underline) >= len(line.strip())
        ):
            sections.append(current)
            current = _Section(title=line.strip(), content=[])
            i += 2
            continue
        current.content.append(line)
        i += 1
    sections.append(current)
    return sections


def _join_sections(sections: list[_Section]) -> str:
    output: list[str] = []
    for section in sections:
        if section.title is not None:
            if output and output[-1] != "":
                output.append("")
            output.append(section.title)
            output.append("-" * len(section.title))
        output.extend(section.content)
    return "\n".join(output).strip()


@dataclass(slots=True)
class _ParameterBlock:
    display_name: str
    description: list[str]


def _parse_parameters(section: _Section) -> dict[str, _ParameterBlock]:
    entries: dict[str, _ParameterBlock] = {}
    lines = section.content
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        name_part, _meta = line.split(":", 1)
        display_name = name_part.strip()
        canonical = display_name.lstrip("*")
        desc: list[str] = []
        i += 1
        while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
            desc.append(lines[i])
            i += 1
        entries[canonical] = _ParameterBlock(display_name=display_name, description=desc)
    return entries


@dataclass(slots=True)
class _ReturnBlock:
    description: list[str]


def _parse_returns(section: _Section) -> _ReturnBlock:
    lines = section.content
    if not lines:
        return _ReturnBlock(description=[])
    desc = [line for line in lines[1:]]
    return _ReturnBlock(description=desc)


def _ensure_description(description: list[str], fallback: str, name: str) -> list[str]:
    if description:
        normalised: list[str] = []
        for line in description:
            stripped = line.strip()
            normalised.append(line if line.startswith("    ") else f"    {stripped}")
        return normalised
    return [f"    {fallback}"]


def _merge_default(description: list[str], default: str | None) -> list[str]:
    if not default:
        return description
    merged: list[str] = []
    added = False
    for line in description:
        stripped = line.strip()
        marker = f"Optional parameter default ``{default}``."
        if stripped.startswith(marker):
            added = True
            tail = stripped.removeprefix(marker).strip()
            merged.append(f"    Defaults to ``{default}``.")
            if tail:
                merged.append(f"    {tail}")
            continue
        merged.append(line)
    if not added:
        merged.append(f"    Defaults to ``{default}``.")
    return merged


def _build_parameters_content(
    signature: inspect.Signature,
    hints: dict[str, Any],
    section: _Section | None,
    module_globals: dict[str, Any],
    symbol: SymbolHarvest,
) -> list[str]:
    blocks = _parse_parameters(section) if section else {}
    harvested = {param.name: param.annotation for param in symbol.parameters}
    lines: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            display = f"*{parameter.name}"
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            display = f"**{parameter.name}"
        else:
            display = parameter.name
        annotation = hints.get(parameter.name, parameter.annotation)
        annotation_text = _format_type(annotation, module_globals)
        if annotation_text is None:
            harvested_text = harvested.get(parameter.name)
            annotation_text = harvested_text or "Any"
        optional = parameter.default is not inspect._empty
        default_text = _format_default(parameter.default)
        block = blocks.get(parameter.name) or _ParameterBlock(
            display_name=display,
            description=[],
        )
        desc_lines = _ensure_description(block.description, f"Describe ``{parameter.name}``.", parameter.name)
        if optional and default_text:
            desc_lines = _merge_default(desc_lines, default_text)
        signature_line = f"{display} : {annotation_text}"
        if optional:
            signature_line += ", optional"
        lines.append(signature_line)
        lines.extend(desc_lines)
    return lines


def _build_returns_content(
    signature: inspect.Signature,
    hints: dict[str, Any],
    section: _Section | None,
    module_globals: dict[str, Any],
    symbol: SymbolHarvest,
) -> list[str]:
    annotation = hints.get("return", signature.return_annotation)
    annotation_text = _format_type(annotation, module_globals)
    if annotation_text in {None, "None"}:
        return []
    block = _parse_returns(section) if section else _ReturnBlock(description=[])
    desc_lines = _ensure_description(block.description, "Describe return value.", symbol.qname)
    return [annotation_text, *desc_lines]


def normalize_docstring(symbol: SymbolHarvest, marker: str) -> str | None:
    """Return a docstring updated to mirror runtime annotations."""

    if not symbol.docstring:
        return None
    obj = _resolve_object(symbol)
    if obj is None:
        return textwrap.dedent(symbol.docstring).strip()

    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - objects without signature
        return textwrap.dedent(symbol.docstring).strip()

    module_globals: dict[str, Any] = {}
    try:
        module = importlib.import_module(symbol.module)
    except Exception:  # pragma: no cover - import guard
        module = None
    if module is not None:
        module_globals = vars(module)

    try:
        hints = get_type_hints(obj, globalns=module_globals, include_extras=True)
    except Exception:
        hints = {}

    body = textwrap.dedent(symbol.docstring).strip()
    sections = _parse_sections(body)
    parameter_section = next((section for section in sections if section.title == "Parameters"), None)
    return_section_title = "Yields" if symbol.is_generator else "Returns"
    return_section = next((section for section in sections if section.title == return_section_title), None)

    parameter_content = _build_parameters_content(signature, hints, parameter_section, module_globals, symbol)
    return_content = _build_returns_content(signature, hints, return_section, module_globals, symbol)

    if parameter_section and parameter_content:
        parameter_section.content = parameter_content
    elif parameter_section and not parameter_content:
        sections.remove(parameter_section)
    elif not parameter_section and parameter_content:
        insertion_index = 1 if sections and sections[0].title is None else 0
        sections.insert(insertion_index, _Section(title="Parameters", content=parameter_content))

    if return_section and return_content:
        return_section.content = return_content
    elif return_section and not return_content:
        sections.remove(return_section)
    elif not return_section and return_content:
        sections.append(_Section(title=return_section_title, content=return_content))

    updated = _join_sections(sections)
    if marker and marker not in updated:
        updated_lines = updated.splitlines()
        insertion_point = 1 if updated_lines else 0
        updated_lines.insert(insertion_point, marker)
        updated = "\n".join(updated_lines)
    return updated


__all__ = ["normalize_docstring"]
