"""Utilities for harmonising existing docstrings with runtime signatures."""

from __future__ import annotations

import importlib
import inspect
import re
import textwrap
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import NoneType
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from tools.docstring_builder.harvest import SymbolHarvest

_ELLIPSIS_PAIR_LENGTH = 2
_SIGNATURE_RE = re.compile(r"^(?:\*{1,2})?[A-Za-z_][\w]*\s*:")


def _resolve_object(symbol: SymbolHarvest) -> object | None:
    """Import the runtime object referenced by a harvested symbol."""
    try:
        module = importlib.import_module(symbol.module)
    except Exception:  # pragma: no cover - runtime import guard
        return None
    else:
        module_parts = symbol.module.split(".")
        qname_parts = symbol.qname.split(".")
        attr_parts = qname_parts[len(module_parts) :]
        obj: object = module
        for part in attr_parts:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None
        return obj


def _resolve_callable(symbol: SymbolHarvest) -> Callable[..., Any] | None:
    """Return a callable object for the harvested symbol when available."""
    obj = _resolve_object(symbol)
    if obj is None or not callable(obj):
        return None
    return cast(Callable[..., Any], obj)


def _import_module_globals(module_name: str) -> dict[str, Any]:
    """Return module globals for the supplied module name."""
    try:
        module = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - import guard
        return {}
    return vars(module)


def _signature_and_hints(
    obj: Callable[..., Any], module_globals: Mapping[str, Any]
) -> tuple[inspect.Signature | None, dict[str, Any]]:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - objects without signature
        return None, {}
    try:
        hints = get_type_hints(obj, globalns=dict(module_globals), include_extras=True)
    except Exception:
        hints = {}
    return signature, hints


def _format_default(value: object) -> str:
    if value is inspect._empty:
        return ""
    if isinstance(value, str):
        return repr(value)
    if value is Ellipsis:
        return "..."
    return repr(value)


def _alias_for_type(annotation: object, module_globals: Mapping[str, object] | None) -> str | None:
    if not module_globals:
        return None
    for name, value in module_globals.items():
        if value is annotation:
            return name
    return None


def _module_alias(module_name: str, module_globals: Mapping[str, object] | None) -> str | None:
    if not module_globals:
        return None
    for name, value in module_globals.items():
        if getattr(value, "__name__", None) == module_name:
            return name
    return None


def _format_simple_type(annotation: object) -> str | None:
    if annotation is inspect._empty:
        return None
    if isinstance(annotation, str):
        return annotation
    if annotation is Any:
        return "Any"
    if annotation in {None, NoneType}:
        return "None"
    return None


def _format_union_type(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    if isinstance(annotation, types.UnionType):
        args = getattr(annotation, "__args__", ())
        parts = [_format_type(arg, module_globals) or "Any" for arg in args]
        return " | ".join(parts)
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        parts = [_format_type(arg, module_globals) or "Any" for arg in args]
        return " | ".join(parts)
    return None


def _format_annotated_type(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    if get_origin(annotation) is Annotated:
        base, *_ = get_args(annotation)
        return _format_type(base, module_globals)
    return None


def _format_literal_type(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    if get_origin(annotation) is Literal:
        values = ", ".join(repr(arg) for arg in get_args(annotation))
        return f"Literal[{values}]"
    return None


def _format_class_type(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    if not isinstance(annotation, type):
        return None
    alias = _alias_for_type(annotation, module_globals)
    if alias:
        return alias
    module = getattr(annotation, "__module__", "")
    qualname = getattr(annotation, "__qualname__", str(annotation))
    if module == "builtins":
        return qualname
    prefix: str | None = None
    module_alias = _module_alias(module, module_globals)
    if module_alias:
        prefix = f"{module_alias}.{qualname}"
    elif module in {"typing", "collections.abc"}:
        prefix = qualname
    elif module == "numpy":
        numpy_alias = _module_alias("numpy", module_globals)
        if numpy_alias:
            prefix = f"{numpy_alias}.{qualname}"
    if prefix:
        return prefix
    return f"{module}.{qualname}"


def _format_collection_type(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    origin = get_origin(annotation)
    if origin not in {list, set, tuple, dict}:
        return None
    args = get_args(annotation)
    origin_type = cast(type, origin)
    name = origin_type.__name__
    if not args:
        return name
    if origin is tuple and len(args) == _ELLIPSIS_PAIR_LENGTH and args[1] is Ellipsis:
        inner = _format_type(args[0], module_globals) or "Any"
        return f"tuple[{inner}, ...]"
    inner = ", ".join(_format_type(arg, module_globals) or "Any" for arg in args)
    return f"{name}[{inner}]"


def _format_generic_type(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    origin = get_origin(annotation)
    if origin is None:
        return None
    module = getattr(origin, "__module__", "")
    qualname = getattr(origin, "__qualname__", str(origin))
    if module == "numpy" and qualname == "ndarray":
        args = get_args(annotation)
        dtype_text = "Any"
        if len(args) >= _ELLIPSIS_PAIR_LENGTH:
            dtype_arg = args[1]
            dtype_args = get_args(dtype_arg)
            dtype_obj = dtype_args[0] if dtype_args else getattr(dtype_arg, "type", dtype_arg)
            dtype_text = _format_type(dtype_obj, module_globals) or "Any"
        base = "NDArray" if module_globals and "NDArray" in module_globals else "numpy.ndarray"
        return f"{base}[{dtype_text}]"
    alias = _alias_for_type(origin, module_globals)
    module_alias = _module_alias(module, module_globals)
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


def _format_module_attribute(
    annotation: object, module_globals: Mapping[str, object] | None
) -> str | None:
    if not hasattr(annotation, "__module__") or not hasattr(annotation, "__qualname__"):
        return None
    module = getattr(annotation, "__module__", "")
    qualname = getattr(annotation, "__qualname__", str(annotation))
    if module in {"builtins", "typing", "collections.abc"}:
        return qualname
    module_alias = _module_alias(module, module_globals)
    if module_alias:
        return f"{module_alias}.{qualname}"
    return f"{module}.{qualname}"


def _format_type(
    annotation: object, module_globals: Mapping[str, object] | None = None
) -> str | None:
    simple = _format_simple_type(annotation)
    if simple is not None:
        return simple

    formatted = _format_union_type(annotation, module_globals)
    if formatted is not None:
        return formatted

    annotated = _format_annotated_type(annotation, module_globals)
    if annotated is not None:
        return annotated

    formatters = (
        _format_union_type,
        _format_annotated_type,
        _format_literal_type,
        _format_class_type,
        _format_collection_type,
        _format_generic_type,
        _format_module_attribute,
    )

    for formatter in formatters:
        formatted = formatter(annotation, module_globals)
        if formatted is not None:
            return formatted

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


def _relayout_marker_block(docstring: str, marker: str) -> str:
    """Ensure the ownership marker sits on its own paragraph."""
    if not marker:
        return docstring

    lines = docstring.splitlines()
    if not lines:
        return marker

    summary = lines[0].replace(marker, "").strip()
    remaining = [line for line in lines[1:] if marker not in line]

    while remaining and not remaining[0].strip():
        remaining.pop(0)
    while remaining and not remaining[-1].strip():
        remaining.pop()

    if summary and summary[-1] not in ".!?":
        summary = summary.rstrip(".!? ") + "."

    parts: list[str] = []
    if summary:
        parts.append(summary)
    else:
        parts.append("")

    parts.append("")
    parts.append(marker)

    if remaining:
        parts.append("")
        parts.extend(remaining)

    return "\n".join(parts).strip()


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
    desc = list(lines[1:])
    return _ReturnBlock(description=desc)


def _ensure_description(description: list[str], fallback: str, name: str) -> list[str]:
    if description:
        normalised: list[str] = []
        has_text = False
        for line in description:
            stripped = line.strip()
            if _SIGNATURE_RE.match(stripped):
                # Drop lines that look like nested parameter signatures.
                continue
            if stripped:
                has_text = True
            normalised.append(line if line.startswith("    ") else f"    {stripped}")
        if has_text:
            return normalised
    return [f"    {fallback}"]


def _merge_default(description: list[str], default: str | None) -> list[str]:
    if not default:
        return description
    merged: list[str] = []
    added = False
    seen_defaults = False
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
        if stripped == f"Defaults to ``{default}``.":
            if not seen_defaults:
                merged.append(line if line.startswith("    ") else f"    {stripped}")
                seen_defaults = True
            added = True
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
        desc_lines = _ensure_description(
            block.description, f"Describe ``{parameter.name}``.", parameter.name
        )
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


def _update_parameter_section(
    sections: list[_Section], parameter_section: _Section | None, content: list[str]
) -> None:
    if parameter_section and content:
        parameter_section.content = content
        return
    if parameter_section and not content:
        sections.remove(parameter_section)
        return
    if not parameter_section and content:
        insertion_index = 1 if sections and sections[0].title is None else 0
        sections.insert(insertion_index, _Section(title="Parameters", content=content))


def _update_return_section(
    sections: list[_Section],
    return_section: _Section | None,
    content: list[str],
    section_title: str,
) -> None:
    if return_section and content:
        return_section.content = content
        return
    if return_section and not content:
        sections.remove(return_section)
        return
    if not return_section and content:
        sections.append(_Section(title=section_title, content=content))


def normalize_docstring(symbol: SymbolHarvest, marker: str) -> str | None:
    """Return a docstring updated to mirror runtime annotations."""
    if not symbol.docstring:
        return None
    original = textwrap.dedent(symbol.docstring).strip()
    callable_obj = _resolve_callable(symbol)
    if callable_obj is None:
        return None

    module_globals = _import_module_globals(symbol.module)
    signature, hints = _signature_and_hints(callable_obj, module_globals)
    if signature is None:
        return None

    body = original
    sections = _parse_sections(body)
    parameter_section = next(
        (section for section in sections if section.title == "Parameters"), None
    )
    return_section_title = "Yields" if symbol.is_generator else "Returns"
    return_section = next(
        (section for section in sections if section.title == return_section_title), None
    )

    parameter_content = _build_parameters_content(
        signature, hints, parameter_section, module_globals, symbol
    )
    return_content = _build_returns_content(
        signature, hints, return_section, module_globals, symbol
    )

    _update_parameter_section(sections, parameter_section, parameter_content)
    _update_return_section(sections, return_section, return_content, return_section_title)

    updated = _join_sections(sections)
    if marker:
        updated = _relayout_marker_block(updated, marker)
    return updated


__all__ = ["normalize_docstring"]
