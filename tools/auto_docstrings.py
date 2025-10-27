#!/usr/bin/env python
"""Fallback docstring generator using the Python AST."""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DocstringChange:
    """Record a docstring update for logging purposes."""

    path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path, help="Directory to process.")
    parser.add_argument("--log", required=False, type=Path, help="Log file for changed paths.")
    return parser.parse_args()


def module_name_for(path: Path) -> str:
    """Return the dotted module path for a source file."""
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        relative = path
    parts = list(relative.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    module = ".".join(parts)
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def summarize(name: str, kind: str) -> str:
    """Return a short imperative summary for the given symbol."""
    base = " ".join(name.replace("_", " ").split()) or "value"
    if kind == "class":
        text = f"Represent {base}."
    elif kind == "module":
        text = f"Provide utilities for {base}."
    else:
        text = f"Return {base}."
    return text if text.endswith(".") else text + "."


def annotation_to_text(node: ast.AST | None) -> str:
    """Return the textual form of an annotation node in NumPy style."""
    if node is None:
        return "Any"
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover
        return "Any"
    text = text.replace("typing.", "")
    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        text = f"{inner} | None"
    replacements = {"list": "List", "dict": "Mapping[str, Any]", "tuple": "Tuple", "set": "Set"}
    if text in replacements:
        return replacements[text]
    text = text.replace("list[", "List[").replace("tuple[", "Tuple[").replace("set[", "Set[")
    if text.startswith("dict["):
        text = text.replace("dict[", "Mapping[")
    return text


def iter_docstring_nodes(tree: ast.Module) -> list[tuple[int, ast.AST, str]]:
    """Return docstring-bearing nodes sorted by descending line number."""
    items: list[tuple[int, ast.AST, str]] = [(0, tree, "module")]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            items.append((node.lineno, node, "class"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append((node.lineno, node, "function"))
    items.sort(key=lambda item: item[0], reverse=True)
    return items


def parameters_for(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str]]:
    """Return (name, type) tuples for a function's parameters."""
    params: list[tuple[str, str]] = []
    args = node.args

    def add(arg: ast.arg, default: ast.AST | None) -> None:
        """Return add.

        Parameters
        ----------
        arg : ast.arg
            Description.
        default : ast.AST | None
            Description.

        Returns
        -------
        None
            Description.
        """
        name = arg.arg
        if name in {"self", "cls"}:
            return
        annotation = annotation_to_text(arg.annotation)
        if default is not None:
            annotation = f"{annotation}, optional"
        params.append((name, annotation))

    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults, strict=True):
        add(arg, default)

    if args.vararg:
        params.append((f"*{args.vararg.arg}", "Any, optional"))

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        add(arg, default)

    if args.kwarg:
        params.append((f"**{args.kwarg.arg}", "Any, optional"))

    return params


def detect_raises(node: ast.AST) -> list[str]:
    """Return exception names raised by the supplied node."""
    seen: OrderedDict[str, None] = OrderedDict()
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise):
            continue
        exc = child.exc
        if exc is None:
            name = "Exception"
        elif isinstance(exc, ast.Call):
            func = exc.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = ast.unparse(func)
            else:  # pragma: no cover - defensive
                name = "Exception"
        elif isinstance(exc, ast.Name):
            name = exc.id
        elif isinstance(exc, ast.Attribute):
            name = ast.unparse(exc)
        else:
            name = "Exception"
        if name not in seen:
            seen[name] = None
    return list(seen.keys())


def build_examples(
    module_name: str, name: str, parameters: list[tuple[str, str]], has_return: bool
) -> list[str]:
    """Construct a doctestable Examples section for the symbol."""
    lines: list[str] = ["Examples", "--------"]
    if module_name:
        lines.append(f">>> from {module_name} import {name}")
    call_args = ["..."] * sum(1 for param, _ in parameters if not param.startswith("*"))
    invocation = f"{name}({', '.join(call_args)})" if call_args else f"{name}()"
    if has_return:
        lines.append(f">>> result = {invocation}")
        lines.append(">>> result  # doctest: +ELLIPSIS")
        lines.append("...")
    else:
        lines.append(f">>> {invocation}  # doctest: +ELLIPSIS")
    return lines


def build_docstring(kind: str, node: ast.AST, module_name: str) -> list[str]:
    """Construct a NumPy-style docstring for a node."""
    name = getattr(node, "name", "module")
    summary = summarize(name, kind)

    parameters: list[tuple[str, str]] = []
    returns: str | None = None
    raises: list[str] = []
    if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = parameters_for(node)
        return_annotation = annotation_to_text(node.returns)
        if return_annotation not in {"None", "NoReturn"}:
            returns = return_annotation
        raises = detect_raises(node)

    lines: list[str] = ['"""', summary]

    if kind == "module":
        lines.extend([
            "",
            "Notes",
            "-----",
            "This module exposes the primary interfaces for the package.",
            "",
            "See Also",
            "--------",
            module_name or name,
        ])
    elif kind == "class" and isinstance(node, ast.ClassDef):
        attributes: list[tuple[str, str]] = []
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                attributes.append((child.target.id, annotation_to_text(child.annotation)))
            elif isinstance(child, ast.Assign):
                names = [t.id for t in child.targets if isinstance(t, ast.Name)]
                for attr in names:
                    attributes.append((attr, "Any"))
        lines.extend(["", "Attributes", "----------"])
        if attributes:
            for attr_name, attr_type in attributes:
                lines.append(f"{attr_name} : {attr_type}")
                lines.append("    Attribute description.")
        else:
            lines.append("None")
            lines.append("    No public attributes documented.")

        methods = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
        if methods:
            lines.extend(["", "Methods", "-------"])
            for method in methods:
                lines.append(f"{method}()")
                lines.append("    Method description.")

        lines.extend(["", *build_examples(module_name, name, [], True)])
        lines.extend([
            "",
            "See Also",
            "--------",
            module_name or name,
            "",
            "Notes",
            "-----",
            "Document class invariants and lifecycle details here.",
        ])
    else:
        if parameters:
            lines.extend(["", "Parameters", "----------"])
            for param_name, annotation in parameters:
                lines.append(f"{param_name} : {annotation}")
                lines.append(f"    Description for ``{param_name}``.")

        if returns:
            lines.extend(["", "Returns", "-------", returns, "    Description of return value."])

        if raises:
            lines.extend(["", "Raises", "------"])
            for exc in raises:
                lines.append(f"{exc}")
                lines.append("    Raised when validation fails.")

        lines.extend(["", *build_examples(module_name, name, parameters, returns is not None)])
        lines.extend([
            "",
            "See Also",
            "--------",
            module_name or name,
            "",
            "Notes",
            "-----",
            "Provide usage considerations, constraints, or complexity notes.",
        ])

    lines.append('"""')
    return lines


def _required_sections(
    kind: str,
    parameters: list[tuple[str, str]],
    returns: str | None,
    raises: list[str],
) -> set[str]:
    """Return the set of sections expected for the supplied symbol."""
    required: set[str] = set()
    if kind == "module":
        required.update({"Notes", "See Also"})
    elif kind == "class":
        required.update({"Attributes", "Methods", "Examples", "See Also", "Notes"})
    else:
        required.update({"Examples", "See Also", "Notes"})
        if parameters:
            required.add("Parameters")
        if returns:
            required.add("Returns")
        if raises:
            required.add("Raises")
    return required


def docstring_text(node: ast.AST) -> tuple[str | None, ast.Expr | None]:
    """Return the existing docstring text and expression node, if any."""
    body = getattr(node, "body", [])
    if not body:
        return None, None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value.value, first
    return None, None


def replace(
    doc_expr: ast.Expr | None, lines: list[str], new_lines: list[str], indent: str, insert_at: int
) -> None:
    """Replace or insert the docstring at the calculated position."""
    formatted = [indent + line + "\n" for line in new_lines]
    if doc_expr is not None:
        start = doc_expr.lineno - 1
        end = doc_expr.end_lineno or doc_expr.lineno
        del lines[start:end]
        lines[start:start] = formatted
        after_index = start + len(formatted)
    else:
        lines[insert_at:insert_at] = formatted
        after_index = insert_at + len(formatted)
    lines.insert(after_index, indent + "\n")


def process_file(path: Path) -> bool:
    """Generate missing or placeholder docstrings for a single file."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    tree = ast.parse(text)
    lines = text.splitlines()
    lines = [line + "\n" for line in lines]
    changed = False

    module_name = module_name_for(path)

    for _, node, kind in iter_docstring_nodes(tree):
        doc, expr = docstring_text(node)
        parameters: list[tuple[str, str]] = []
        returns: str | None = None
        raises: list[str] = []
        if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = parameters_for(node)
            return_annotation = annotation_to_text(node.returns)
            if return_annotation not in {"None", "NoReturn"}:
                returns = return_annotation
            raises = detect_raises(node)

        required_sections = _required_sections(kind, parameters, returns, raises)
        needs_update = doc is None or "TODO" in (doc or "") or "NavMap:" in (doc or "")
        if not needs_update and required_sections:
            needs_update = not all(section in doc for section in required_sections)
        if not needs_update and doc:
            lower_markers = (" list[", " tuple[", " set[", " dict[", " list ", " dict ")
            if any(marker in doc for marker in lower_markers):
                needs_update = True
        if not needs_update:
            continue

        if kind == "module":
            indent = ""
            new_lines = build_docstring(kind, node, module_name)
            insert_at = 1 if lines and lines[0].startswith("#!") else 0
            replace(expr, lines, new_lines, indent, insert_at)
        else:
            indent = " " * (node.col_offset + 4)
            new_lines = build_docstring(kind, node, module_name)
            body = getattr(node, "body", [])
            insert_at = body[0].lineno - 1 if body else node.lineno
            replace(expr, lines, new_lines, indent, insert_at)
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> None:
    """Entry point for the fallback generator."""
    args = parse_args()
    target = args.target.resolve()
    changed: list[DocstringChange] = []

    for file_path in sorted(target.rglob("*.py")):
        if process_file(file_path):
            changed.append(DocstringChange(file_path))

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a", encoding="utf-8") as handle:
            for item in changed:
                handle.write(f"{item.path}\n")


if __name__ == "__main__":
    main()
