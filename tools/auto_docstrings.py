#!/usr/bin/env python
"""Fallback docstring generator using the Python AST."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


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
    """Return the textual form of an annotation node."""
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return "Any"


def iter_docstring_nodes(tree: ast.Module) -> Iterable[tuple[ast.AST, str]]:
    """Yield (node, kind) pairs for module, classes, and functions."""
    yield tree, "module"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node, "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, "function"


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


def build_docstring(kind: str, node: ast.AST) -> list[str]:
    """Construct a NumPy-style docstring for a node."""
    name = getattr(node, "name", "module")
    summary = summarize(name, kind)

    parameters: list[tuple[str, str]] = []
    returns: str | None = None
    if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = parameters_for(node)
        return_annotation = annotation_to_text(node.returns)
        if return_annotation not in {"None", "NoReturn"}:
            returns = return_annotation

    lines: list[str] = ['"""', summary]

    if parameters:
        lines.extend(["", "Parameters", "----------"])
        for name, annotation in parameters:
            lines.append(f"{name} : {annotation}")
            lines.append("    Description.")

    if returns:
        lines.extend(["", "Returns", "-------", returns, "    Description."])

    lines.append('"""')
    return lines


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

    for node, kind in iter_docstring_nodes(tree):
        doc, expr = docstring_text(node)
        needs_update = doc is None or (doc and "TODO" in doc)
        if not needs_update:
            continue
        if kind == "module":
            indent = ""
            new_lines = build_docstring(kind, node)
            insert_at = 1 if lines and lines[0].startswith("#!") else 0
            replace(expr, lines, new_lines, indent, insert_at)
        else:
            indent = " " * (node.col_offset + 4)
            new_lines = build_docstring(kind, node)
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
