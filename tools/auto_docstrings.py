"""Auto-generate lightweight docstrings across the source tree."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

VERB_PREFIXES = {
    "add",
    "apply",
    "build",
    "cache",
    "calibrate",
    "close",
    "commit",
    "compute",
    "configure",
    "convert",
    "create",
    "download",
    "emit",
    "encode",
    "fetch",
    "generate",
    "get",
    "index",
    "insert",
    "load",
    "log",
    "normalize",
    "open",
    "prepare",
    "process",
    "register",
    "run",
    "save",
    "search",
    "train",
    "update",
    "write",
}

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

Action = tuple[str, int, str | list[str]]


def words_from_name(name: str) -> str:
    """Return a human-readable phrase from a snake_case identifier.

    Parameters
    ----------
    name : str
        Identifier to normalise.

    Returns
    -------
    str
        Identifier with underscores replaced by spaces.
    """
    return name.replace("_", " ")


def summary_for(name: str, kind: str) -> str:
    """Return a summary line tailored to the symbol type.

    Parameters
    ----------
    name : str
        Candidate identifier.
    kind : str
        Symbol kind: 'function', 'class', or 'module'.

    Returns
    -------
    str
        One-sentence summary ending with a period.
    """
    words = words_from_name(name)
    first = words.split()[0].lower() if words else ""
    if kind == "function":
        if first in VERB_PREFIXES:
            summary = words.capitalize()
        else:
            summary = f"Handle {words}" if words else "Handle value"
    elif kind == "class":
        summary = f"Represent {words}" if words else "Represent entity"
    else:
        summary = f"Module for {words}" if words else "Module summary"
    if not summary.endswith("."):
        summary += "."
    return summary


def ensure_module_docstring(path: Path, lines: list[str]) -> list[tuple[int, list[str]]]:
    """Return insertion actions for a module docstring when one is missing.

    Parameters
    ----------
    path : Path
        File path being processed.
    lines : list[str]
        File contents split into individual lines (including newlines).

    Returns
    -------
    list[tuple[int, list[str]]]
        Insertion actions or an empty list when a docstring already exists.
    """
    text = "".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    if ast.get_docstring(tree, clean=False) is not None:
        return []
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    module_name = str(rel).replace("/", ".")
    doc = summary_for(module_name, "module")
    doc_line = f'"""{doc}"""\n'
    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    return [(insert_idx, [doc_line, "\n"])]


def collect_defs(text: str) -> ast.AST:
    """Parse source text into an abstract syntax tree.

    Parameters
    ----------
    text : str
        Python source code.
    """
    return ast.parse(text)


def need_docstring(node: ast.AST) -> bool:
    """Return True if the AST node lacks a docstring."""
    return ast.get_docstring(node, clean=False) is None


def action_for_node(node: ast.AST, lines: list[str]) -> list[Action]:
    """Return docstring insertion actions for the supplied AST node.

    Parameters
    ----------
    node : ast.AST
        Node to inspect.
    lines : list[str]
        Source lines for the containing file.

    Returns
    -------
    list[Action]
        Actions required to insert a docstring (possibly empty).
    """
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return actions_for_function(node, lines)
    if isinstance(node, ast.ClassDef):
        return actions_for_class(node, lines)
    return []


def actions_for_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]
) -> list[Action]:
    """Generate docstring insertion actions for a function or coroutine.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        Function node to inspect.
    lines : list[str]
        Source lines for the containing file.

    Returns
    -------
    list[Action]
        Actions describing how to insert the summary docstring.
    """
    if ast.get_docstring(node, clean=False) is not None:
        return []
    summary = summary_for(node.name, "function")
    indent = " " * (node.col_offset + 4)
    same_line = False
    trailing = None
    insert_line = None
    if node.body:
        first = node.body[0]
        same_line = first.lineno == node.lineno
        if same_line:
            line_index = node.lineno - 1
            line = lines[line_index]
            colon_idx = line.rfind(":")
            signature = line[: colon_idx + 1]
            trailing = line[colon_idx + 1 :].strip()
            actions = [("replace_line", line_index, signature + "\n")]
            insert_line = line_index + 1
        else:
            insert_line = node.body[0].lineno - 1
            actions = []
    else:
        line_index = node.lineno - 1
        line = lines[line_index]
        colon_idx = line.rfind(":")
        signature = line[: colon_idx + 1]
        trailing = line[colon_idx + 1 :].strip()
        actions = [("replace_line", line_index, signature + "\n")]
        insert_line = line_index + 1
        same_line = True
    doc_line = indent + f'"""{summary}"""\n'
    actions.append(("insert_line", insert_line, doc_line))
    if trailing:
        actions.append(("insert_line", insert_line + 1, indent + trailing + "\n"))
    return actions


def actions_for_class(node: ast.ClassDef, lines: list[str]) -> list[Action]:
    """Generate docstring insertion actions for a class definition.

    Parameters
    ----------
    node : ast.ClassDef
        Class node to inspect.
    lines : list[str]
        Source lines for the containing file.

    Returns
    -------
    list[Action]
        Actions describing how to insert the summary docstring.
    """
    if ast.get_docstring(node, clean=False) is not None:
        return []
    summary = summary_for(node.name, "class")
    indent = " " * (node.col_offset + 4)
    actions: list[Action] = []
    trailing = None
    if node.body:
        first = node.body[0]
        same_line = first.lineno == node.lineno
        if same_line:
            line_index = node.lineno - 1
            line = lines[line_index]
            colon_idx = line.rfind(":")
            signature = line[: colon_idx + 1]
            trailing = line[colon_idx + 1 :].strip()
            actions.append(("replace_line", line_index, signature + "\n"))
            insert_line = line_index + 1
        else:
            insert_line = node.body[0].lineno - 1
    else:
        line_index = node.lineno - 1
        line = lines[line_index]
        colon_idx = line.rfind(":")
        signature = line[: colon_idx + 1]
        trailing = line[colon_idx + 1 :].strip()
        actions.append(("replace_line", line_index, signature + "\n"))
        insert_line = line_index + 1
    doc_line = indent + f'"""{summary}"""\n'
    actions.append(("insert_line", insert_line, doc_line))
    if trailing:
        actions.append(("insert_line", insert_line + 1, indent + trailing + "\n"))
    return actions


def gather_actions(path: Path, lines: list[str]) -> list[Action]:
    """Collect all docstring insertion actions for a source file.

    Parameters
    ----------
    path : Path
        File being processed.
    lines : list[str]
        Source lines (including newlines).

    Returns
    -------
    list[Action]
        Combined list of module, class, and function actions.
    """
    text = "".join(lines)
    try:
        tree = collect_defs(text)
    except SyntaxError:
        return []
    actions: list[Action] = []
    for idx, block in ensure_module_docstring(path, lines):
        actions.append(("module_block", idx, block))
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            actions.extend(action_for_node(node, lines))
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return actions


def apply_actions(lines: list[str], actions: Iterable[Action]) -> None:
    """Apply the generated actions to a mutable list of source lines.

    Parameters
    ----------
    lines : list[str]
        Source lines to mutate in-place.
    actions : Iterable[Action]
        Actions describing how to update the lines.
    """
    sortable: list[tuple[int, Action]] = []
    for action in actions:
        if action[0] == "module_block":
            idx = action[1]
            content = action[2]
            sortable.append((idx, ("insert_block", idx, content)))
        else:
            kind, line_idx, payload = action
            sortable.append((line_idx, (kind, line_idx, payload)))
    for _, (kind, line_idx, payload) in sorted(sortable, key=lambda x: x[0], reverse=True):
        if kind == "replace_line" and isinstance(payload, str):
            lines[line_idx] = payload
        elif kind == "insert_line" and isinstance(payload, str):
            lines.insert(line_idx, payload)
        elif kind == "insert_block" and isinstance(payload, list):
            for entry in reversed(payload):
                lines.insert(line_idx, entry)


def main() -> None:
    """Walk the src tree and synthesise missing docstrings."""
    for path in SRC_ROOT.rglob("*.py"):
        original = path.read_text()
        lines = list(original.splitlines(keepends=True))
        actions = gather_actions(path, lines)
        if not actions:
            continue
        apply_actions(lines, actions)
        new_text = "".join(lines)
        path.write_text(new_text)


if __name__ == "__main__":
    main()
