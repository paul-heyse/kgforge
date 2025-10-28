#!/usr/bin/env python
"""Auto Docstrings utilities."""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

QUALIFIED_NAME_OVERRIDES: dict[str, str] = {
    "FloatArray": "src.vectorstore_faiss.gpu.FloatArray",
    "IntArray": "src.vectorstore_faiss.gpu.IntArray",
    "StrArray": "src.vectorstore_faiss.gpu.StrArray",
    "VecArray": "src.search_api.faiss_adapter.VecArray",
    "_SupportsHttp": "src.search_client.client._SupportsHttp",
    "_SupportsResponse": "src.search_client.client._SupportsResponse",
    "NavMap": "src.kgfoundry_common.navmap_types.NavMap",
    "Doc": "src.kgfoundry_common.models.Doc",
    "DoctagsAsset": "src.kgfoundry_common.models.DoctagsAsset",
    "Chunk": "src.kgfoundry_common.models.Chunk",
    "LinkAssertion": "src.kgfoundry_common.models.LinkAssertion",
    "DownloadError": "src.kgfoundry_common.errors.DownloadError",
    "UnsupportedMIMEError": "src.kgfoundry_common.errors.UnsupportedMIMEError",
    "SparseEncoder": "src.embeddings_sparse.base.SparseEncoder",
    "SparseIndex": "src.embeddings_sparse.base.SparseIndex",
    "DenseEmbeddingModel": "src.embeddings_dense.base.DenseEmbeddingModel",
    "SearchRequest": "src.search_api.schemas.SearchRequest",
    "SearchResult": "src.search_api.schemas.SearchResult",
    "FixtureDoc": "src.search_api.fixture_index.FixtureDoc",
    "SpladeDoc": "src.search_api.splade_index.SpladeDoc",
    "Concept": "src.ontology.catalog.ConceptMeta",
    "Neo4jStore": "src.kg_builder.neo4j_store.Neo4jStore",
    "BaseModel": "pydantic.BaseModel",
    "NDArray": "numpy.typing.NDArray",
    "ArrayLike": "numpy.typing.ArrayLike",
    "Iterable": "typing.Iterable",
    "Iterator": "typing.Iterator",
    "Mapping": "typing.Mapping",
    "MutableMapping": "typing.MutableMapping",
    "Sequence": "typing.Sequence",
    "MutableSequence": "typing.MutableSequence",
    "Optional": "typing.Optional",
    "Callable": "typing.Callable",
    "Any": "typing.Any",
    "duckdb.DuckDBPyConnection": "duckdb.DuckDBPyConnection",
    "DuckDBPyConnection": "duckdb.DuckDBPyConnection",
    "pyarrow.Schema": "pyarrow.Schema",
    "pyarrow.schema": "pyarrow.schema",
    "HTTPException": "fastapi.HTTPException",
    "Exit": "typer.Exit",
}


@dataclass
class DocstringChange:
    """Describe DocstringChange."""

    path: Path


def parse_args() -> argparse.Namespace:
    """Return parse args.

    Returns
    -------
    argparse.Namespace
        Description of return value.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path, help="Directory to process.")
    parser.add_argument("--log", required=False, type=Path, help="Log file for changed paths.")
    return parser.parse_args()


def module_name_for(path: Path) -> str:
    """Return module name for.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    str
        Description of return value.
    """
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
    if module and path.is_relative_to(SRC_ROOT):
        module = f"src.{module}" if module else "src"
    return module


def summarize(name: str, kind: str) -> str:
    """Return summarize.

    Parameters
    ----------
    name : str
        Description for ``name``.
    kind : str
        Description for ``kind``.

    Returns
    -------
    str
        Description of return value.
    """
    base = " ".join(name.replace("_", " ").split()).strip()
    if kind == "class":
        text = f"Describe {base or 'object'}."
    elif kind == "module":
        text = f"{base.title() or 'Module'} utilities."
    else:
        text = f"Return {base or 'value'}."
    return text if text.endswith(".") else text + "."


def extended_summary(kind: str, name: str, module_name: str) -> str:
    """Return extended summary.

    Parameters
    ----------
    kind : str
        Description for ``kind``.
    name : str
        Description for ``name``.
    module_name : str
        Description for ``module_name``.

    Returns
    -------
    str
        Description of return value.
    """
    return ""


def annotation_to_text(node: ast.AST | None) -> str:
    """Return annotation to text.

    Parameters
    ----------
    node : ast.AST | None
        Description for ``node``.

    Returns
    -------
    str
        Description of return value.
    """
    if node is None:
        return "Any"
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover
        return "Any"
    text = text.replace("typing.", "")
    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        text = f"Optional {inner}"
    replacements = {"list": "List", "dict": "Mapping[str, Any]", "tuple": "Tuple", "set": "Set"}
    if text in replacements:
        return replacements[text]
    text = text.replace("list[", "List[").replace("tuple[", "Tuple[").replace("set[", "Set[")
    if text.startswith("dict["):
        text = text.replace("dict[", "Mapping[")
    resolved = QUALIFIED_NAME_OVERRIDES.get(text, text)
    if resolved.startswith("Optional[") and resolved.endswith("]"):
        inner = resolved[len("Optional[") : -1]
        return f"Optional {inner}"
    return resolved


def iter_docstring_nodes(tree: ast.Module) -> list[tuple[int, ast.AST, str]]:
    """Return iter docstring nodes.

    Parameters
    ----------
    tree : ast.Module
        Description for ``tree``.

    Returns
    -------
    List[Tuple[int, ast.AST, str]]
        Description of return value.
    """
    items: list[tuple[int, ast.AST, str]] = [(0, tree, "module")]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            items.append((node.lineno, node, "class"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append((node.lineno, node, "function"))
    items.sort(key=lambda item: item[0], reverse=True)
    return items


def parameters_for(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str]]:
    """Return parameters for.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        Description for ``node``.

    Returns
    -------
    List[Tuple[str, str]]
        Description of return value.
    """
    params: list[tuple[str, str]] = []
    args = node.args

    def add(arg: ast.arg, default: ast.AST | None) -> None:
        """Return add.

        Parameters
        ----------
        arg : ast.arg
            Description for ``arg``.
        default : ast.AST | None
            Description for ``default``.
        """
        name = arg.arg
        if name in {"self", "cls"}:
            return
        annotation = annotation_to_text(arg.annotation)
        is_optional = default is not None
        if is_optional and annotation.endswith(" | None"):
            annotation = annotation[: -len(" | None")]
        if is_optional:
            cleaned = annotation or "Any"
            annotation = f"{cleaned} | None"
        params.append((name, annotation or "Any"))

    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults, strict=True):
        add(arg, default)

    if args.vararg:
        params.append((f"*{args.vararg.arg}", "Any"))

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        add(arg, default)

    if args.kwarg:
        params.append((f"**{args.kwarg.arg}", "Any"))

    return params


def detect_raises(node: ast.AST) -> list[str]:
    """Return detect raises.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.

    Returns
    -------
    List[str]
        Description of return value.
    """
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
    """Return build examples.

    Parameters
    ----------
    module_name : str
        Description for ``module_name``.
    name : str
        Description for ``name``.
    parameters : List[Tuple[str, str]]
        Description for ``parameters``.
    has_return : bool
        Description for ``has_return``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    lines: list[str] = ["Examples", "--------"]
    if module_name and not name.startswith("__"):
        lines.append(f">>> from {module_name} import {name}")
    call_args = ["..."] * sum(1 for param, _ in parameters if not param.startswith("*"))
    invocation = f"{name}({', '.join(call_args)})" if call_args else f"{name}()"
    if not name.startswith("__"):
        if has_return:
            lines.append(f">>> result = {invocation}")
            lines.append(">>> result  # doctest: +ELLIPSIS")
            lines.append("...")
        else:
            lines.append(f">>> {invocation}  # doctest: +ELLIPSIS")
    return lines


def build_docstring(kind: str, node: ast.AST, module_name: str) -> list[str]:
    """Return build docstring.

    Parameters
    ----------
    kind : str
        Description for ``kind``.
    node : ast.AST
        Description for ``node``.
    module_name : str
        Description for ``module_name``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    if kind == "module":
        module_display = module_name.split(".")[-1] if module_name else "module"
        summary = summarize(module_display, kind)
        extended = extended_summary(kind, module_display, module_name)
    else:
        object_name = getattr(node, "name", "value")
        summary = summarize(object_name, kind)
        extended = extended_summary(kind, object_name, module_name)

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
    if extended:
        lines.extend(["", extended])

    if kind == "module":
        lines.append('"""')
        return lines

    if kind == "class" and isinstance(node, ast.ClassDef):
        lines.append('"""')
        return lines

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

    lines.append('"""')
    return lines


def _required_sections(
    kind: str,
    parameters: list[tuple[str, str]],
    returns: str | None,
    raises: list[str],
) -> set[str]:
    """Return required sections.

    Parameters
    ----------
    kind : str
        Description for ``kind``.
    parameters : List[Tuple[str, str]]
        Description for ``parameters``.
    returns : str | None
        Description for ``returns``.
    raises : List[str]
        Description for ``raises``.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    if kind in {"module", "class"}:
        return set()
    required: set[str] = {"Examples"}
    if parameters:
        required.add("Parameters")
    if returns:
        required.add("Returns")
    if raises:
        required.add("Raises")
    return required


def docstring_text(node: ast.AST) -> tuple[str | None, ast.Expr | None]:
    """Return docstring text.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.

    Returns
    -------
    Tuple[str | None, ast.Expr | None]
        Description of return value.
    """
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
    """Return replace.

    Parameters
    ----------
    doc_expr : ast.Expr | None
        Description for ``doc_expr``.
    lines : List[str]
        Description for ``lines``.
    new_lines : List[str]
        Description for ``new_lines``.
    indent : str
        Description for ``indent``.
    insert_at : int
        Description for ``insert_at``.
    """
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
    """Return process file.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.
    """
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
        if not needs_update and doc:
            boilerplate_tokens = (
                "Provide utilities for module.",
                "This module exposes the primary interfaces for the package.",
                "See Also\n--------\n",
                "Attributes\n----------\nNone",
                "Method description.",
                "Provide usage considerations, constraints, or complexity notes.",
            )
            if any(token in doc for token in boilerplate_tokens):
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
    """Return main."""
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
