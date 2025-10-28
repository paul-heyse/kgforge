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


def _humanize_identifier(value: str) -> str:
    """Compute humanize identifier.

    Carry out the humanize identifier operation.

    Parameters
    ----------
    value : str
        Description for ``value``.

    Returns
    -------
    str
        Description of return value.
    """
    cleaned = value.replace("_", " ").strip()
    return " ".join(cleaned.split())


def _is_magic(name: str | None) -> bool:
    """Return whether ``name`` is a Python magic method."""
    return bool(name and name.startswith("__") and name.endswith("__"))


def _is_pydantic_model(node: ast.ClassDef) -> bool:
    """Return whether ``node`` inherits from pydantic BaseModel."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


def _is_pydantic_artifact(name: str | None) -> bool:
    """Return ``True`` when ``name`` refers to a Pydantic helper attribute."""
    if not name:
        return False
    return bool(
        name.startswith("model_")
        or name.startswith("__pydantic_")
        or name in PYDANTIC_ARTIFACT_SUMMARIES
        or name
        in {
            "__class_vars__",
            "__private_attributes__",
            "__signature__",
        }
    )


def _is_dataclass_artifact(name: str | None) -> bool:
    """Return ``True`` when ``name`` matches a dataclass support attribute."""
    if not name:
        return False
    return name in {
        "__dataclass_fields__",
        "__dataclass_params__",
        "__match_args__",
        "asdict",
        "astuple",
        "replace",
    }


MAGIC_METHOD_EXTENDED_SUMMARIES: dict[str, str] = {
    "__repr__": "Return an unambiguous representation useful for debugging.",
    "__str__": "Render a human-readable string describing the instance.",
    "__len__": "Report the number of items stored on the instance.",
    "__iter__": "Yield each item from the instance in order.",
    "__aiter__": "Yield items from the instance asynchronously.",
    "__next__": "Return the next value produced by the iterator.",
    "__anext__": "Return the next value produced by the asynchronous iterator.",
    "__getitem__": "Fetch a value by key or positional index.",
    "__setitem__": "Store a value for the provided key or positional index.",
    "__delitem__": "Remove the value associated with the provided key or index.",
    "__contains__": "Report whether the provided value exists on the instance.",
    "__bool__": "Indicate whether the instance should be treated as truthy.",
    "__eq__": "Determine whether two instances represent the same value.",
    "__ne__": "Determine whether two instances represent different values.",
    "__lt__": "Compare whether this instance is ordered before another.",
    "__le__": "Compare whether this instance is ordered before or equal to another.",
    "__gt__": "Compare whether this instance is ordered after another.",
    "__ge__": "Compare whether this instance is ordered after or equal to another.",
    "__hash__": "Produce the hash used when storing the instance in sets or dictionaries.",
    "__call__": "Invoke the instance as though it were a function.",
    "__enter__": "Enter the runtime context and return the managed resource.",
    "__exit__": "Clean up the runtime context when the block finishes.",
    "__aenter__": "Enter the asynchronous runtime context for the instance.",
    "__aexit__": "Leave the asynchronous runtime context and release resources.",
    "__await__": "Enable awaiting the instance to obtain a result.",
    "__copy__": "Produce a shallow copy of the instance.",
    "__deepcopy__": "Produce a deep copy of the instance and its contents.",
}


PYDANTIC_ARTIFACT_SUMMARIES: dict[str, str] = {
    "model_config": "Configuration options controlling validation and serialisation.",
    "model_fields": "Mapping describing each field declared on the model.",
    "model_computed_fields": "Mapping of computed field descriptors registered on the model.",
    "model_fields_set": "Set of field names provided when the instance was created.",
    "model_extra": "Dictionary containing extra values captured during validation.",
    "model_post_init": "Hook executed once Pydantic completes ``__init__`` validation.",
    "model_rebuild": "Trigger Pydantic to rebuild the cached schema information.",
    "model_parametrized_name": "Return the runtime generated name for parametrised models.",
    "model_dump": "Serialise the model instance into a standard Python mapping.",
    "model_dump_json": "Serialise the model instance into a JSON string.",
    "model_validate": "Construct an instance from validated input data.",
    "model_validate_json": "Parse a JSON string and return a validated model instance.",
    "model_copy": "Return a shallow or deep copy of the model instance.",
    "model_construct": "Instantiate the model without performing validation.",
    "model_serializer": "Return the serializer callable registered for the model.",
    "model_json_schema": "Generate the JSON Schema describing the model structure.",
    "schema": "Generate a JSON-compatible schema describing the model.",
    "schema_json": "Return the model schema as a JSON string.",
    "dict": "Serialise the model into a dictionary of field values.",
    "json": "Serialise the model into a JSON string.",
    "copy": "Return a shallow copy of the model instance.",
    "__pydantic_core_schema__": "Low-level schema object created by Pydantic's core runtime.",
    "__pydantic_core_config__": "Core configuration driving validation for this model.",
    "__pydantic_decorators__": "Registry tracking validators and serializers declared on the model.",
    "__pydantic_extra__": "Container storing extra attributes permitted on the model.",
    "__pydantic_fields_set__": "Set recording which fields were explicitly provided by the caller.",
    "__pydantic_parent_namespace__": "Namespace used by Pydantic to resolve forward references.",
    "__pydantic_generic_metadata__": "Metadata describing how the generic model was specialised.",
    "__pydantic_model_complete__": "Flag indicating whether Pydantic finished configuring the model.",
    "__pydantic_serializer__": "Callable responsible for serialising model instances.",
    "__pydantic_validator__": "Callable responsible for validating model instances.",
    "__pydantic_custom_init__": "Marker showing that the model defines a custom ``__init__``.",
}


QUALIFIED_NAME_OVERRIDES: dict[str, str] = {
    # === Project-specific vector store aliases ===
    "FloatArray": "src.vectorstore_faiss.gpu.FloatArray",
    "IntArray": "src.vectorstore_faiss.gpu.IntArray",
    "StrArray": "src.vectorstore_faiss.gpu.StrArray",
    "VecArray": "src.search_api.faiss_adapter.VecArray",
    # === Project-specific client helpers ===
    "_SupportsHttp": "src.search_client.client._SupportsHttp",
    "_SupportsResponse": "src.search_client.client._SupportsResponse",
    # === Project-specific models ===
    "Chunk": "src.kgfoundry_common.models.Chunk",
    "Concept": "src.ontology.catalog.Concept",
    "ConceptMeta": "src.ontology.catalog.ConceptMeta",
    "DatasetVersion": "src.kgfoundry_common.parquet_io.DatasetVersion",
    "Doc": "src.kgfoundry_common.models.Doc",
    "DoctagsAsset": "src.kgfoundry_common.models.DoctagsAsset",
    "FixtureDoc": "src.search_api.fixture_index.FixtureDoc",
    "Id": "src.kgfoundry_common.models.Id",
    "kgfoundry.kgfoundry_common.models.Chunk": "src.kgfoundry_common.models.Chunk",
    "kgfoundry.kgfoundry_common.models.Concept": "src.ontology.catalog.Concept",
    "kgfoundry.kgfoundry_common.models.Doc": "src.kgfoundry_common.models.Doc",
    "kgfoundry.kgfoundry_common.models.DoctagsAsset": "src.kgfoundry_common.models.DoctagsAsset",
    "kgfoundry.kgfoundry_common.models.LinkAssertion": "src.kgfoundry_common.models.LinkAssertion",
    "LinkAssertion": "src.kgfoundry_common.models.LinkAssertion",
    "NavMap": "src.kgfoundry_common.navmap_types.NavMap",
    "Neo4jStore": "src.kg_builder.neo4j_store.Neo4jStore",
    "ParsedSchema": "src.kgfoundry_common.parquet_io.ParsedSchema",
    "SearchRequest": "src.search_api.schemas.SearchRequest",
    "SearchResult": "src.search_api.schemas.SearchResult",
    "SpladeDoc": "src.search_api.splade_index.SpladeDoc",
    # === Project-specific errors ===
    "DownloadError": "src.kgfoundry_common.errors.DownloadError",
    "UnsupportedMIMEError": "src.kgfoundry_common.errors.UnsupportedMIMEError",
    # === Project-specific embedding abstractions ===
    "DenseEmbeddingModel": "src.embeddings_dense.base.DenseEmbeddingModel",
    "SparseEncoder": "src.embeddings_sparse.base.SparseEncoder",
    "SparseIndex": "src.embeddings_sparse.base.SparseIndex",
    # === Pydantic base models ===
    "BaseModel": "pydantic.BaseModel",
    "pydantic.BaseModel": "pydantic.BaseModel",
    # === NumPy core ndarray and typing utilities ===
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.typing.NDArray",
    "numpy.dtype": "numpy.dtype",
    "numpy.ndarray": "numpy.ndarray",
    "numpy.random.Generator": "numpy.random.Generator",
    "numpy.typing.ArrayLike": "numpy.typing.ArrayLike",
    "numpy.typing.NDArray": "numpy.typing.NDArray",
    "np.dtype": "numpy.dtype",
    "np.ndarray": "numpy.ndarray",
    "np.typing.NDArray": "numpy.typing.NDArray",
    # === NumPy scalar types (fully qualified) ===
    "numpy.complex128": "numpy.complex128",
    "numpy.complex64": "numpy.complex64",
    "numpy.float16": "numpy.float16",
    "numpy.float32": "numpy.float32",
    "numpy.float64": "numpy.float64",
    "numpy.int16": "numpy.int16",
    "numpy.int32": "numpy.int32",
    "numpy.int64": "numpy.int64",
    "numpy.int8": "numpy.int8",
    "numpy.uint16": "numpy.uint16",
    "numpy.uint32": "numpy.uint32",
    "numpy.uint64": "numpy.uint64",
    "numpy.uint8": "numpy.uint8",
    # === NumPy scalar types (short aliases) ===
    "np.complex128": "numpy.complex128",
    "np.complex64": "numpy.complex64",
    "np.float16": "numpy.float16",
    "np.float32": "numpy.float32",
    "np.float64": "numpy.float64",
    "np.int16": "numpy.int16",
    "np.int32": "numpy.int32",
    "np.int64": "numpy.int64",
    "np.int8": "numpy.int8",
    "np.uint16": "numpy.uint16",
    "np.uint32": "numpy.uint32",
    "np.uint64": "numpy.uint64",
    "np.uint8": "numpy.uint8",
    # === PyArrow core types ===
    "pyarrow.Array": "pyarrow.Array",
    "pyarrow.DataType": "pyarrow.DataType",
    "pyarrow.Field": "pyarrow.Field",
    "pyarrow.Int64Type": "pyarrow.Int64Type",
    "pyarrow.RecordBatch": "pyarrow.RecordBatch",
    "pyarrow.Schema": "pyarrow.Schema",
    "pyarrow.StringType": "pyarrow.StringType",
    "pyarrow.Table": "pyarrow.Table",
    "pyarrow.TimestampType": "pyarrow.TimestampType",
    "pyarrow.schema": "pyarrow.schema",
    # === Pydantic helpers and validators ===
    "pydantic.AliasChoices": "pydantic.AliasChoices",
    "pydantic.ConfigDict": "pydantic.ConfigDict",
    "pydantic.Field": "pydantic.Field",
    "pydantic.TypeAdapter": "pydantic.TypeAdapter",
    "pydantic.ValidationError": "pydantic.ValidationError",
    "pydantic.field_validator": "pydantic.field_validator",
    "pydantic.fields.Field": "pydantic.fields.Field",
    "pydantic.model_validator": "pydantic.model_validator",
    # === typing and typing_extensions utilities ===
    "Annotated": "typing.Annotated",
    "Any": "typing.Any",
    "Callable": "collections.abc.Callable",
    "Final": "typing.Final",
    "Iterable": "collections.abc.Iterable",
    "Iterator": "collections.abc.Iterator",
    "Literal": "typing.Literal",
    "Mapping": "collections.abc.Mapping",
    "MutableMapping": "collections.abc.MutableMapping",
    "MutableSequence": "collections.abc.MutableSequence",
    "Optional": "typing.Optional",
    "Sequence": "collections.abc.Sequence",
    "Set": "collections.abc.Set",
    "Type": "typing.Type",
    "typing_extensions.Annotated": "typing_extensions.Annotated",
    "typing_extensions.NotRequired": "typing_extensions.NotRequired",
    "typing_extensions.Self": "typing_extensions.Self",
    "typing_extensions.TypeAlias": "typing_extensions.TypeAlias",
    "typing_extensions.TypedDict": "typing_extensions.TypedDict",
    # === Standard library collections and utilities ===
    "collections.Counter": "collections.Counter",
    "collections.defaultdict": "collections.defaultdict",
    "collections.deque": "collections.deque",
    "collections.OrderedDict": "collections.OrderedDict",
    "datetime.date": "datetime.date",
    "datetime.datetime": "datetime.datetime",
    "datetime.timedelta": "datetime.timedelta",
    "pathlib.Path": "pathlib.Path",
    "pathlib.PurePath": "pathlib.PurePath",
    "uuid.UUID": "uuid.UUID",
    # === External service integrations ===
    "duckdb.DuckDBPyConnection": "duckdb.DuckDBPyConnection",
    "Exit": "typer.Exit",
    "fastapi.HTTPException": "fastapi.HTTPException",
    "HTTPException": "fastapi.HTTPException",
    "typer.Exit": "typer.Exit",
}


def _split_generic_arguments(text: str) -> list[str]:
    """Return a list of comma-separated generic arguments respecting nesting."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_qualified_name(text: str) -> str:
    """Map an annotation string to a fully qualified name when possible."""
    cleaned = text.strip()
    override = QUALIFIED_NAME_OVERRIDES.get(cleaned)
    if override:
        return override
    if "[" in cleaned and cleaned.endswith("]"):
        base, remainder = cleaned.split("[", 1)
        base = base.strip()
        normalized_base = QUALIFIED_NAME_OVERRIDES.get(base, base)
        inner = remainder[:-1]
        arguments = _split_generic_arguments(inner)
        normalized_args = [_normalize_qualified_name(argument) for argument in arguments]
        if QUALIFIED_NAME_OVERRIDES.get(base):
            # Sphinx cannot resolve cross-references that include generic
            # parameters, so we fall back to the canonical base name.
            return normalized_base
        joined = ", ".join(normalized_args) if normalized_args else ""
        if joined:
            return f"{normalized_base}[{joined}]"
        return normalized_base
    return cleaned


def _format_annotation_string(value: str) -> str:
    """Normalise an annotation string for consistent documentation output."""
    text = value.strip().replace("typing.", "")
    replacements = {"list": "List", "dict": "Mapping[str, Any]", "tuple": "Tuple", "set": "Set"}
    if text in replacements:
        text = replacements[text]
    text = text.replace("list[", "List[").replace("tuple[", "Tuple[").replace("set[", "Set[")
    if text.startswith("dict["):
        text = text.replace("dict[", "Mapping[")
    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        inner_text = _normalize_qualified_name(_format_annotation_string(inner))
        return f"Optional {inner_text}"
    text = _normalize_qualified_name(text)
    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        inner_text = _normalize_qualified_name(_format_annotation_string(inner))
        return f"Optional {inner_text}"
    return text


@dataclass
class DocstringChange:
    """Describe DocstringChange."""

    path: Path


def parse_args() -> argparse.Namespace:
    """Compute parse args.

    Carry out the parse args operation.

    Returns
    -------
    argparse.Namespace
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import parse_args
    >>> result = parse_args()
    >>> result  # doctest: +ELLIPSIS
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path, help="Directory to process.")
    parser.add_argument("--log", required=False, type=Path, help="Log file for changed paths.")
    return parser.parse_args()


def module_name_for(path: Path) -> str:
    """Compute module name for.

    Carry out the module name for operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import module_name_for
    >>> result = module_name_for(...)
    >>> result  # doctest: +ELLIPSIS
    """
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        relative = path

    in_src = path.is_relative_to(SRC_ROOT)
    parts = list(relative.with_suffix("").parts)
    if in_src and parts and parts[0] == "src":
        parts = parts[1:]

    module = ".".join(parts)
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]

    return module


def summarize(name: str, kind: str) -> str:
    """Compute summarize.

    Carry out the summarize operation.

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

    Examples
    --------
    >>> from tools.auto_docstrings import summarize
    >>> result = summarize(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    base = _humanize_identifier(name) or "value"
    if kind == "module":
        text = f"Overview of {base}."
    elif kind == "class":
        text = f"Model the {base}."
    elif kind == "function":
        text = f"Compute {base}."
    else:
        text = f"Describe {base}."
    return text if text.endswith(".") else text + "."


def extended_summary(kind: str, name: str, module_name: str, node: ast.AST | None = None) -> str:
    """Compute extended summary.

    Carry out the extended summary operation.

    Parameters
    ----------
    kind : str
        Description for ``kind``.
    name : str
        Description for ``name``.
    module_name : str
        Description for ``module_name``.
    node : ast.AST | None
        Description for ``node``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import extended_summary
    >>> result = extended_summary(..., ..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    pretty = _humanize_identifier(name)
    if kind == "module":
        module_pretty = _humanize_identifier(module_name.split(".")[-1] if module_name else name)
        return (
            f"This module bundles {module_pretty.lower()} logic for the kgfoundry stack."
            if module_pretty
            else "Utility module providing KGFoundry helpers."
        )
    if kind == "class" and isinstance(node, ast.ClassDef):
        if _is_pydantic_model(node):
            return "Pydantic model defining the structured payload used across the system."
        return (
            f"Represents the {pretty.lower()} data structure used throughout the project."
            if pretty
            else "Core data structure used within kgfoundry."
        )
    if _is_pydantic_artifact(name):
        return PYDANTIC_ARTIFACT_SUMMARIES.get(
            name,
            "Internal helper generated by Pydantic for model configuration and validation.",
        )
    if kind == "function" and name == "__init__":
        return "Initialise a new instance with validated parameters."
    if kind == "function" and name in MAGIC_METHOD_EXTENDED_SUMMARIES:
        return MAGIC_METHOD_EXTENDED_SUMMARIES[name]
    if kind == "function" and _is_magic(name):
        return "Special method customising Python's object protocol for this class."
    if kind == "function":
        return (
            f"Carry out the {pretty.lower()} operation."
            if pretty
            else "Perform the requested operation."
        )
    return "Auto-generated reference for project internals."


def annotation_to_text(node: ast.AST | None) -> str:
    """Compute annotation to text.

    Carry out the annotation to text operation.

    Parameters
    ----------
    node : ast.AST | None
        Description for ``node``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import annotation_to_text
    >>> result = annotation_to_text(...)
    >>> result  # doctest: +ELLIPSIS
    """
    if node is None:
        return "Any"
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover
        return "Any"
    return _format_annotation_string(text)


def iter_docstring_nodes(tree: ast.Module) -> list[tuple[int, ast.AST, str]]:
    """Compute iter docstring nodes.

    Carry out the iter docstring nodes operation.

    Parameters
    ----------
    tree : ast.Module
        Description for ``tree``.

    Returns
    -------
    List[Tuple[int, ast.AST, str]]
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import iter_docstring_nodes
    >>> result = iter_docstring_nodes(...)
    >>> result  # doctest: +ELLIPSIS
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
    """Compute parameters for.

    Carry out the parameters for operation.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        Description for ``node``.

    Returns
    -------
    List[Tuple[str, str]]
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import parameters_for
    >>> result = parameters_for(...)
    >>> result  # doctest: +ELLIPSIS
    """
    params: list[tuple[str, str]] = []
    args = node.args

    def add(arg: ast.arg, default: ast.AST | None) -> None:
        """Compute add.

        Carry out the add operation.

        Parameters
        ----------
        arg : ast.arg
            Description for ``arg``.
        default : ast.AST | None
            Description for ``default``.

        Examples
        --------
        >>> from tools.auto_docstrings import add
        >>> add(..., ...)  # doctest: +ELLIPSIS
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
    """Compute detect raises.

    Carry out the detect raises operation.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import detect_raises
    >>> result = detect_raises(...)
    >>> result  # doctest: +ELLIPSIS
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
    """Compute build examples.

    Carry out the build examples operation.

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

    Examples
    --------
    >>> from tools.auto_docstrings import build_examples
    >>> result = build_examples(..., ..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
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
    """Compute build docstring.

    Carry out the build docstring operation.

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

    Examples
    --------
    >>> from tools.auto_docstrings import build_docstring
    >>> result = build_docstring(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    if kind == "module":
        module_display = module_name.split(".")[-1] if module_name else "module"
        summary = summarize(module_display, kind)
        extended = extended_summary(kind, module_display, module_name, node)
    else:
        object_name = getattr(node, "name", "value")
        summary = summarize(object_name, kind)
        extended = extended_summary(kind, object_name, module_name, node)

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

    if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        name = getattr(node, "name", "")
        is_public = bool(name) and not name.startswith("_")
        if is_public:
            example_lines = build_examples(module_name, name, parameters, bool(returns))
            if example_lines:
                lines.extend(["", *example_lines])

    lines.append('"""')
    return lines


def _required_sections(
    kind: str,
    parameters: list[tuple[str, str]],
    returns: str | None,
    raises: list[str],
) -> set[str]:
    """Compute required sections.

    Carry out the required sections operation.

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
    """Compute docstring text.

    Carry out the docstring text operation.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.

    Returns
    -------
    Tuple[str | None, ast.Expr | None]
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import docstring_text
    >>> result = docstring_text(...)
    >>> result  # doctest: +ELLIPSIS
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
    """Compute replace.

    Carry out the replace operation.

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

    Examples
    --------
    >>> from tools.auto_docstrings import replace
    >>> replace(..., ..., ..., ..., ...)  # doctest: +ELLIPSIS
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
    """Compute process file.

    Carry out the process file operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.auto_docstrings import process_file
    >>> result = process_file(...)
    >>> result  # doctest: +ELLIPSIS
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
        node_name = getattr(node, "name", None)
        if kind != "module" and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if (
                node_name
                and node_name.startswith("__")
                and node_name.endswith("__")
                and node_name != "__init__"
            ):
                continue
            if node_name and node_name.startswith("_") and not node_name.startswith("__"):
                continue

        doc, expr = docstring_text(node)
        parameters: list[tuple[str, str]] = []
        returns: str | None = None
        raises: list[str] = []
        if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = parameters_for(node)
            return_annotation: str = annotation_to_text(node.returns)
            if return_annotation not in {"None", "NoReturn"}:
                returns = return_annotation
            raises = detect_raises(node)

        required_sections = _required_sections(kind, parameters, returns, raises)
        needs_update = doc is None or "TODO" in (doc or "") or "NavMap:" in (doc or "")
        if not needs_update and required_sections:
            if doc is None:
                needs_update = True
            else:
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
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                continue
            indent = " " * (node.col_offset + 4)
            new_lines = build_docstring(kind, node, module_name)
            body: list[ast.stmt] = node.body
            insert_at = body[0].lineno - 1 if body else node.lineno
            replace(expr, lines, new_lines, indent, insert_at)
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> None:
    """Compute main.

    Carry out the main operation.

    Examples
    --------
    >>> from tools.auto_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    """
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
