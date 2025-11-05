"""Orchestration CLI commands for kgfoundry stack.

This module provides command-line interface for building indexes (BM25, FAISS) and running end-to-
end pipelines using Prefect orchestration.
"""
# [nav:section public-api]

from __future__ import annotations

import contextlib
import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, cast
from uuid import uuid4

import typer

from kgfoundry.embeddings_sparse.bm25 import get_bm25
from kgfoundry_common.errors import ConfigurationError, IndexBuildError
from kgfoundry_common.jsonschema_utils import (
    create_draft202012_validator,
)
from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.problem_details import (
    build_configuration_problem,
    build_problem_details,
    render_problem,
)
from kgfoundry_common.schema_helpers import load_schema
from kgfoundry_common.vector_types import (
    VectorValidationError,
    coerce_vector_batch,
)
from orchestration import safe_pickle
from orchestration.config import IndexCliConfig

if TYPE_CHECKING:
    # Type signature for e2e_flow: takes no args, returns list of strings
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from kgfoundry_common.jsonschema_utils import (
        Draft202012ValidatorProtocol,
        ValidationErrorProtocol,
    )
    from kgfoundry_common.problem_details import ProblemDetails
    from kgfoundry_common.types import JsonValue
    from kgfoundry_common.vector_types import (
        VectorBatch,
    )

    type _E2EFlow = Callable[[], list[str]]


class _UvicornRun(Protocol):
    def __call__(self, app: str, *, host: str, port: int, reload: bool = False) -> None:
        """Run uvicorn server.

        Parameters
        ----------
        app : str
            Application module path.
        host : str
            Host to bind to.
        port : int
            Port to bind to.
        reload : bool, optional
            Whether to enable auto-reload.
            Defaults to False.
        """
        del self, app, host, port, reload
        raise NotImplementedError


# Runtime: may be None if Prefect not installed
_e2e_flow: _E2EFlow | None = None

with contextlib.suppress(ImportError):
    from orchestration.flows import e2e_flow as _loaded_flow

    _e2e_flow = _loaded_flow


logger = logging.getLogger(__name__)

__all__ = [
    "api",
    "e2e",
    "index_bm25",
    "index_faiss",
    "run_index_faiss",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


app = typer.Typer(help="kgfoundry orchestration CLI")


class _BM25Builder(Protocol):
    def build(self, docs: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Build BM25 index from documents.

        Parameters
        ----------
        docs : Iterable[tuple[str, dict[str, str]]]
            Iterable of (document_id, document_fields) tuples.
        """
        ...


@dataclass(frozen=True)
class BM25BuildConfig:
    """Configuration for BM25 index building."""

    chunks_path: str
    backend: str
    index_dir: str


@dataclass(frozen=True)
class FaissIndexConfig:
    """Configuration for FAISS index building."""

    dense_vectors: str
    index_path: str
    factory: str
    metric: str


def _extract_bm25_document(record: Mapping[str, object]) -> tuple[str, dict[str, str]] | None:
    """Extract BM25 document from a record mapping.

    Parameters
    ----------
    record : Mapping[str, object]
        Record to extract from.

    Returns
    -------
    tuple[str, dict[str, str]] | None
        (chunk_id, fields_dict) or None if record is invalid.
    """
    chunk_id = record.get("chunk_id")
    if not isinstance(chunk_id, str):
        return None
    title = record.get("title")
    section = record.get("section")
    text = record.get("text")
    return (
        chunk_id,
        {
            "title": title if isinstance(title, str) else "",
            "section": section if isinstance(section, str) else "",
            "body": text if isinstance(text, str) else "",
        },
    )


def _load_bm25_documents(chunks_path: str) -> list[tuple[str, dict[str, str]]]:
    """Load BM25 documents from Parquet or JSONL file.

    Parameters
    ----------
    chunks_path : str
        Path to Parquet or JSONL file.

    Returns
    -------
    list[tuple[str, dict[str, str]]]
        List of (chunk_id, fields_dict) tuples.

    Raises
    ------
    TypeError
        If dataset format is invalid.

    Notes
    -----
    Propagates :class:`FileNotFoundError` when the input path is missing and
    :class:`json.JSONDecodeError` for malformed JSON payloads in non-JSONL
    inputs. Invalid JSON lines in JSONL inputs are logged and skipped.
    """
    docs: list[tuple[str, dict[str, str]]] = []

    if chunks_path.endswith(".jsonl"):
        with Path(chunks_path).open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec_obj: object = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON line: %s", exc)
                    continue
                if isinstance(rec_obj, Mapping):
                    document = _extract_bm25_document(rec_obj)
                    if document:
                        docs.append(document)
    else:
        with Path(chunks_path).open(encoding="utf-8") as fh:
            raw_data: object = json.load(fh)
        if not isinstance(raw_data, Sequence) or isinstance(raw_data, (str, bytes)):
            msg = "Chunk dataset must be a sequence of mapping objects"
            raise TypeError(msg)
        for rec_obj in raw_data:
            if isinstance(rec_obj, Mapping):
                document = _extract_bm25_document(rec_obj)
                if document:
                    docs.append(document)

    return docs


def _get_bm25_index_path(index_dir: Path, backend: str) -> Path:
    """Determine BM25 index file path based on backend.

    Parameters
    ----------
    index_dir : Path
        Base index directory.
    backend : str
        Backend type: "lucene" or "pure".

    Returns
    -------
    Path
        Expected index file path.
    """
    return index_dir / "pure_bm25.pkl" if backend == "pure" else index_dir / "bm25_index"


def _build_bm25_index(config: BM25BuildConfig) -> str:
    """Build BM25 index from configuration.

    Parameters
    ----------
    config : BM25BuildConfig
        Build configuration.

    Raises
    ------
    RuntimeError
        If index building fails after attempting both backends.

    Notes
    -----
    Propagates :class:`FileNotFoundError`, :class:`TypeError`, and
    :class:`json.JSONDecodeError` from dataset loading helpers when the source
    data is invalid.

    Returns
    -------
    str
        Backend identifier that successfully produced the index.
    """
    docs = _load_bm25_documents(config.chunks_path)
    builder, backend_used = _instantiate_bm25_builder(config)

    try:
        builder.build(docs)
    except RuntimeError as exc:
        if backend_used != "lucene":
            raise
        logger.warning(
            "Lucene backend failed during build; retrying with pure backend",
            extra={
                "operation": "index_bm25",
                "error": type(exc).__name__,
                "backend": backend_used,
                "fallback_backend": "pure",
                "phase": "build",
            },
            exc_info=exc,
        )
        fallback = cast(
            "_BM25Builder",
            get_bm25("pure", config.index_dir, k1=0.9, b=0.4, load_existing=False),
        )
        try:
            fallback.build(docs)
        except Exception as fallback_exc:  # pragma: no cover - defensive fallback path
            msg = "Failed to build BM25 index with fallback backend"
            raise RuntimeError(msg) from fallback_exc
        backend_used = "pure"
    except (AttributeError, ValueError, KeyError) as exc:
        msg = f"Failed to build BM25 index: {exc}"
        raise RuntimeError(msg) from exc

    return backend_used


def _instantiate_bm25_builder(config: BM25BuildConfig) -> tuple[_BM25Builder, str]:
    """Instantiate a BM25 builder, falling back to pure backend if Lucene fails.

    Parameters
    ----------
    config : BM25BuildConfig
        Build configuration.

    Returns
    -------
    tuple[_BM25Builder, str]
        (builder instance, backend identifier).

    Raises
    ------
    RuntimeError
        If builder instantiation fails and fallback is unavailable.
    """
    requested_backend = config.backend.strip().lower()
    try:
        builder = cast(
            "_BM25Builder",
            get_bm25(
                requested_backend,
                config.index_dir,
                k1=0.9,
                b=0.4,
                load_existing=False,
            ),
        )
    except RuntimeError as exc:
        if requested_backend != "lucene":
            raise
        logger.warning(
            "Lucene backend unavailable during instantiation; using pure backend",
            extra={
                "operation": "index_bm25",
                "error": type(exc).__name__,
                "backend": requested_backend,
                "fallback_backend": "pure",
                "phase": "instantiate",
            },
            exc_info=exc,
        )
        fallback_builder = cast(
            "_BM25Builder",
            get_bm25("pure", config.index_dir, k1=0.9, b=0.4, load_existing=False),
        )
        return fallback_builder, "pure"
    else:
        return builder, requested_backend


_VECTOR_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schema/vector-ingestion/vector-batch.v1.schema.json"
)
_VECTOR_SCHEMA_ID = "https://kgfoundry.dev/schema/vector-ingestion/vector-batch.v1.json"
_VECTOR_PROBLEM_TYPE = "https://kgfoundry.dev/problems/vector-ingestion/invalid-payload"
_VECTOR_SCHEMA_ERROR_LIMIT = 5
_VECTOR_VALIDATOR_CACHE: dict[str, Draft202012ValidatorProtocol] = {}


def _vector_batch_validator() -> Draft202012ValidatorProtocol:
    """Return a cached JSON Schema validator for vector ingestion payloads.

    Returns
    -------
    Draft202012ValidatorProtocol
        Cached validator instance.
    """
    validator = _VECTOR_VALIDATOR_CACHE.get("validator")
    if validator is None:
        schema = load_schema(_VECTOR_SCHEMA_PATH)
        validator = create_draft202012_validator(cast("dict[str, object]", schema))
        _VECTOR_VALIDATOR_CACHE["validator"] = validator
    return validator


def _error_sort_key(error: ValidationErrorProtocol) -> tuple[str, ...]:
    """Build a sortable key for JSON Schema validation errors.

    Parameters
    ----------
    error : ValidationErrorProtocol
        Validation error to extract path from.

    Returns
    -------
    tuple[str, ...]
        Sortable tuple of path components.
    """
    return tuple(str(part) for part in error.path)


def _validate_vector_payload(payload: object) -> None:
    """Validate vector ingestion payloads against the canonical schema.

    Parameters
    ----------
    payload : object
        Payload to validate.

    Raises
    ------
    VectorValidationError
        If validation fails.
    """
    validator = _vector_batch_validator()
    errors_iter = validator.iter_errors(payload)
    errors = sorted(errors_iter, key=_error_sort_key)
    if not errors:
        return

    messages: list[str] = []
    for error in errors[:_VECTOR_SCHEMA_ERROR_LIMIT]:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")

    if len(errors) > _VECTOR_SCHEMA_ERROR_LIMIT:
        remaining = len(errors) - _VECTOR_SCHEMA_ERROR_LIMIT
        messages.append(f"... {remaining} additional validation errors")

    raise VectorValidationError(messages[0], errors=messages)


def _build_vector_problem_details(
    *,
    detail: str,
    correlation_id: str,
    vector_path: str,
    errors: Sequence[str],
    instance: str,
) -> ProblemDetails:
    """Create Problem Details payload for vector ingestion failures.

    Parameters
    ----------
    detail : str
        Human-readable error detail.
    correlation_id : str
        Correlation identifier for tracing.
    vector_path : str
        Path to the vector file that failed.
    errors : Sequence[str]
        Validation error messages.
    instance : str
        Problem instance URI.

    Returns
    -------
    ProblemDetails
        RFC 9457 Problem Details payload.
    """
    validation_errors = cast("list[JsonValue]", list(errors))
    errors_payload: dict[str, JsonValue] = {
        "schema_id": _VECTOR_SCHEMA_ID,
        "vector_path": vector_path,
        "validation_errors": validation_errors,
    }
    problem_extensions: dict[str, JsonValue] = {
        "correlation_id": correlation_id,
        "errors": errors_payload,
    }

    return build_problem_details(
        problem_type=_VECTOR_PROBLEM_TYPE,
        title="Vector payload failed validation",
        status=422,
        detail=detail,
        instance=instance,
        extensions=problem_extensions,
    )


def load_vector_batch_from_json(vectors_path: str) -> VectorBatch:
    """Load and validate dense vectors from JSON file.

    Parameters
    ----------
    vectors_path : str
        Path to JSON file containing vectors.

    Returns
    -------
    VectorBatch
        Validated vector batch.

    Raises
    ------
    FileNotFoundError
        If the vectors file does not exist.
    VectorValidationError
        If payload validation fails.
    """
    vectors_file = Path(vectors_path)
    if not vectors_file.exists():
        msg = f"Vectors file not found: {vectors_path}"
        raise FileNotFoundError(msg)

    with vectors_file.open("r", encoding="utf-8") as file_obj:
        payload: object = json.load(file_obj)

    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        msg = "Dense vectors payload must be a sequence of mapping objects"
        raise VectorValidationError(msg, errors=[msg])

    _validate_vector_payload(payload)
    records = cast("Iterable[Mapping[str, object]]", payload)
    return coerce_vector_batch(records)


def _prepare_index_directory(index_path: str) -> None:
    """Create index output directory.

    Parameters
    ----------
    index_path : str
        Path where index will be written.

    Notes
    -----
    Propagates :class:`OSError` when directory creation fails.
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)


# [nav:anchor index_bm25]
def index_bm25(
    chunks_parquet: Annotated[str, typer.Argument(..., help="Path to Parquet/JSONL with chunks")],
    backend: Annotated[str, typer.Option(help="lucene|pure")] = "lucene",
    index_dir: Annotated[str, typer.Option(help="Output index directory")] = "./_indices/bm25",
) -> None:
    """Build BM25 index from chunk data.

    This command builds a BM25 index from chunk data. The operation is
    **idempotent**: if an index already exists at the output directory, it
    will be rebuilt from scratch. No side effects occur beyond writing
    the index files.

    Parameters
    ----------
    chunks_parquet : str
        Path to Parquet or JSONL file containing chunks.
    backend : str, optional
        Backend to use: "lucene" or "pure". Defaults to "lucene".
    index_dir : str, optional
        Output directory for the index. Defaults to "./_indices/bm25".

    Notes
    -----
    - **Idempotency**: Running twice with identical inputs rebuilds the index.
    - **Retries**: No automatic retries. On failure, check logs and re-run.

    Raises
    ------
    FileNotFoundError
        If the chunk dataset cannot be located.
    TypeError
        If the dataset format is invalid.
    json.JSONDecodeError
        If JSON chunk payloads are malformed.
    typer.Exit
        If index building fails (exit code 1).
    """
    config = BM25BuildConfig(
        chunks_path=chunks_parquet,
        backend=backend,
        index_dir=index_dir,
    )

    try:
        _prepare_index_directory(config.index_dir)
        logger.info(
            "Building BM25 index",
            extra={
                "operation": "index_bm25",
                "backend": backend,
                "chunks_path": chunks_parquet,
            },
        )
        backend_used = _build_bm25_index(config)
        typer.echo(f"BM25 index built at {config.index_dir} using backend={backend_used}")
        logger.info(
            "BM25 index build completed",
            extra={
                "operation": "index_bm25",
                "backend": backend_used,
                "path": index_dir,
                "chunks_path": chunks_parquet,
            },
        )
    except FileNotFoundError as exc:
        logger.exception(
            "Document loading failed",
            extra={"operation": "index_bm25", "error": type(exc).__name__},
        )
        typer.echo(f"Error loading documents: {exc}", err=True)
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        logger.exception(
            "Document loading failed",
            extra={"operation": "index_bm25", "error": type(exc).__name__},
        )
        typer.echo(f"Error loading documents: {exc}", err=True)
        raise
    except RuntimeError as exc:
        logger.exception(
            "BM25 index build failed",
            extra={"operation": "index_bm25", "error": type(exc).__name__},
        )
        typer.echo(f"Error building index: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _index_bm25_cli(
    chunks_parquet: Annotated[str, typer.Argument(..., help="Path to Parquet/JSONL with chunks")],
    backend: Annotated[str, typer.Option(help="lucene|pure", show_default=True)] = "lucene",
    index_dir: Annotated[
        str,
        typer.Option(help="Output index directory", show_default=True),
    ] = "./_indices/bm25",
) -> None:
    index_bm25(chunks_parquet, backend, index_dir)


# [nav:anchor index_faiss]
# [nav:anchor run_index_faiss]
def run_index_faiss(*, config: IndexCliConfig) -> None:
    """Build FAISS index from dense vectors using typed configuration.

    This function builds a FAISS index from dense vector data with structured
    observability and comprehensive error handling. The operation is **idempotent**:
    if an index already exists at the output path, it will be rebuilt from scratch.

    Parameters
    ----------
    config : IndexCliConfig
        Typed configuration with dense_vectors path, index_path, factory string,
        and metric type.

    Raises
    ------
    typer.Exit
        On any error with exit code 1. Errors are logged with correlation ID
        and rendered as RFC 9457 Problem Details JSON to stderr.

    Notes
    -----
    - **Idempotency**: Running twice with identical inputs rebuilds the index.
    - **Retries**: No automatic retries. On failure, check logs and re-run.
    - **GPU Fallback**: If GPU unavailable, CPU index is built automatically.
    - **Error Handling**: Configuration errors are rendered as Problem Details with
      correlation IDs for observability and debugging.

    Examples
    --------
    Build Flat CPU index for testing:

    >>> from orchestration.config import IndexCliConfig
    >>> config = IndexCliConfig(
    ...     dense_vectors="vectors.json",
    ...     index_path="./_indices/faiss/shard_000.idx",
    ...     factory="Flat",
    ...     metric="ip",
    ... )
    >>> # run_index_faiss(config=config)

    Build quantized GPU index:

    >>> config = IndexCliConfig(
    ...     dense_vectors="vectors.json",
    ...     index_path="./_indices/faiss/shard_000.idx",
    ...     factory="OPQ64,IVF8192,PQ64",
    ...     metric="ip",
    ... )
    >>> # run_index_faiss(config=config)
    """
    correlation_id = uuid4().hex
    instance_uri = f"urn:orchestration:index-faiss:{correlation_id}"

    try:
        _prepare_index_directory(config.index_path)
        batch = load_vector_batch_from_json(config.dense_vectors)

        logger.info(
            "Building FAISS index",
            extra={
                "operation": "index_faiss",
                "factory": config.factory,
                "metric": config.metric,
                "vectors": batch.count,
                "dimension": batch.dimension,
                "correlation_id": correlation_id,
            },
        )

        matrix_rows = cast("list[list[float]]", batch.matrix.tolist())
        vectors_payload: list[list[float]] = [
            [float(component) for component in row] for row in matrix_rows
        ]
        index_data: dict[str, object] = {
            "keys": [str(vector_id) for vector_id in batch.ids],
            "vectors": vectors_payload,
            "factory": config.factory,
            "metric": config.metric,
        }
        with Path(config.index_path).open("wb") as file_obj:
            safe_pickle.dump(index_data, file_obj)

        logger.info(
            "Index saved successfully",
            extra={
                "operation": "index_faiss",
                "path": config.index_path,
                "correlation_id": correlation_id,
            },
        )
        typer.echo(f"FAISS index vectors stored at {config.index_path}")
        logger.info(
            "FAISS index build completed",
            extra={
                "operation": "index_faiss",
                "path": config.index_path,
                "factory": config.factory,
                "metric": config.metric,
                "correlation_id": correlation_id,
            },
        )

    except VectorValidationError as exc:
        logger.exception(
            "Vector validation failed",
            extra={
                "operation": "index_faiss",
                "error": type(exc).__name__,
                "correlation_id": correlation_id,
            },
        )
        problem = _build_vector_problem_details(
            detail=str(exc),
            correlation_id=correlation_id,
            vector_path=config.dense_vectors,
            errors=exc.errors,
            instance=instance_uri,
        )
        typer.echo(render_problem(problem), err=True)
        index_error = IndexBuildError(
            "Vector payload failed validation",
            cause=exc,
            context={"problem": problem, "correlation_id": correlation_id},
        )
        raise typer.Exit(code=1) from index_error

    except ConfigurationError as exc:
        logger.exception(
            "Configuration validation failed",
            extra={
                "operation": "index_faiss",
                "error": type(exc).__name__,
                "correlation_id": correlation_id,
            },
        )
        problem = build_configuration_problem(exc)
        typer.echo(render_problem(problem), err=True)
        raise typer.Exit(code=2) from exc

    except (TypeError, json.JSONDecodeError, FileNotFoundError) as exc:
        logger.exception(
            "Vector loading failed",
            extra={
                "operation": "index_faiss",
                "error": type(exc).__name__,
                "correlation_id": correlation_id,
            },
        )
        typer.echo(f"Error loading vectors: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception(
            "Index save failed",
            extra={
                "operation": "index_faiss",
                "error": type(exc).__name__,
                "correlation_id": correlation_id,
            },
        )
        typer.echo(f"Error saving index: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def index_faiss(
    dense_vectors: Annotated[
        str, typer.Argument(..., help="Path to dense vectors JSON (skeleton)")
    ],
    index_path: Annotated[
        str, typer.Option(help="Output index (CPU .idx)")
    ] = "./_indices/faiss/shard_000.idx",
    factory: Annotated[str, typer.Option(help="FAISS factory string")] = "Flat",
    metric: Annotated[str, typer.Option(help="Metric: 'ip' or 'l2'")] = "ip",
) -> None:
    r"""Build FAISS index from dense vectors.

    This command builds a FAISS index from dense vector data with structured
    observability and comprehensive error handling. The operation is **idempotent**:
    if an index already exists at the output path, it will be rebuilt from scratch.

    Parameters
    ----------
    dense_vectors : str
        Path to JSON file containing vectors in skeleton format.
        Expected format: list of {"key": "id", "vector": [float, ...]} objects.
    index_path : str, optional
        Output path for the index file. Defaults to "./_indices/faiss/shard_000.idx".
    factory : str, optional
        FAISS factory string (e.g., "Flat", "OPQ64,IVF8192,PQ64").
        Defaults to "Flat" for testing; production uses "OPQ64,IVF8192,PQ64".
    metric : str, optional
        Metric type: "ip" (inner product) or "l2" (L2 distance).
        Defaults to "ip".

    Notes
    -----
    - **Idempotency**: Running twice with identical inputs rebuilds the index.
    - **Retries**: No automatic retries. On failure, check logs and re-run.
    - **GPU Fallback**: If GPU unavailable, CPU index is built automatically.

    Notes
    -----
    Propagates :class:`typer.Exit` from :func:`run_index_faiss` when the index
    build fails.

    Examples
    --------
    Build Flat CPU index for testing::

        kgfoundry orchestration cli index_faiss vectors.json

    Build quantized GPU index::

        kgfoundry orchestration cli index_faiss vectors.json \\
            --factory "OPQ64,IVF8192,PQ64"
    """
    config = IndexCliConfig(
        dense_vectors=dense_vectors,
        index_path=index_path,
        factory=factory,
        metric=metric,
    )
    run_index_faiss(config=config)


# [nav:anchor api]
def api(port: int = 8080) -> None:
    """Start FastAPI search service.

    Parameters
    ----------
    port : int, optional
        Port to bind to. Defaults to 8080.

    Raises
    ------
    typer.Exit
        If uvicorn is not available or entry point is missing (exit code 1).
    """
    try:
        uvicorn_module: ModuleType = importlib.import_module("uvicorn")
    except ImportError as exc:
        typer.echo(
            "uvicorn is required to run the API server",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    module_attrs = cast("Mapping[str, object]", vars(uvicorn_module))
    run_attr = module_attrs.get("run")
    if not callable(run_attr):
        typer.echo("uvicorn.run entry point not available", err=True)
        raise typer.Exit(code=1)
    run_server = cast("_UvicornRun", run_attr)
    run_server("search_api.app:app", host="127.0.0.1", port=port, reload=False)


def _run_e2e_flow() -> list[str]:
    """Safely invoke e2e_flow with type narrowing.

    Returns
    -------
    list[str]
        Stages from the e2e flow.

    Raises
    ------
    typer.Exit
        If e2e_flow is not available.
    """
    if _e2e_flow is None:
        raise typer.Exit(code=1)
    return _e2e_flow()  # Now properly typed via _flows_module.e2e_flow


# [nav:anchor e2e]
def e2e() -> None:
    """Execute end-to-end orchestration pipeline.

    This command runs the complete e2e flow using Prefect orchestration.
    Requires Prefect to be installed.

    Raises
    ------
    typer.Exit
        If Prefect is not installed (exit code 1).
    """
    try:
        stages = _run_e2e_flow()
    except typer.Exit:
        typer.echo(
            "Prefect is required for the e2e pipeline command. "
            "Install it via `pip install -e '.[gpu]'` or add `prefect` manually.",
            err=True,
        )
        raise

    for step in stages:
        typer.echo(step)


app.command("index_bm25")(_index_bm25_cli)
app.command("index_faiss")(index_faiss)
app.command()(api)
app.command()(e2e)


if __name__ == "__main__":
    app()
