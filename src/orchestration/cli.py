"""Orchestration CLI commands for kgfoundry stack.

This module provides command-line interface for building indexes (BM25, FAISS)
and running end-to-end pipelines using Prefect orchestration.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from uuid import uuid4

import typer
from jsonschema import Draft202012Validator, ValidationError

from kgfoundry.embeddings_sparse.bm25 import get_bm25
from kgfoundry_common.errors import IndexBuildError
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import ProblemDetails, build_problem_details, render_problem
from kgfoundry_common.schema_helpers import load_schema
from kgfoundry_common.types import JsonValue
from kgfoundry_common.vector_types import (
    VectorBatch,
    VectorValidationError,
    coerce_vector_batch,
)
from orchestration import safe_pickle

if TYPE_CHECKING:
    from collections.abc import Callable

    # Type signature for e2e_flow: takes no args, returns list of strings
    type _E2EFlow = Callable[[], list[str]]

# Runtime: may be None if Prefect not installed
_e2e_flow: _E2EFlow | None = None

with contextlib.suppress(ImportError):
    from orchestration.flows import e2e_flow as _loaded_flow

    _e2e_flow = _loaded_flow


logger = logging.getLogger(__name__)

__all__ = ["api", "e2e", "index_bm25", "index_faiss"]

__navmap__: Final[NavMap] = {
    "title": "orchestration.cli",
    "synopsis": "Prefect command-line entrypoints for orchestration flows",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@orchestration",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@orchestration",
            "stability": "beta",
            "since": "0.1.0",
        }
        for name in __all__
    },
}

app = typer.Typer(help="kgfoundry orchestration CLI")


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
    FileNotFoundError
        If file not found.
    TypeError
        If dataset format is invalid.
    json.JSONDecodeError
        If JSON parsing fails.
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


def _build_bm25_index(config: BM25BuildConfig) -> None:
    """Build BM25 index from configuration.

    Parameters
    ----------
    config : BM25BuildConfig
        Build configuration.

    Raises
    ------
    FileNotFoundError
        If input file missing.
    TypeError
        If dataset is invalid.
    json.JSONDecodeError
        If JSON parsing fails.
    RuntimeError
        If index building fails.
    """
    docs = _load_bm25_documents(config.chunks_path)
    idx = get_bm25(config.backend, config.index_dir, k1=0.9, b=0.4)
    try:
        idx.build(docs)
    except (AttributeError, ValueError, KeyError) as exc:
        msg = f"Failed to build BM25 index: {exc}"
        raise RuntimeError(msg) from exc


_VECTOR_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schema/vector-ingestion/vector-batch.v1.schema.json"
)
_VECTOR_SCHEMA_ID = "https://kgfoundry.dev/schema/vector-ingestion/vector-batch.v1.json"
_VECTOR_PROBLEM_TYPE = "https://kgfoundry.dev/problems/vector-ingestion/invalid-payload"
_VECTOR_SCHEMA_ERROR_LIMIT = 5
_VECTOR_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def _vector_batch_validator() -> Draft202012Validator:
    """Return a cached JSON Schema validator for vector ingestion payloads."""
    validator = _VECTOR_VALIDATOR_CACHE.get("validator")
    if validator is None:
        schema = load_schema(_VECTOR_SCHEMA_PATH)
        validator = Draft202012Validator(cast(dict[str, object], schema))
        _VECTOR_VALIDATOR_CACHE["validator"] = validator
    return validator


def _error_sort_key(error: ValidationError) -> tuple[str, ...]:
    """Build a sortable key for JSON Schema validation errors."""
    return tuple(str(part) for part in error.path)


def _validate_vector_payload(payload: object) -> None:
    """Validate vector ingestion payloads against the canonical schema."""
    validator = _vector_batch_validator()
    errors = sorted(validator.iter_errors(payload), key=_error_sort_key)
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
    """Create Problem Details payload for vector ingestion failures."""
    problem_extensions: dict[str, JsonValue] = {
        "schema_id": _VECTOR_SCHEMA_ID,
        "vector_path": vector_path,
        "correlation_id": correlation_id,
        "validation_errors": list(errors),
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
    """Load and validate dense vectors from JSON file."""
    vectors_file = Path(vectors_path)
    if not vectors_file.exists():
        msg = f"Vectors file not found: {vectors_path}"
        raise FileNotFoundError(msg)

    with vectors_file.open("r", encoding="utf-8") as file_obj:
        payload: object = json.load(file_obj)

    if not isinstance(payload, Sequence):
        msg = "Dense vectors payload must be a sequence of mapping objects"
        raise TypeError(msg)

    _validate_vector_payload(payload)
    records = cast(Iterable[Mapping[str, object]], payload)
    return coerce_vector_batch(records)


def _prepare_index_directory(index_path: str) -> None:
    """Create index output directory.

    Parameters
    ----------
    index_path : str
        Path where index will be written.

    Raises
    ------
    OSError
        If directory creation fails.
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)


# [nav:anchor index_bm25]
def index_bm25(
    chunks_parquet: str = typer.Argument(..., help="Path to Parquet/JSONL with chunks"),
    backend: str = typer.Option("lucene", help="lucene|pure"),
    index_dir: str = typer.Option("./_indices/bm25", help="Output index directory"),
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
    """
    config = BM25BuildConfig(
        chunks_path=chunks_parquet,
        backend=backend,
        index_dir=index_dir,
    )

    try:
        _prepare_index_directory(config.index_dir)
        _build_bm25_index(config)
        typer.echo(f"BM25 index built at {config.index_dir} using backend={config.backend}")
    except (TypeError, json.JSONDecodeError, FileNotFoundError) as exc:
        logger.exception(
            "Document loading failed",
            extra={"operation": "index_bm25", "error": type(exc).__name__},
        )
        typer.echo(f"Error loading documents: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        logger.exception(
            "BM25 index build failed",
            extra={"operation": "index_bm25", "error": type(exc).__name__},
        )
        typer.echo(f"Error building index: {exc}", err=True)
        raise typer.Exit(code=1) from exc


# [nav:anchor index_faiss]
def index_faiss(
    dense_vectors: str = typer.Argument(..., help="Path to dense vectors JSON (skeleton)"),
    index_path: str = typer.Option(
        "./_indices/faiss/shard_000.idx", help="Output index (CPU .idx)"
    ),
    factory: str = typer.Option("Flat", help="FAISS factory string"),
    metric: str = typer.Option("ip", help="Metric: 'ip' or 'l2'"),
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

    Raises
    ------
    typer.Exit
        On any error with exit code 1.

    Examples
    --------
    Build Flat CPU index for testing::

        kgfoundry orchestration cli index_faiss vectors.json

    Build quantized GPU index::

        kgfoundry orchestration cli index_faiss vectors.json \\
            --factory "OPQ64,IVF8192,PQ64"
    """
    config = FaissIndexConfig(
        dense_vectors=dense_vectors,
        index_path=index_path,
        factory=factory,
        metric=metric,
    )

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

        index_data = {
            "keys": list(batch.ids),
            "vectors": batch.matrix,
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


# [nav:anchor api]
def api(port: int = 8080) -> None:
    """Start FastAPI search service.

    Parameters
    ----------
    port : int, optional
        Port to bind to. Defaults to 8080.
    """
    try:
        uvicorn_module = importlib.import_module("uvicorn")
    except ImportError as exc:
        typer.echo(
            "uvicorn is required to run the API server",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    uvicorn_module.run("search_api.app:app", host="127.0.0.1", port=port, reload=False)


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


app.command()(index_bm25)
app.command()(index_faiss)
app.command()(api)
app.command()(e2e)


if __name__ == "__main__":
    app()
