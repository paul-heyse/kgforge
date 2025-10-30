from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import os
import pathlib
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kgfoundry_common.gpu import has_gpu_stack
from kgfoundry_common.parquet_io import ParquetChunkWriter, ParquetVectorWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"

pytest_plugins = ["tests.plugins.pytest_requires"]


warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy\w* has no __module__ attribute",
    category=DeprecationWarning,
    module=r"faiss.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
    module=r"faiss.*",
)

_swig_warning_rules = [
    "ignore:builtin type SwigPy",
    "ignore:builtin type swigvarlink",
]
current_warnings = os.environ.get("PYTHONWARNINGS")
if current_warnings:
    extras = [rule for rule in _swig_warning_rules if rule not in current_warnings]
    if extras:
        os.environ["PYTHONWARNINGS"] = ",".join([current_warnings, *extras])
else:
    os.environ["PYTHONWARNINGS"] = ",".join(_swig_warning_rules)


HAS_GPU_STACK: bool = has_gpu_stack()


def _resolve_prefect_logging() -> tuple[Callable[..., None], type[logging.Handler]] | None:
    try:
        from prefect.logging.configuration import setup_logging
    except Exception:  # pragma: no cover - prefect optional
        return None

    try:
        handlers_mod = importlib.import_module("prefect.logging.handlers")
    except Exception:  # pragma: no cover - prefect optional
        return None

    handler_cls = getattr(
        handlers_mod,
        "RichConsoleHandler",
        getattr(handlers_mod, "PrefectConsoleHandler", None),
    )
    if handler_cls is None:
        return None

    return setup_logging, cast(type[logging.Handler], handler_cls)


def _configure_prefect_logging() -> None:
    """Disable Prefect's rich console handlers to avoid closed-file errors in tests."""
    quiet_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "null": {
                "class": "logging.NullHandler",
            }
        },
        "root": {"handlers": ["null"], "level": "WARNING"},
        "loggers": {
            "prefect": {"handlers": ["null"], "propagate": False},
            "prefect.server": {"handlers": ["null"], "propagate": False},
        },
    }
    logging_config_path = pathlib.Path(tempfile.gettempdir()) / "prefect_logging_quiet.json"
    if not logging_config_path.exists():
        logging_config_path.write_text(json.dumps(quiet_config), encoding="utf-8")
    os.environ.setdefault("PREFECT_LOGGING_SETTINGS_PATH", str(logging_config_path))
    os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "")

    resolution = _resolve_prefect_logging()
    if resolution is None:
        return
    setup_logging, rich_handler_type = resolution

    setup_logging(incremental=False)

    if not hasattr(rich_handler_type, "_kgf_emit_patched"):
        original_emit = rich_handler_type.emit

        def safe_emit(self: logging.Handler, record: logging.LogRecord) -> None:
            try:
                original_emit(self, record)
            except ValueError:
                logging.getLogger(__name__).debug(
                    "Suppressed Prefect RichConsoleHandler ValueError during teardown",
                    exc_info=True,
                )

        rich_handler_type.emit = safe_emit  # type: ignore[method-assign]
        cast("Any", rich_handler_type)._kgf_emit_patched = True

    logger_names = (
        "prefect",
        "prefect.server",
        "prefect.server.api",
        "prefect.server.api.server",
    )
    for name in logger_names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, rich_handler_type):
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(logging.ERROR)
    os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")





def require_modules(
    modules: Iterable[str],
    *,
    minversions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """
    Import requested modules or skip tests when unavailable.

    Parameters
    ----------
    modules : Iterable[str]
        Module names to load.
    minversions : Mapping[str, str], optional
        Optional minimum versions keyed by module name.

    Returns
    -------
    dict[str, Any]
        Mapping of module names to the imported modules.
    """
    minversions = minversions or {}
    loaded: dict[str, Any] = {}
    for name in modules:
        min_version = minversions.get(name)
        reason = f"Requires {name}{f' >= {min_version}' if min_version else ''}"
        loaded[name] = pytest.importorskip(name, minversion=min_version, reason=reason)
    return loaded


def _write_table(path: pathlib.Path, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _ensure_chunks() -> None:
    target = FIXTURES / "chunks.parquet"
    if target.exists() and target.stat().st_size > 0:
        return
    source = FIXTURES / "chunks_fixture.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    now = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(data):
        base_text = f"{entry['text']} concept"
        tokens = len(base_text.split())
        span = {"node_id": f"node-{idx + 1}", "start": 0, "end": len(base_text)}
        rows.append(
            {
                "chunk_id": entry["chunk_id"],
                "doc_id": f"urn:doc:fixture:{idx:04d}",
                "section": entry["section"],
                "start_char": 0,
                "end_char": len(base_text),
                "doctags_span": span,
                "text": base_text,
                "tokens": tokens,
                "created_at": now,
            }
        )
    _write_table(target, ParquetChunkWriter.chunk_schema(), rows)


def _ensure_dense() -> None:
    target = FIXTURES / "dense_qwen3.parquet"
    if target.exists() and target.stat().st_size > 0:
        return
    data = json.loads((FIXTURES / "dense_vectors.json").read_text(encoding="utf-8"))
    dim = len(data[0]["vector"])
    schema = ParquetVectorWriter.dense_schema(dim)
    now = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    rows: list[dict[str, Any]] = []
    for entry in data:
        vec = np.asarray(entry["vector"], dtype=np.float32)
        l2_norm = float(np.linalg.norm(vec))
        rows.append(
            {
                "chunk_id": entry["key"],
                "model": "Qwen3-Embedding-4B",
                "run_id": "fixture",
                "dim": dim,
                "vector": vec.tolist(),
                "l2_norm": l2_norm if l2_norm else 1.0,
                "created_at": now,
            }
        )
    _write_table(target, schema, rows)


def _ensure_sparse() -> None:
    target = FIXTURES / "sparse_splade.parquet"
    if target.exists() and target.stat().st_size > 0:
        return
    rows = [
        {
            "chunk_id": "chunk:1",
            "model": "SPLADE-v3-distilbert",
            "run_id": "fixture",
            "vocab_ids": [1, 7, 42],
            "weights": [0.3, 0.2, 0.1],
            "nnz": 3,
            "created_at": dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        }
    ]
    _write_table(target, ParquetVectorWriter.splade_schema(), rows)


def _ensure_fixture_config() -> None:
    config_path = FIXTURES / "config.fixture.yaml"
    if not config_path.exists():
        indices_root = FIXTURES / "_indices"
        (indices_root / "bm25").mkdir(parents=True, exist_ok=True)
        (indices_root / "splade_impact").mkdir(parents=True, exist_ok=True)
        (indices_root / "faiss").mkdir(parents=True, exist_ok=True)
        cfg = {
            "system": {
                "parquet_root": str(FIXTURES),
                "duckdb_path": str(FIXTURES / "catalog.duckdb"),
            },
            "search": {
                "sparse_backend": "pure",
                "k": 5,
                "dense_candidates": 50,
                "sparse_candidates": 50,
                "rrf_k": 20,
                "kg_boosts": {"direct": 0.1, "one_hop": 0.05},
            },
            "sparse_embedding": {
                "bm25": {"index_dir": str(indices_root / "bm25"), "k1": 0.9, "b": 0.4},
                "splade": {"index_dir": str(indices_root / "splade_impact"), "topk": 256},
            },
            "faiss": {
                "gpu": False,
                "cuvs": False,
                "index_factory": "Flat",
                "nprobe": 1,
            },
        }
        import yaml

        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ.setdefault("KGF_CONFIG", str(config_path))


_ensure_chunks()
_ensure_dense()
_ensure_sparse()
_ensure_fixture_config()
