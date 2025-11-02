from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from kgfoundry_common.observability import MetricsProvider

class SearchOptionsPayload(TypedDict, total=False):
    """Typed payload for SearchOptions configuration."""

    alpha: float
    facets: Mapping[str, str]
    candidate_pool: int
    model_loader: Callable[[str], EmbeddingModelProtocol]
    embedding_model: str
    batch_size: int

class SearchDocumentPayload(TypedDict):
    """Typed payload for SearchDocument construction."""

    symbol_id: str
    package: str
    module: str
    qname: str
    kind: str
    stability: str | None
    deprecated: bool
    summary: str | None
    docstring: str | None
    anchor_start: int | None
    anchor_end: int | None
    text: str
    tokens: Any  # collections.Counter[str]
    row: int

@dataclass(slots=True)
class SearchRequest:
    """Request parameters for searching the agent catalog."""

    repo_root: Path
    query: str
    k: int

class EmbeddingModelProtocol(Protocol):
    """Protocol describing the embedding model encode interface."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> Sequence[Sequence[float]]: ...

@dataclass(slots=True)
class SearchOptions:
    """Optional tuning parameters for hybrid search."""

    alpha: float | None = None
    facets: Mapping[str, str] | None = None
    candidate_pool: int | None = None
    model_loader: Callable[[str], EmbeddingModelProtocol] | None = None
    embedding_model: str | None = None
    batch_size: int | None = None

@dataclass(slots=True)
class SearchConfig:
    """Configuration used for hybrid search against the catalog."""

    alpha: float
    candidate_pool: int
    lexical_fields: list[str]

@dataclass(slots=True)
class SearchDocument:
    """Intermediate representation used to build or query the semantic index."""

    symbol_id: str
    package: str
    module: str
    qname: str
    kind: str
    stability: str | None
    deprecated: bool
    summary: str | None
    docstring: str | None
    anchor_start: int | None
    anchor_end: int | None
    text: str
    tokens: Any  # collections.Counter[str]
    row: int

@dataclass(slots=True)
class SearchResult:
    """Result record returned by hybrid search."""

    symbol_id: str
    score: float
    lexical_score: float
    vector_score: float
    package: str
    module: str
    qname: str
    kind: str
    stability: str | None
    deprecated: bool
    summary: str | None
    docstring: str | None
    anchor: dict[str, int | None]

@dataclass(slots=True)
class VectorSearchContext:
    """Supporting data required to execute a vector search."""

    semantic_meta: Mapping[str, Any]
    mapping_payload: Mapping[str, Any]
    index_path: Path
    documents: Sequence[SearchDocument]
    candidate_limit: int
    k: int
    candidate_ids: set[str]
    row_to_document: Mapping[int, SearchDocument]

@dataclass(slots=True)
class PreparedSearchArtifacts:
    """Documents and optional semantic metadata extracted from the catalog."""

    documents: list[SearchDocument]
    semantic_meta: tuple[Mapping[str, Any], Path, Path] | None
    mapping_payload: Mapping[str, Any] | None

def build_default_search_options(
    *,
    alpha: float | None = None,
    candidate_pool: int | None = None,
    batch_size: int | None = None,
    embedding_model: str | None = None,
    model_loader: Callable[[str], EmbeddingModelProtocol] | None = None,
) -> SearchOptions: ...
def build_faceted_search_options(
    *,
    facets: Mapping[str, str],
    alpha: float | None = None,
    candidate_pool: int | None = None,
    batch_size: int | None = None,
    embedding_model: str | None = None,
    model_loader: Callable[[str], EmbeddingModelProtocol] | None = None,
) -> SearchOptions: ...
def build_embedding_aware_search_options(
    *,
    embedding_model: str,
    model_loader: Callable[[str], EmbeddingModelProtocol],
    alpha: float | None = None,
    candidate_pool: int | None = None,
    batch_size: int | None = None,
    facets: Mapping[str, str] | None = None,
) -> SearchOptions: ...
def make_search_document(
    *,
    symbol_id: str,
    package: str,
    module: str,
    qname: str,
    kind: str,
    stability: str | None = None,
    deprecated: bool = False,
    summary: str | None = None,
    docstring: str | None = None,
    anchor_start: int | None = None,
    anchor_end: int | None = None,
    row: int = -1,
) -> SearchDocument: ...
def documents_from_catalog(
    catalog: Mapping[str, Any],
    row_lookup: Mapping[str, int] | None = None,
) -> list[SearchDocument]: ...
def compute_vector_scores(
    query: str,
    options: SearchOptions,
    context: VectorSearchContext,
) -> dict[str, float]: ...
def search_catalog(
    catalog: Mapping[str, object],
    *,
    request: SearchRequest,
    options: SearchOptions | None = ...,
    metrics: MetricsProvider | None = ...,
) -> list[SearchResult]: ...
