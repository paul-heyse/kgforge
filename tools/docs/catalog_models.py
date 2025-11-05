"""Dataclass-based models for agent catalog artefacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import cast, overload

__all__ = [
    "AgentCatalog",
    "AgentHints",
    "Anchors",
    "ChangeImpact",
    "LinkPolicy",
    "Metrics",
    "ModuleRecord",
    "PackageRecord",
    "SemanticIndexMetadata",
    "SymbolRecord",
]


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


JsonDict = dict[str, object]
JsonList = list[object]


@overload
def _strip_none(value: JsonDict) -> JsonDict:  # pragma: no cover - runtime dispatch
    ...


@overload
def _strip_none(value: JsonList) -> JsonList:  # pragma: no cover - runtime dispatch
    ...


@overload
def _strip_none(value: object) -> object:  # pragma: no cover - runtime dispatch
    ...


def _strip_none(value: object) -> object:
    if isinstance(value, dict):
        sanitized: JsonDict = {}
        for key, raw_val in value.items():
            if not isinstance(key, str) or raw_val is None:
                continue
            sanitized[key] = _strip_none(raw_val)
        return sanitized
    if isinstance(value, list):
        sanitized_list: JsonList = []
        for item in value:
            if item is None:
                continue
            sanitized_list.append(_strip_none(item))
        return sanitized_list
    return value


@dataclass(slots=True)
class CatalogStruct:
    """Base class providing a ``model_dump`` helper."""

    def model_dump(self) -> dict[str, object]:
        """Serialize model to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation with None values stripped.
        """
        raw = cast("JsonDict", asdict(self))
        return _strip_none(raw)


@dataclass(slots=True)
class LinkPolicy(CatalogStruct):
    """Link policy configuration used for catalog anchor templates."""

    mode: str
    editor_template: str
    github_template: str
    github: dict[str, str] | None = None


@dataclass(slots=True)
class Anchors(CatalogStruct):
    """Anchor metadata for a symbol."""

    start_line: int | None = None
    end_line: int | None = None
    cst_fingerprint: str | None = None
    remap_order: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class Quality(CatalogStruct):
    """Quality signals for a symbol."""

    pyright_status: str
    pyrefly_status: str
    ruff_rules: list[str]
    pydoclint_parity: bool | None = None
    docstring_coverage: float | None = None
    doctest_status: str | None = None


@dataclass(slots=True)
class Metrics(CatalogStruct):
    """Metric summary for a symbol."""

    complexity: float | None = None
    loc: int | None = None
    last_modified: str | None = None
    codeowners: list[str] = field(default_factory=list)
    stability: str | None = None
    deprecated: bool = False


@dataclass(slots=True)
class AgentHints(CatalogStruct):
    """Agent hint bundle for downstream consumers."""

    intent_tags: list[str] = field(default_factory=list)
    safe_ops: list[str] = field(default_factory=list)
    tests_to_run: list[str] = field(default_factory=list)
    perf_budgets: list[str] = field(default_factory=list)
    breaking_change_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SemanticIndexMetadata(CatalogStruct):
    """Metadata describing persisted semantic index artefacts."""

    index: str
    mapping: str
    model: str
    dimension: int
    count: int


@dataclass(slots=True)
class ChangeImpact(CatalogStruct):
    """Change impact metadata per symbol."""

    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    tests: list[dict[str, object]] = field(default_factory=list)
    codeowners: list[str] = field(default_factory=list)
    churn_last_n: int = 0


@dataclass(slots=True)
class SymbolRecord(CatalogStruct):
    """Serializable representation of a symbol."""

    qname: str
    kind: str
    symbol_id: str
    docfacts: dict[str, object] | None = None
    anchors: Anchors | None = None
    quality: Quality | None = None
    metrics: Metrics | None = None
    agent_hints: AgentHints | None = None
    change_impact: ChangeImpact | None = None
    exemplars: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class ModuleRecord(CatalogStruct):
    """Serializable representation of a module and its symbols."""

    name: str
    qualified: str
    source: dict[str, object]
    pages: dict[str, str | None]
    imports: list[str]
    symbols: list[SymbolRecord]
    graph: dict[str, object]


@dataclass(slots=True)
class PackageRecord(CatalogStruct):
    """Serializable representation of a package."""

    name: str
    modules: list[ModuleRecord]


@dataclass(slots=True)
class AgentCatalog(CatalogStruct):
    """Top-level agent catalog representation."""

    version: str
    generated_at: str = field(default_factory=_now_iso)
    repo: dict[str, str] = field(default_factory=dict)
    link_policy: LinkPolicy | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    packages: list[PackageRecord] = field(default_factory=list)
    shards: dict[str, object] | None = None
    semantic_index: SemanticIndexMetadata | None = None
    search: dict[str, object] | None = None
