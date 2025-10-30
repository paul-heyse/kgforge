"""Pydantic models describing the Agent Catalog payload."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AnchorsModel(BaseModel):
    """Source anchor metadata for a symbol."""

    start_line: int | None = None
    end_line: int | None = None
    cst_fingerprint: str | None = None
    remap_order: list[dict[str, Any]] = Field(default_factory=list)


class QualityModel(BaseModel):
    """Quality signals captured for a symbol."""

    mypy_status: str | None = None
    ruff_rules: list[str] = Field(default_factory=list)
    pydoclint_parity: bool | None = None
    docstring_coverage: float | None = None
    doctest_status: str | None = None


class MetricsModel(BaseModel):
    """Metric snapshot for a symbol."""

    complexity: float | None = None
    loc: int | None = None
    last_modified: str | None = None
    codeowners: list[str] = Field(default_factory=list)
    stability: str | None = None
    deprecated: bool | None = None


class AgentHintsModel(BaseModel):
    """Structured hints for downstream agents."""

    intent_tags: list[str] = Field(default_factory=list)
    safe_ops: list[str] = Field(default_factory=list)
    tests_to_run: list[str] = Field(default_factory=list)
    perf_budgets: list[str] = Field(default_factory=list)
    breaking_change_notes: list[str] = Field(default_factory=list)


class ChangeImpactModel(BaseModel):
    """Change impact metadata for a symbol."""

    callers: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    codeowners: list[str] = Field(default_factory=list)
    churn_last_n: int | None = None


class SymbolModel(BaseModel):
    """Catalog entry for a concrete symbol."""

    qname: str
    kind: str
    symbol_id: str
    docfacts: dict[str, Any] | None = None
    anchors: AnchorsModel
    quality: QualityModel
    metrics: MetricsModel
    agent_hints: AgentHintsModel
    change_impact: ChangeImpactModel
    exemplars: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {
        "extra": "ignore",
    }


class ModuleGraphModel(BaseModel):
    """Graph adjacency lists for a module."""

    imports: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)


class ModuleModel(BaseModel):
    """Representation of a module within a package."""

    name: str
    qualified: str
    source: dict[str, str]
    pages: dict[str, str | None]
    imports: list[str] = Field(default_factory=list)
    symbols: list[SymbolModel] = Field(default_factory=list)
    graph: ModuleGraphModel

    model_config = {
        "extra": "ignore",
    }


class PackageModel(BaseModel):
    """Grouping of modules under a package name."""

    name: str
    modules: list[ModuleModel] = Field(default_factory=list)


class SemanticIndexModel(BaseModel):
    """Metadata describing persisted semantic index artifacts."""

    index: str
    mapping: str
    model: str | None = None
    dimension: int | None = None
    count: int | None = None


class ShardEntryModel(BaseModel):
    """Entry describing a package shard."""

    name: str
    path: str
    modules: int | None = None


class ShardsModel(BaseModel):
    """Shard index metadata when the catalog is split."""

    index: str
    packages: list[ShardEntryModel] = Field(default_factory=list)


class AgentCatalogModel(BaseModel):
    """Top-level representation of the Agent Catalog."""

    version: str
    generated_at: str
    repo: dict[str, str]
    link_policy: dict[str, Any]
    artifacts: dict[str, Any]
    packages: list[PackageModel] = Field(default_factory=list)
    shards: ShardsModel | None = None
    semantic_index: SemanticIndexModel | None = None
    search: dict[str, Any] | None = None

    model_config = {
        "extra": "ignore",
    }

    def iter_symbols(self) -> Iterable[SymbolModel]:
        """Yield all symbol entries in the catalog."""
        for package in self.packages:
            for module in package.modules:
                yield from module.symbols

    def get_symbol(self, symbol_id: str) -> SymbolModel | None:
        """Return the symbol entry for ``symbol_id`` when available."""
        for symbol in self.iter_symbols():
            if symbol.symbol_id == symbol_id:
                return symbol
        return None

    def get_module(self, qualified: str) -> ModuleModel | None:
        """Return the module entry with the given qualified name."""
        for package in self.packages:
            for module in package.modules:
                if module.qualified == qualified:
                    return module
        return None


def load_catalog_payload(path: Path, *, load_shards: bool = True) -> dict[str, Any]:
    """Load a catalog payload from disk, expanding shards if requested."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    shards = payload.get("shards")
    if load_shards and not payload.get("packages") and isinstance(shards, dict):
        index_rel = shards.get("index")
        if isinstance(index_rel, str):
            index_path = Path(index_rel)
            if not index_path.is_absolute():
                index_path = (path.parent / index_path).resolve()
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            packages: list[dict[str, Any]] = []
            for entry in index_payload.get("packages", []):
                shard_path = Path(entry.get("path", ""))
                if not shard_path:
                    continue
                if not shard_path.is_absolute():
                    shard_path = (index_path.parent / shard_path).resolve()
                shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
                packages.append(shard_payload)
            payload["packages"] = packages
    return payload


def load_catalog_model(path: Path, *, load_shards: bool = True) -> AgentCatalogModel:
    """Return a validated catalog model from the JSON artifact."""
    payload = load_catalog_payload(path, load_shards=load_shards)
    return AgentCatalogModel.model_validate(payload)
