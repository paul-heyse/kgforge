"""Pydantic models describing the Agent Catalog payload."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kgfoundry.agent_catalog.sqlite import load_catalog_from_sqlite, sqlite_candidates


class AnchorsModel(BaseModel):
    """Source anchor metadata for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    start_line : int | None, optional
        Describe ``start_line``.
        Defaults to ``None``.
    end_line : int | None, optional
        Describe ``end_line``.
        Defaults to ``None``.
    cst_fingerprint : str | None, optional
        Describe ``cst_fingerprint``.
        Defaults to ``None``.
    remap_order : list[dict[str, Any]], optional
        Describe ``remap_order``.
        Defaults to ``<factory>``.
    """

    start_line: int | None = None
    end_line: int | None = None
    cst_fingerprint: str | None = None
    remap_order: list[dict[str, Any]] = Field(default_factory=list)


class QualityModel(BaseModel):
    """Quality signals captured for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    mypy_status : str | None, optional
        Describe ``mypy_status``.
        Defaults to ``None``.
    ruff_rules : list[str], optional
        Describe ``ruff_rules``.
        Defaults to ``<factory>``.
    pydoclint_parity : bool | None, optional
        Describe ``pydoclint_parity``.
        Defaults to ``None``.
    docstring_coverage : float | None, optional
        Describe ``docstring_coverage``.
        Defaults to ``None``.
    doctest_status : str | None, optional
        Describe ``doctest_status``.
        Defaults to ``None``.
    """

    mypy_status: str | None = None
    ruff_rules: list[str] = Field(default_factory=list)
    pydoclint_parity: bool | None = None
    docstring_coverage: float | None = None
    doctest_status: str | None = None


class MetricsModel(BaseModel):
    """Metric snapshot for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    complexity : float | None, optional
        Describe ``complexity``.
        Defaults to ``None``.
    loc : int | None, optional
        Describe ``loc``.
        Defaults to ``None``.
    last_modified : str | None, optional
        Describe ``last_modified``.
        Defaults to ``None``.
    codeowners : list[str], optional
        Describe ``codeowners``.
        Defaults to ``<factory>``.
    stability : str | None, optional
        Describe ``stability``.
        Defaults to ``None``.
    deprecated : bool | None, optional
        Describe ``deprecated``.
        Defaults to ``None``.
    """

    complexity: float | None = None
    loc: int | None = None
    last_modified: str | None = None
    codeowners: list[str] = Field(default_factory=list)
    stability: str | None = None
    deprecated: bool | None = None


class AgentHintsModel(BaseModel):
    """Structured hints for downstream agents.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    intent_tags : list[str], optional
        Describe ``intent_tags``.
        Defaults to ``<factory>``.
    safe_ops : list[str], optional
        Describe ``safe_ops``.
        Defaults to ``<factory>``.
    tests_to_run : list[str], optional
        Describe ``tests_to_run``.
        Defaults to ``<factory>``.
    perf_budgets : list[str], optional
        Describe ``perf_budgets``.
        Defaults to ``<factory>``.
    breaking_change_notes : list[str], optional
        Describe ``breaking_change_notes``.
        Defaults to ``<factory>``.
    """

    intent_tags: list[str] = Field(default_factory=list)
    safe_ops: list[str] = Field(default_factory=list)
    tests_to_run: list[str] = Field(default_factory=list)
    perf_budgets: list[str] = Field(default_factory=list)
    breaking_change_notes: list[str] = Field(default_factory=list)


class ChangeImpactModel(BaseModel):
    """Change impact metadata for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    callers : list[str], optional
        Describe ``callers``.
        Defaults to ``<factory>``.
    callees : list[str], optional
        Describe ``callees``.
        Defaults to ``<factory>``.
    tests : list[dict[str, Any]], optional
        Describe ``tests``.
        Defaults to ``<factory>``.
    codeowners : list[str], optional
        Describe ``codeowners``.
        Defaults to ``<factory>``.
    churn_last_n : int | None, optional
        Describe ``churn_last_n``.
        Defaults to ``None``.
    """

    callers: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    codeowners: list[str] = Field(default_factory=list)
    churn_last_n: int | None = None


class SymbolModel(BaseModel):
    """Catalog entry for a concrete symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    qname : str
        Describe ``qname``.
    kind : str
        Describe ``kind``.
    symbol_id : str
        Describe ``symbol_id``.
    docfacts : dict[str, Any] | None, optional
        Describe ``docfacts``.
        Defaults to ``None``.
    anchors : AnchorsModel
        Describe ``anchors``.
    quality : QualityModel
        Describe ``quality``.
    metrics : MetricsModel
        Describe ``metrics``.
    agent_hints : AgentHintsModel
        Describe ``agent_hints``.
    change_impact : ChangeImpactModel
        Describe ``change_impact``.
    exemplars : list[dict[str, Any]], optional
        Describe ``exemplars``.
        Defaults to ``<factory>``.
    """

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
    """Graph adjacency lists for a module.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    imports : list[str], optional
        Describe ``imports``.
        Defaults to ``<factory>``.
    calls : list[dict[str, Any]], optional
        Describe ``calls``.
        Defaults to ``<factory>``.
    """

    imports: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)


class ModuleModel(BaseModel):
    """Representation of a module within a package.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    name : str
        Describe ``name``.
    qualified : str
        Describe ``qualified``.
    source : dict[str, str]
        Describe ``source``.
    pages : dict[str, str | None]
        Describe ``pages``.
    imports : list[str], optional
        Describe ``imports``.
        Defaults to ``<factory>``.
    symbols : list[SymbolModel], optional
        Describe ``symbols``.
        Defaults to ``<factory>``.
    graph : ModuleGraphModel
        Describe ``graph``.
    """

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
    """Grouping of modules under a package name.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    name : str
        Describe ``name``.
    modules : list[ModuleModel], optional
        Describe ``modules``.
        Defaults to ``<factory>``.
    """

    name: str
    modules: list[ModuleModel] = Field(default_factory=list)


class SemanticIndexModel(BaseModel):
    """Metadata describing persisted semantic index artifacts.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    index : str
        Describe ``index``.
    mapping : str
        Describe ``mapping``.
    model : str | None, optional
        Describe ``model``.
        Defaults to ``None``.
    dimension : int | None, optional
        Describe ``dimension``.
        Defaults to ``None``.
    count : int | None, optional
        Describe ``count``.
        Defaults to ``None``.
    """

    index: str
    mapping: str
    model: str | None = None
    dimension: int | None = None
    count: int | None = None


class ShardEntryModel(BaseModel):
    """Entry describing a package shard.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    name : str
        Describe ``name``.
    path : str
        Describe ``path``.
    modules : int | None, optional
        Describe ``modules``.
        Defaults to ``None``.
    """

    name: str
    path: str
    modules: int | None = None


class ShardsModel(BaseModel):
    """Shard index metadata when the catalog is split.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    index : str
        Describe ``index``.
    packages : list[ShardEntryModel], optional
        Describe ``packages``.
        Defaults to ``<factory>``.
    """

    index: str
    packages: list[ShardEntryModel] = Field(default_factory=list)


class AgentCatalogModel(BaseModel):
    """Top-level representation of the Agent Catalog.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    version : str
        Describe ``version``.
    generated_at : str
        Describe ``generated_at``.
    repo : dict[str, str]
        Describe ``repo``.
    link_policy : dict[str, Any]
        Describe ``link_policy``.
    artifacts : dict[str, Any]
        Describe ``artifacts``.
    packages : list[PackageModel], optional
        Describe ``packages``.
        Defaults to ``<factory>``.
    shards : ShardsModel | None, optional
        Describe ``shards``.
        Defaults to ``None``.
    semantic_index : SemanticIndexModel | None, optional
        Describe ``semantic_index``.
        Defaults to ``None``.
    search : dict[str, Any] | None, optional
        Describe ``search``.
        Defaults to ``None``.
    """

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
        """Yield all symbol entries in the catalog.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        Iterable[SymbolModel]
            Describe return value.
        """
        for package in self.packages:
            for module in package.modules:
                yield from module.symbols

    def get_symbol(self, symbol_id: str) -> SymbolModel | None:
        """Return the symbol entry for ``symbol_id`` when available.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        SymbolModel | None
            Describe return value.
        """
        for symbol in self.iter_symbols():
            if symbol.symbol_id == symbol_id:
                return symbol
        return None

    def get_module(self, qualified: str) -> ModuleModel | None:
        """Return the module entry with the given qualified name.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        qualified : str
            Describe ``qualified``.

        Returns
        -------
        ModuleModel | None
            Describe return value.
        """
        for package in self.packages:
            for module in package.modules:
                if module.qualified == qualified:
                    return module
        return None


def load_catalog_payload(path: Path, *, load_shards: bool = True) -> dict[str, Any]:
    """Load a catalog payload from disk, expanding shards if requested.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Describe ``path``.
    load_shards : bool, optional
        Describe ``load_shards``.
        Defaults to ``True``.

    Returns
    -------
    dict[str, Any]
        Describe return value.
    """
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
    """Return a validated catalog model from the JSON artifact.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Describe ``path``.
    load_shards : bool, optional
        Describe ``load_shards``.
        Defaults to ``True``.

    Returns
    -------
    AgentCatalogModel
        Describe return value.
    """
    for candidate in sqlite_candidates(path):
        if candidate.exists():
            payload = load_catalog_from_sqlite(candidate)
            return AgentCatalogModel.model_validate(payload)
    payload = load_catalog_payload(path, load_shards=load_shards)
    return AgentCatalogModel.model_validate(payload)
