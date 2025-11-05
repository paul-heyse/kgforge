"""Pydantic models describing the Agent Catalog payload."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from kgfoundry.agent_catalog.sqlite import load_catalog_from_sqlite, sqlite_candidates
from kgfoundry_common.errors import CatalogLoadError
from kgfoundry_common.problem_details import JsonValue

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AnchorsModel(BaseModel):
    """Source anchor metadata for a symbol.

    Contains source location information including line numbers,
    fingerprint, and remap order for symbol anchors.

    Parameters
    ----------
    start_line : int | None, optional
        Starting line number for the symbol. Defaults to None.
    end_line : int | None, optional
        Ending line number for the symbol. Defaults to None.
    cst_fingerprint : str | None, optional
        CST fingerprint for the symbol. Defaults to None.
    remap_order : list[dict[str, JsonValue]], optional
        Remap order information. Defaults to empty list.
    """

    start_line: int | None = None
    end_line: int | None = None
    cst_fingerprint: str | None = None
    remap_order: list[dict[str, JsonValue]] = Field(default_factory=list)


class QualityModel(BaseModel):
    """Quality signals captured for a symbol.

    Contains code quality metrics including type checking status,
    linting rules, docstring coverage, and doctest status.

    Parameters
    ----------
    pyright_status : str | None, optional
        Pyright type checking status. Defaults to None.
    pyrefly_status : str | None, optional
        Pyrefly type checking status. Defaults to None.
    ruff_rules : list[str], optional
        List of Ruff rule violations. Defaults to empty list.
    pydoclint_parity : bool | None, optional
        Pydoclint parity status. Defaults to None.
    docstring_coverage : float | None, optional
        Docstring coverage percentage. Defaults to None.
    doctest_status : str | None, optional
        Doctest execution status. Defaults to None.
    """

    pyright_status: str | None = None
    pyrefly_status: str | None = None
    ruff_rules: list[str] = Field(default_factory=list)
    pydoclint_parity: bool | None = None
    docstring_coverage: float | None = None
    doctest_status: str | None = None


class MetricsModel(BaseModel):
    """Metric snapshot for a symbol.

    Contains runtime and code metrics including complexity, lines of code,
    last modification time, codeowners, stability, and deprecation status.

    Parameters
    ----------
    complexity : float | None, optional
        Cyclomatic complexity score. Defaults to None.
    loc : int | None, optional
        Lines of code count. Defaults to None.
    last_modified : str | None, optional
        Last modification timestamp. Defaults to None.
    codeowners : list[str], optional
        List of code owner identifiers. Defaults to empty list.
    stability : str | None, optional
        Stability level (e.g., "stable", "experimental"). Defaults to None.
    deprecated : bool | None, optional
        Whether the symbol is deprecated. Defaults to None.
    """

    complexity: float | None = None
    loc: int | None = None
    last_modified: str | None = None
    codeowners: list[str] = Field(default_factory=list)
    stability: str | None = None
    deprecated: bool | None = None


class AgentHintsModel(BaseModel):
    """Structured hints for downstream agents.

    Contains metadata hints for AI agents including intent tags,
    safe operations, test suggestions, performance budgets, and
    breaking change notes.

    Parameters
    ----------
    intent_tags : list[str], optional
        Tags describing the symbol's intent. Defaults to empty list.
    safe_ops : list[str], optional
        List of safe operations for this symbol. Defaults to empty list.
    tests_to_run : list[str], optional
        List of tests that should be run when this symbol changes.
        Defaults to empty list.
    perf_budgets : list[str], optional
        Performance budget constraints. Defaults to empty list.
    breaking_change_notes : list[str], optional
        Notes about breaking changes. Defaults to empty list.
    """

    intent_tags: list[str] = Field(default_factory=list)
    safe_ops: list[str] = Field(default_factory=list)
    tests_to_run: list[str] = Field(default_factory=list)
    perf_budgets: list[str] = Field(default_factory=list)
    breaking_change_notes: list[str] = Field(default_factory=list)


class ChangeImpactModel(BaseModel):
    """Change impact metadata for a symbol.

    Contains information about which symbols and modules would be
    affected by changes to this symbol, including callers, callees,
    test suggestions, and churn metrics.

    Parameters
    ----------
    callers : list[str], optional
        List of symbol IDs that call this symbol. Defaults to empty list.
    callees : list[str], optional
        List of symbol IDs called by this symbol. Defaults to empty list.
    tests : list[dict[str, JsonValue]], optional
        List of test suggestions. Defaults to empty list.
    codeowners : list[str], optional
        List of code owner identifiers. Defaults to empty list.
    churn_last_n : int | None, optional
        Code churn metric for recent changes. Defaults to None.
    """

    callers: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    tests: list[dict[str, JsonValue]] = Field(default_factory=list)
    codeowners: list[str] = Field(default_factory=list)
    churn_last_n: int | None = None


class SymbolModel(BaseModel):
    """Catalog entry for a concrete symbol.

    Represents a symbol (function, class, module, etc.) in the catalog
    with all associated metadata including docfacts, anchors, quality
    metrics, and change impact.

    Parameters
    ----------
    qname : str
        Fully qualified symbol name.
    kind : str
        Symbol kind (e.g., "class", "function", "module").
    symbol_id : str
        Unique symbol identifier.
    docfacts : dict[str, JsonValue] | None, optional
        Documentation facts (summary, docstring). Defaults to None.
    anchors : AnchorsModel
        Source anchor metadata.
    quality : QualityModel
        Quality metrics and signals.
    metrics : MetricsModel
        Runtime and code metrics.
    agent_hints : AgentHintsModel
        Agent-specific hints and metadata.
    change_impact : ChangeImpactModel
        Change impact analysis data.
    exemplars : list[dict[str, JsonValue]], optional
        Example usage snippets. Defaults to empty list.
    """

    qname: str
    kind: str
    symbol_id: str
    docfacts: dict[str, JsonValue] | None = None
    anchors: AnchorsModel
    quality: QualityModel
    metrics: MetricsModel
    agent_hints: AgentHintsModel
    change_impact: ChangeImpactModel
    exemplars: list[dict[str, JsonValue]] = Field(default_factory=list)

    model_config = {
        "extra": "ignore",
    }


class ModuleGraphModel(BaseModel):
    """Graph adjacency lists for a module.

    Contains import relationships and call graph edges for a module.

    Parameters
    ----------
    imports : list[str], optional
        List of imported module names. Defaults to empty list.
    calls : list[dict[str, JsonValue]], optional
        List of call graph edges. Defaults to empty list.
    """

    imports: list[str] = Field(default_factory=list)
    calls: list[dict[str, JsonValue]] = Field(default_factory=list)


class ModuleModel(BaseModel):
    """Representation of a module within a package.

    Represents a Python module with its symbols, source information,
    documentation pages, imports, and call graph.

    Parameters
    ----------
    name : str
        Module name (short name).
    qualified : str
        Fully qualified module name.
    source : dict[str, str]
        Source file information.
    pages : dict[str, str | None]
        Documentation page mappings.
    imports : list[str], optional
        List of imported module names. Defaults to empty list.
    symbols : list[SymbolModel], optional
        List of symbols in this module. Defaults to empty list.
    graph : ModuleGraphModel
        Import and call graph for this module.
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

    Represents a Python package containing multiple modules.

    Parameters
    ----------
    name : str
        Package name.
    modules : list[ModuleModel], optional
        List of modules in this package. Defaults to empty list.
    """

    name: str
    modules: list[ModuleModel] = Field(default_factory=list)


class SemanticIndexModel(BaseModel):
    """Metadata describing persisted semantic index artifacts.

    Contains paths and metadata for semantic search index files.

    Parameters
    ----------
    index : str
        Relative path to FAISS index file.
    mapping : str
        Relative path to symbol-to-row mapping file.
    model : str | None, optional
        Embedding model name used. Defaults to None.
    dimension : int | None, optional
        Vector dimension. Defaults to None.
    count : int | None, optional
        Number of indexed vectors. Defaults to None.
    """

    index: str
    mapping: str
    model: str | None = None
    dimension: int | None = None
    count: int | None = None


class ShardEntryModel(BaseModel):
    """Entry describing a package shard.

    Represents a single shard entry in a sharded catalog.

    Parameters
    ----------
    name : str
        Package name in this shard.
    path : str
        Relative path to shard JSON file.
    modules : int | None, optional
        Number of modules in this shard. Defaults to None.
    """

    name: str
    path: str
    modules: int | None = None


class ShardsModel(BaseModel):
    """Shard index metadata when the catalog is split.

    Contains metadata for sharded catalogs where packages are split
    across multiple files.

    Parameters
    ----------
    index : str
        Relative path to shard index file.
    packages : list[ShardEntryModel], optional
        List of shard entries. Defaults to empty list.
    """

    index: str
    packages: list[ShardEntryModel] = Field(default_factory=list)


class AgentCatalogModel(BaseModel):
    """Top-level representation of the Agent Catalog.

    Root model for the entire catalog containing version, metadata,
    packages, optional shards, and semantic index information.

    Parameters
    ----------
    version : str
        Catalog schema version.
    generated_at : str
        ISO timestamp of catalog generation.
    repo : dict[str, str]
        Repository metadata.
    link_policy : dict[str, JsonValue]
        Link policy configuration.
    artifacts : dict[str, JsonValue]
        Artifact metadata.
    packages : list[PackageModel], optional
        List of packages in the catalog. Defaults to empty list.
    shards : ShardsModel | None, optional
        Shard metadata if catalog is sharded. Defaults to None.
    semantic_index : SemanticIndexModel | None, optional
        Semantic index metadata if available. Defaults to None.
    search : dict[str, JsonValue] | None, optional
        Search configuration. Defaults to None.
    """

    version: str
    generated_at: str
    repo: dict[str, str]
    link_policy: dict[str, JsonValue]
    artifacts: dict[str, JsonValue]
    packages: list[PackageModel] = Field(default_factory=list)
    shards: ShardsModel | None = None
    semantic_index: SemanticIndexModel | None = None
    search: dict[str, JsonValue] | None = None

    model_config = {
        "extra": "ignore",
    }

    def iter_symbols(self) -> Iterable[SymbolModel]:
        """Yield all symbol entries in the catalog.

        Iterates through all packages and modules to yield every symbol.

        Yields
        ------
        SymbolModel
            Symbol entry from the catalog.
        """
        for package in self.packages:
            for module in package.modules:
                yield from module.symbols

    def get_symbol(self, symbol_id: str) -> SymbolModel | None:
        """Return the symbol entry for ``symbol_id`` when available.

        Searches all packages and modules to find a symbol by ID.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier to search for.

        Returns
        -------
        SymbolModel | None
            Symbol model if found, None otherwise.
        """
        for symbol in self.iter_symbols():
            if symbol.symbol_id == symbol_id:
                return symbol
        return None

    def get_module(self, qualified: str) -> ModuleModel | None:
        """Return the module entry with the given qualified name.

        Searches all packages to find a module by qualified name.

        Parameters
        ----------
        qualified : str
            Fully qualified module name.

        Returns
        -------
        ModuleModel | None
            Module model if found, None otherwise.
        """
        for package in self.packages:
            for module in package.modules:
                if module.qualified == qualified:
                    return module
        return None


def load_catalog_payload(path: Path, *, load_shards: bool = True) -> dict[str, JsonValue]:
    """Load a catalog payload from disk, expanding shards if requested.

    Loads JSON catalog payload from file or SQLite database, optionally
    expanding sharded catalogs by loading referenced shard files.

    Parameters
    ----------
    path : Path
        Path to catalog JSON file or SQLite database.
    load_shards : bool, optional
        Whether to expand shards if present. Defaults to True.

    Returns
    -------
    dict[str, JsonValue]
        Catalog payload dictionary.

    Raises
    ------
    CatalogLoadError
        If the catalog file is not found, contains invalid JSON, or has
        an invalid format.
    """
    try:
        payload_raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        msg = f"Catalog file not found: {path}"
        raise CatalogLoadError(msg, cause=e) from e
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in catalog: {path}"
        raise CatalogLoadError(msg, cause=e) from e

    # json.loads returns object (JsonValue at runtime), narrow to dict
    if not isinstance(payload_raw, dict):
        msg = f"Invalid catalog format: expected dict, got {type(payload_raw)}"
        raise CatalogLoadError(msg)

    payload: dict[str, JsonValue] = cast("dict[str, JsonValue]", payload_raw)

    if load_shards and not payload.get("packages"):
        payload = _expand_shards_if_present(path, payload)

    return payload


def _expand_shards_if_present(
    base_path: Path, payload: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Expand shard payloads into the main catalog if shards are present.

    Checks for shard metadata and loads referenced shard files to populate
    the packages list.

    Parameters
    ----------
    base_path : Path
        Base path for resolving relative shard paths.
    payload : dict[str, JsonValue]
        Payload dict potentially containing shards metadata.

    Returns
    -------
    dict[str, JsonValue]
        Payload with packages field populated from shards if applicable.

    Raises
    ------
    CatalogLoadError
        If shard index file is not found or contains invalid JSON.
    """
    shards = payload.get("shards")
    if not isinstance(shards, dict):
        return payload

    index_rel = shards.get("index")
    if not isinstance(index_rel, str):
        return payload

    index_path = Path(index_rel)
    if not index_path.is_absolute():
        index_path = (base_path.parent / index_path).resolve()

    try:
        index_payload_raw: object = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        msg = f"Shard index file not found: {index_path}"
        raise CatalogLoadError(msg, cause=e) from e
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in shard index: {index_path}"
        raise CatalogLoadError(msg, cause=e) from e

    if not isinstance(index_payload_raw, dict):
        msg = f"Invalid shard index format: expected dict, got {type(index_payload_raw)}"
        raise CatalogLoadError(msg)

    index_payload: dict[str, JsonValue] = cast("dict[str, JsonValue]", index_payload_raw)

    packages = _load_shard_packages(base_path / index_path.parent, index_payload)
    # Cast packages to JsonValue for assignment to payload dict
    payload["packages"] = cast("JsonValue", packages)

    return payload


def _load_shard_packages(
    base_path: Path, index_payload: dict[str, JsonValue]
) -> list[dict[str, JsonValue]]:
    """Load all package shards referenced in the index payload.

    Loads individual shard JSON files referenced in the shard index.

    Parameters
    ----------
    base_path : Path
        Base directory for resolving relative shard paths.
    index_payload : dict[str, JsonValue]
        Index payload containing list of shard entries.

    Returns
    -------
    list[dict[str, JsonValue]]
        List of loaded shard package dictionaries.
    """
    packages: list[dict[str, JsonValue]] = []
    packages_raw: JsonValue = index_payload.get("packages", [])

    if not isinstance(packages_raw, list):
        return packages

    for entry in packages_raw:
        if not isinstance(entry, dict):
            continue

        # isinstance check narrows type - static type checkers understand this
        entry_dict: dict[str, JsonValue] = entry
        shard_path_raw = entry_dict.get("path", "")

        if not isinstance(shard_path_raw, str) or not shard_path_raw:
            continue

        shard_path = Path(shard_path_raw)
        if not shard_path.is_absolute():
            shard_path = (base_path / shard_path).resolve()

        try:
            shard_payload_raw: object = json.loads(shard_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Shard package file not found: %s", shard_path)
            continue
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in shard package %s: %s", shard_path, e)
            continue

        if not isinstance(shard_payload_raw, dict):
            logger.warning("Invalid shard package format (expected dict) at %s", shard_path)
            continue

        packages.append(cast("dict[str, JsonValue]", shard_payload_raw))

    return packages


# Provide typed access to Pydantic's ``model_validate`` without propagating
# ``Any`` into callers when type checking.
if TYPE_CHECKING:  # pragma: no cover - typing only

    def _model_validate_catalog(payload: dict[str, JsonValue]) -> AgentCatalogModel: ...

else:  # pragma: no cover - runtime behaviour

    def _model_validate_catalog(payload: dict[str, JsonValue]) -> AgentCatalogModel:
        return AgentCatalogModel.model_validate(payload)


def load_catalog_model(path: Path, *, load_shards: bool = True) -> AgentCatalogModel:
    """Return a validated catalog model from the JSON artifact.

    Loads catalog from JSON file or SQLite database and validates it
    against the Pydantic schema.

    Parameters
    ----------
    path : Path
        Path to catalog JSON file or SQLite database.
    load_shards : bool, optional
        Whether to expand shards if present. Defaults to True.

    Returns
    -------
    AgentCatalogModel
        Validated catalog model instance.
    """

    # Pydantic BaseModel classes contain Any in their type signatures.
    # We use cast() to explicitly acknowledge this typing limitation while
    # maintaining runtime safety through schema validation.
    def _validate_catalog(payload: dict[str, JsonValue]) -> AgentCatalogModel:
        """Validate and create AgentCatalogModel from payload.

        Parameters
        ----------
        payload : dict[str, JsonValue]
            The catalog payload to validate.

        Returns
        -------
        AgentCatalogModel
            Validated catalog model instance.
        """
        return _model_validate_catalog(payload)

    for candidate in sqlite_candidates(path):
        if candidate.exists():
            payload = load_catalog_from_sqlite(candidate)
            # Use cast() to acknowledge Pydantic's Any constraints while
            # maintaining runtime safety through validation
            return _validate_catalog(payload)
    payload = load_catalog_payload(path, load_shards=load_shards)
    # Use cast() to acknowledge Pydantic's Any constraints while
    # maintaining runtime safety through validation
    return _validate_catalog(payload)
