"""Agent catalog generator for documentation artifacts."""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import datetime as dt
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tokenize
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import jsonschema
import numpy as np

from kgfoundry.agent_catalog import search as catalog_search
from kgfoundry.agent_catalog.search import (
    LEXICAL_FIELDS,
    CatalogSearchError,
    EmbeddingModelProtocol,
    SearchDocument,
    SearchOptions,
    SearchResult,
    VectorArray,
    load_faiss,
)
from kgfoundry.agent_catalog.sqlite import write_sqlite_catalog
from tools._shared.logging import get_logger, with_fields
from tools._shared.proc import ToolExecutionError, run_tool
from tools.docs.errors import CatalogBuildError


@dataclass
class LinkPolicy:
    """Link policy configuration used for catalog anchor templates."""

    mode: str
    editor_template: str
    github_template: str
    github: dict[str, str] | None = None


@dataclass
class Anchors:
    """Anchor metadata for a symbol."""

    start_line: int | None
    end_line: int | None
    cst_fingerprint: str | None
    remap_order: list[dict[str, Any]]


@dataclass
class Quality:
    """Quality signals for a symbol."""

    mypy_status: str
    ruff_rules: list[str]
    pydoclint_parity: bool | None
    docstring_coverage: float | None
    doctest_status: str


@dataclass
class Metrics:
    """Metric summary for a symbol."""

    complexity: float | None
    loc: int | None
    last_modified: str | None
    codeowners: list[str]
    stability: str | None
    deprecated: bool


@dataclass
class AgentHints:
    """Agent hint bundle for downstream consumers."""

    intent_tags: list[str] = field(default_factory=list)
    safe_ops: list[str] = field(default_factory=list)
    tests_to_run: list[str] = field(default_factory=list)
    perf_budgets: list[str] = field(default_factory=list)
    breaking_change_notes: list[str] = field(default_factory=list)


@dataclass
class SemanticIndexMetadata:
    """Metadata describing the persisted semantic index artifacts."""

    index: str
    mapping: str
    model: str
    dimension: int
    count: int


@dataclass
class ChangeImpact:
    """Change impact metadata per symbol."""

    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    codeowners: list[str] = field(default_factory=list)
    churn_last_n: int = 0


@dataclass
class SymbolRecord:
    """Serializable representation of a symbol."""

    qname: str
    kind: str
    symbol_id: str
    docfacts: dict[str, Any] | None
    anchors: Anchors
    quality: Quality
    metrics: Metrics
    agent_hints: AgentHints
    change_impact: ChangeImpact
    exemplars: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModuleRecord:
    """Serializable representation of a module and its symbols."""

    name: str
    qualified: str
    source: dict[str, str]
    pages: dict[str, str | None]
    imports: list[str]
    symbols: list[SymbolRecord]
    graph: dict[str, Any]


@dataclass
class PackageRecord:
    """Serializable representation of a package."""

    name: str
    modules: list[ModuleRecord]


@dataclass
class AgentCatalog:
    """Top-level agent catalog representation."""

    version: str
    generated_at: str
    repo: dict[str, str]
    link_policy: dict[str, Any]
    artifacts: dict[str, str]
    packages: list[PackageRecord]
    shards: dict[str, Any] | None = None
    semantic_index: SemanticIndexMetadata | None = None
    search: catalog_search.SearchConfig | None = None


STD_LIB_MODULES = set(sys.stdlib_module_names)
TRIGRAM_LENGTH = 3
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
EMBEDDING_MATRIX_RANK = 2
logger = get_logger(__name__)


class ModuleAnalyzer:
    """Collect definitions, imports, and call graph data for a module."""

    def __init__(self, module: str, path: Path) -> None:
        self.module = module
        self.path = path
        self.imports: set[str] = set()
        self.import_aliases: dict[str, str] = {}
        self.symbol_nodes: dict[str, ast.AST] = {}
        self.symbol_scopes: dict[str, tuple[str, ...]] = {}
        self.local_name_map: dict[str, str] = {}
        self.call_edges: dict[str, set[str]] = collections.defaultdict(set)
        self.source_lines: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        self.source_lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return
        _attach_parents(tree)
        collector = _DefinitionCollector(self.module)
        collector.visit(tree)
        self.imports = collector.imports
        self.import_aliases = collector.import_aliases
        self.symbol_nodes = collector.symbol_nodes
        self.symbol_scopes = collector.symbol_scopes
        self.local_name_map = collector.local_name_map
        for qname, node in self.symbol_nodes.items():
            scope = self.symbol_scopes.get(qname, ())
            call_collector = _CallCollector(
                module=self.module,
                scope=scope,
                import_aliases=self.import_aliases,
                local_name_map=self.local_name_map,
            )
            call_collector.visit(node)
            if call_collector.calls:
                self.call_edges[qname].update(call_collector.calls)

    def get_node(self, qname: str) -> ast.AST | None:
        """Return the AST node for the qualified name, if present."""
        return self.symbol_nodes.get(qname)

    def get_scope(self, qname: str) -> tuple[str, ...]:
        """Return the lexical scope for a qualified name."""
        return self.symbol_scopes.get(qname, ())

    def get_source_segment(self, node: ast.AST) -> str:
        """Return the source segment for a node."""
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        start = max(int(getattr(node, "lineno", 0)) - 1, 0)
        end = max(int(getattr(node, "end_lineno", start)) - 1, start)
        if not self.source_lines:
            return ""
        return "\n".join(self.source_lines[start : end + 1])

    def module_imports(self) -> list[str]:
        """Return normalized imports for the module."""
        imports: set[str] = set()
        for value in self.imports:
            root = value.split(".")[0]
            if root and root not in STD_LIB_MODULES:
                imports.add(value)
        return sorted(imports)

    def symbol_calls(self, qname: str) -> set[str]:
        """Return call targets for a symbol."""
        return self.call_edges.get(qname, set())


class _DefinitionCollector(ast.NodeVisitor):
    """Collect symbol definitions and imports."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.imports: set[str] = set()
        self.import_aliases: dict[str, str] = {}
        self.symbol_nodes: dict[str, ast.AST] = {}
        self.symbol_scopes: dict[str, tuple[str, ...]] = {}
        self.local_name_map: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target = alias.name
            self.imports.add(target)
            self.import_aliases[alias.asname or alias.name] = target

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module:
            self.imports.add(module)
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            self.import_aliases[alias.asname or alias.name] = full

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = ".".join([self.module, *self.scope, node.name])
        self.symbol_nodes[qname] = node
        self.symbol_scopes[qname] = tuple(self.scope)
        self.local_name_map[node.name] = qname
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = node.name
        qname = ".".join([self.module, *self.scope, name])
        self.symbol_nodes[qname] = node
        self.symbol_scopes[qname] = tuple(self.scope)
        self.local_name_map[name] = qname
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()


class _CallCollector(ast.NodeVisitor):
    """Collect call targets for a symbol."""

    def __init__(
        self,
        *,
        module: str,
        scope: tuple[str, ...],
        import_aliases: Mapping[str, str],
        local_name_map: Mapping[str, str],
    ) -> None:
        self.module = module
        self.scope = scope
        self.import_aliases = import_aliases
        self.local_name_map = local_name_map
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        target = self._resolve_call(node.func)
        if target:
            self.calls.add(target)
        self.generic_visit(node)

    def _resolve_call(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        if isinstance(node, ast.Attribute):
            chain = self._attribute_chain(node)
            if not chain:
                return None
            return self._resolve_attribute_chain(chain)
        return None

    def _resolve_name(self, name: str) -> str | None:
        if name in self.local_name_map:
            return self.local_name_map[name]
        if name in self.import_aliases:
            return self.import_aliases[name]
        return None

    def _attribute_chain(self, node: ast.Attribute) -> list[str]:
        chain: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            chain.append(current.id)
            chain.reverse()
            return chain
        return []

    def _resolve_attribute_chain(self, chain: list[str]) -> str | None:
        root = chain[0]
        remainder = chain[1:]
        if root in {"self", "cls"} and self.scope and remainder:
            class_name = self.scope[0]
            method = remainder[0]
            return ".".join([self.module, class_name, method])
        if root in self.import_aliases:
            mapped = self.import_aliases[root]
            return ".".join([mapped, *remainder]) if remainder else mapped
        if root in self.local_name_map:
            mapped = self.local_name_map[root]
            return ".".join([mapped, *remainder]) if remainder else mapped
        return ".".join(chain)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate the agent catalog.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_build/agent_catalog.json"),
        help="Catalog JSON output path.",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path("docs/_build/catalog.sqlite"),
        help="Optional SQLite catalogue output path.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/_build/schema_agent_catalog.json"),
        help="Schema JSON path used for validation.",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("docs/_build/agent_catalog"),
        help="Directory where catalog shards should be written.",
    )
    parser.add_argument(
        "--max-modules-per-shard",
        type=int,
        default=0,
        help="Maximum number of modules before sharding is enabled (0 disables sharding).",
    )
    parser.add_argument(
        "--max-symbols-per-shard",
        type=int,
        default=0,
        help="Maximum number of symbols before sharding is enabled (0 disables sharding).",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "editor", "github"],
        default="auto",
        help="Link mode (CLI overrides environment).",
    )
    parser.add_argument("--github-org", help="GitHub organization override.")
    parser.add_argument("--github-repo", help="GitHub repository override.")
    parser.add_argument("--github-sha", help="GitHub commit SHA override.")
    parser.add_argument("--repo-sha", help="Repository commit SHA override.")
    parser.add_argument("--version", default="1.0.0", help="Catalog version string.")
    parser.add_argument(
        "--docfacts",
        type=Path,
        default=Path("docs/_build/docfacts.json"),
        help="Path to docfacts JSON artifact.",
    )
    parser.add_argument(
        "--navmap",
        type=Path,
        default=Path("site/_build/navmap/navmap.json"),
        help="Path to navmap artifact.",
    )
    parser.add_argument(
        "--by-module",
        type=Path,
        default=Path("site/_build/by_module.json"),
        help="Path to by_module JSON artifact.",
    )
    parser.add_argument(
        "--by-file",
        type=Path,
        default=Path("site/_build/by_file.json"),
        help="Path to by_file JSON artifact.",
    )
    parser.add_argument(
        "--symbols",
        type=Path,
        default=Path("site/_build/symbols.json"),
        help="Path to symbols JSON artifact.",
    )
    parser.add_argument(
        "--test-map",
        type=Path,
        default=Path("site/_build/test_map.json"),
        help="Path to test map JSON artifact.",
    )
    parser.add_argument(
        "--test-map-coverage",
        type=Path,
        default=Path("site/_build/test_map_coverage.json"),
        help="Path to test map coverage JSON artifact.",
    )
    parser.add_argument(
        "--test-map-summary",
        type=Path,
        default=Path("site/_build/test_map_summary.json"),
        help="Path to test map summary JSON artifact.",
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=Path("docs/_build/agent_catalog/semantic"),
        help="Directory for semantic index artifacts.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name used for embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size to use when computing embeddings.",
    )
    parser.add_argument(
        "--search-alpha",
        type=float,
        default=0.6,
        help="Hybrid search weight for vector similarity (0-1).",
    )
    parser.add_argument(
        "--search-candidates",
        type=int,
        default=50,
        help="Number of lexical candidates considered before vector rerank.",
    )
    parser.add_argument(
        "--search-query",
        help="If provided, run hybrid search against an existing catalog instead of building.",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=10,
        help="Number of search results to return when --search-query is used.",
    )
    parser.add_argument(
        "--search-facet",
        action="append",
        default=[],
        help="Facet filter expressed as key=value (may be repeated).",
    )
    return parser.parse_args(argv)


class AgentCatalogBuilder:
    """Builds the agent catalog from documentation artifacts."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        repo_root_arg = cast(Path, args.repo_root)
        self.repo_root = repo_root_arg.resolve()
        self.artifact_paths = {
            "docfacts": cast(Path, args.docfacts),
            "navmap": cast(Path, args.navmap),
            "by_module": cast(Path, args.by_module),
            "by_file": cast(Path, args.by_file),
            "symbols": cast(Path, args.symbols),
            "test_map": cast(Path, args.test_map),
            "test_map_coverage": cast(Path, args.test_map_coverage),
            "test_map_summary": cast(Path, args.test_map_summary),
        }
        self.semantic_dir = cast(Path, args.semantic_dir)
        self.embedding_model = str(args.embedding_model)
        self.embedding_batch_size = int(args.embedding_batch_size)
        self.search_alpha = float(args.search_alpha)
        self.search_candidates = int(args.search_candidates)
        self.sqlite_path = cast(Path, args.sqlite)
        self._symbol_records: dict[str, SymbolRecord] = {}
        self._git_churn_cache: dict[Path, int] = {}
        self._git_modified_cache: dict[Path, str | None] = {}
        self._symbol_index: dict[str, dict[str, Any]] = {}
        self._docfacts_index: dict[str, dict[str, Any]] = {}
        self._test_map: dict[str, list[dict[str, Any]]] = {}
        self._coverage_map: dict[str, Any] = {}
        self._packages_snapshot: list[dict[str, Any]] = []

    def build(self) -> AgentCatalog:
        """Build the agent catalog."""
        self._ensure_artifacts()
        link_policy = self._resolve_link_policy()
        repo_sha = self._resolve_repo_sha()
        generated_at = dt.datetime.now(tz=dt.UTC).isoformat()
        packages = self._collect_packages()
        self._packages_snapshot = [dataclasses.asdict(pkg) for pkg in packages]
        self._apply_call_graph(packages)
        semantic_index = self._build_semantic_index(packages)
        link_policy_dict = {
            key: value
            for key, value in dataclasses.asdict(link_policy).items()
            if value is not None
        }
        artifacts = {
            key: self._relative_string(self._resolve_artifact_path(path))
            for key, path in self.artifact_paths.items()
        }
        if semantic_index is not None:
            artifacts["semantic_index"] = semantic_index.index
            artifacts["semantic_mapping"] = semantic_index.mapping
        sqlite_target = self._resolve_artifact_path(self.sqlite_path)
        artifacts["catalog_sqlite"] = self._relative_string(sqlite_target)
        search_config = catalog_search.SearchConfig(
            alpha=self.search_alpha,
            candidate_pool=self.search_candidates,
            lexical_fields=list(LEXICAL_FIELDS),
        )
        catalog = AgentCatalog(
            version=self.args.version,
            generated_at=generated_at,
            repo={"sha": repo_sha, "root": str(self.repo_root)},
            link_policy=link_policy_dict,
            artifacts=artifacts,
            packages=packages,
            semantic_index=semantic_index,
            search=search_config,
        )
        self._maybe_shard(catalog)
        return catalog

    def _ensure_artifacts(self) -> None:
        missing = []
        for name, path in self.artifact_paths.items():
            if not self._resolve_artifact_path(path).exists():
                missing.append(name)
        if missing:
            message = "Missing required artifacts: " + ", ".join(sorted(missing))
            raise CatalogBuildError(message)

    def _resolve_artifact_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.repo_root / path)

    def _resolve_link_policy(self) -> LinkPolicy:
        env_mode = os.environ.get("DOCS_LINK_MODE")
        mode = self.args.link_mode
        if mode == "auto":
            mode = env_mode or "editor"
        editor_template = "vscode://file/{path}:{line}"
        github_template = "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}"
        env_org = os.environ.get("DOCS_GITHUB_ORG")
        env_repo = os.environ.get("DOCS_GITHUB_REPO")
        env_sha = os.environ.get("DOCS_GITHUB_SHA")
        raw_github: dict[str, str | None] = {
            "org": self.args.github_org or env_org,
            "repo": self.args.github_repo or env_repo,
            "sha": self.args.github_sha or env_sha,
        }
        if mode == "github":
            for key, value in raw_github.items():
                if not value:
                    message = f"GitHub mode selected but missing value for {key}."
                    raise CatalogBuildError(message)
        resolved_github: dict[str, str] | None
        if all(value is not None for value in raw_github.values()):
            resolved_github = {key: cast(str, value) for key, value in raw_github.items()}
        else:
            resolved_github = None
        return LinkPolicy(
            mode=mode,
            editor_template=editor_template,
            github_template=github_template,
            github=resolved_github,
        )

    def _resolve_repo_sha(self) -> str:
        if self.args.repo_sha:
            return cast(str, self.args.repo_sha)
        log_adapter = with_fields(logger, command=["git", "rev-parse", "--short", "HEAD"])
        try:
            result = run_tool(
                ["git", "rev-parse", "--short", "HEAD"],
                timeout=10.0,
                cwd=self.repo_root,
            )
        except ToolExecutionError as exc:
            log_adapter.exception("Failed to resolve repository SHA")
            message = "Unable to resolve repository SHA"
            raise CatalogBuildError(message) from exc
        if result.returncode != 0:
            log_adapter.error("Git returned non-zero exit code: %s", result.returncode)
            message = "Unable to resolve repository SHA"
            raise CatalogBuildError(message)
        return result.stdout.strip()

    def _collect_packages(self) -> list[PackageRecord]:
        symbols_data = cast(
            list[dict[str, Any]], self._load_json(self.artifact_paths["symbols"], [])
        )
        docfacts_raw = self._load_json(self.artifact_paths["docfacts"], {})
        if isinstance(docfacts_raw, dict):
            docfacts_data = cast(list[dict[str, Any]], docfacts_raw.get("entries", []))
        else:
            docfacts_data = cast(list[dict[str, Any]], docfacts_raw or [])
        by_module = cast(
            dict[str, list[str]], self._load_json(self.artifact_paths["by_module"], {})
        )
        self._test_map = cast(
            dict[str, list[dict[str, Any]]],
            self._load_json(self.artifact_paths["test_map"], {}),
        )
        self._coverage_map = cast(
            dict[str, Any],
            self._load_json(self.artifact_paths["test_map_coverage"], {}),
        )
        module_index = self._index_modules(symbols_data)
        self._symbol_index = self._index_symbols(symbols_data)
        self._docfacts_index = self._index_docfacts(docfacts_data)
        packages: dict[str, list[ModuleRecord]] = collections.defaultdict(list)
        for module_name in sorted(by_module):
            qnames = sorted(set(by_module[module_name]))
            module_record = self._build_module(
                module_name=module_name,
                qnames=qnames,
                module_entry=module_index.get(module_name),
            )
            if module_record is None:
                continue
            package_name = module_name.split(".")[0]
            packages[package_name].append(module_record)
        return [
            PackageRecord(
                name=name,
                modules=sorted(mods, key=lambda module: module.qualified),
            )
            for name, mods in sorted(packages.items())
        ]

    def _build_module(
        self,
        *,
        module_name: str,
        qnames: list[str],
        module_entry: dict[str, Any] | None,
    ) -> ModuleRecord | None:
        source_path = self._resolve_source_path(module_name, module_entry)
        analyzer = ModuleAnalyzer(module_name, source_path)
        symbols: list[SymbolRecord] = []
        for qname in qnames:
            symbol_record = self._build_symbol_record(
                module_name=module_name,
                qname=qname,
                analyzer=analyzer,
                symbol_entry=self._lookup_symbol(qname, self._symbol_index),
                docfacts=self._lookup_docfacts(qname, self._docfacts_index),
            )
            if symbol_record is not None:
                symbols.append(symbol_record)
        if not symbols:
            return None
        symbols.sort(key=lambda record: record.qname)
        module_imports = analyzer.module_imports()
        calls = []
        for caller, targets in analyzer.call_edges.items():
            if caller not in self._symbol_records:
                continue
            for callee in sorted(targets):
                calls.append(
                    {
                        "caller": caller,
                        "callee": callee,
                        "confidence": "static",
                    }
                )
        calls.sort(key=lambda entry: (entry["caller"], entry["callee"]))
        graph = {"imports": module_imports, "calls": calls}
        return ModuleRecord(
            name=module_name.rsplit(".", maxsplit=1)[-1],
            qualified=module_name,
            source={"path": self._relative_string(source_path)},
            pages={
                "html": self._resolve_module_page(module_name, "html"),
                "fjson": self._resolve_module_page(module_name, "json"),
            },
            imports=module_imports,
            symbols=symbols,
            graph=graph,
        )

    def _build_symbol_record(
        self,
        *,
        module_name: str,  # noqa: ARG002
        qname: str,
        analyzer: ModuleAnalyzer,
        symbol_entry: dict[str, Any] | None,
        docfacts: dict[str, Any] | None,
    ) -> SymbolRecord | None:
        node = analyzer.get_node(qname)
        if symbol_entry is None and node is None:
            return None
        kind = symbol_entry.get("kind") if symbol_entry else "object"
        anchors = self._build_anchors(qname, analyzer, node, symbol_entry)
        symbol_id = self._compute_symbol_id(qname, node)
        tests = self._normalize_tests(self._test_map.get(qname, []))
        hints = AgentHints(
            intent_tags=[],
            safe_ops=[],
            tests_to_run=sorted({test["file"] for test in tests}),
            perf_budgets=[],
            breaking_change_notes=[],
        )
        coverage_entry = self._coverage_map.get(qname)
        quality = Quality(
            mypy_status="unknown",
            ruff_rules=[],
            pydoclint_parity=None,
            docstring_coverage=(
                coverage_entry.get("ratio") if isinstance(coverage_entry, Mapping) else None
            ),
            doctest_status="unknown",
        )
        metrics = Metrics(
            complexity=self._compute_complexity(node),
            loc=self._compute_loc(anchors),
            last_modified=self._git_last_modified(analyzer.path),
            codeowners=[],
            stability=symbol_entry.get("stability") if symbol_entry else None,
            deprecated=bool(symbol_entry.get("deprecated_in")) if symbol_entry else False,
        )
        change_impact = ChangeImpact(
            callers=[],
            callees=[],
            tests=tests,
            codeowners=[],
            churn_last_n=self._git_churn(analyzer.path),
        )
        record = SymbolRecord(
            qname=qname,
            kind=str(kind),
            symbol_id=symbol_id,
            docfacts=docfacts,
            anchors=anchors,
            quality=quality,
            metrics=metrics,
            agent_hints=hints,
            change_impact=change_impact,
            exemplars=[],
        )
        self._symbol_records[qname] = record
        return record

    def _build_anchors(
        self,
        qname: str,
        analyzer: ModuleAnalyzer,
        node: ast.AST | None,
        symbol_entry: dict[str, Any] | None,
    ) -> Anchors:
        start_line: int | None
        end_line: int | None
        if node is not None and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            start_line = int(node.lineno)
            end_line = int(node.end_lineno)
        else:
            start_line = None
            end_line = None
            if symbol_entry is not None:
                raw_start = _docfacts_int(symbol_entry, "lineno")
                if raw_start is not None:
                    start_line = raw_start
                raw_end = _docfacts_int(symbol_entry, "end_lineno", "endlineno")
                if raw_end is not None:
                    end_line = raw_end
        fingerprint = self._compute_fingerprint(analyzer, node)
        scope = analyzer.get_scope(qname)
        name_arity = self._compute_name_arity(node, scope)
        nearest_text = self._nearest_text(analyzer, node)
        symbol_id = self._compute_symbol_id(qname, node)
        remap = [
            {
                "symbol_id": symbol_id,
                "cst_fingerprint": fingerprint,
                "name_arity": name_arity,
                "nearest_text": nearest_text,
            }
        ]
        return Anchors(
            start_line=start_line,
            end_line=end_line,
            cst_fingerprint=fingerprint,
            remap_order=remap,
        )

    def _compute_symbol_id(self, qname: str, node: ast.AST | None) -> str:
        if node is None:
            return hashlib.sha256(qname.encode()).hexdigest()
        normalized = _normalize_ast(node)
        payload = f"{qname}:{normalized}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _compute_fingerprint(self, analyzer: ModuleAnalyzer, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        segment = analyzer.get_source_segment(node)
        if not segment:
            return None
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(segment).readline))
        except tokenize.TokenError:
            return None
        skip_types = {
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        }
        normalized: list[str] = []
        for token in tokens:
            if token.type in skip_types:
                continue
            value = self._normalize_token(token)
            if value:
                normalized.append(value)
        if not normalized:
            return None
        if len(normalized) < TRIGRAM_LENGTH:
            return "|".join(normalized)
        trigrams = [
            " ".join(normalized[i : i + TRIGRAM_LENGTH])
            for i in range(len(normalized) - (TRIGRAM_LENGTH - 1))
        ]
        return "|".join(trigrams)

    def _normalize_token(self, token: tokenize.TokenInfo) -> str | None:
        if token.type in {tokenize.NAME, tokenize.OP}:
            return token.string
        if token.type == tokenize.NUMBER:
            return "NUMBER"
        if token.type == tokenize.STRING:
            return "STRING"
        return None

    def _compute_name_arity(self, node: ast.AST | None, scope: tuple[str, ...]) -> int:
        if node is None:
            return 0
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            total = len(args.args) + len(args.kwonlyargs) + len(args.posonlyargs)
            if args.vararg is not None:
                total += 1
            if args.kwarg is not None:
                total += 1
            if scope and args.args and args.args[0].arg in {"self", "cls"}:
                total -= 1
            return max(total, 0)
        return 0

    def _nearest_text(self, analyzer: ModuleAnalyzer, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        segment = analyzer.get_source_segment(node)
        if not segment:
            return None
        for line in segment.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def _compute_complexity(self, node: ast.AST | None) -> float | None:
        if node is None:
            return None
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.AsyncFor,
                    ast.IfExp,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                    ast.BoolOp,
                    ast.comprehension,
                ),
            ):
                complexity += 1
        return float(complexity)

    def _compute_loc(self, anchors: Anchors) -> int | None:
        if anchors.start_line is None or anchors.end_line is None:
            return None
        return max(anchors.end_line - anchors.start_line + 1, 0)

    def _normalize_tests(self, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for test in tests:
            normalized.append(
                {
                    "file": test.get("file", ""),
                    "lines": list(test.get("lines", [])),
                    "reason": test.get("reason"),
                    "windows": [
                        {
                            "start": window.get("start"),
                            "end": window.get("end"),
                        }
                        for window in test.get("windows", [])
                    ],
                }
            )
        normalized.sort(key=lambda item: (item["file"], item["lines"]))
        return normalized

    def _git_churn(self, path: Path) -> int:
        path = path.resolve()
        if path in self._git_churn_cache:
            return self._git_churn_cache[path]
        if not path.exists():
            self._git_churn_cache[path] = 0
            return 0
        log_adapter = with_fields(logger, command=["git", "log"], file=str(path))
        try:
            result = run_tool(
                ["git", "log", "--follow", "--pretty=%h", "-n", "30", str(path)],
                timeout=10.0,
                cwd=self.repo_root,
            )
        except ToolExecutionError:
            log_adapter.debug("Git churn lookup failed for %s", path)
            self._git_churn_cache[path] = 0
            return 0
        if result.returncode != 0:
            self._git_churn_cache[path] = 0
            return 0
        churn = len([line for line in result.stdout.splitlines() if line.strip()])
        self._git_churn_cache[path] = churn
        return churn

    def _git_last_modified(self, path: Path) -> str | None:
        path = path.resolve()
        if path in self._git_modified_cache:
            return self._git_modified_cache[path]
        if not path.exists():
            self._git_modified_cache[path] = None
            return None
        log_adapter = with_fields(logger, command=["git", "log"], file=str(path))
        try:
            result = run_tool(
                ["git", "log", "-1", "--format=%cI", str(path)],
                timeout=10.0,
                cwd=self.repo_root,
            )
        except ToolExecutionError:
            log_adapter.debug("Git last-modified lookup failed for %s", path)
            self._git_modified_cache[path] = None
            return None
        if result.returncode != 0:
            self._git_modified_cache[path] = None
            return None
        value = result.stdout.strip() or None
        self._git_modified_cache[path] = value
        return value

    def _load_json(self, relative: Path, default: object) -> object:
        path = self._resolve_artifact_path(relative)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def _index_modules(self, symbols_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        modules: dict[str, dict[str, Any]] = {}
        for entry in symbols_data:
            if entry.get("kind") == "module":
                canonical = entry.get("canonical_path") or entry.get("module")
                if canonical:
                    modules[str(canonical)] = entry
        return modules

    def _index_symbols(self, symbols_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for entry in symbols_data:
            canonical = entry.get("canonical_path") or entry.get("path")
            if canonical and canonical not in index:
                index[str(canonical)] = entry
        return index

    def _index_docfacts(self, docfacts_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for entry in docfacts_data:
            qname = entry.get("qname")
            if qname:
                index[str(qname)] = entry
        return index

    def _lookup_symbol(self, qname: str, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        if qname in index:
            return index[qname]
        tail = qname.split(".", 1)[-1]
        matches = [value for key, value in index.items() if key.endswith(tail)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _lookup_docfacts(
        self, qname: str, index: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        if qname in index:
            return index[qname]
        tail = qname.split(".", 1)[-1]
        matches = [value for key, value in index.items() if key.endswith(tail)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_source_path(self, module_name: str, module_entry: dict[str, Any] | None) -> Path:
        if module_entry:
            file_value = module_entry.get("file")
            if isinstance(file_value, str):
                return (self.repo_root / file_value).resolve()
        inferred = module_name.replace(".", "/") + ".py"
        return (self.repo_root / inferred).resolve()

    def _resolve_module_page(self, module_name: str, flavor: str) -> str | None:
        suffix = "html" if flavor == "html" else "json"
        filename = "index.html" if suffix == "html" else "index.fjson"
        path = (
            self.repo_root
            / "site/_build"
            / suffix
            / "autoapi"
            / "src"
            / Path("/".join(module_name.split(".")))
            / filename
        )
        if not path.exists():
            return None
        return self._relative_string(path)

    def _relative_string(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _apply_call_graph(self, packages: Sequence[PackageRecord]) -> None:
        callers: dict[str, set[str]] = collections.defaultdict(set)
        callees: dict[str, set[str]] = collections.defaultdict(set)
        for package in packages:
            for module in package.modules:
                for edge in module.graph["calls"]:
                    caller = edge["caller"]
                    callee = edge["callee"]
                    callers[callee].add(caller)
                    callees[caller].add(callee)
        for qname, record in self._symbol_records.items():
            record.change_impact.callers = sorted(callers.get(qname, []))
            record.change_impact.callees = sorted(callees.get(qname, []))

    def _collect_search_documents(self, packages: Sequence[PackageRecord]) -> list[SearchDocument]:
        documents: list[SearchDocument] = []
        for package in packages:
            for module in package.modules:
                for symbol in module.symbols:
                    document = self._build_search_document(
                        package=package,
                        module=module,
                        symbol=symbol,
                    )
                    documents.append(document)
        return documents

    def _build_search_document(
        self,
        *,
        package: PackageRecord,
        module: ModuleRecord,
        symbol: SymbolRecord,
    ) -> SearchDocument:
        summary, docstring = _extract_docfacts_text(symbol.docfacts)
        text_parts = [
            symbol.qname,
            module.qualified,
            package.name,
            summary,
            docstring,
            " ".join(symbol.agent_hints.intent_tags),
            " ".join(symbol.agent_hints.tests_to_run),
        ]
        normalized_text = " ".join(part for part in text_parts if part)
        tokens = collections.Counter(_tokenize(normalized_text))
        return SearchDocument(
            symbol_id=symbol.symbol_id,
            package=package.name,
            module=module.qualified,
            qname=symbol.qname,
            kind=symbol.kind,
            stability=symbol.metrics.stability,
            deprecated=symbol.metrics.deprecated,
            summary=summary,
            docstring=docstring,
            anchor_start=symbol.anchors.start_line,
            anchor_end=symbol.anchors.end_line,
            text=normalized_text,
            tokens=tokens,
        )

    def _build_semantic_index(
        self, packages: Sequence[PackageRecord]
    ) -> SemanticIndexMetadata | None:
        documents = self._collect_search_documents(packages)
        if not documents:
            return None
        texts = [document.text for document in documents]
        model = _load_embedding_model(self.embedding_model)
        try:
            embeddings = model.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"Embedding model failed to encode texts: {exc}"
            raise CatalogBuildError(message) from exc
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != EMBEDDING_MATRIX_RANK:
            message = "Embedding model returned an unexpected vector shape"
            raise CatalogBuildError(message)
        if vectors.shape[0] != len(documents):
            message = "Embedding count mismatch between texts and vectors"
            raise CatalogBuildError(message)
        if vectors.shape[1] == 0:
            message = "Embedding vectors must have a positive dimension"
            raise CatalogBuildError(message)
        semantic_dir = (self.repo_root / self.semantic_dir).resolve()
        semantic_dir.mkdir(parents=True, exist_ok=True)
        faiss_module = load_faiss("build the semantic index")
        index = faiss_module.IndexFlatIP(int(vectors.shape[1]))
        contiguous_vectors: VectorArray = np.ascontiguousarray(vectors)
        index.add(contiguous_vectors)
        index_path = semantic_dir / "symbols.faiss"
        faiss_module.write_index(index, str(index_path))
        mapping_entries = []
        for row, document in enumerate(documents):
            document.row = row
            mapping_entries.append(
                {
                    "symbol_id": document.symbol_id,
                    "row": row,
                    "package": document.package,
                    "module": document.module,
                    "qname": document.qname,
                    "kind": document.kind,
                    "stability": document.stability,
                    "deprecated": document.deprecated,
                }
            )
        mapping_payload = {
            "model": self.embedding_model,
            "dimension": int(vectors.shape[1]),
            "count": len(documents),
            "symbols": mapping_entries,
        }
        mapping_path = semantic_dir / "symbols.json"
        mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")
        return SemanticIndexMetadata(
            index=self._relative_string(index_path),
            mapping=self._relative_string(mapping_path),
            model=self.embedding_model,
            dimension=int(vectors.shape[1]),
            count=len(documents),
        )

    def _maybe_shard(self, catalog: AgentCatalog) -> None:
        total_modules = sum(len(pkg.modules) for pkg in catalog.packages)
        total_symbols = sum(len(mod.symbols) for pkg in catalog.packages for mod in pkg.modules)
        module_limit = self.args.max_modules_per_shard or 0
        symbol_limit = self.args.max_symbols_per_shard or 0
        if module_limit <= 0:
            module_limit = float("inf")
        if symbol_limit <= 0:
            symbol_limit = float("inf")
        if total_modules <= module_limit and total_symbols <= symbol_limit:
            catalog.shards = None
            return
        shard_dir = (self.repo_root / self.args.shard_dir).resolve()
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_entries = []
        for package in catalog.packages:
            package_dict = dataclasses.asdict(package)
            shard_path = shard_dir / f"{package.name}.json"
            shard_path.write_text(json.dumps(package_dict, indent=2), encoding="utf-8")
            shard_entries.append(
                {
                    "name": package.name,
                    "path": self._relative_string(shard_path),
                    "modules": len(package.modules),
                }
            )
        index_path = shard_dir / "index.json"
        index_path.write_text(json.dumps({"packages": shard_entries}, indent=2), encoding="utf-8")
        catalog.packages = []
        catalog.shards = {
            "index": self._relative_string(index_path),
            "packages": shard_entries,
        }

    def write(self, catalog: AgentCatalog, path: Path, schema: Path) -> None:
        """Write the catalog and validate against the schema."""
        catalog_dict = dataclasses.asdict(catalog)
        output_path = self._resolve_artifact_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(catalog_dict, indent=2), encoding="utf-8")
        schema_path = self._resolve_artifact_path(schema)
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema_data)
        errors = sorted(validator.iter_errors(catalog_dict), key=lambda err: tuple(err.path))
        if errors:
            rendered = "; ".join(
                f"{'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors
            )
            message = f"Catalog validation failed: {rendered}"
            raise CatalogBuildError(message)
        sqlite_target = self._resolve_artifact_path(self.sqlite_path)
        packages_override: list[dict[str, Any]] = []
        if self._packages_snapshot:
            packages_override = json.loads(json.dumps(self._packages_snapshot))
        elif catalog_dict.get("packages"):
            packages_override = catalog_dict.get("packages", [])
        write_sqlite_catalog(
            catalog_dict,
            sqlite_target,
            packages_override=packages_override,
        )


def _load_embedding_model(model_name: str) -> EmbeddingModelProtocol:
    """Return the configured embedding model, raising on failure."""
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:  # pragma: no cover - runtime guard
        message = "sentence-transformers is required for semantic indexing"
        raise CatalogBuildError(message) from exc
    factory = getattr(module, "SentenceTransformer", None)
    if not callable(factory):  # pragma: no cover - defensive guard
        message = "sentence-transformers is missing the SentenceTransformer class"
        raise CatalogBuildError(message)
    try:
        model = factory(model_name)
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Unable to load embedding model '{model_name}'"
        raise CatalogBuildError(message) from exc
    return cast(EmbeddingModelProtocol, model)


def _tokenize(text: str) -> list[str]:
    """Return normalized word tokens for the given text."""
    return WORD_RE.findall(text.lower())


def _docfacts_int(entry: Mapping[str, Any], *names: str) -> int | None:
    """Return the first integer value found for ``names`` within ``entry``."""
    for name in names:
        if not name:
            continue
        value = entry.get(name)
        if isinstance(value, int):
            return value
    return None


def _extract_docfacts_text(
    docfacts: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    """Extract summary and docstring text from docfacts payloads."""
    if not isinstance(docfacts, Mapping):
        return None, None
    summary = _stringify(docfacts.get("summary"))
    if summary is None:
        summary = _stringify(docfacts.get("short_summary"))
    docstring = _stringify(docfacts.get("docstring"))
    if docstring is None:
        docstring = _stringify(docfacts.get("description"))
    return summary, docstring


def _stringify(value: object) -> str | None:
    """Convert nested docfacts data into a normalized string."""
    if value is None:
        return None
    result: str | None
    if isinstance(value, str):
        stripped = value.strip()
        result = stripped or None
    elif isinstance(value, bool):
        result = "true" if value else "false"
    elif isinstance(value, (int, float)):
        result = str(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_stringify(item) for item in value]
        joined = " ".join(part for part in parts if part)
        result = joined or None
    elif isinstance(value, Mapping):
        parts = [_stringify(item) for item in value.values()]
        joined = " ".join(part for part in parts if part)
        result = joined or None
    else:
        result = str(value)
    return result


def _iter_symbol_entries(
    catalog: Mapping[str, Any],
) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    """Yield package, module, and symbol payloads from the catalog."""
    packages = catalog.get("packages")
    if not isinstance(packages, list):
        return
    for package in packages:
        if not isinstance(package, Mapping):
            continue
        package_name = str(package.get("name", ""))
        modules = package.get("modules")
        if not isinstance(modules, list):
            continue
        for module in modules:
            if not isinstance(module, Mapping):
                continue
            module_name = str(module.get("qualified") or module.get("name") or "")
            symbols = module.get("symbols")
            if not isinstance(symbols, list):
                continue
            for symbol in symbols:
                if isinstance(symbol, Mapping):
                    yield package_name, module_name, symbol


def _extract_agent_hints_payload(symbol: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return normalized agent hints for a symbol."""
    intent_tags: list[str] = []
    tests_to_run: list[str] = []
    agent_hints = symbol.get("agent_hints")
    if isinstance(agent_hints, Mapping):
        raw_tags = agent_hints.get("intent_tags")
        if isinstance(raw_tags, list):
            intent_tags = [str(tag) for tag in raw_tags if tag]
        raw_tests = agent_hints.get("tests_to_run")
        if isinstance(raw_tests, list):
            tests_to_run = [str(test) for test in raw_tests if test]
    return intent_tags, tests_to_run


def _extract_anchor_lines(symbol: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Return anchor line information for a symbol payload."""
    anchors = symbol.get("anchors")
    start_line: int | None = None
    end_line: int | None = None
    if isinstance(anchors, Mapping):
        raw_start = anchors.get("start_line")
        raw_end = anchors.get("end_line")
        if isinstance(raw_start, int):
            start_line = raw_start
        if isinstance(raw_end, int):
            end_line = raw_end
    return start_line, end_line


def _build_document_from_payload(
    package_name: str,
    module_name: str,
    symbol: Mapping[str, Any],
    symbol_id: str,
    row: int,
) -> SearchDocument:
    """Construct a search document from serialized symbol payloads."""
    docfacts_payload = symbol.get("docfacts")
    summary, docstring = _extract_docfacts_text(
        docfacts_payload if isinstance(docfacts_payload, Mapping) else None
    )
    intent_tags, tests_to_run = _extract_agent_hints_payload(symbol)
    qname_value = _stringify(symbol.get("qname")) or symbol_id
    text_parts = [
        qname_value,
        module_name,
        package_name,
        summary or "",
        docstring or "",
        " ".join(intent_tags),
        " ".join(tests_to_run),
    ]
    normalized_text = " ".join(part for part in text_parts if part)
    tokens = collections.Counter(_tokenize(normalized_text))
    start_line, end_line = _extract_anchor_lines(symbol)
    metrics = symbol.get("metrics")
    stability = None
    deprecated = False
    if isinstance(metrics, Mapping):
        stability = _stringify(metrics.get("stability"))
        deprecated = bool(metrics.get("deprecated"))
    return SearchDocument(
        symbol_id=symbol_id,
        package=package_name,
        module=module_name,
        qname=qname_value,
        kind=str(symbol.get("kind", "object")),
        stability=stability,
        deprecated=deprecated,
        summary=summary,
        docstring=docstring,
        anchor_start=start_line,
        anchor_end=end_line,
        text=normalized_text,
        tokens=tokens,
        row=row,
    )


def search_catalog(
    catalog: Mapping[str, Any],
    *,
    repo_root: Path,
    query: str,
    k: int = 10,
    options: SearchOptions | None = None,
) -> list[SearchResult]:
    """Execute hybrid lexical/vector search against the catalog."""
    try:
        return catalog_search.search_catalog(
            catalog, repo_root=repo_root, query=query, k=k, options=options
        )
    except CatalogSearchError as exc:
        raise CatalogBuildError(str(exc)) from exc


def load_catalog(path: Path, *, load_shards: bool = True) -> dict[str, Any]:
    """Load a catalog JSON file and expand shards if requested."""
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    shards = data.get("shards")
    if load_shards and not data.get("packages") and isinstance(shards, Mapping):
        packages: list[dict[str, Any]] = []
        base_dir = path.parent
        for entry in shards.get("packages", []):
            if not isinstance(entry, Mapping):
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                continue
            shard_path = Path(raw_path)
            if not shard_path.is_absolute():
                shard_path = (base_dir / shard_path).resolve()
            shard_data = cast(
                dict[str, Any],
                json.loads(shard_path.read_text(encoding="utf-8")),
            )
            packages.append(shard_data)
        data["packages"] = packages
    return data


def _parse_facet_args(values: Sequence[str]) -> dict[str, str]:
    """Parse CLI facet arguments into a mapping."""
    facets: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            message = f"Facet must be expressed as key=value: {raw}"
            raise CatalogBuildError(message)
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            message = f"Facet must include both key and value: {raw}"
            raise CatalogBuildError(message)
        facets[key] = value
    return facets


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    builder = AgentCatalogBuilder(args)
    if args.search_query:
        try:
            facets = _parse_facet_args(args.search_facet)
            catalog_path = builder._resolve_artifact_path(args.output)
            catalog_data = load_catalog(catalog_path, load_shards=True)
            options = SearchOptions(
                alpha=float(args.search_alpha),
                facets=facets,
                candidate_pool=int(args.search_candidates),
            )
            results = search_catalog(
                catalog_data,
                repo_root=builder.repo_root,
                query=args.search_query,
                k=max(1, int(args.search_k)),
                options=options,
            )
        except CatalogBuildError as exc:
            logger.exception("Catalog build failed")
            sys.stderr.write(f"error: {exc}\n")
            return 1
        logger.info("Search results: %d", len(results))
        sys.stdout.write(
            json.dumps([dataclasses.asdict(result) for result in results], indent=2) + "\n"
        )
        return 0
    try:
        catalog = builder.build()
        builder.write(catalog, args.output, args.schema)
    except CatalogBuildError as exc:
        logger.exception("Catalog build failed")
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _normalize_ast(node: ast.AST) -> str:
    """Return a normalized AST dump with docstrings stripped."""
    transformer = _DocstringStripper()
    cleaned = transformer.visit(copy.deepcopy(node))
    return ast.dump(cleaned, include_attributes=False)


def _attach_parents(tree: ast.AST) -> None:
    """Attach parent references to AST nodes for method detection."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


DocNode = TypeVar("DocNode", ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstrings from AST nodes."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._strip_doc(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._strip_doc(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._strip_doc(node)

    def _strip_doc(self, node: DocNode) -> DocNode:
        updated = cast(DocNode, self.generic_visit(node))
        body = list(updated.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            updated.body = body[1:]
        return updated


if __name__ == "__main__":
    sys.exit(main())
