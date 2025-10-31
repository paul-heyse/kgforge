"""High-level client for querying the Agent Catalog."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kgfoundry.agent_catalog import search as catalog_search
from kgfoundry.agent_catalog.models import (
    AgentCatalogModel,
    ChangeImpactModel,
    ModuleModel,
    PackageModel,
    SymbolModel,
    load_catalog_model,
)


class AgentCatalogClientError(RuntimeError):
    """Raised when catalog operations cannot be completed."""


class AgentCatalogClient:
    """Typed client for interacting with the Agent Catalog."""

    def __init__(
        self,
        catalog: AgentCatalogModel,
        *,
        repo_root: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._catalog_path = catalog_path
        self._repo_root = repo_root or (catalog_path.parent if catalog_path else Path.cwd())

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        repo_root: Path | None = None,
        load_shards: bool = True,
    ) -> AgentCatalogClient:
        """Construct a client by loading a catalog JSON artifact."""
        model = load_catalog_model(path, load_shards=load_shards)
        root = repo_root or path.parent
        return cls(model, repo_root=root, catalog_path=path)

    def _require_symbol(self, symbol_id: str) -> SymbolModel:
        """Return the symbol for ``symbol_id`` or raise a descriptive error."""
        symbol = self.get_symbol(symbol_id)
        if symbol is None:
            message = f"Unknown symbol: {symbol_id}"
            raise AgentCatalogClientError(message)
        return symbol

    def _require_module(self, qualified: str) -> ModuleModel:
        """Return the module for ``qualified`` or raise a descriptive error."""
        module = self.get_module(qualified)
        if module is None:
            message = f"Unknown module: {qualified}"
            raise AgentCatalogClientError(message)
        return module

    @property
    def catalog(self) -> AgentCatalogModel:
        """Return the underlying catalog model."""
        return self._catalog

    @property
    def repo_root(self) -> Path:
        """Return the repository root used for relative links."""
        return self._repo_root

    def list_packages(self) -> list[PackageModel]:
        """Return the packages present in the catalog."""
        return list(self._catalog.packages)

    def list_modules(self, package: str | PackageModel) -> list[ModuleModel]:
        """Return modules for the specified package."""
        package_name = package.name if isinstance(package, PackageModel) else package
        for pkg in self._catalog.packages:
            if pkg.name == package_name:
                return list(pkg.modules)
        message = f"Unknown package: {package_name}"
        raise AgentCatalogClientError(message)

    def get_module(self, qualified: str) -> ModuleModel | None:
        """Return module metadata for ``qualified``."""
        return self._catalog.get_module(qualified)

    def iter_symbols(self) -> Iterable[SymbolModel]:
        """Yield all symbol records within the catalog."""
        yield from self._catalog.iter_symbols()

    def get_symbol(self, symbol_id: str) -> SymbolModel | None:
        """Return symbol metadata for ``symbol_id``."""
        return self._catalog.get_symbol(symbol_id)

    def find_callers(self, symbol_id: str) -> list[str]:
        """Return callers recorded for ``symbol_id``."""
        symbol = self._require_symbol(symbol_id)
        return list(symbol.change_impact.callers)

    def find_callees(self, symbol_id: str) -> list[str]:
        """Return callees recorded for ``symbol_id``."""
        symbol = self._require_symbol(symbol_id)
        return list(symbol.change_impact.callees)

    def change_impact(self, symbol_id: str) -> ChangeImpactModel:
        """Return change impact metadata for ``symbol_id``."""
        symbol = self._require_symbol(symbol_id)
        return symbol.change_impact

    def suggest_tests(self, symbol_id: str) -> list[dict[str, Any]]:
        """Return suggested tests to run for ``symbol_id``."""
        return list(self.change_impact(symbol_id).tests)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        facets: dict[str, str] | None = None,
    ) -> list[catalog_search.SearchResult]:
        """Execute hybrid search against the catalog."""
        options = catalog_search.SearchOptions(facets=facets)
        request = catalog_search.SearchRequest(
            repo_root=self.repo_root,
            query=query,
            k=k,
        )
        return catalog_search.search_catalog(
            self._catalog.model_dump(),
            request=request,
            options=options,
        )

    def open_anchor(self, symbol_id: str) -> dict[str, str]:
        """Return anchor URLs for the requested symbol."""
        symbol = self._require_symbol(symbol_id)
        module_name = symbol.qname.rsplit(".", 1)[0] if "." in symbol.qname else symbol.qname
        module = self._require_module(module_name)
        source_path = module.source.get("path")
        if not source_path:
            message = "Symbol source path missing from catalog"
            raise AgentCatalogClientError(message)
        start_line = symbol.anchors.start_line or 1
        link_policy = self._catalog.link_policy
        editor_template = link_policy.get("editor_template", "vscode://file/{path}:{line}")
        github_template = link_policy.get(
            "github_template",
            "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}",
        )
        rel_path = Path(source_path)
        if rel_path.is_absolute():
            rel_path = rel_path.relative_to(self.repo_root)
        editor_link = editor_template.format(path=str(self.repo_root / rel_path), line=start_line)
        github_vars = link_policy.get("github") or {}
        github_link = github_template.format(
            org=github_vars.get("org", ""),
            repo=github_vars.get("repo", ""),
            sha=github_vars.get("sha", ""),
            path=str(rel_path),
            line=start_line,
        )
        return {"editor": editor_link, "github": github_link}


__all__ = [
    "AgentCatalogClient",
    "AgentCatalogClientError",
]
