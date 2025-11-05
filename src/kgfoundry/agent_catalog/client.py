"""High-level client for querying the Agent Catalog."""
# [nav:section public-api]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from kgfoundry.agent_catalog import search as catalog_search
from kgfoundry.agent_catalog.models import (
    load_catalog_model,
)
from kgfoundry_common.navmap_loader import load_nav_metadata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kgfoundry.agent_catalog.models import (
        AgentCatalogModel,
        ChangeImpactModel,
        ModuleModel,
        PackageModel,
        SymbolModel,
    )
    from kgfoundry.agent_catalog.search import (
        SearchOptions,
        SearchRequest,
        SearchResult,
        build_faceted_search_options,
        search_catalog,
    )
    from kgfoundry_common.problem_details import JsonObject, JsonValue
else:
    SearchOptions = catalog_search.SearchOptions
    SearchRequest = catalog_search.SearchRequest
    SearchResult = catalog_search.SearchResult
    search_catalog = catalog_search.search_catalog
    build_faceted_search_options = catalog_search.build_faceted_search_options


class GithubLinkPolicy(TypedDict, total=False):
    """Optional GitHub link configuration."""

    org: str
    repo: str
    sha: str


class LinkPolicy(TypedDict, total=False):
    """Catalog link policy schema."""

    editor_template: str
    github_template: str
    github: NotRequired[GithubLinkPolicy]


class _SafeFormatDict(dict[str, object]):
    """Dictionary that returns an empty string for missing keys."""

    def __missing__(self, key: str) -> object:
        """Return an empty string for missing keys.

        Parameters
        ----------
        key : str
            Missing dictionary key.

        Returns
        -------
        str
            Empty string ("").
        """
        return ""


_DEFAULT_EDITOR_TEMPLATE = "vscode://file/{path}:{line}"
_DEFAULT_GITHUB_TEMPLATE = "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}"


_EMPTY_GITHUB_POLICY: GithubLinkPolicy = cast("GithubLinkPolicy", {})


def _coerce_str(value: object, default: str = "") -> str:
    """Return *value* if it is a string; otherwise return *default*.

    Parameters
    ----------
    value : object
        Value to coerce.
    default : str, optional
        Default value if coercion fails. Defaults to empty string.

    Returns
    -------
    str
        Coerced string value.
    """
    return value if isinstance(value, str) else default


@dataclass(slots=True)
class _SymbolAnchorContext:
    """Resolved source location data for rendering anchor links."""

    absolute_path: Path
    relative_path: Path
    line: int


DEFAULT_EDITOR_TEMPLATE = "vscode://file/{path}:{line}"
DEFAULT_GITHUB_TEMPLATE = "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}"


# [nav:anchor AgentCatalogClientError]
class AgentCatalogClientError(RuntimeError):
    """Exception raised for Agent Catalog client errors."""


# [nav:anchor AgentCatalogClient]
class AgentCatalogClient:
    """Typed client for interacting with the Agent Catalog.

    Provides a high-level interface for querying catalog data including
    packages, modules, symbols, search, and change impact analysis.

    Parameters
    ----------
    catalog : AgentCatalogModel
        Loaded catalog model instance.
    repo_root : Path | None, optional
        Repository root for resolving relative paths. Defaults to None.
    catalog_path : Path | None, optional
        Filesystem path to the catalog file. Defaults to None.
    """

    def __init__(
        self,
        catalog: AgentCatalogModel,
        *,
        repo_root: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        """Initialize catalog client.

        Sets up the client with a catalog model and resolves repository root.

        Parameters
        ----------
        catalog : AgentCatalogModel
            Loaded catalog model instance.
        repo_root : Path | None, optional
            Repository root path. If None, uses catalog_path parent or cwd.
            Defaults to None.
        catalog_path : Path | None, optional
            Filesystem path to the catalog file. Defaults to None.
        """
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
        """Construct a client by loading a catalog JSON artifact.

        Factory method that loads a catalog from file and creates a client.

        Parameters
        ----------
        path : Path
            Path to catalog JSON file or SQLite database.
        repo_root : Path | None, optional
            Repository root path. If None, uses catalog path parent.
            Defaults to None.
        load_shards : bool, optional
            Whether to expand shards if present. Defaults to True.

        Returns
        -------
        AgentCatalogClient
            Configured client instance.
        """
        model = load_catalog_model(path, load_shards=load_shards)
        root = repo_root or path.parent
        return cls(model, repo_root=root, catalog_path=path)

    def _require_symbol(self, symbol_id: str) -> SymbolModel:
        """Return the symbol for ``symbol_id`` or raise a descriptive error.

        Internal helper that ensures a symbol exists before returning it.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        SymbolModel
            Symbol model instance.

        Raises
        ------
        AgentCatalogClientError
            If the symbol is not found.
        """
        symbol = self.get_symbol(symbol_id)
        if symbol is None:
            message = f"Unknown symbol: {symbol_id}"
            raise AgentCatalogClientError(message)
        return symbol

    def _require_module(self, qualified: str) -> ModuleModel:
        """Return the module for ``qualified`` or raise a descriptive error.

        Internal helper that ensures a module exists before returning it.

        Parameters
        ----------
        qualified : str
            Fully qualified module name.

        Returns
        -------
        ModuleModel
            Module model instance.

        Raises
        ------
        AgentCatalogClientError
            If the module is not found.
        """
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
        """Return the packages present in the catalog.

        Returns
        -------
        list[PackageModel]
            List of package models in the catalog.
        """
        return list(self._catalog.packages)

    def list_modules(self, package: str | PackageModel) -> list[ModuleModel]:
        """Return modules for the specified package.

        Lists all modules contained in a package.

        Parameters
        ----------
        package : str | PackageModel
            Package name or PackageModel instance.

        Returns
        -------
        list[ModuleModel]
            List of module models in the package.

        Raises
        ------
        AgentCatalogClientError
            If the package is not found.
        """
        package_name = package if isinstance(package, str) else package.name
        for pkg in self._catalog.packages:
            if pkg.name == package_name:
                return list(pkg.modules)
        message = f"Unknown package: {package_name}"
        raise AgentCatalogClientError(message)

    def get_module(self, qualified: str) -> ModuleModel | None:
        """Return module metadata for ``qualified``.

        Retrieves a module by its fully qualified name.

        Parameters
        ----------
        qualified : str
            Fully qualified module name.

        Returns
        -------
        ModuleModel | None
            Module model if found, None otherwise.
        """
        return self._catalog.get_module(qualified)

    def iter_symbols(self) -> Iterable[SymbolModel]:
        """Yield all symbol records within the catalog.

        Iterates through all packages and modules to yield every symbol.

        Yields
        ------
        SymbolModel
            Symbol model from the catalog.
        """
        yield from self._catalog.iter_symbols()

    def get_symbol(self, symbol_id: str) -> SymbolModel | None:
        """Return symbol metadata for ``symbol_id``.

        Retrieves a symbol by its unique identifier.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        SymbolModel | None
            Symbol model if found, None otherwise.
        """
        return self._catalog.get_symbol(symbol_id)

    def find_callers(self, symbol_id: str) -> list[str]:
        """Return callers recorded for ``symbol_id``.

        Finds all symbols that reference the given symbol.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        list[str]
            List of caller symbol IDs.
        """
        symbol = self._require_symbol(symbol_id)
        return list(symbol.change_impact.callers)

    def find_callees(self, symbol_id: str) -> list[str]:
        """Return callees recorded for ``symbol_id``.

        Finds all symbols that are called by the given symbol.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        list[str]
            List of callee symbol IDs.
        """
        symbol = self._require_symbol(symbol_id)
        return list(symbol.change_impact.callees)

    def change_impact(self, symbol_id: str) -> ChangeImpactModel:
        """Return change impact metadata for ``symbol_id``.

        Retrieves change impact analysis showing affected symbols and modules.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        ChangeImpactModel
            Change impact model with callers, callees, and test suggestions.
        """
        symbol = self._require_symbol(symbol_id)
        return symbol.change_impact

    def suggest_tests(self, symbol_id: str) -> list[dict[str, JsonValue]]:
        """Return suggested tests to run for ``symbol_id``.

        Retrieves test suggestions from change impact metadata.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        list[dict[str, JsonValue]]
            List of test suggestion dictionaries.
        """
        # ChangeImpactModel.tests is a list of dicts from Pydantic model_dump
        # Cast to JsonValue since these are JSON-serializable structures
        return list(self.change_impact(symbol_id).tests)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        facets: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Execute hybrid search against the catalog.

        Performs semantic search combining lexical and vector search
        with optional facet filters.

        Parameters
        ----------
        query : str
            Search query text.
        k : int, optional
            Number of results to return. Defaults to 10.
        facets : dict[str, str] | None, optional
            Facet filters (package, module, kind, stability). Defaults to None.

        Returns
        -------
        list[SearchResult]
            List of search results with scores and metadata.
        """
        if facets:
            options = build_faceted_search_options(facets=facets)
        else:
            options = catalog_search.SearchOptions()
        request = SearchRequest(
            repo_root=self.repo_root,
            query=query,
            k=k,
        )
        catalog_payload = cast(
            "Mapping[str, str | int | float | bool | list[object] | dict[str, object] | None]",
            self._catalog.model_dump(),
        )
        return search_catalog(
            catalog_payload,
            request=request,
            options=options,
        )

    def open_anchor(self, symbol_id: str) -> dict[str, str]:
        """Return anchor URLs for the requested symbol.

        Generates editor and GitHub links for opening the symbol in source.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        dict[str, str]
            Dictionary with "editor" and "github" keys containing URLs.
        """
        symbol = self._require_symbol(symbol_id)
        module_name = _module_name_from_qname(symbol.qname)
        module = self._require_module(module_name)

        location = _resolve_symbol_anchor_context(symbol, module, self.repo_root)
        raw_policy = cast("JsonObject | None", self._catalog.link_policy)
        policy = _normalize_link_policy(raw_policy)
        return _build_anchor_links(policy, location)


__all__ = [
    "AgentCatalogClient",
    "AgentCatalogClientError",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


def _module_name_from_qname(qname: str) -> str:
    if "." in qname:
        return qname.rsplit(".", 1)[0]
    return qname


def _resolve_symbol_anchor_context(
    symbol: SymbolModel,
    module: ModuleModel,
    repo_root: Path,
) -> _SymbolAnchorContext:
    source_payload = module.source if isinstance(module.source, Mapping) else {}
    raw_path = source_payload.get("path") if isinstance(source_payload, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path:
        message = "Symbol source path missing from catalog"
        raise AgentCatalogClientError(message)

    resolved_path = Path(raw_path)
    if resolved_path.is_absolute():
        absolute_path = resolved_path
        try:
            relative_path = resolved_path.relative_to(repo_root)
        except ValueError as exc:  # pragma: no cover - defensive guard
            message = f"Symbol source path {resolved_path} is outside repo root {repo_root}"
            raise AgentCatalogClientError(message) from exc
    else:
        relative_path = resolved_path
        absolute_path = repo_root / relative_path

    start_line = symbol.anchors.start_line
    line_number = start_line if isinstance(start_line, int) and start_line > 0 else 1
    return _SymbolAnchorContext(
        absolute_path=absolute_path, relative_path=relative_path, line=line_number
    )


def _format_anchor_template(
    template: str, context: Mapping[str, object], error_message: str
) -> str:
    render_context = _SafeFormatDict({key: str(value) for key, value in context.items()})
    try:
        return template.format_map(render_context)
    except Exception as exc:  # pragma: no cover - formatting issues are exceptional
        raise AgentCatalogClientError(error_message) from exc


def _build_anchor_links(policy: LinkPolicy, location: _SymbolAnchorContext) -> dict[str, str]:
    editor_template = _coerce_str(policy.get("editor_template"), _DEFAULT_EDITOR_TEMPLATE)
    github_template = _coerce_str(policy.get("github_template"), _DEFAULT_GITHUB_TEMPLATE)

    editor_link = _format_anchor_template(
        editor_template,
        {"path": str(location.absolute_path), "line": location.line},
        "Invalid editor_template in catalog link_policy",
    )

    github_policy = policy.get("github", _EMPTY_GITHUB_POLICY)
    github_link = _format_anchor_template(
        github_template,
        {
            "org": _coerce_str(github_policy.get("org")),
            "repo": _coerce_str(github_policy.get("repo")),
            "sha": _coerce_str(github_policy.get("sha")),
            "path": str(location.relative_path),
            "line": location.line,
        },
        "Invalid github_template in catalog link_policy",
    )

    return {"editor": editor_link, "github": github_link}


def _normalize_link_policy(raw_policy: JsonObject | None) -> LinkPolicy:
    """Normalize catalog link policy structure.

    Parameters
    ----------
    raw_policy : JsonObject | None
        Raw link policy dictionary.

    Returns
    -------
    LinkPolicy
        Normalized link policy dictionary.
    """
    if raw_policy is None or not isinstance(raw_policy, Mapping):
        return {}

    normalized: dict[str, object] = {}

    editor_template = raw_policy.get("editor_template")
    if isinstance(editor_template, str):
        normalized["editor_template"] = editor_template

    github_template = raw_policy.get("github_template")
    if isinstance(github_template, str):
        normalized["github_template"] = github_template

    github_raw = raw_policy.get("github")
    if isinstance(github_raw, Mapping):
        github_values: dict[str, str] = {}
        for key in ("org", "repo", "sha"):
            value = github_raw.get(key)
            if isinstance(value, str):
                github_values[key] = value
        if github_values:
            normalized["github"] = github_values

    return cast("LinkPolicy", normalized)
