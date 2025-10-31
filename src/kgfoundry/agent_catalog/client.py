"""High-level client for querying the Agent Catalog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from kgfoundry.agent_catalog import search as catalog_search
from kgfoundry.agent_catalog.models import (
    AgentCatalogModel,
    ChangeImpactModel,
    ModuleModel,
    PackageModel,
    SymbolModel,
    load_catalog_model,
)
from kgfoundry_common.problem_details import JsonObject, JsonValue


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

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return ""


_DEFAULT_EDITOR_TEMPLATE = "vscode://file/{path}:{line}"
_DEFAULT_GITHUB_TEMPLATE = "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}"


def _coerce_str(value: object, default: str = "") -> str:
    """Return *value* if it is a string; otherwise return *default*."""
    return value if isinstance(value, str) else default


def _coerce_mapping(value: object) -> Mapping[str, object]:
    """Return *value* as a mapping when possible; otherwise return an empty mapping."""
    return value if isinstance(value, Mapping) else {}


DEFAULT_EDITOR_TEMPLATE = "vscode://file/{path}:{line}"
DEFAULT_GITHUB_TEMPLATE = "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}"


class AgentCatalogClientError(RuntimeError):
    """Document AgentCatalogClientError.

    &lt;!-- auto:docstring-builder v1 --&gt;

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
    """


class AgentCatalogClient:
    """Typed client for interacting with the Agent Catalog.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog : AgentCatalogModel
        Describe ``catalog``.
    repo_root : Path | None, optional
        Describe ``repo_root``.
        Defaults to ``None``.
    catalog_path : Path | None, optional
        Describe ``catalog_path``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        catalog: AgentCatalogModel,
        *,
        repo_root: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        """Document   init  .

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        catalog : AgentCatalogModel
            Configure the catalog.
        repo_root : Path | NoneType, optional
            Configure the repo root. Defaults to ``None``.
            Defaults to ``None``.
        catalog_path : Path | NoneType, optional
            Filesystem path for the catalog. Defaults to ``None``.
            Defaults to ``None``.
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

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        path : Path
            Describe ``path``.
        repo_root : Path | NoneType, optional
            Describe ``repo_root``.
            Defaults to ``None``.
        load_shards : bool, optional
            Describe ``load_shards``.
            Defaults to ``True``.

        Returns
        -------
        AgentCatalogClient
            Describe return value.
        """
        model = load_catalog_model(path, load_shards=load_shards)
        root = repo_root or path.parent
        return cls(model, repo_root=root, catalog_path=path)

    def _require_symbol(self, symbol_id: str) -> SymbolModel:
        """Return the symbol for ``symbol_id`` or raise a descriptive error.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        SymbolModel
            Describe return value.
        """
        symbol = self.get_symbol(symbol_id)
        if symbol is None:
            message = f"Unknown symbol: {symbol_id}"
            raise AgentCatalogClientError(message)
        return symbol

    def _require_module(self, qualified: str) -> ModuleModel:
        """Return the module for ``qualified`` or raise a descriptive error.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        qualified : str
            Describe ``qualified``.

        Returns
        -------
        ModuleModel
            Describe return value.
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

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        list[PackageModel]
            Describe return value.
        """
        return list(self._catalog.packages)

    def list_modules(self, package: str | PackageModel) -> list[ModuleModel]:
        """Return modules for the specified package.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        package : str | PackageModel
            Describe ``package``.

        Returns
        -------
        list[ModuleModel]
            Describe return value.
        """
        package_name = package.name if isinstance(package, PackageModel) else package
        for pkg in self._catalog.packages:
            if pkg.name == package_name:
                return list(pkg.modules)
        message = f"Unknown package: {package_name}"
        raise AgentCatalogClientError(message)

    def get_module(self, qualified: str) -> ModuleModel | None:
        """Return module metadata for ``qualified``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        qualified : str
            Describe ``qualified``.

        Returns
        -------
        ModuleModel | NoneType
            Describe return value.
        """
        return self._catalog.get_module(qualified)

    def iter_symbols(self) -> Iterable[SymbolModel]:
        """Yield all symbol records within the catalog.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        SymbolModel
            Describe return value.
        """
        yield from self._catalog.iter_symbols()

    def get_symbol(self, symbol_id: str) -> SymbolModel | None:
        """Return symbol metadata for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        SymbolModel | NoneType
            Describe return value.
        """
        return self._catalog.get_symbol(symbol_id)

    def find_callers(self, symbol_id: str) -> list[str]:
        """Return callers recorded for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        list[str]
            Describe return value.
        """
        symbol = self._require_symbol(symbol_id)
        return list(symbol.change_impact.callers)

    def find_callees(self, symbol_id: str) -> list[str]:
        """Return callees recorded for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        list[str]
            Describe return value.
        """
        symbol = self._require_symbol(symbol_id)
        return list(symbol.change_impact.callees)

    def change_impact(self, symbol_id: str) -> ChangeImpactModel:
        """Return change impact metadata for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        ChangeImpactModel
            Describe return value.
        """
        symbol = self._require_symbol(symbol_id)
        return symbol.change_impact

    def suggest_tests(self, symbol_id: str) -> list[dict[str, JsonValue]]:
        """Return suggested tests to run for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        list[dict[str, object]]
            Describe return value.
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
    ) -> list[catalog_search.SearchResult]:
        """Execute hybrid search against the catalog.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int, optional
            Describe ``k``.
            Defaults to ``10``.
        facets : dict[str, str] | NoneType, optional
            Describe ``facets``.
            Defaults to ``None``.

        Returns
        -------
        list[catalog_search.SearchResult]
            Describe return value.
        """
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
        """Return anchor URLs for the requested symbol.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        dict[str, str]
            Describe return value.
        """
        symbol = self._require_symbol(symbol_id)
        module_name = symbol.qname.rsplit(".", 1)[0] if "." in symbol.qname else symbol.qname
        module = self._require_module(module_name)
        source_path = module.source.get("path")
        if not source_path:
            message = "Symbol source path missing from catalog"
            raise AgentCatalogClientError(message)
        start_line = symbol.anchors.start_line or 1

        raw_policy = cast(JsonObject | None, self._catalog.link_policy)
        policy: LinkPolicy = cast(LinkPolicy, raw_policy)
        editor_template = _coerce_str(policy.get("editor_template"), _DEFAULT_EDITOR_TEMPLATE)
        github_template = _coerce_str(policy.get("github_template"), _DEFAULT_GITHUB_TEMPLATE)
        rel_path = Path(source_path)
        if rel_path.is_absolute():
            rel_path = rel_path.relative_to(self.repo_root)
        editor_vars = _SafeFormatDict({"path": str(self.repo_root / rel_path), "line": start_line})
        try:
            editor_link = editor_template.format_map(editor_vars)
        except Exception as exc:
            message = "Invalid editor_template in catalog link_policy"
            raise AgentCatalogClientError(message) from exc

        raw_github = _coerce_mapping(policy.get("github"))
        github_policy = cast(GithubLinkPolicy, raw_github)
        github_vars = _SafeFormatDict(
            {
                "org": _coerce_str(github_policy.get("org")),
                "repo": _coerce_str(github_policy.get("repo")),
                "sha": _coerce_str(github_policy.get("sha")),
                "path": str(rel_path),
                "line": start_line,
            }
        )
        try:
            github_link = github_template.format_map(github_vars)
        except Exception as exc:
            message = "Invalid github_template in catalog link_policy"
            raise AgentCatalogClientError(message) from exc
        return {"editor": editor_link, "github": github_link}


__all__ = [
    "AgentCatalogClient",
    "AgentCatalogClientError",
]


def _normalize_link_policy(raw_policy: JsonObject | None) -> LinkPolicy:
    """Normalize catalog link policy structure."""
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

    return cast(LinkPolicy, normalized)


def _format_template(template: str, context: Mapping[str, object], fallback: str) -> str:
    """Format a template with graceful fallback on KeyError/ValueError."""
    safe_context = {key: str(value) for key, value in context.items()}
    try:
        return template.format(**safe_context)
    except KeyError:
        return fallback
    except ValueError:
        return fallback
