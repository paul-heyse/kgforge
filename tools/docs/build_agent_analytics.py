"""Generate docs/_build/analytics.json summarising catalog generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import msgspec
from msgspec import DecodeError
from msgspec import json as msgspec_json

if TYPE_CHECKING:
    from tools import validate_tools_payload as _validate_tools_payload
else:
    from tools import validate_tools_payload as _validate_tools_payload

from kgfoundry.agent_catalog.models import load_catalog_payload
from tools.docs.analytics_models import (
    ANALYTICS_SCHEMA,
    AgentAnalyticsDocument,
    AnalyticsErrors,
    BrokenLinkDetail,
    CatalogMetrics,
    PortalAnalytics,
    PortalSessions,
    RepoInfo,
)

type JSONMapping = Mapping[str, object]

validate_tools_payload: Callable[[Mapping[str, object], str], None]
validate_tools_payload = _validate_tools_payload


def _get_mapping(value: object) -> JSONMapping | None:
    if isinstance(value, Mapping):
        return cast(JSONMapping, value)
    return None


def _get_list(value: object) -> list[object] | None:
    return value if isinstance(value, list) else None


def _int_or_default(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


@dataclass(frozen=True, slots=True)
class CatalogModuleSnapshot:
    """Normalized view of catalog module metadata used during analytics."""

    qualified: str
    source_path: str | None
    pages: dict[str, str]
    symbol_count: int


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the analytics builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/_build/agent_catalog.json"),
        help="Path to the agent catalog JSON artefact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_build/analytics.json"),
        help="Destination for the analytics JSON file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to validate relative paths.",
    )
    parser.add_argument(
        "--link-sample",
        type=int,
        default=25,
        help="Number of catalog links to validate when computing errors.",
    )
    return parser


def _load_previous(path: Path) -> AgentAnalyticsDocument | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        decoded: AgentAnalyticsDocument = msgspec_json.decode(raw, type=AgentAnalyticsDocument)
    except DecodeError:
        try:
            legacy_raw: object = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        legacy_mapping = _get_mapping(legacy_raw)
        if legacy_mapping is None:
            return None
        return _legacy_to_document(legacy_mapping)
    return decoded


def _snapshot_module(module: JSONMapping) -> CatalogModuleSnapshot:
    qualified_value = module.get("qualified")
    qualified = str(qualified_value) if qualified_value is not None else ""

    source_mapping = _get_mapping(module.get("source"))
    path_value_obj = source_mapping.get("path") if source_mapping is not None else None
    source_path = path_value_obj if isinstance(path_value_obj, str) else None

    pages_mapping = _get_mapping(module.get("pages"))
    pages: dict[str, str] = {}
    if pages_mapping is not None:
        pages = {
            str(kind): value
            for kind, value in pages_mapping.items()
            if isinstance(kind, str) and isinstance(value, str)
        }

    symbols = _get_list(module.get("symbols"))
    symbol_count = len(symbols) if symbols is not None else 0

    return CatalogModuleSnapshot(
        qualified=qualified,
        source_path=source_path,
        pages=pages,
        symbol_count=symbol_count,
    )


def _package_modules(package: JSONMapping) -> list[CatalogModuleSnapshot]:
    modules: list[CatalogModuleSnapshot] = []
    module_list = _get_list(package.get("modules"))
    if module_list is None:
        return modules
    for module in module_list:
        module_mapping = _get_mapping(module)
        if module_mapping is None:
            continue
        modules.append(_snapshot_module(module_mapping))
    return modules


def _catalog_metrics(catalog: JSONMapping) -> CatalogMetrics:
    packages = _get_list(catalog.get("packages"))
    package_count = 0
    module_count = 0
    symbol_count = 0
    if packages is not None:
        for package in packages:
            package_mapping = _get_mapping(package)
            if package_mapping is None:
                continue
            package_count += 1
            package_modules = _package_modules(package_mapping)
            module_count += len(package_modules)
            symbol_count += sum(item.symbol_count for item in package_modules)

    shard_count = 0
    shards_mapping = _get_mapping(catalog.get("shards"))
    if shards_mapping is not None:
        shard_entries = _get_list(shards_mapping.get("packages"))
        if shard_entries is not None:
            shard_count = len(shard_entries)

    return CatalogMetrics(
        packages=package_count,
        modules=module_count,
        symbols=symbol_count,
        shards=shard_count,
    )


def _iter_catalog_modules(catalog: JSONMapping) -> list[CatalogModuleSnapshot]:
    """Return all module snapshots contained in ``catalog``."""
    modules: list[CatalogModuleSnapshot] = []
    packages = _get_list(catalog.get("packages"))
    if packages is None:
        return modules
    for package in packages:
        package_mapping = _get_mapping(package)
        if package_mapping is None:
            continue
        modules.extend(_package_modules(package_mapping))
    return modules


def _module_source_issue(module: CatalogModuleSnapshot, repo_root: Path) -> BrokenLinkDetail | None:
    """Return a broken link detail when the module source path is missing."""
    path_value = module.source_path
    if path_value is None:
        return None
    if (repo_root / path_value).exists():
        return None
    return BrokenLinkDetail(module=module.qualified, path=path_value)


def _module_page_issues(module: CatalogModuleSnapshot, repo_root: Path) -> list[BrokenLinkDetail]:
    """Return missing documentation pages for ``module``."""
    issues: list[BrokenLinkDetail] = []
    for kind, value in module.pages.items():
        if (repo_root / value).exists():
            continue
        issues.append(BrokenLinkDetail(module=module.qualified, page=value, kind=kind))
    return issues


def _check_links(catalog: JSONMapping, repo_root: Path, sample: int) -> list[BrokenLinkDetail]:
    broken: list[BrokenLinkDetail] = []
    for idx, module in enumerate(_iter_catalog_modules(catalog)):
        source_issue = _module_source_issue(module, repo_root)
        if source_issue is not None:
            broken.append(source_issue)
        broken.extend(_module_page_issues(module, repo_root))
        if idx + 1 >= sample:
            break
    return broken


def _portal_sessions(previous: AgentAnalyticsDocument | None) -> PortalSessions:
    if previous is None:
        return PortalSessions(builds=1, unique_users=0)
    builds = previous.portal.sessions.builds + 1
    return PortalSessions(builds=builds, unique_users=previous.portal.sessions.unique_users)


def _legacy_repo_root(payload: JSONMapping) -> str:
    repo_section = _get_mapping(payload.get("repo"))
    if repo_section is None:
        return ""
    root_value = repo_section.get("root")
    return root_value if isinstance(root_value, str) else ""


def _legacy_catalog_section_metrics(section: JSONMapping | None) -> CatalogMetrics:
    return CatalogMetrics(
        packages=_int_or_default(section.get("packages") if section is not None else 0),
        modules=_int_or_default(section.get("modules") if section is not None else 0),
        symbols=_int_or_default(section.get("symbols") if section is not None else 0),
        shards=_int_or_default(section.get("shards") if section is not None else 0),
    )


def _legacy_portal_sessions(section: JSONMapping | None) -> PortalSessions:
    sessions_mapping = _get_mapping(section.get("sessions")) if section is not None else None
    return PortalSessions(
        builds=_int_or_default(sessions_mapping.get("builds") if sessions_mapping else 0),
        unique_users=_int_or_default(
            sessions_mapping.get("unique_users") if sessions_mapping else 0
        ),
    )


def _broken_link_from_mapping(mapping: JSONMapping) -> BrokenLinkDetail:
    module_name_raw = mapping.get("module")
    module_name = str(module_name_raw) if module_name_raw is not None else ""
    path_value = mapping.get("path")
    page_value = mapping.get("page")
    kind_value = mapping.get("kind")
    return BrokenLinkDetail(
        module=module_name,
        path=path_value if isinstance(path_value, str) else None,
        page=page_value if isinstance(page_value, str) else None,
        kind=kind_value if isinstance(kind_value, str) else None,
    )


def _legacy_error_bundle(section: JSONMapping | None) -> tuple[list[BrokenLinkDetail], int]:
    broken_links: list[BrokenLinkDetail] = []
    details = _get_list(section.get("details")) if section is not None else None
    if details is not None:
        for item in details:
            item_mapping = _get_mapping(item)
            if item_mapping is None:
                continue
            broken_links.append(_broken_link_from_mapping(item_mapping))

    total = _int_or_default(
        section.get("broken_links") if section is not None else None,
        default=len(broken_links),
    )
    return broken_links, total


def _legacy_generated_at(value: object) -> str:
    return value if isinstance(value, str) else datetime.now(tz=UTC).isoformat()


def build_analytics(args: argparse.Namespace) -> AgentAnalyticsDocument:
    """Return the analytics document for the current catalog snapshot."""
    catalog_path = cast(Path, args.catalog)
    output_path = cast(Path, args.output)
    repo_root = cast(Path, args.repo_root)
    link_sample = cast(int, args.link_sample)

    catalog_payload: object = load_catalog_payload(catalog_path, load_shards=True)
    catalog = cast(JSONMapping, catalog_payload)
    previous = _load_previous(output_path)
    metrics = _catalog_metrics(catalog)
    broken_links = _check_links(catalog, repo_root, link_sample)
    return AgentAnalyticsDocument(
        repo=RepoInfo(root=str(repo_root)),
        catalog=metrics,
        portal=PortalAnalytics(sessions=_portal_sessions(previous)),
        errors=AnalyticsErrors(broken_links=len(broken_links), details=broken_links),
    )


def write_analytics(args: argparse.Namespace) -> None:
    """Persist the analytics document to disk after schema validation."""
    document = build_analytics(args)
    payload = cast(dict[str, object], msgspec.to_builtins(document))
    validate_tools_payload(payload, ANALYTICS_SCHEMA)
    output = cast(Path, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2)
    output.write_text(encoded, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the analytics builder."""
    parser = build_parser()
    args = parser.parse_args(argv)
    write_analytics(args)
    return 0


def _legacy_to_document(payload: JSONMapping) -> AgentAnalyticsDocument:
    """Convert legacy analytics payloads into the typed document model."""
    repo_root = _legacy_repo_root(payload)
    metrics = _legacy_catalog_section_metrics(_get_mapping(payload.get("catalog")))
    sessions = _legacy_portal_sessions(_get_mapping(payload.get("portal")))
    broken_links, broken_links_total = _legacy_error_bundle(_get_mapping(payload.get("errors")))
    generated_at = _legacy_generated_at(payload.get("generated_at"))

    return AgentAnalyticsDocument(
        generatedAt=generated_at,
        repo=RepoInfo(root=repo_root),
        catalog=metrics,
        portal=PortalAnalytics(sessions=sessions),
        errors=AnalyticsErrors(broken_links=broken_links_total, details=broken_links),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
