"""Generate docs/_build/analytics.json summarising catalog generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec
from tools import validate_tools_payload
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

from kgfoundry.agent_catalog.models import load_catalog_payload


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
        return msgspec.json.decode(raw, type=AgentAnalyticsDocument)
    except msgspec.DecodeError:
        try:
            legacy = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        return _legacy_to_document(legacy)


def _catalog_metrics(catalog: Mapping[str, Any]) -> CatalogMetrics:
    packages = catalog.get("packages")
    package_count = 0
    module_count = 0
    symbol_count = 0
    if isinstance(packages, list):
        for package in packages:
            if not isinstance(package, Mapping):
                continue
            package_count += 1
            modules = package.get("modules")
            if not isinstance(modules, list):
                continue
            for module in modules:
                if not isinstance(module, Mapping):
                    continue
                module_count += 1
                symbols = module.get("symbols")
                if isinstance(symbols, list):
                    symbol_count += len(symbols)
    shards = catalog.get("shards")
    shard_count = 0
    if isinstance(shards, Mapping):
        shard_entries = shards.get("packages")
        if isinstance(shard_entries, list):
            shard_count = len(shard_entries)
    return CatalogMetrics(
        packages=package_count,
        modules=module_count,
        symbols=symbol_count,
        shards=shard_count,
    )


def _iter_catalog_modules(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return all module mappings contained in ``catalog``."""
    modules: list[Mapping[str, Any]] = []
    packages = catalog.get("packages")
    if not isinstance(packages, list):
        return modules
    for package in packages:
        if not isinstance(package, Mapping):
            continue
        package_modules = package.get("modules")
        if not isinstance(package_modules, list):
            continue
        for module in package_modules:
            if isinstance(module, Mapping):
                modules.append(module)
    return modules


def _module_source_issue(module: Mapping[str, Any], repo_root: Path) -> BrokenLinkDetail | None:
    """Return a broken link detail when the module source path is missing."""
    source = module.get("source")
    if not isinstance(source, Mapping):
        return None
    path_value = source.get("path")
    if not isinstance(path_value, str):
        return None
    if (repo_root / path_value).exists():
        return None
    return BrokenLinkDetail(module=str(module.get("qualified", "")), path=path_value)


def _module_page_issues(module: Mapping[str, Any], repo_root: Path) -> list[BrokenLinkDetail]:
    """Return missing documentation pages for ``module``."""
    pages = module.get("pages")
    if not isinstance(pages, Mapping):
        return []
    issues: list[BrokenLinkDetail] = []
    module_name = str(module.get("qualified", ""))
    for key, value in pages.items():
        if not isinstance(value, str):
            continue
        if not (repo_root / value).exists():
            issues.append(BrokenLinkDetail(module=module_name, page=value, kind=str(key)))
    return issues


def _check_links(
    catalog: Mapping[str, Any], repo_root: Path, sample: int
) -> list[BrokenLinkDetail]:
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


def build_analytics(args: argparse.Namespace) -> AgentAnalyticsDocument:
    """Return the analytics document for the current catalog snapshot."""
    catalog = load_catalog_payload(args.catalog, load_shards=True)
    previous = _load_previous(args.output)
    metrics = _catalog_metrics(catalog)
    broken_links = _check_links(catalog, args.repo_root, args.link_sample)
    return AgentAnalyticsDocument(
        repo=RepoInfo(root=str(args.repo_root)),
        catalog=metrics,
        portal=PortalAnalytics(sessions=_portal_sessions(previous)),
        errors=AnalyticsErrors(broken_links=len(broken_links), details=broken_links),
    )


def write_analytics(args: argparse.Namespace) -> None:
    """Persist the analytics document to disk after schema validation."""
    document = build_analytics(args)
    payload = msgspec.to_builtins(document)
    validate_tools_payload(payload, ANALYTICS_SCHEMA)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2)
    output.write_text(encoded, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the analytics builder."""
    parser = build_parser()
    args = parser.parse_args(argv)
    write_analytics(args)
    return 0


def _legacy_to_document(payload: Mapping[str, Any]) -> AgentAnalyticsDocument:
    """Convert legacy analytics payloads into the typed document model."""
    repo_root = str(payload.get("repo", {}).get("root", ""))
    catalog_section = payload.get("catalog", {})
    portal_section = payload.get("portal", {})
    error_section = payload.get("errors", {})

    metrics = CatalogMetrics(
        packages=int(catalog_section.get("packages", 0)),
        modules=int(catalog_section.get("modules", 0)),
        symbols=int(catalog_section.get("symbols", 0)),
        shards=int(catalog_section.get("shards", 0)),
    )

    sessions_payload = (
        portal_section.get("sessions", {}) if isinstance(portal_section, Mapping) else {}
    )
    sessions = PortalSessions(
        builds=int(sessions_payload.get("builds", 0)),
        unique_users=int(sessions_payload.get("unique_users", 0)),
    )

    broken_links_payload = (
        error_section.get("details", []) if isinstance(error_section, Mapping) else []
    )
    broken_links: list[BrokenLinkDetail] = []
    if isinstance(broken_links_payload, list):
        for item in broken_links_payload:
            if not isinstance(item, Mapping):
                continue
            broken_links.append(
                BrokenLinkDetail(
                    module=str(item.get("module", "")),
                    path=item.get("path"),
                    page=item.get("page"),
                    kind=item.get("kind"),
                )
            )

    return AgentAnalyticsDocument(
        generatedAt=str(payload.get("generated_at", datetime.now(tz=UTC).isoformat())),
        repo=RepoInfo(root=repo_root),
        catalog=metrics,
        portal=PortalAnalytics(sessions=sessions),
        errors=AnalyticsErrors(
            broken_links=(
                int(error_section.get("broken_links", len(broken_links)))
                if isinstance(error_section, Mapping)
                else len(broken_links)
            ),
            details=broken_links,
        ),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
