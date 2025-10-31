"""Generate docs/_build/analytics.json summarising catalog generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def _load_previous(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _catalog_metrics(catalog: Mapping[str, Any]) -> dict[str, int]:
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
    return {
        "packages": package_count,
        "modules": module_count,
        "symbols": symbol_count,
        "shards": shard_count,
    }


def _check_links(catalog: Mapping[str, Any], repo_root: Path, sample: int) -> list[dict[str, str]]:
    broken: list[dict[str, str]] = []
    checked = 0
    packages = catalog.get("packages")
    if not isinstance(packages, list):
        return broken
    for package in packages:
        if not isinstance(package, Mapping):
            continue
        modules = package.get("modules")
        if not isinstance(modules, list):
            continue
        for module in modules:
            if not isinstance(module, Mapping):
                continue
            source = module.get("source")
            if isinstance(source, Mapping):
                path_value = source.get("path")
                if isinstance(path_value, str):
                    candidate = repo_root / path_value
                    if not candidate.exists():
                        broken.append(
                            {"module": str(module.get("qualified", "")), "path": path_value}
                        )
            pages = module.get("pages")
            if isinstance(pages, Mapping):
                for key, value in pages.items():
                    if not isinstance(value, str):
                        continue
                    candidate = repo_root / value
                    if not candidate.exists():
                        broken.append(
                            {
                                "module": str(module.get("qualified", "")),
                                "page": value,
                                "kind": str(key),
                            }
                        )
            checked += 1
            if checked >= sample:
                return broken
    return broken


def _portal_sessions(previous: Mapping[str, Any]) -> dict[str, Any]:
    sessions = previous.get("portal", {}).get("sessions", {})
    builds = int(sessions.get("builds", 0)) + 1
    return {"builds": builds, "unique_users": sessions.get("unique_users", 0)}


def build_analytics(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog_payload(args.catalog, load_shards=True)
    previous = _load_previous(args.output)
    metrics = _catalog_metrics(catalog)
    broken_links = _check_links(catalog, args.repo_root, args.link_sample)
    analytics = {
        "version": "1.0",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "repo": {"root": str(args.repo_root)},
        "catalog": metrics,
        "portal": {"sessions": _portal_sessions(previous)},
        "errors": {"broken_links": len(broken_links), "details": broken_links},
    }
    return analytics


def write_analytics(args: argparse.Namespace) -> None:
    payload = build_analytics(args)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    write_analytics(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
