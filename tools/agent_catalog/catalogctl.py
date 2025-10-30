"""Command-line interface for querying the Agent Catalog artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from kgfoundry.agent_catalog import search as catalog_search
from kgfoundry.agent_catalog.client import AgentCatalogClient, AgentCatalogClientError
from kgfoundry.agent_catalog.search import SearchOptions
from kgfoundry.agent_catalog.sqlite import sqlite_candidates

EXIT_CONFIG = 2
EXIT_INTERNAL = 3

DEFAULT_CATALOG = Path("docs/_build/agent_catalog.json")

CommandHandler = Callable[[AgentCatalogClient, argparse.Namespace], None]


def _parse_facets(raw: list[str]) -> dict[str, str]:
    """Return a mapping of facet filters parsed from ``raw`` expressions."""
    facets: dict[str, str] = {}
    for value in raw:
        if "=" not in value:
            message = f"Invalid facet expression '{value}', expected key=value"
            raise AgentCatalogClientError(message)
        key, raw_val = value.split("=", 1)
        key = key.strip()
        if not key:
            message = "Facet key must not be empty"
            raise AgentCatalogClientError(message)
        facets[key] = raw_val.strip()
    return facets


def _render_json(payload: object) -> None:
    """Write ``payload`` as formatted JSON to stdout."""
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _render_error(message: str) -> None:
    """Render ``message`` to stderr with a trailing newline."""
    sys.stderr.write(f"{message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser configured for the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to the agent catalog JSON artifact.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for resolving relative links.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("capabilities", help="List packages present in the catalog.")

    symbol_parser = subparsers.add_parser("symbol", help="Show metadata for a symbol.")
    symbol_parser.add_argument("symbol_id", help="Symbol identifier to inspect.")

    callers_parser = subparsers.add_parser(
        "find-callers", help="List callers recorded for a symbol."
    )
    callers_parser.add_argument("symbol_id", help="Symbol identifier to inspect.")

    callees_parser = subparsers.add_parser(
        "find-callees", help="List callees recorded for a symbol."
    )
    callees_parser.add_argument("symbol_id", help="Symbol identifier to inspect.")

    change_parser = subparsers.add_parser("change-impact", help="Show change impact metadata.")
    change_parser.add_argument("symbol_id", help="Symbol identifier to inspect.")

    tests_parser = subparsers.add_parser("suggest-tests", help="List suggested tests for a symbol.")
    tests_parser.add_argument("symbol_id", help="Symbol identifier to inspect.")

    anchor_parser = subparsers.add_parser("open-anchor", help="Render anchor links for a symbol.")
    anchor_parser.add_argument("symbol_id", help="Symbol identifier to inspect.")

    search_parser = subparsers.add_parser("search", help="Execute hybrid search over the catalog.")
    search_parser.add_argument("query", help="Search query text.")
    search_parser.add_argument("--k", type=int, default=10, help="Number of results to return.")
    search_parser.add_argument(
        "--facet",
        action="append",
        default=[],
        help="Facet filter expressed as key=value (may be repeated).",
    )

    explain_parser = subparsers.add_parser(
        "explain-ranking",
        help="Show search results with lexical and vector scores.",
    )
    explain_parser.add_argument("query", help="Search query text.")
    explain_parser.add_argument("--k", type=int, default=5, help="Number of results to inspect.")

    modules_parser = subparsers.add_parser("list-modules", help="List modules for a package.")
    modules_parser.add_argument("package", help="Package name to inspect.")

    return parser


def _load_client(args: argparse.Namespace) -> AgentCatalogClient:
    """Return an ``AgentCatalogClient`` configured from CLI arguments."""
    catalog_path = args.catalog
    if not catalog_path.exists():
        if not any(candidate.exists() for candidate in sqlite_candidates(catalog_path)):
            message = f"Catalog not found at {catalog_path}"
            raise AgentCatalogClientError(message)
    repo_root = args.repo_root
    if not repo_root.exists():
        message = f"Repository root does not exist: {repo_root}"
        raise AgentCatalogClientError(message)
    return AgentCatalogClient.from_path(catalog_path, repo_root=repo_root)


def _cmd_capabilities(client: AgentCatalogClient, _: argparse.Namespace) -> None:
    """Render the available packages in the catalog."""
    packages = [pkg.name for pkg in client.list_packages()]
    _render_json(packages)


def _cmd_symbol(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render the catalog entry for a specific symbol."""
    symbol = client.get_symbol(args.symbol_id)
    if symbol is None:
        message = f"Unknown symbol: {args.symbol_id}"
        raise AgentCatalogClientError(message)
    _render_json(symbol.model_dump())


def _cmd_find_callers(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render callers recorded for a symbol."""
    _render_json(client.find_callers(args.symbol_id))


def _cmd_find_callees(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render callees recorded for a symbol."""
    _render_json(client.find_callees(args.symbol_id))


def _cmd_change_impact(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render change impact metadata for a symbol."""
    _render_json(client.change_impact(args.symbol_id).model_dump())


def _cmd_suggest_tests(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render suggested tests for a symbol."""
    _render_json(client.suggest_tests(args.symbol_id))


def _cmd_open_anchor(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render editor and GitHub anchors for a symbol."""
    _render_json(client.open_anchor(args.symbol_id))


def _cmd_search(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render hybrid search results for the provided query."""
    facets = _parse_facets(args.facet) if args.facet else None
    options = SearchOptions(facets=facets)
    results = catalog_search.search_catalog(
        client.catalog.model_dump(),
        repo_root=client.repo_root,
        query=args.query,
        k=args.k,
        options=options,
    )
    payload = [asdict(result) for result in results]
    _render_json(payload)


def _cmd_explain_ranking(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render search results with lexical/vector scores for inspection."""
    results = client.search(args.query, k=args.k)
    payload = [
        {
            "symbol_id": result.symbol_id,
            "score": result.score,
            "lexical_score": result.lexical_score,
            "vector_score": result.vector_score,
            "package": result.package,
            "module": result.module,
            "qname": result.qname,
        }
        for result in results
    ]
    _render_json(payload)


def _cmd_list_modules(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render modules available within ``args.package``."""
    modules = [module.qualified for module in client.list_modules(args.package)]
    _render_json(modules)


COMMANDS: dict[str, CommandHandler] = {
    "capabilities": _cmd_capabilities,
    "symbol": _cmd_symbol,
    "find-callers": _cmd_find_callers,
    "find-callees": _cmd_find_callees,
    "change-impact": _cmd_change_impact,
    "suggest-tests": _cmd_suggest_tests,
    "open-anchor": _cmd_open_anchor,
    "search": _cmd_search,
    "explain-ranking": _cmd_explain_ranking,
    "list-modules": _cmd_list_modules,
}


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces choices
        message = f"Unknown command: {args.command}"
        parser.error(message)
    try:
        client = _load_client(args)
        handler(client, args)
    except (AgentCatalogClientError, catalog_search.CatalogSearchError) as exc:
        _render_error(f"error: {exc}")
        return EXIT_CONFIG
    except Exception as exc:  # pragma: no cover - defensive guard  # noqa: BLE001
        _render_error(f"unexpected error: {exc}")
        return EXIT_INTERNAL
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
