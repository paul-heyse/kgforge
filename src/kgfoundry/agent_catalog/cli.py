"""Command-line interface for querying the Agent Catalog artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from kgfoundry.agent_catalog import search as catalog_search
from kgfoundry.agent_catalog.client import AgentCatalogClient, AgentCatalogClientError
from kgfoundry.agent_catalog.search import SearchOptions, SearchRequest, SearchResult
from kgfoundry.agent_catalog.sqlite import sqlite_candidates
from kgfoundry_common.errors import AgentCatalogSearchError
from kgfoundry_common.logging import get_logger, set_correlation_id, with_fields
from kgfoundry_common.problem_details import JsonValue

CommandHandler = Callable[[AgentCatalogClient, argparse.Namespace], None]

logger = get_logger(__name__)

# Allowed facet keys (from schema/search/mcp_payload.json)
ALLOWED_FACET_KEYS = {"package", "module", "kind", "stability", "deprecated"}


# Feature flag for typed CLI output
def _should_use_typed_envelope() -> bool:
    """Check if typed envelope should be used by default.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    bool
        Describe return value.
"""
    return os.getenv("AGENT_SEARCH_TYPED", "0").lower() in {"1", "true", "yes"}


class CLIEnvelope(TypedDict, total=False):
    """Document CLIEnvelope.

    &lt;!-- auto:docstring-builder v1 --&gt;

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    schemaVersion: str
    schemaId: str
    generatedAt: str
    status: str
    command: str
    subcommand: str
    durationSeconds: float
    files: list[dict[str, JsonValue]]
    errors: list[dict[str, JsonValue]]
    problem: dict[str, JsonValue]
    correlation_id: str
    payload: dict[str, JsonValue]


class CatalogctlError(RuntimeError):
    """Document CatalogctlError.

    &lt;!-- auto:docstring-builder v1 --&gt;

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""


def _parse_facets(raw: list[str]) -> dict[str, str]:
    """Return a mapping of facet filters parsed from ``raw`` expressions.

    <!-- auto:docstring-builder v1 -->

    Validates that facet keys are from the allowed set (package, module, kind,
    stability, deprecated) and raises CatalogctlError with friendly messages
    for invalid keys.

    Parameters
    ----------
    raw : list[str]
        Raw facet expressions in "key=value" format.

    Returns
    -------
    dict[str, str]
        Validated facet mapping.

    Raises
    ------
    CatalogctlError
        If any facet expression is malformed or uses an invalid key.
"""
    facets: dict[str, str] = {}
    for value in raw:
        if "=" not in value:
            message = f"Invalid facet expression '{value}', expected key=value"
            raise CatalogctlError(message)
        key, raw_val = value.split("=", 1)
        key = key.strip()
        if not key:
            message = "Facet key must not be empty"
            raise CatalogctlError(message)
        if key not in ALLOWED_FACET_KEYS:
            allowed = ", ".join(sorted(ALLOWED_FACET_KEYS))
            message = f"Invalid facet key '{key}'. Allowed keys: {allowed}"
            raise CatalogctlError(message)
        facets[key] = raw_val.strip()
    return facets


def _render_json(payload: object) -> None:
    """Write ``payload`` as formatted JSON to stdout.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    payload : object
        Describe ``payload``.
"""
    json.dump(payload, fp=sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _render_error(message: str, problem: dict[str, JsonValue] | None = None) -> None:
    """Render ``message`` to stderr with optional Problem Details.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    message : str
        Human-readable error message.
    problem : dict[str, object] | NoneType, optional
        RFC 9457 Problem Details payload to include in JSON output.
        Defaults to ``None``.
"""
    if problem is not None:
        error_payload = {"message": message, "problem": problem}
        json.dump(error_payload, fp=sys.stderr, indent=2, ensure_ascii=False)
        sys.stderr.write("\n")
    else:
        sys.stderr.write(f"{message}\n")


def _search_result_to_dict(result: SearchResult) -> dict[str, JsonValue]:
    """Convert SearchResult dataclass to VectorSearchResultTypedDict-compatible dict.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    result : SearchResult
        Describe ``result``.

    Returns
    -------
    dict[str, object]
        Describe return value.
"""
    return {
        "symbol_id": result.symbol_id,
        "score": result.score,
        "lexical_score": result.lexical_score,
        "vector_score": result.vector_score,
        "package": result.package,
        "module": result.module,
        "qname": result.qname,
        "kind": result.kind,
        "anchor": dict(result.anchor),
        "metadata": {
            "stability": result.stability,
            "deprecated": result.deprecated,
            "summary": result.summary,
            "docstring": result.docstring,
        },
    }


def _build_cli_envelope(  # noqa: PLR0913
    subcommand: str,
    status: str,
    duration_seconds: float,
    correlation_id: str,
    payload: dict[str, JsonValue] | None = None,
    errors: list[dict[str, JsonValue]] | None = None,
    problem: dict[str, JsonValue] | None = None,
) -> CLIEnvelope:
    """Build a typed CLI envelope conforming to schema/tools/cli_envelope.json.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    subcommand : str
        Subcommand name (empty string if none).
    status : str
        Status ("success", "error", "violation", "config").
    duration_seconds : float
        Execution duration in seconds.
    correlation_id : str
        Correlation ID for tracking.
    payload : dict[str, object] | NoneType, optional
        Command-specific payload (e.g., search results).
        Defaults to ``None``.
    errors : list[dict[str, object]] | NoneType, optional
        Error entries (for non-success status).
        Defaults to ``None``.
    problem : dict[str, object] | NoneType, optional
        RFC 9457 Problem Details (for error status).
        Defaults to ``None``.

    Returns
    -------
    CLIEnvelope
        Typed envelope ready for JSON serialization.
"""
    envelope: CLIEnvelope = {
        "schemaVersion": "1.0.0",
        "schemaId": "https://kgfoundry.dev/schema/cli-envelope.json",
        "generatedAt": datetime.now(tz=UTC).isoformat(),
        "status": status,
        "command": "agent_catalog",
        "subcommand": subcommand,
        "durationSeconds": duration_seconds,
        "files": [],
        "errors": errors or [],
        "correlation_id": correlation_id,
    }
    if payload is not None:
        envelope["payload"] = payload
    if problem is not None:
        envelope["problem"] = problem
    return envelope


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser configured for the CLI.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    argparse.ArgumentParser
        Describe return value.
"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/_build/agent_catalog.json"),
        help="Path to the agent catalog JSON artifact.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for resolving relative links.",
    )
    use_typed_default = _should_use_typed_envelope()
    parser.add_argument(
        "--json",
        action="store_true",
        default=use_typed_default,
        help="Emit typed JSON envelope (default if AGENT_SEARCH_TYPED=1).",
    )
    parser.add_argument(
        "--legacy-json",
        action="store_true",
        help="Use legacy JSON format (plain output without envelope).",
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
        default=[],  # type: ignore[misc]  # argparse default list typing limitation
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


def _raise_unknown_command_error(command: str) -> None:
    """Raise CatalogctlError for unknown command.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    command : str
        Describe ``command``.
"""
    message = f"Unknown command: {command}"
    raise CatalogctlError(message)


def _load_client(args: argparse.Namespace) -> AgentCatalogClient:
    """Return an ``AgentCatalogClient`` configured from CLI arguments.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    args : argparse.Namespace
        Describe ``args``.

    Returns
    -------
    AgentCatalogClient
        Describe return value.
"""
    catalog_path: Path = args.catalog
    if not catalog_path.exists() and not any(
        candidate.exists() for candidate in sqlite_candidates(catalog_path)
    ):
        message = f"Catalog not found at {catalog_path}"
        raise CatalogctlError(message)
    repo_root: Path = args.repo_root
    if not repo_root.exists():
        message = f"Repository root does not exist: {repo_root}"
        raise CatalogctlError(message)
    return AgentCatalogClient.from_path(catalog_path, repo_root=repo_root)


def _determine_output_format(args: argparse.Namespace) -> tuple[bool, bool]:
    """Determine output format based on flags and feature flag.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    args : argparse.Namespace
        Describe ``args``.

    Returns
    -------
    tuple[bool, bool]
        (use_envelope, use_legacy) flags indicating output format.
"""
    legacy_json: bool = args.legacy_json
    if legacy_json:
        return False, True
    json_flag: bool = args.json
    if json_flag:
        return True, False
    # Default: use envelope if feature flag enabled
    return _should_use_typed_envelope(), False


def _cmd_capabilities(client: AgentCatalogClient, _: argparse.Namespace) -> None:
    """Render the available packages in the catalog.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    _ : argparse.Namespace
        Describe ``_``.
"""
    packages = [pkg.name for pkg in client.list_packages()]
    _render_json(packages)


def _cmd_symbol(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render the catalog entry for a specific symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    symbol_id: str = args.symbol_id
    symbol = client.get_symbol(symbol_id)
    if symbol is None:
        message = f"Unknown symbol: {symbol_id}"
        raise CatalogctlError(message)
    # model_dump returns dict[str, object], cast to JsonValue since it's JSON-serializable
    _render_json(cast(dict[str, JsonValue], symbol.model_dump()))


def _cmd_find_callers(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render callers recorded for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    symbol_id: str = args.symbol_id
    _render_json(client.find_callers(symbol_id))


def _cmd_find_callees(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render callees recorded for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    symbol_id: str = args.symbol_id
    _render_json(client.find_callees(symbol_id))


def _cmd_change_impact(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render change impact metadata for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    symbol_id: str = args.symbol_id
    # model_dump returns dict[str, object], cast to JsonValue since it's JSON-serializable
    _render_json(cast(dict[str, JsonValue], client.change_impact(symbol_id).model_dump()))


def _cmd_suggest_tests(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render suggested tests for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    symbol_id: str = args.symbol_id
    # suggest_tests already returns list[dict[str, JsonValue]], no cast needed
    _render_json(client.suggest_tests(symbol_id))


def _cmd_open_anchor(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render editor and GitHub anchors for a symbol.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    symbol_id: str = args.symbol_id
    _render_json(client.open_anchor(symbol_id))


def _cmd_search(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Execute hybrid search and render the resulting documents.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    use_envelope, _use_legacy = _determine_output_format(args)
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    start_time = time.perf_counter()

    with with_fields(
        logger, operation="cli_search", correlation_id=correlation_id, status="started"
    ) as log_adapter:
        try:
            facet_list: list[str] = args.facet
            facets = _parse_facets(facet_list)
        except CatalogctlError as exc:
            duration = time.perf_counter() - start_time
            if use_envelope:
                problem = {
                    "type": "https://kgfoundry.dev/problems/invalid-input",
                    "title": "Invalid Facet",
                    "status": 400,
                    "detail": str(exc),
                    "instance": "urn:cli:agent_catalog:search",
                }
                envelope = _build_cli_envelope(
                    "search",
                    "error",
                    duration,
                    correlation_id,
                    problem=cast(dict[str, JsonValue], problem),
                )
                _render_json(envelope)
            else:
                _render_error(str(exc))
            log_adapter.exception("Search failed", exc_info=exc)
            return

        options = SearchOptions(facets=facets)
        query: str = args.query
        k: int = args.k
        request = SearchRequest(
            repo_root=client.repo_root,
            query=query,
            k=max(1, k),
        )
        try:
            # model_dump returns dict[str, object] - cast immediately to avoid Any expression
            # Cast to expected search_catalog type: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]]
            results = catalog_search.search_catalog(
                cast(
                    Mapping[
                        str, str | int | float | bool | None | list[object] | dict[str, object]
                    ],
                    client.catalog.model_dump(),
                ),
                request=request,
                options=options,
            )
            duration = time.perf_counter() - start_time
            if use_envelope:
                payload: dict[str, JsonValue] = {
                    "query": query,
                    "results": [_search_result_to_dict(result) for result in results],
                    "total": len(results),
                    "took_ms": int(duration * 1000),
                    "metadata": {},
                }
                envelope = _build_cli_envelope(
                    "search",
                    "success",
                    duration,
                    correlation_id,
                    payload=payload,
                )
                _render_json(envelope)
            else:
                # asdict returns dict[str, Any], cast to JsonValue since SearchResult is JSON-serializable
                _render_json([cast(dict[str, JsonValue], asdict(result)) for result in results])
            log_adapter.info(
                "Search completed", extra={"status": "success", "result_count": len(results)}
            )
        except AgentCatalogSearchError as exc:
            duration = time.perf_counter() - start_time
            problem_details = exc.to_problem_details(instance="urn:cli:agent_catalog:search")
            if use_envelope:
                errors = [{"status": "error", "message": str(exc), "problem": problem_details}]
                envelope = _build_cli_envelope(
                    "search",
                    "error",
                    duration,
                    correlation_id,
                    errors=cast(list[dict[str, JsonValue]], errors),
                    problem=cast(dict[str, JsonValue], problem_details),
                )
                _render_json(envelope)
            else:
                _render_error(str(exc), problem=cast(dict[str, JsonValue], problem_details))
            log_adapter.exception("Search failed", exc_info=exc)
            raise


def _cmd_explain_ranking(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render hybrid search results with detailed scoring metadata.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    use_envelope, _use_legacy = _determine_output_format(args)
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    start_time = time.perf_counter()

    with with_fields(
        logger, operation="cli_explain_ranking", correlation_id=correlation_id, status="started"
    ) as log_adapter:
        query: str = args.query
        k: int = args.k
        options = SearchOptions(candidate_pool=max(10, k))
        request = SearchRequest(
            repo_root=client.repo_root,
            query=query,
            k=max(1, k),
        )
        try:
            # model_dump returns dict[str, object] - cast immediately to avoid Any expression
            # Cast to expected search_catalog type: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]]
            results = catalog_search.search_catalog(
                cast(
                    Mapping[
                        str, str | int | float | bool | None | list[object] | dict[str, object]
                    ],
                    client.catalog.model_dump(),
                ),
                request=request,
                options=options,
            )
            duration = time.perf_counter() - start_time
            if use_envelope:
                payload: dict[str, JsonValue] = {
                    "query": query,
                    "results": [_search_result_to_dict(result) for result in results],
                    "total": len(results),
                    "took_ms": int(duration * 1000),
                    "metadata": {},
                }
                envelope = _build_cli_envelope(
                    "explain-ranking",
                    "success",
                    duration,
                    correlation_id,
                    payload=payload,
                )
                _render_json(envelope)
            else:
                # asdict returns dict[str, Any], cast to JsonValue since SearchResult is JSON-serializable
                _render_json([cast(dict[str, JsonValue], asdict(result)) for result in results])
            log_adapter.info(
                "Explain ranking completed",
                extra={"status": "success", "result_count": len(results)},
            )
        except AgentCatalogSearchError as exc:
            duration = time.perf_counter() - start_time
            problem_details = exc.to_problem_details(
                instance="urn:cli:agent_catalog:explain-ranking"
            )
            if use_envelope:
                errors = [{"status": "error", "message": str(exc), "problem": problem_details}]
                envelope = _build_cli_envelope(
                    "explain-ranking",
                    "error",
                    duration,
                    correlation_id,
                    errors=cast(list[dict[str, JsonValue]], errors),
                    problem=cast(dict[str, JsonValue], problem_details),
                )
                _render_json(envelope)
            else:
                _render_error(str(exc), problem=cast(dict[str, JsonValue], problem_details))
            log_adapter.exception("Explain ranking failed", exc_info=exc)
            raise


def _cmd_list_modules(client: AgentCatalogClient, args: argparse.Namespace) -> None:
    """Render the module names for a package.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    args : argparse.Namespace
        Describe ``args``.
"""
    package: str = args.package
    modules = client.list_modules(package)
    _render_json([module.qualified for module in modules])


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
    """Execute the CLI and return an exit code.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    argv : list[str] | NoneType, optional
        Describe ``argv``.
        Defaults to ``None``.

    Returns
    -------
    int
        Describe return value.
"""
    parser = build_parser()
    args = parser.parse_args(argv)
    use_envelope, _use_legacy = _determine_output_format(args)
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    start_time = time.perf_counter()

    with with_fields(
        logger,
        operation="cli",
        correlation_id=correlation_id,
        command=args.command,  # type: ignore[misc]  # argparse.Namespace attribute typing
        status="started",
    ) as log_adapter:
        command_str: str = args.command
        try:
            client = _load_client(args)
            handler = COMMANDS.get(command_str)
            if handler is None:
                _raise_unknown_command_error(command_str)
            handler(client, args)  # type: ignore[misc]  # handler is guaranteed non-None
            duration = time.perf_counter() - start_time
            log_adapter.info("CLI completed", extra={"status": "success"})
            return 0  # noqa: TRY300 - handler is guaranteed to be non-None due to exception above
        except (CatalogctlError, AgentCatalogClientError) as exc:
            duration = time.perf_counter() - start_time
            if use_envelope:
                problem = {
                    "type": "https://kgfoundry.dev/problems/runtime-error",
                    "title": "CLI Error",
                    "status": 400 if isinstance(exc, CatalogctlError) else 404,
                    "detail": str(exc),
                    "instance": f"urn:cli:agent_catalog:{command_str}",
                }
                errors = [{"status": "error", "message": str(exc), "problem": problem}]
                envelope = _build_cli_envelope(
                    command_str,
                    "error",
                    duration,
                    correlation_id,
                    errors=cast(list[dict[str, JsonValue]], errors),
                    problem=cast(dict[str, JsonValue], problem),
                )
                _render_json(envelope)
            else:
                _render_error(str(exc))
            log_adapter.exception("CLI failed", exc_info=exc)
            return 2
        except AgentCatalogSearchError as exc:
            duration = time.perf_counter() - start_time
            problem_details = exc.to_problem_details(
                instance=f"urn:cli:agent_catalog:{command_str}"
            )
            if use_envelope:
                errors = [{"status": "error", "message": str(exc), "problem": problem_details}]
                envelope = _build_cli_envelope(
                    command_str,
                    "error",
                    duration,
                    correlation_id,
                    errors=cast(list[dict[str, JsonValue]], errors),
                    problem=cast(dict[str, JsonValue], problem_details),
                )
                _render_json(envelope)
            else:
                _render_error(str(exc), problem=cast(dict[str, JsonValue], problem_details))
            log_adapter.exception("CLI failed", exc_info=exc)
            return 2
        except Exception as exc:  # pragma: no cover - defensive guard
            duration = time.perf_counter() - start_time
            if use_envelope:
                problem = {
                    "type": "https://kgfoundry.dev/problems/runtime-error",
                    "title": "Internal Error",
                    "status": 500,
                    "detail": f"Internal error: {exc}",
                    "instance": f"urn:cli:agent_catalog:{command_str}",
                }
                errors = [{"status": "error", "message": str(exc), "problem": problem}]
                envelope = _build_cli_envelope(
                    command_str,
                    "error",
                    duration,
                    correlation_id,
                    errors=cast(list[dict[str, JsonValue]], errors),
                    problem=cast(dict[str, JsonValue], problem),
                )
                _render_json(envelope)
            else:
                _render_error(f"Internal error: {exc}")
            log_adapter.exception("CLI internal error", exc_info=exc)
            return 3


__all__ = ["build_parser", "main"]
