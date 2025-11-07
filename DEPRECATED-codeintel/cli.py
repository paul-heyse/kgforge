"""Unified CodeIntel CLI aligned with the shared tooling metadata contracts.

This CLI provides:
- Developer tooling: query, symbols (Tree-sitter powered)
- Operational commands: mcp serve, index build
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, NoReturn, cast
from uuid import uuid4

import anyio
import typer
from tools import (
    CliEnvelope,
    CliEnvelopeBuilder,
    CliErrorStatus,
    CliStatus,
    JsonValue,
    ProblemDetailsDict,
    ProblemDetailsParams,
    build_problem_details,
    get_logger,
    render_cli_envelope,
    with_fields,
)
from tools._shared.logging import LoggerAdapter
from tree_sitter import Language

from codeintel import cli_context
from codeintel.index.store import IndexStore, ensure_schema, index_incremental
from codeintel.indexer.tscore import (
    LANGUAGE_NAMES,
    get_language,
    load_langs,
    parse_bytes,
    run_query,
)
from codeintel.mcp_server import tools as mcp_tools
from codeintel.mcp_server.server import amain as mcp_amain
from codeintel.observability import INDEX_BUILD_DURATION_SECONDS, update_index_metrics
from kgfoundry_common.errors import ConfigurationError

CLI_COMMAND = cli_context.CLI_COMMAND
CLI_OPERATION_IDS = cli_context.CLI_OPERATION_IDS
CLI_INTERFACE_ID = cli_context.CLI_INTERFACE_ID
CLI_TITLE = cli_context.CLI_TITLE
CLI_SETTINGS = cli_context.get_cli_settings()
CLI_CONFIG = cli_context.get_cli_config()

REPO_ROOT = cli_context.REPO_ROOT
CLI_ENVELOPE_DIR = REPO_ROOT / "site" / "_build" / "cli"

LOGGER = get_logger(__name__)

STATUS_BAD_REQUEST = 400
STATUS_NOT_FOUND = 404
STATUS_UNPROCESSABLE_ENTITY = 422
STATUS_INTERNAL_ERROR = 500

CLI_PROBLEM_BASE = "https://kgfoundry.dev/problems/codeintel/indexer"

DEFAULT_NAMED_ONLY: Final[bool] = True

LANGUAGE_CHOICES: Sequence[str] = tuple(sorted(LANGUAGE_NAMES))
DEFAULT_LANGUAGE = (
    "python"
    if "python" in LANGUAGE_NAMES
    else (LANGUAGE_CHOICES[0] if LANGUAGE_CHOICES else "python")
)

QUERY_OVERRIDE = cli_context.get_operation_override("query")
SYMBOLS_OVERRIDE = cli_context.get_operation_override("symbols")

QUERY_HELP = (
    QUERY_OVERRIDE.description
    if QUERY_OVERRIDE and QUERY_OVERRIDE.description
    else "Run a Tree-sitter query against a source file and emit capture records."
)
SYMBOLS_HELP = (
    SYMBOLS_OVERRIDE.description
    if SYMBOLS_OVERRIDE and SYMBOLS_OVERRIDE.description
    else "List Python symbol captures for a directory tree using the MCP server tools."
)


def _resolve_cli_help() -> str:
    title = CLI_CONFIG.title or CLI_TITLE
    return f"{title} ({CLI_CONFIG.version})"


app = typer.Typer(help=_resolve_cli_help(), no_args_is_help=True, add_completion=False)

LanguageOption = Annotated[
    str,
    typer.Option(
        "--language",
        "-l",
        help="Tree-sitter language to use when parsing the input file.",
        show_default=True,
        case_sensitive=False,
    ),
]
QueryFileOption = Annotated[
    Path,
    typer.Option(
        ...,
        "--query",
        "-q",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the Tree-sitter S-expression query file.",
    ),
]
NamedOnlyOption = Annotated[
    bool,
    typer.Option(
        "--named-only/--all-captures",
        help="Emit only named captures unless --all-captures is supplied.",
        show_default=True,
    ),
]
SourcePathArgument = Annotated[
    Path,
    typer.Argument(
        help="Source file to parse before applying the query.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
]
DirectoryArgument = Annotated[
    Path,
    typer.Argument(
        help="Directory containing Python source files to analyse.",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
    ),
]


@dataclass(slots=True)
class _CommandContext:
    """Structured metadata shared across a single command invocation."""

    subcommand: str
    operation_id: str
    correlation_id: str
    logger: LoggerAdapter
    builder: CliEnvelopeBuilder
    start: float

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def extensions(self, extras: Mapping[str, object] | None = None) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "command": CLI_COMMAND,
            "subcommand": self.subcommand,
            "operation_id": self.operation_id,
            "correlation_id": self.correlation_id,
        }
        if extras:
            payload.update(_normalize_json_mapping(extras))
        return payload


@dataclass(slots=True)
class _FailureOptions:
    """Configuration for synthesising failure envelopes and exits."""

    kind: str
    extras: Mapping[str, object] | None = None
    exit_code: int | None = None
    exc: BaseException | None = None


def _normalize_json_mapping(values: Mapping[str, object]) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for key, value in values.items():
        payload[str(key)] = _coerce_json_value(value)
    return payload


def _coerce_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _coerce_json_value(v) for k, v in value.items()}
    return str(value)


def _start_command(subcommand: str, **fields: object) -> _CommandContext:
    operation_id = CLI_OPERATION_IDS.get(subcommand, subcommand)
    operation_alias = subcommand.replace("-", "_")
    correlation_id = uuid4().hex
    logger_fields = {key: value for key, value in fields.items() if value is not None}
    logger = with_fields(
        LOGGER,
        correlation_id=correlation_id,
        operation_id=operation_id,
        operation=operation_alias,
        command=CLI_COMMAND,
        subcommand=subcommand,
        **logger_fields,
    )
    logger.info("Command started", extra={"status": "start"})
    builder = CliEnvelopeBuilder.create(
        command=CLI_COMMAND, status="success", subcommand=subcommand
    )
    return _CommandContext(
        subcommand=subcommand,
        operation_id=operation_id,
        correlation_id=correlation_id,
        logger=logger,
        builder=builder,
        start=time.monotonic(),
    )


def _cli_status_from_error(error_status: CliErrorStatus) -> CliStatus:
    if error_status in {"violation", "config"}:
        return error_status
    return "error"


def _exit_code_for(error_status: CliErrorStatus) -> int:
    return 2 if error_status in {"violation", "config"} else 1


def _envelope_path(subcommand: str) -> Path:
    safe_subcommand = subcommand or "root"
    filename = f"{CLI_SETTINGS.bin_name}-{CLI_COMMAND}-{safe_subcommand.replace('/', '-')}.json"
    return CLI_ENVELOPE_DIR / filename


def _emit_envelope(envelope: CliEnvelope, *, subcommand: str, logger: LoggerAdapter) -> Path:
    path = _envelope_path(subcommand)
    CLI_ENVELOPE_DIR.mkdir(parents=True, exist_ok=True)
    rendered = render_cli_envelope(envelope)
    path.write_text(rendered + "\n", encoding="utf-8")
    logger.debug(
        "CLI envelope written",
        extra={"status": envelope.status, "cli_envelope": str(path)},
    )
    return path


def _finish_success(context: _CommandContext) -> CliEnvelope:
    envelope = context.builder.finish(duration_seconds=context.elapsed())
    path = _emit_envelope(envelope, subcommand=context.subcommand, logger=context.logger)
    context.logger.info(
        "Command completed",
        extra={
            "status": "success",
            "cli_envelope": str(path),
            "duration_seconds": envelope.duration_seconds,
        },
    )
    return envelope


def _build_problem(
    context: _CommandContext,
    *,
    detail: str,
    status: int,
    kind: str,
    extras: Mapping[str, object] | None = None,
) -> ProblemDetailsDict:
    extensions = context.extensions({"error": kind, **(extras or {})})
    return build_problem_details(
        ProblemDetailsParams(
            type=f"{CLI_PROBLEM_BASE}/{context.subcommand}/{kind}",
            title=f"{CLI_TITLE} {context.subcommand.replace('_', ' ')} command failed",
            status=status,
            detail=detail,
            instance=f"urn:cli:{CLI_INTERFACE_ID}:{context.subcommand}",
            extensions=extensions,
        )
    )


def _fail_command(
    context: _CommandContext,
    *,
    detail: str,
    status: int,
    error_status: CliErrorStatus,
    options: _FailureOptions | None = None,
) -> NoReturn:
    failure = options or _FailureOptions(kind="generic-error")
    problem = _build_problem(
        context,
        detail=detail,
        status=status,
        kind=failure.kind,
        extras=failure.extras,
    )
    failure_builder = CliEnvelopeBuilder.create(
        command=CLI_COMMAND,
        status=_cli_status_from_error(error_status),
        subcommand=context.subcommand,
    )
    failure_builder.add_error(status=error_status, message=detail, problem=problem)
    failure_builder.set_problem(problem)
    envelope = failure_builder.finish(duration_seconds=context.elapsed())
    path = _emit_envelope(envelope, subcommand=context.subcommand, logger=context.logger)
    log_extra = {
        "status": error_status,
        "cli_envelope": str(path),
        "duration_seconds": envelope.duration_seconds,
    }
    if failure.exc is not None:
        context.logger.error("Command failed", extra=log_extra, exc_info=failure.exc)
    else:
        context.logger.error("Command failed", extra=log_extra)
    typer.echo(detail, err=True)
    exit_code = failure.exit_code if failure.exit_code is not None else _exit_code_for(error_status)
    raise typer.Exit(code=exit_code)


def _load_source_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:  # pragma: no cover - I/O error handling
        message = f"Failed to read source file '{path}': {exc}"
        raise RuntimeError(message) from exc


def _resolve_language_or_exit(context: _CommandContext, language: str) -> Language:
    try:
        langs = load_langs()
        return get_language(langs, language)
    except (ConfigurationError, RuntimeError, ValueError) as exc:  # pragma: no cover
        _fail_command(
            context,
            detail=str(exc),
            status=STATUS_INTERNAL_ERROR,
            error_status="config",
            options=_FailureOptions(
                kind="language-load-error",
                extras={"language": language},
                exc=exc,
            ),
        )


def _load_source_or_exit(context: _CommandContext, path: Path) -> bytes:
    try:
        return _load_source_bytes(path)
    except FileNotFoundError as exc:
        _fail_command(
            context,
            detail=f"Source file not found: {path}",
            status=STATUS_NOT_FOUND,
            error_status="config",
            options=_FailureOptions(
                kind="source-missing",
                extras={"path": path},
                exc=exc,
            ),
        )
    except RuntimeError as exc:
        _fail_command(
            context,
            detail=str(exc),
            status=STATUS_BAD_REQUEST,
            error_status="config",
            options=_FailureOptions(
                kind="source-unreadable",
                extras={"path": path},
                exc=exc,
            ),
        )


def _read_query_or_exit(context: _CommandContext, query_file: Path) -> str:
    try:
        return query_file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        _fail_command(
            context,
            detail=f"Query file not found: {query_file}",
            status=STATUS_NOT_FOUND,
            error_status="config",
            options=_FailureOptions(
                kind="query-missing",
                extras={"query": query_file},
                exc=exc,
            ),
        )
    except OSError as exc:  # pragma: no cover - I/O error handling
        _fail_command(
            context,
            detail=f"Failed to read query file '{query_file}': {exc}",
            status=STATUS_BAD_REQUEST,
            error_status="config",
            options=_FailureOptions(
                kind="query-unreadable",
                extras={"query": query_file},
                exc=exc,
            ),
        )


def _execute_query_or_exit(
    context: _CommandContext,
    *,
    lang: Language,
    query_text: str,
    data: bytes,
    normalized_language: str,
) -> list[dict[str, object]]:
    try:
        tree = parse_bytes(lang, data)
        return run_query(lang, query_text, tree, data)
    except ValueError as exc:
        _fail_command(
            context,
            detail=str(exc),
            status=STATUS_UNPROCESSABLE_ENTITY,
            error_status="violation",
            options=_FailureOptions(
                kind="query-invalid",
                extras={"language": normalized_language},
                exc=exc,
            ),
        )
    except (ConfigurationError, RuntimeError) as exc:  # pragma: no cover - defensive catch
        _fail_command(
            context,
            detail=str(exc),
            status=STATUS_INTERNAL_ERROR,
            error_status="error",
            options=_FailureOptions(
                kind="query-execution",
                extras={"language": normalized_language},
                exc=exc,
            ),
        )


def _infer_repo_root(repo: str | None) -> str:
    """Resolve repository root path.

    Parameters
    ----------
    repo : str | None
        Repository path, defaults to current directory.

    Returns
    -------
    str
        Absolute path to repository root.
    """
    return str(Path(repo or ".").resolve())


# Developer tooling commands (Tree-sitter powered)
@app.command(name="query", help=QUERY_HELP)
def query(
    path: SourcePathArgument,
    query_file: QueryFileOption,
    language_name: LanguageOption = DEFAULT_LANGUAGE,
    named_only: NamedOnlyOption = DEFAULT_NAMED_ONLY,
) -> None:
    """Execute a Tree-sitter query and emit a structured CLI envelope."""
    normalized_language = language_name.strip().lower()
    context = _start_command(
        "query",
        path=str(path),
        language=normalized_language,
        query=str(query_file),
        named_only=named_only,
    )

    if normalized_language not in LANGUAGE_NAMES:
        supported = list(LANGUAGE_CHOICES)
        supported_str = ", ".join(supported)
        detail = f"Unsupported language '{normalized_language}'. Supported: {supported_str}."
        _fail_command(
            context,
            detail=detail,
            status=STATUS_UNPROCESSABLE_ENTITY,
            error_status="violation",
            options=_FailureOptions(
                kind="unsupported-language",
                extras={"supported_languages": supported},
                exit_code=2,
            ),
        )
        return

    lang = _resolve_language_or_exit(context, normalized_language)
    data = _load_source_or_exit(context, path)
    query_text = _read_query_or_exit(context, query_file)
    hits = _execute_query_or_exit(
        context,
        lang=lang,
        query_text=query_text,
        data=data,
        normalized_language=normalized_language,
    )

    captures = hits
    if named_only:
        captures = [capture for capture in hits if not str(capture.get("kind", "")).startswith("[")]

    context.builder.add_file(
        path=str(path),
        status="success",
        message=f"Parsed source file using language={normalized_language}",
    )
    context.builder.add_file(
        path=str(query_file),
        status="success",
        message="Executed Tree-sitter query",
    )
    context.builder.add_file(
        path="<captures>",
        status="success",
        message=f"Produced {len(captures)} captures (named_only={named_only})",
    )

    typer.echo(json.dumps(captures, indent=2, ensure_ascii=False))
    _finish_success(context)


@app.command(name="symbols", help=SYMBOLS_HELP)
def symbols(dirpath: DirectoryArgument) -> None:
    """List Python symbol captures and emit a standard CLI envelope."""
    context = _start_command("symbols", directory=str(dirpath))

    try:
        results = mcp_tools.list_python_symbols(str(dirpath))
    except FileNotFoundError as exc:
        _fail_command(
            context,
            detail=f"Directory not found: {dirpath}",
            status=STATUS_NOT_FOUND,
            error_status="config",
            options=_FailureOptions(
                kind="directory-missing",
                extras={"directory": dirpath},
                exc=exc,
            ),
        )
        return
    except (
        RuntimeError,
        OSError,
        ValueError,
    ) as exc:  # pragma: no cover - defensive catch for MCP errors
        _fail_command(
            context,
            detail=str(exc),
            status=STATUS_INTERNAL_ERROR,
            error_status="error",
            options=_FailureOptions(
                kind="symbols-error",
                extras={"directory": dirpath},
                exc=exc,
            ),
        )
        return

    context.builder.add_file(
        path=str(dirpath),
        status="success",
        message=f"Enumerated symbols using code-intel MCP tools (count={len(results)})",
    )

    typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
    _finish_success(context)


# Operational commands (MCP server and index management)
mcp_group = typer.Typer()
app.add_typer(mcp_group, name="mcp")

RepoOption = Annotated[
    str,
    typer.Option(
        "--repo",
        "-r",
        help="Repository root path.",
    ),
]


@mcp_group.command("serve")
def serve(repo: RepoOption = ".") -> None:
    """Start the CodeIntel MCP server over stdio.

    Parameters
    ----------
    repo : str, optional
        Repository root path, by default ".".
    """
    context = _start_command("mcp/serve", repo=repo)
    repo_root = _infer_repo_root(repo)
    # Make repo visible to tools in a predictable way:
    os.environ["KGF_REPO_ROOT"] = repo_root

    try:
        # Run server inside AnyIO; façade gives you correlation ids + envelopes
        # Cast to help type checker understand the async function signature
        anyio.run(cast("Any", mcp_amain))
        context.builder.add_file(
            path=str(repo_root),
            status="success",
            message=f"MCP server stopped (repo={repo_root})",
        )
        _finish_success(context)
    except (RuntimeError, OSError, ValueError) as exc:
        _fail_command(
            context,
            detail=f"MCP server error: {exc}",
            status=STATUS_INTERNAL_ERROR,
            error_status="error",
            options=_FailureOptions(
                kind="mcp-server-error",
                extras={"repo": repo_root},
                exc=exc,
            ),
        )


index_group = typer.Typer()
app.add_typer(index_group, name="index")

FreshOption = Annotated[
    bool,
    typer.Option(
        "--fresh/--incremental",
        help="Rebuild entire index (fresh) or only changed files (incremental).",
    ),
]


@index_group.command("build")
def build_index(repo: RepoOption = ".", *, fresh: FreshOption = False) -> None:
    """Build or update the persistent symbol index.

    Parameters
    ----------
    repo : str, optional
        Repository root path, by default ".".
    fresh : bool, optional
        If True, rebuild entire index; otherwise incremental, by default False.
    """
    context = _start_command("index/build", repo=repo, fresh=fresh)
    repo_path = Path(repo).resolve()

    try:
        db = repo_path / ".kgf" / "codeintel.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        with IndexStore(db) as store:
            ensure_schema(store)
            count = index_incremental(store, repo_path, changed_only=not fresh)
        context.builder.add_file(
            path=str(db),
            status="success",
            message=f"Indexed {count} files from repo={repo_path} into {db}",
        )
        _finish_success(context)
    except FileNotFoundError as exc:
        _fail_command(
            context,
            detail=f"Repository not found: {repo_path}",
            status=STATUS_NOT_FOUND,
            error_status="config",
            options=_FailureOptions(
                kind="repo-missing",
                extras={"repo": repo_path},
                exc=exc,
            ),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        _fail_command(
            context,
            detail=f"Index build error: {exc}",
            status=STATUS_INTERNAL_ERROR,
            error_status="error",
            options=_FailureOptions(
                kind="index-build-error",
                extras={"repo": repo_path, "fresh": fresh},
                exc=exc,
            ),
        )


@index_group.command("rebuild")
def rebuild_index(repo: RepoOption = ".", *, confirm: bool = False) -> None:
    """Completely rebuild the symbol index from scratch.

    This deletes the existing index database and re-indexes all files. Use this
    if the index becomes corrupted, after upgrading Tree-sitter grammars, or when
    the index schema changes.

    WARNING: This operation cannot be undone. The existing index will be permanently deleted.

    Parameters
    ----------
    repo : str, optional
        Repository root path, by default ".".
    confirm : bool, optional
        Skip confirmation prompt if True, by default False.
    """
    context = _start_command("index/rebuild", repo=repo, confirm=confirm)
    repo_path = Path(repo).resolve()
    db = repo_path / ".kgf" / "codeintel.db"

    try:
        if db.exists():
            if not confirm:
                typer.confirm(
                    f"Delete {db} and rebuild from scratch?",
                    abort=True,
                )
            db.unlink()
            typer.echo(f"✓ Deleted existing index: {db}")

        # Now build fresh (reuse build_index logic)
        typer.echo("Building fresh index...")
        db.parent.mkdir(parents=True, exist_ok=True)

        with IndexStore(db) as store:
            ensure_schema(store)
            count = index_incremental(store, repo_path, changed_only=False)

            # Collect statistics
            cursor = store.execute("SELECT COUNT(*) FROM symbols")
            symbol_count = cursor.fetchone()[0]
            cursor = store.execute("SELECT COUNT(*) FROM refs")
            ref_count = cursor.fetchone()[0]
            cursor = store.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]

            # Update metrics
            cursor = store.execute("SELECT lang, COUNT(*) FROM symbols GROUP BY lang")
            symbol_counts = dict(cursor.fetchall())
            update_index_metrics(symbol_counts, ref_count, file_count)

        duration = time.time() - context.start
        INDEX_BUILD_DURATION_SECONDS.observe(duration)

        context.builder.add_file(
            path=str(db),
            status="success",
            message=f"Rebuilt index with {count} files ({symbol_count} symbols) in {duration:.2f}s",
        )
        typer.echo(f"✓ Rebuilt index with {count} files in {duration:.2f}s")
        typer.echo(f"  Symbols: {symbol_count}, References: {ref_count}, Files: {file_count}")
        _finish_success(context)

    except (RuntimeError, OSError, ValueError) as exc:
        _fail_command(
            context,
            detail=f"Index rebuild error: {exc}",
            status=STATUS_INTERNAL_ERROR,
            error_status="error",
            options=_FailureOptions(
                kind="index-rebuild-error",
                extras={"repo": repo_path},
                exc=exc,
            ),
        )


if __name__ == "__main__":  # pragma: no cover - script entry point
    app()
