"""Domain orchestration for the docstring builder pipeline."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import enum
import json
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from tools.docstring_builder import BUILDER_VERSION
from tools.docstring_builder.apply import apply_edits
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config import (
    BuilderConfig,
    ConfigSelection,
    load_config_with_selection,
)
from tools.docstring_builder.docfacts import (
    DOCFACTS_VERSION,
    DocFact,
    DocfactsProvenance,
    build_docfacts,
    build_docfacts_document,
    validate_docfacts_payload,
    write_docfacts,
)
from tools.docstring_builder.harvest import HarvestResult, harvest_file
from tools.docstring_builder.io import (
    InvalidPathError,
    dependents_for,
    hash_file,
    matches_patterns,
    module_to_path,
    select_files,
    should_ignore,
)
from tools.docstring_builder.ir import (
    IRDocstring,
    build_ir,
    validate_ir,
)
from tools.docstring_builder.models import (
    CacheSummary,
    CliResult,
    DocfactsDocumentLike,
    DocfactsDocumentPayload,
    DocfactsProvenancePayload,
    DocstringBuilderError,
    ErrorReport,
    FileReport,
    InputHash,
    RunStatus,
    RunSummary,
    SchemaViolationError,
    StatusCounts,
    build_docfacts_document_payload,
)
from tools.docstring_builder.models import (
    ProblemDetails as ModelProblemDetails,
)
from tools.docstring_builder.normalizer import normalize_docstring
from tools.docstring_builder.observability import (
    get_metrics_registry,
    record_operation_metrics,
)
from tools.docstring_builder.orchestration.artifact_generator import (
    ArtifactGenerator,
    ArtifactGeneratorContext,
)
from tools.docstring_builder.orchestration.context_builder import (
    PipelineContextBuilder,
)
from tools.docstring_builder.orchestration.payload_builder import (
    PayloadBuilder,
    PayloadBuilderContext,
)
from tools.docstring_builder.orchestration.state_accumulator import (
    FileProcessingContext,
    PipelineStateAccumulator,
)
from tools.docstring_builder.paths import (
    CACHE_PATH,
    DOCFACTS_DIFF_PATH,
    DOCFACTS_PATH,
    DOCSTRINGS_DIFF_PATH,
    MANIFEST_PATH,
    OBSERVABILITY_MAX_ERRORS,
    OBSERVABILITY_PATH,
    REPO_ROOT,
)
from tools.docstring_builder.plugins import (
    PluginManager,
)
from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult, build_semantic_schemas
from tools.drift_preview import write_html_diff
from tools.shared.logging import get_logger, with_fields
from tools.shared.proc import ToolExecutionError, run_tool

try:  # pragma: no cover - optional dependency at runtime
    from libcst import ParserSyntaxError as _ParserSyntaxError
except ModuleNotFoundError:  # pragma: no cover - defensive guard for optional import
    _PARSER_SYNTAX_ERRORS: tuple[type[BaseException], ...] = ()
else:
    _PARSER_SYNTAX_ERRORS = (_ParserSyntaxError,)

_HARVEST_ERRORS: tuple[type[BaseException], ...] = (
    *_PARSER_SYNTAX_ERRORS,
    DocstringBuilderError,
    OSError,
)

_LOGGER = get_logger(__name__)
METRICS = get_metrics_registry()

MISSING_MODULE_PATTERNS = ("docs/_build/**",)


def _coerce_provenance_payload(data: object) -> DocfactsProvenancePayload | None:
    """Return a typed DocFacts provenance payload when ``data`` is valid."""
    if not isinstance(data, dict):
        return None
    builder_version = data.get("builderVersion")
    if not isinstance(builder_version, str):
        return None
    config_hash = data.get("configHash")
    if not isinstance(config_hash, str):
        return None
    commit_hash = data.get("commitHash")
    if not isinstance(commit_hash, str):
        return None
    generated_at = data.get("generatedAt")
    if not isinstance(generated_at, str):
        return None
    return DocfactsProvenancePayload(
        builderVersion=builder_version,
        configHash=config_hash,
        commitHash=commit_hash,
        generatedAt=generated_at,
    )


def _typed_pipeline_enabled() -> bool:
    value = os.environ.get("DOCSTRINGS_TYPED_IR", "1").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


TYPED_PIPELINE_ENABLED = _typed_pipeline_enabled()


class ExitStatus(enum.IntEnum):
    """Standardised exit codes for CLI subcommands."""

    SUCCESS = 0
    VIOLATION = 1
    CONFIG = 2
    ERROR = 3


STATUS_LABELS = {
    ExitStatus.SUCCESS: "success",
    ExitStatus.VIOLATION: "violation",
    ExitStatus.CONFIG: "config",
    ExitStatus.ERROR: "error",
}

EXIT_SUCCESS = int(ExitStatus.SUCCESS)
EXIT_VIOLATION = int(ExitStatus.VIOLATION)
EXIT_CONFIG = int(ExitStatus.CONFIG)
EXIT_ERROR = int(ExitStatus.ERROR)

_EXIT_TO_RUN_STATUS: dict[ExitStatus, RunStatus] = {
    ExitStatus.SUCCESS: RunStatus.SUCCESS,
    ExitStatus.VIOLATION: RunStatus.VIOLATION,
    ExitStatus.CONFIG: RunStatus.CONFIG,
    ExitStatus.ERROR: RunStatus.ERROR,
}


def _status_from_exit(status: ExitStatus) -> RunStatus:
    return _EXIT_TO_RUN_STATUS.get(status, RunStatus.ERROR)


def _status_from_label(label: str) -> RunStatus:
    lowered = label.lower()
    match lowered:
        case "success":
            return RunStatus.SUCCESS
        case "violation":
            return RunStatus.VIOLATION
        case "config":
            return RunStatus.CONFIG
        case "error":
            return RunStatus.ERROR
        case "warn" | "autofix":
            return RunStatus.VIOLATION
        case _:
            return RunStatus.ERROR


def _http_status_for_exit(status: ExitStatus) -> int:
    match status:
        case ExitStatus.SUCCESS:
            return 200
        case ExitStatus.VIOLATION:
            return 422
        case ExitStatus.CONFIG:
            return 400
        case ExitStatus.ERROR:
            return 500
    raise AssertionError(status)


@dataclasses.dataclass(slots=True)
class ProcessingOptions:
    """Runtime options controlling how a file is processed."""

    command: str
    force: bool
    ignore_missing: bool
    missing_patterns: tuple[str, ...]
    skip_docfacts: bool
    baseline: str | None = None


@dataclasses.dataclass(slots=True)
class FileOutcome:
    """Result of processing a single file."""

    status: ExitStatus
    docfacts: list[DocFact]
    preview: str | None
    changed: bool
    skipped: bool
    message: str | None = None
    cache_hit: bool = False
    semantics: list[SemanticResult] = dataclasses.field(default_factory=list)
    ir: list[IRDocstring] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class DocfactsOutcome:
    """Outcome of reconciling DocFacts artifacts."""

    status: ExitStatus
    message: str | None = None


@dataclasses.dataclass(slots=True)
class DocstringBuildRequest:
    """Typed request describing a docstring builder run."""

    command: str
    subcommand: str
    module: str | None = None
    since: str | None = None
    changed_only: bool = False
    explicit_paths: tuple[str, ...] = ()
    force: bool = False
    diff: bool = False
    ignore_missing: bool = False
    skip_docfacts: bool = False
    json_output: bool = False
    jobs: int = 1
    baseline: str | None = None
    only_plugins: tuple[str, ...] = ()
    disable_plugins: tuple[str, ...] = ()
    policy_overrides: Mapping[str, str] = dataclasses.field(default_factory=dict)
    llm_summary: bool = False
    llm_dry_run: bool = False
    normalize_sections: bool = False
    invoked_subcommand: str | None = None


@dataclasses.dataclass(slots=True)
class DocstringBuildResult:
    """Structured result produced by a docstring builder run."""

    exit_status: ExitStatus
    errors: list[ErrorReport]
    file_reports: list[FileReport]
    observability_payload: Mapping[str, object]
    cli_payload: CliResult | None
    manifest_path: Path | None
    problem_details: ModelProblemDetails | None
    config_selection: ConfigSelection | None
    diff_previews: list[tuple[Path, str]] = dataclasses.field(default_factory=list)


def build_problem_details(
    status: ExitStatus,
    command: str,
    subcommand: str,
    detail: str,
    *,
    instance: str | None = None,
    errors: Sequence[ErrorReport] | None = None,
) -> ModelProblemDetails:
    """Create a RFC 9457 Problem Details payload for CLI failures."""
    problem_dict = build_problem_details(
        type="https://kgfoundry.dev/problems/docbuilder/run-failed",
        title="Docstring builder run failed",
        status=_http_status_for_exit(status),
        detail=detail,
        instance=instance or "",
        extensions=None,
    )
    problem_dict["extensions"] = {
        "command": command,
        "subcommand": subcommand,
        "errorCount": len(errors) if errors is not None else 0,
    }
    return cast(ModelProblemDetails, problem_dict)


def _handle_schema_violation(context: str, exc: SchemaViolationError) -> None:
    if TYPED_PIPELINE_ENABLED:
        raise exc
    log_extra = {"problem": exc.problem} if exc.problem else None
    _LOGGER.warning("%s validation failed: %s", context, exc, extra=log_extra)


def _git_output(arguments: Sequence[str]) -> str | None:
    """Return stripped stdout for a git command or ``None`` on failure."""
    command_args: list[str] = list(arguments)
    adapter = with_fields(_LOGGER, command=command_args)
    try:
        result = run_tool(arguments, timeout=10.0)
    except ToolExecutionError as exc:  # pragma: no cover - git unavailable
        adapter.debug("git invocation failed: %s", exc)
        return None
    if result.returncode != 0:
        adapter.debug("git returned non-zero exit code: %s", result.returncode)
        return None
    output = result.stdout.strip()
    return output or None


def _resolve_commit_hash() -> str:
    command = ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
    return _git_output(command) or "unknown"


def _resolve_commit_timestamp(commit_hash: str) -> str:
    if not commit_hash or commit_hash == "unknown":
        return "1970-01-01T00:00:00Z"
    command = ["git", "-C", str(REPO_ROOT), "show", "-s", "--format=%cI", commit_hash]
    return _git_output(command) or "1970-01-01T00:00:00Z"


def _build_docfacts_provenance(config: BuilderConfig) -> DocfactsProvenance:
    commit_hash = _resolve_commit_hash()
    generated_at = _resolve_commit_timestamp(commit_hash)
    return DocfactsProvenance(
        builder_version=BUILDER_VERSION,
        config_hash=config.config_hash,
        commit_hash=commit_hash,
        generated_at=generated_at,
    )


def _collect_edits(
    result: HarvestResult,
    config: BuilderConfig,
    plugin_manager: PluginManager | None,
    *,
    format_only: bool = False,
) -> tuple[list[DocstringEdit], list[SemanticResult], list[IRDocstring]]:
    semantics = build_semantic_schemas(result, config)
    if plugin_manager is not None:
        semantics = plugin_manager.apply_transformers(result.filepath, semantics)
    edits: list[DocstringEdit] = []
    ir_entries: list[IRDocstring] = []
    for entry in semantics:
        ir_entry = build_ir(entry)
        validate_ir(ir_entry)
        ir_entries.append(ir_entry)
        text: str | None
        if config.normalize_sections:
            normalized = normalize_docstring(entry.symbol, config.ownership_marker)
            if normalized is not None:
                text = normalized
            elif format_only:
                continue
            else:
                text = render_docstring(
                    entry.schema,
                    config.ownership_marker,
                    include_signature=config.render_signature,
                )
        else:
            text = render_docstring(
                entry.schema,
                config.ownership_marker,
                include_signature=config.render_signature,
            )
        edits.append(DocstringEdit(qname=entry.symbol.qname, text=text))
    if plugin_manager is not None:
        edits = plugin_manager.apply_formatters(result.filepath, edits)
    return edits, semantics, ir_entries


def _handle_docfacts(
    docfacts: list[DocFact],
    config: BuilderConfig,
    check_mode: bool,
) -> DocfactsOutcome:
    provenance = _build_docfacts_provenance(config)
    document = build_docfacts_document(docfacts, provenance, DOCFACTS_VERSION)
    payload = build_docfacts_document_payload(cast(DocfactsDocumentLike, document))
    if check_mode:
        if not DOCFACTS_PATH.exists():
            _LOGGER.error("DocFacts missing at %s", DOCFACTS_PATH)
            return DocfactsOutcome(ExitStatus.CONFIG, "docfacts missing")
        try:
            existing_text = DOCFACTS_PATH.read_text(encoding="utf-8")
            existing_raw: object = json.loads(existing_text)
        except json.JSONDecodeError:  # pragma: no cover - defensive guard
            _LOGGER.exception("DocFacts payload at %s is not valid JSON", DOCFACTS_PATH)
            return DocfactsOutcome(ExitStatus.CONFIG, "docfacts invalid json")
        if not isinstance(existing_raw, dict):
            _LOGGER.error("DocFacts payload at %s is not a mapping", DOCFACTS_PATH)
            return DocfactsOutcome(ExitStatus.CONFIG, "docfacts invalid structure")
        existing_payload = cast(DocfactsDocumentPayload, existing_raw)
        try:
            validate_docfacts_payload(existing_payload)
        except SchemaViolationError as exc:
            _handle_schema_violation("DocFacts (check)", exc)
            return DocfactsOutcome(ExitStatus.SUCCESS)
        comparison_payload = cast(DocfactsDocumentPayload, json.loads(json.dumps(payload)))
        provenance_existing = _coerce_provenance_payload(existing_payload.get("provenance"))
        comparison_provenance = comparison_payload["provenance"]
        if provenance_existing is not None:
            commit_hash = comparison_provenance["commitHash"]
            generated_at = comparison_provenance["generatedAt"]
            existing_commit = provenance_existing.get("commitHash")
            if isinstance(existing_commit, str):
                commit_hash = existing_commit
            existing_generated = provenance_existing.get("generatedAt")
            if isinstance(existing_generated, str):
                generated_at = existing_generated
            comparison_payload["provenance"] = DocfactsProvenancePayload(
                builderVersion=comparison_provenance["builderVersion"],
                configHash=comparison_provenance["configHash"],
                commitHash=commit_hash,
                generatedAt=generated_at,
            )
        if existing_payload != comparison_payload:
            before = json.dumps(existing_payload, indent=2, sort_keys=True)
            after = json.dumps(comparison_payload, indent=2, sort_keys=True)
            write_html_diff(before, after, DOCFACTS_DIFF_PATH, "DocFacts drift")
            diff_rel = DOCFACTS_DIFF_PATH.relative_to(REPO_ROOT)
            _LOGGER.error("DocFacts drift detected; run update mode to refresh (see %s)", diff_rel)
            return DocfactsOutcome(ExitStatus.VIOLATION, "docfacts drift")
        DOCFACTS_DIFF_PATH.unlink(missing_ok=True)
        return DocfactsOutcome(ExitStatus.SUCCESS)
    written_payload = write_docfacts(DOCFACTS_PATH, document, validate=TYPED_PIPELINE_ENABLED)
    if not TYPED_PIPELINE_ENABLED:
        try:
            validate_docfacts_payload(written_payload)
        except SchemaViolationError as exc:
            _handle_schema_violation("DocFacts (update)", exc)
    DOCFACTS_DIFF_PATH.unlink(missing_ok=True)
    return DocfactsOutcome(ExitStatus.SUCCESS)


def _load_docfacts_from_disk() -> dict[str, DocFact]:
    """Load previously generated DocFacts entries keyed by qualified name."""
    if not DOCFACTS_PATH.exists():
        return {}
    try:
        raw_text = DOCFACTS_PATH.read_text(encoding="utf-8")
        raw: object = json.loads(raw_text)
    except json.JSONDecodeError:
        _LOGGER.warning("DocFacts cache is not valid JSON; ignoring existing data.")
        return {}
    entries: dict[str, DocFact] = {}
    payload_items: Iterable[Mapping[str, object]]
    if isinstance(raw, Mapping):
        entries_field: object = raw.get("entries", [])
        if isinstance(entries_field, Sequence) and not isinstance(entries_field, (str, bytes)):
            payload_items = [item for item in entries_field if isinstance(item, Mapping)]
        else:
            payload_items = []
    elif isinstance(raw, list):  # pragma: no cover - legacy fallback
        payload_items = [item for item in raw if isinstance(item, Mapping)]
    else:  # pragma: no cover - defensive guard
        return {}
    for item in payload_items:
        fact = DocFact.from_mapping(item)
        if fact is None:
            continue
        entries[fact.qname] = fact
    return entries


def _load_docfact_state() -> tuple[dict[str, DocFact], dict[str, Path]]:
    """Load docfact entries along with best-effort source mapping."""
    entries = _load_docfacts_from_disk()
    sources: dict[str, Path] = {}
    for qname, fact in entries.items():
        candidate: Path | None = None
        if fact.filepath:
            candidate_path = (REPO_ROOT / fact.filepath).resolve()
            if candidate_path.exists():
                candidate = candidate_path
        if candidate is None:
            candidate = module_to_path(fact.module)
        if candidate is not None:
            sources[qname] = candidate
    return entries, sources


def _record_docfacts(
    facts: Iterable[DocFact],
    file_path: Path,
    entries: dict[str, DocFact],
    sources: dict[str, Path],
) -> None:
    for fact in facts:
        entries[fact.qname] = fact
        sources[fact.qname] = file_path


def _filter_docfacts_for_output(
    entries: dict[str, DocFact],
    sources: dict[str, Path],
    config: BuilderConfig,
) -> list[DocFact]:
    filtered: list[DocFact] = []
    for qname, fact in entries.items():
        source = sources.get(qname)
        if source is None and fact.filepath:
            candidate = (REPO_ROOT / fact.filepath).resolve()
            if candidate.exists():
                source = candidate
        if source is None:
            source = module_to_path(fact.module)
        if source is not None and should_ignore(source, config):
            _LOGGER.debug("Dropping docfact %s due to ignore rules", qname)
            continue
        filtered.append(fact)
    return filtered


def _ordered_outcomes(
    files: Sequence[Path],
    jobs: int,
    config: BuilderConfig,
    cache: BuilderCache,
    options: ProcessingOptions,
    plugin_manager: PluginManager | None,
) -> Iterable[tuple[Path, FileOutcome]]:
    if jobs <= 1:
        for file_path in files:
            yield file_path, _process_file(file_path, config, cache, options, plugin_manager)
        return

    futures: list[tuple[int, Path, concurrent.futures.Future[FileOutcome]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        for index, candidate in enumerate(files):
            future = executor.submit(
                _process_file,
                candidate,
                config,
                cache,
                options,
                plugin_manager,
            )
            futures.append((index, candidate, future))

        ordered: list[tuple[int, Path, FileOutcome]] = []
        for index, candidate, future in futures:
            exception = future.exception()
            if exception is None:
                outcome = future.result()
            else:
                if isinstance(exception, KeyboardInterrupt):  # pragma: no cover - propagate
                    raise exception
                _LOGGER.error("Processing failed for %s", candidate, exc_info=exception)
                outcome = FileOutcome(
                    ExitStatus.ERROR,
                    [],
                    None,
                    False,
                    False,
                    str(exception),
                )
            ordered.append((index, candidate, outcome))

    for _, candidate, outcome in sorted(ordered, key=_ordered_index):
        yield candidate, outcome


def _ordered_index(item: tuple[int, Path, FileOutcome]) -> int:
    return item[0]


def _process_file(
    file_path: Path,
    config: BuilderConfig,
    cache: BuilderCache,
    options: ProcessingOptions,
    plugin_manager: PluginManager | None,
) -> FileOutcome:
    """Harvest, render, and apply docstrings for a single file."""
    command = options.command
    is_update = command in {"update", "fmt"}
    is_check = command == "check"
    docfacts: list[DocFact] = []
    preview: str | None = None
    changed = False
    skipped = False
    message: str | None = None
    if (
        command != "harvest"
        and not options.force
        and not cache.needs_update(file_path, config.config_hash)
    ):
        _LOGGER.debug("Skipping %s; cache is fresh", file_path)
        skipped = True
        message = "cache fresh"
        return FileOutcome(
            ExitStatus.SUCCESS,
            docfacts,
            preview,
            changed,
            skipped,
            message,
            cache_hit=True,
        )
    try:
        with record_operation_metrics("harvest", status="success"):
            result = harvest_file(file_path, config, REPO_ROOT)
        if plugin_manager is not None:
            result = plugin_manager.apply_harvest(file_path, result)
    except ModuleNotFoundError as exc:
        relative = file_path.relative_to(REPO_ROOT)
        message = f"missing dependency: {exc}"
        if options.ignore_missing and matches_patterns(file_path, options.missing_patterns):
            _LOGGER.info("Skipping %s due to missing dependency: %s", relative, exc)
            return FileOutcome(
                ExitStatus.SUCCESS,
                docfacts,
                preview,
                False,
                True,
                message,
            )
        _LOGGER.exception("Failed to harvest %s", relative)
        return FileOutcome(
            ExitStatus.CONFIG,
            docfacts,
            preview,
            changed,
            skipped,
            message,
        )
    except _HARVEST_ERRORS as exc:
        _LOGGER.exception("Failed to harvest %s", file_path)
        return FileOutcome(
            ExitStatus.ERROR,
            docfacts,
            preview,
            changed,
            skipped,
            str(exc),
        )

    edits, semantics, ir_entries = _collect_edits(
        result,
        config,
        plugin_manager,
        format_only=command == "fmt",
    )
    if command == "harvest":
        docfacts = build_docfacts(semantics)
        return FileOutcome(
            ExitStatus.SUCCESS,
            docfacts,
            preview,
            changed,
            skipped,
            message,
            semantics=list(semantics),
            ir=ir_entries,
        )

    if not semantics:
        if is_update:
            cache.update(file_path, config.config_hash)
        message = "no managed symbols"
        return FileOutcome(
            ExitStatus.SUCCESS,
            docfacts,
            preview,
            changed,
            skipped,
            message,
            semantics=list(semantics),
            ir=ir_entries,
        )

    changed, preview = apply_edits(result, edits, write=is_update)
    status = ExitStatus.SUCCESS
    if is_check and changed:
        relative = file_path.relative_to(REPO_ROOT)
        _LOGGER.error("Docstrings out of date in %s", relative)
        status = ExitStatus.VIOLATION
        message = "docstrings drift"
    if is_update:
        cache.update(file_path, config.config_hash)
    docfacts = build_docfacts(semantics)
    return FileOutcome(
        status,
        docfacts,
        preview,
        changed,
        skipped,
        message,
        semantics=list(semantics),
        ir=ir_entries,
    )


def _print_failure_summary(payload: Mapping[str, object]) -> None:
    """Emit a concise summary to stderr when the CLI exits non-zero."""
    summary_obj = payload.get("summary")
    if isinstance(summary_obj, Mapping):
        summary: Mapping[str, object] = summary_obj
    else:
        summary = {}

    errors_obj = payload.get("errors")
    if isinstance(errors_obj, Sequence):
        error_entries = [entry for entry in errors_obj if isinstance(entry, Mapping)]
    else:
        error_entries = []

    def _coerce_int(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    def _coerce_str(value: object, fallback: str) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return fallback
        return str(value)

    considered = _coerce_int(summary.get("considered"))
    processed = _coerce_int(summary.get("processed"))
    changed = _coerce_int(summary.get("changed"))

    status_counts_obj = summary.get("status_counts")
    if isinstance(status_counts_obj, Mapping):
        status_counts = {str(key): _coerce_int(value) for key, value in status_counts_obj.items()}
    else:
        status_counts = {}

    lines = [
        "[SUMMARY] Docstring builder reported issues.",
        f"  Considered files: {considered}",
        f"  Processed files: {processed}",
        f"  Changed files: {changed}",
        f"  Status counts: {status_counts}",
        f"  Observability log: {OBSERVABILITY_PATH}",
    ]
    if error_entries:
        lines.append("  Top errors:")
        for entry in error_entries[:5]:
            file_name = _coerce_str(entry.get("file"), "<unknown>")
            status = _coerce_str(entry.get("status"), "unknown")
            message = _coerce_str(entry.get("message"), "no additional details")
            lines.append(f"    - {file_name}: {status} ({message})")
    for line in lines:
        _LOGGER.error(line)


def render_cli_result(result: DocstringBuildResult) -> CliResult | None:
    """Return the CLI payload for JSON output when available."""
    return result.cli_payload


def render_failure_summary(result: DocstringBuildResult) -> None:
    """Log a structured failure summary for non-successful runs."""
    if result.exit_status is ExitStatus.SUCCESS:
        return
    _print_failure_summary(result.observability_payload)


def load_builder_config(
    override: str | None = None,
) -> tuple[BuilderConfig, ConfigSelection]:
    """Load builder configuration honouring CLI/environment precedence."""
    config, selection = load_config_with_selection(override)
    return config, selection


def _initial_summary(subcommand: str) -> RunSummary:
    summary: RunSummary = {
        "considered": 0,
        "processed": 0,
        "skipped": 0,
        "changed": 0,
        "duration_seconds": 0.0,
        "status_counts": {
            "success": 0,
            "violation": 0,
            "config": 0,
            "error": 0,
        },
        "cache_hits": 0,
        "cache_misses": 0,
        "subcommand": subcommand,
        "docfacts_checked": False,
    }
    return summary


def _build_observability_payload(
    *,
    status: ExitStatus,
    summary: RunSummary,
    errors: Sequence[ErrorReport],
    selection: ConfigSelection | None,
) -> dict[str, object]:
    limited_errors = list(errors)[:OBSERVABILITY_MAX_ERRORS]
    cache_hits_value = summary.get("cache_hits")
    cache_misses_value = summary.get("cache_misses")
    cache_hits = cache_hits_value if isinstance(cache_hits_value, int) else 0
    cache_misses = cache_misses_value if isinstance(cache_misses_value, int) else 0
    payload: dict[str, object] = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": STATUS_LABELS[status],
        "summary": summary,
        "errors": limited_errors,
    }
    if selection is not None:
        payload["config"] = {"path": str(selection.path), "source": selection.source}
    payload["cache"] = {
        "path": str(CACHE_PATH),
        "exists": CACHE_PATH.exists(),
        "hits": cache_hits,
        "misses": cache_misses,
    }
    return payload


def _build_error_result(
    status: ExitStatus,
    request: DocstringBuildRequest,
    detail: str,
    *,
    selection: ConfigSelection | None,
) -> DocstringBuildResult:
    command = request.command or "unknown"
    subcommand = request.invoked_subcommand or request.subcommand or command
    errors: list[ErrorReport] = [
        {
            "file": "<command>",
            "status": _status_from_exit(status),
            "message": detail,
        }
    ]
    summary = _initial_summary(subcommand)
    status_counts_block = summary["status_counts"]
    if status is ExitStatus.SUCCESS:
        status_counts_block["success"] += 1
    elif status is ExitStatus.VIOLATION:
        status_counts_block["violation"] += 1
    elif status is ExitStatus.CONFIG:
        status_counts_block["config"] += 1
    else:
        status_counts_block["error"] += 1
    observability_payload = _build_observability_payload(
        status=status,
        summary=summary,
        errors=errors,
        selection=selection,
    )
    problem = build_problem_details(status, command, subcommand, detail, errors=errors)
    return DocstringBuildResult(
        exit_status=status,
        errors=errors,
        file_reports=[],
        observability_payload=observability_payload,
        cli_payload=None,
        manifest_path=None,
        problem_details=problem,
        config_selection=selection,
    )


def _run_pipeline(
    files: Iterable[Path],
    request: DocstringBuildRequest,
    config: BuilderConfig,
    selection: ConfigSelection | None,
) -> DocstringBuildResult:
    """Execute the docstring building pipeline.

    This function orchestrates the entire pipeline by delegating to modular
    helper classes that manage specific concerns: context setup, file processing,
    artifact generation, and payload building. This design provides clear separation
    of concerns and makes the pipeline logic easy to test and understand.

    Parameters
    ----------
    files : Iterable[Path]
        Paths to Python files for processing.
    request : DocstringBuildRequest
        User request with command, options, and flags.
    config : BuilderConfig
        Loaded builder configuration.
    selection : ConfigSelection | None
        Resolved configuration selection (if any).

    Returns
    -------
    DocstringBuildResult
        Complete result including exit status, errors, reports, and payloads.

    Notes
    -----
    All I/O and side effects are encapsulated in helper classes:
    - PipelineContextBuilder: setup/teardown and dependency resolution
    - PipelineStateAccumulator: mutable state from file processing
    - ArtifactGenerator: diff and manifest generation
    - PayloadBuilder: observability and CLI payloads
    """
    files_list = list(files)
    start = time.perf_counter()

    # Phase 1: Build execution context
    context_builder = PipelineContextBuilder(request, config, selection, files_list)
    build_result = context_builder.build()
    if isinstance(build_result, DocstringBuildResult):
        return build_result
    _, plugin_manager, policy_engine, options = build_result

    try:
        # Phase 2: Process files and accumulate state
        accumulator = PipelineStateAccumulator()
        cache = BuilderCache(CACHE_PATH)
        docfact_entries, docfact_sources = _load_docfact_state()
        docfacts_checked = False
        docfacts_result: DocfactsOutcome | None = None
        docfacts_payload_text: str | None = None

        jobs = request.jobs or 1
        if jobs <= 0:
            jobs = max(1, os.cpu_count() or 1)

        outcomes = _ordered_outcomes(files_list, jobs, config, cache, options, plugin_manager)
        for file_path, outcome in outcomes:
            ctx = FileProcessingContext(
                file_path=file_path,
                outcome=outcome,
                options=options,
                request_command=request.command,
                request_json_output=request.json_output,
                request_diff=request.diff,
            )
            accumulator.process_file_outcome(ctx)
            _record_docfacts(outcome.docfacts, file_path, docfact_entries, docfact_sources)
            if outcome.semantics:
                policy_engine.record(outcome.semantics)

        if request.command in {"update", "check"} and not options.skip_docfacts:
            filtered = _filter_docfacts_for_output(docfact_entries, docfact_sources, config)
            docfacts_result = _handle_docfacts(
                filtered, config, check_mode=request.command == "check"
            )
            docfacts_checked = True
            accumulator.status_counts[docfacts_result.status] += 1
            if docfacts_result.status is not ExitStatus.SUCCESS:
                accumulator.errors.append(
                    {
                        "file": "<docfacts>",
                        "status": _status_from_exit(docfacts_result.status),
                        "message": docfacts_result.message or "",
                    }
                )
            else:
                try:
                    docfacts_payload_text = DOCFACTS_PATH.read_text(encoding="utf-8")
                except FileNotFoundError:
                    docfacts_payload_text = None

        if request.command == "update":
            cache.write()

        policy_report = policy_engine.finalize()
        for violation in policy_report.violations:
            accumulator.errors.append(
                {
                    "file": violation.symbol,
                    "status": _status_from_label(str(violation.action)),
                    "message": violation.message,
                }
            )
            if violation.fatal:
                accumulator.status_counts[ExitStatus.VIOLATION] += 1

        # Phase 3: Build metadata
        duration = time.perf_counter() - start
        exit_status = max(
            (status for status, count in accumulator.status_counts.items() if count),
            default=ExitStatus.SUCCESS,
        )
        status_label = STATUS_LABELS[exit_status]
        METRICS.cli_duration_seconds.labels(
            command=request.command or "unknown", status=status_label
        ).observe(duration)
        METRICS.runs_total.labels(status=status_label).inc()

        status_counts_full = _build_status_counts(accumulator.status_counts)
        cache_payload = _build_cache_summary(accumulator.cache_hits, accumulator.cache_misses)
        input_hashes = _build_input_hashes(files_list)
        dependency_map = _build_dependency_map(files_list)

        # Phase 4: Generate artifacts
        artifact_gen = ArtifactGenerator()
        artifact_ctx = ArtifactGeneratorContext(
            docstring_diffs=accumulator.docstring_diffs,
            all_ir=accumulator.all_ir,
            cache_payload=cache_payload,
            input_hashes=input_hashes,
            dependency_map=dependency_map,
            policy_report=policy_report,
            request=request,
            selection=selection,
            docfacts_payload_text=docfacts_payload_text,
            baseline=options.baseline,
            files_count=len(files_list),
            processed_count=accumulator.processed_count,
            skipped_count=accumulator.skipped_count,
            changed_count=accumulator.changed_count,
        )
        diff_links = artifact_gen.generate_and_persist(artifact_ctx)

        # Phase 5: Build observability and CLI payloads
        payload_builder = PayloadBuilder()
        payload_ctx = PayloadBuilderContext(
            exit_status=exit_status,
            command=request.command or "unknown",
            invoked=str(request.invoked_subcommand or request.subcommand or request.command or ""),
            duration=duration,
            files_count=len(files_list),
            processed_count=accumulator.processed_count,
            skipped_count=accumulator.skipped_count,
            changed_count=accumulator.changed_count,
            cache_hits=accumulator.cache_hits,
            cache_misses=accumulator.cache_misses,
            status_counts=status_counts_full,
            cache_payload=cache_payload,
            input_hashes=input_hashes,
            errors=accumulator.errors,
            file_reports=accumulator.file_reports,
            policy_report=policy_report,
            plugin_manager=plugin_manager,
            selection=selection,
            docfacts_checked=docfacts_checked,
            baseline=options.baseline,
            diff_links=diff_links,
        )
        observability_payload, cli_result, problem_details = payload_builder.build_all(payload_ctx)

        # Phase 6: Persist payloads and assemble result
        OBSERVABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        OBSERVABILITY_PATH.write_text(
            json.dumps(observability_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return DocstringBuildResult(
            exit_status=exit_status,
            errors=accumulator.errors,
            file_reports=accumulator.file_reports,
            observability_payload=observability_payload,
            cli_payload=cli_result,
            manifest_path=MANIFEST_PATH,
            problem_details=problem_details,
            config_selection=selection,
            diff_previews=accumulator.diff_previews,
        )
    finally:
        if not files_list and request.command == "check" and request.diff:
            DOCSTRINGS_DIFF_PATH.unlink(missing_ok=True)
            DOCFACTS_DIFF_PATH.unlink(missing_ok=True)


def _build_status_counts(status_counts: Counter[ExitStatus]) -> StatusCounts:
    """Build StatusCounts from counter.

    Parameters
    ----------
    status_counts : Counter[ExitStatus]
        Status counter from file processing.

    Returns
    -------
    StatusCounts
        Typed status counts dictionary.
    """
    status_counts.setdefault(ExitStatus.SUCCESS, 0)
    return {
        "success": status_counts.get(ExitStatus.SUCCESS, 0),
        "violation": status_counts.get(ExitStatus.VIOLATION, 0),
        "config": status_counts.get(ExitStatus.CONFIG, 0),
        "error": status_counts.get(ExitStatus.ERROR, 0),
    }


def _build_cache_summary(cache_hits: int, cache_misses: int) -> CacheSummary:
    """Build cache summary payload.

    Parameters
    ----------
    cache_hits : int
        Number of cache hits.
    cache_misses : int
        Number of cache misses.

    Returns
    -------
    CacheSummary
        Cache metadata.
    """
    return {
        "path": str(CACHE_PATH),
        "exists": CACHE_PATH.exists(),
        "mtime": None,
        "hits": cache_hits,
        "misses": cache_misses,
    }


def _build_input_hashes(files_list: list[Path]) -> dict[str, InputHash]:
    """Build input file hashes.

    Parameters
    ----------
    files_list : list[Path]
        Files to hash.

    Returns
    -------
    dict[str, InputHash]
        File path to hash metadata mapping.
    """
    input_hashes: dict[str, InputHash] = {}
    for path in files_list:
        rel = str(path.relative_to(REPO_ROOT))
        if path.exists():
            input_hashes[rel] = {
                "hash": hash_file(path),
                "mtime": datetime.datetime.fromtimestamp(
                    path.stat().st_mtime, tz=datetime.UTC
                ).isoformat(),
            }
        else:
            input_hashes[rel] = {"hash": "", "mtime": None}
    return input_hashes


def _build_dependency_map(files_list: list[Path]) -> dict[str, list[str]]:
    """Build file dependency map.

    Parameters
    ----------
    files_list : list[Path]
        Files to build dependency map for.

    Returns
    -------
    dict[str, list[str]]
        File path to dependent paths mapping.
    """
    return {
        str(path.relative_to(REPO_ROOT)): [
            str(dependent.relative_to(REPO_ROOT)) for dependent in dependents_for(path)
        ]
        for path in files_list
    }


def run_docstring_builder(
    request: DocstringBuildRequest,
    *,
    config_override: str | None = None,
) -> DocstringBuildResult:
    """Execute the docstring builder for ``request`` and return a structured result."""
    config, selection = load_builder_config(config_override)
    if request.llm_summary:
        config.llm_summary_mode = "apply"
    elif request.llm_dry_run:
        config.llm_summary_mode = "dry-run"
    if request.normalize_sections:
        config.normalize_sections = True
    try:
        files = select_files(
            config,
            module=request.module or None,
            since=request.since or None,
            changed_only=request.changed_only,
            explicit_paths=list(request.explicit_paths) or None,
        )
    except InvalidPathError:
        _LOGGER.exception("Invalid path supplied to docstring builder")
        return _build_error_result(
            ExitStatus.CONFIG,
            request,
            "Invalid path supplied to docstring builder",
            selection=selection,
        )
    return _run_pipeline(files, request, config, selection)


__all__ = [
    "DocstringBuildRequest",
    "DocstringBuildResult",
    "ExitStatus",
    "InvalidPathError",
    "build_problem_details",
    "load_builder_config",
    "render_cli_result",
    "render_failure_summary",
    "run_docstring_builder",
]
