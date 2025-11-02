"""Domain orchestration for the docstring builder pipeline."""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tools.docstring_builder import BUILDER_VERSION
from tools.docstring_builder.builder_types import (
    STATUS_LABELS,
    DocstringBuildRequest,
    DocstringBuildResult,
    ExitStatus,
    _status_from_exit,
    build_problem_details,
)
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config import (
    BuilderConfig,
    ConfigSelection,
    load_config_with_selection,
)
from tools.docstring_builder.diff_manager import DiffManager
from tools.docstring_builder.docfacts import (
    DocFact,
    DocfactsProvenance,
)
from tools.docstring_builder.docfacts_coordinator import DocfactsCoordinator
from tools.docstring_builder.failure_summary import FailureSummaryRenderer
from tools.docstring_builder.file_processor import FileProcessor
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.io import (
    InvalidPathError,
    module_to_path,
    select_files,
    should_ignore,
)
from tools.docstring_builder.ir import IRDocstring, build_ir, validate_ir
from tools.docstring_builder.metrics import MetricsRecorder
from tools.docstring_builder.models import (
    DocfactsProvenancePayload,
    ErrorReport,
    SchemaViolationError,
)
from tools.docstring_builder.models import (
    ProblemDetails as ModelProblemDetails,
)
from tools.docstring_builder.normalizer import normalize_docstring
from tools.docstring_builder.observability import get_metrics_registry
from tools.docstring_builder.paths import (
    CACHE_PATH,
    DOCFACTS_DIFF_PATH,
    DOCFACTS_PATH,
    DOCSTRINGS_DIFF_PATH,
    OBSERVABILITY_PATH,
    REPO_ROOT,
)
from tools.docstring_builder.pipeline import PipelineConfig, PipelineRunner
from tools.docstring_builder.plugins import PluginManager
from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult, build_semantic_schemas
from tools.shared.logging import get_logger, with_fields
from tools.shared.proc import ToolExecutionError, run_tool

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


def _ordered_index(item: tuple[int, Path, FileOutcome]) -> int:
    return item[0]


def render_cli_result(result: DocstringBuildResult) -> CliResult | None:
    """Return the CLI payload for JSON output when available."""
    return result.cli_payload


def render_failure_summary(result: DocstringBuildResult) -> None:
    """Log a structured failure summary for non-successful runs."""
    if result.exit_status is ExitStatus.SUCCESS:
        return

    summary_obj = result.observability_payload.get("summary", {})
    if isinstance(summary_obj, Mapping):
        considered = int(summary_obj.get("considered", 0))
        processed = int(summary_obj.get("processed", 0))
        changed = int(summary_obj.get("changed", 0))
        status_counts = summary_obj.get("status_counts", {})
    else:
        considered = processed = changed = 0
        status_counts = {}

    _LOGGER.error("[SUMMARY] Docstring builder reported issues.")
    _LOGGER.error("  Considered files: %s", considered)
    _LOGGER.error("  Processed files: %s", processed)
    _LOGGER.error("  Changed files: %s", changed)
    _LOGGER.error("  Status counts: %s", status_counts)
    _LOGGER.error("  Observability log: %s", OBSERVABILITY_PATH)

    if result.errors:
        _LOGGER.error("  Top errors:")
        for entry in result.errors[:5]:
            file_name = entry.get("file", "<unknown>")
            status_label = entry.get("status", "unknown")
            message = entry.get("message", "no additional details")
            _LOGGER.error("    - %s: %s (%s)", file_name, status_label, message)


def load_builder_config(
    override: str | None = None,
) -> tuple[BuilderConfig, ConfigSelection]:
    """Load builder configuration honouring CLI/environment precedence."""
    config, selection = load_config_with_selection(override)
    return config, selection


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
    summary = {
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
    summary["status_counts"][STATUS_LABELS[status]] += 1

    observability_payload: dict[str, object] = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": STATUS_LABELS[status],
        "summary": summary,
        "errors": errors,
        "cache": {
            "path": str(CACHE_PATH),
            "exists": CACHE_PATH.exists(),
            "hits": 0,
            "misses": 0,
        },
    }
    if selection is not None:
        observability_payload["config"] = {
            "path": str(selection.path),
            "source": selection.source,
        }

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
    from tools.docstring_builder.orchestration.context_builder import PipelineContextBuilder

    files_list = list(files)

    context_builder = PipelineContextBuilder(request, config, selection, files_list)
    build_result = context_builder.build()
    if isinstance(build_result, DocstringBuildResult):
        return build_result
    logger, plugin_manager, policy_engine, options = build_result

    cache = BuilderCache(CACHE_PATH)
    docfact_entries, docfact_sources = _load_docfact_state()
    diff_manager = DiffManager(options)
    metrics = MetricsRecorder(
        cli_duration_seconds=METRICS.cli_duration_seconds,
        runs_total=METRICS.runs_total,
    )
    file_processor = FileProcessor(
        config=config,
        cache=cache,
        options=options,
        collect_edits=_collect_edits,
        plugin_manager=plugin_manager,
        logger=logger,
    )

    def record_docfacts(facts: Iterable[DocFact], file_path: Path) -> None:
        _record_docfacts(facts, file_path, docfact_entries, docfact_sources)

    def filter_docfacts() -> list[DocFact]:
        return _filter_docfacts_for_output(docfact_entries, docfact_sources, config)

    def docfacts_coordinator_factory(check_mode: bool) -> DocfactsCoordinator:
        return DocfactsCoordinator(
            config=config,
            build_provenance=_build_docfacts_provenance,
            handle_schema_violation=_handle_schema_violation,
            typed_pipeline_enabled=TYPED_PIPELINE_ENABLED,
            check_mode=check_mode,
            logger=logger,
        )

    def build_problem_details_wrapper(
        status: ExitStatus,
        command: str,
        subcommand: str,
        detail: str,
        errors: Sequence[ErrorReport] | None,
    ) -> ModelProblemDetails:
        return build_problem_details(status, command, subcommand, detail, errors=errors)

    pipeline_config = PipelineConfig(
        request=request,
        config=config,
        selection=selection,
        options=options,
        cache=cache,
        file_processor=file_processor,
        record_docfacts=record_docfacts,
        filter_docfacts=filter_docfacts,
        docfacts_coordinator_factory=docfacts_coordinator_factory,
        plugin_manager=plugin_manager,
        policy_engine=policy_engine,
        metrics=metrics,
        diff_manager=diff_manager,
        failure_renderer=FailureSummaryRenderer(logger),
        logger=logger,
        status_from_exit=_status_from_exit,
        status_labels=STATUS_LABELS,
        build_problem_details=build_problem_details_wrapper,
        success_status=ExitStatus.SUCCESS,
        violation_status=ExitStatus.VIOLATION,
        config_status=ExitStatus.CONFIG,
        error_status=ExitStatus.ERROR,
    )

    runner = PipelineRunner(pipeline_config)
    result = runner.run(files_list)

    if request.command == "update":
        cache.write()

    if not files_list and request.command == "check" and request.diff:
        DOCSTRINGS_DIFF_PATH.unlink(missing_ok=True)
        DOCFACTS_DIFF_PATH.unlink(missing_ok=True)

    return result


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
