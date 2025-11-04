"""Core pipeline orchestration for the docstring builder."""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING

from tools.docstring_builder.builder_types import (
    DocstringBuildResult,
)
from tools.docstring_builder.docfacts import DOCFACTS_VERSION
from tools.docstring_builder.io import dependents_for, hash_file
from tools.docstring_builder.manifest_builder import ManifestContext, write_manifest
from tools.docstring_builder.models import (
    build_cli_result_skeleton,
    validate_cli_output,
)
from tools.docstring_builder.paths import (
    CACHE_PATH,
    DOCFACTS_DIFF_PATH,
    DOCFACTS_PATH,
    OBSERVABILITY_MAX_ERRORS,
    OBSERVABILITY_PATH,
    REPO_ROOT,
)
from tools.docstring_builder.pipeline_types import (
    ErrorEnvelope,
    FileOutcome,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tools.docstring_builder.builder_types import (
        DocstringBuildRequest,
        ExitStatus,
        LoggerLike,
    )
    from tools.docstring_builder.cache import BuilderCache
    from tools.docstring_builder.config import BuilderConfig, ConfigSelection
    from tools.docstring_builder.diff_manager import DiffManager
    from tools.docstring_builder.docfacts import DocFact
    from tools.docstring_builder.docfacts_coordinator import DocfactsCoordinator
    from tools.docstring_builder.file_processor import FileProcessor
    from tools.docstring_builder.ir import IRDocstring
    from tools.docstring_builder.metrics import MetricsRecorder
    from tools.docstring_builder.models import (
        CacheSummary,
        CliResult,
        DocfactsReport,
        ErrorReport,
        FileReport,
        InputHash,
        ObservabilityReport,
        PluginReport,
        RunStatus,
        RunSummary,
        StatusCounts,
    )
    from tools.docstring_builder.pipeline_types import (
        ProcessingOptions,
    )
    from tools.docstring_builder.plugins import PluginManager
    from tools.docstring_builder.policy import PolicyEngine, PolicyReport


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.as_posix()


def _coerce_int(value: object) -> int:
    """Return ``value`` coerced to ``int`` when possible."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:  # pragma: no cover - defensive path
            return 0
    return 0


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Protocol

    from tools.docstring_builder.models import ProblemDetails as ModelProblemDetails

    class ProblemDetailsBuilder(Protocol):
        """Callable signature for building Problem Details envelopes."""

        def __call__(
            self,
            status: ExitStatus,
            request: DocstringBuildRequest,
            detail: str,
            *,
            instance: str | None = ...,  # Protocol stub body (required for TYPE_CHECKING)
            errors: Sequence[ErrorReport] | None = ...,
        ) -> ModelProblemDetails:
            """Return a Problem Details payload for the supplied context."""
            ...  # Protocol stub body (required for TYPE_CHECKING)

    class DocfactsCoordinatorFactory(Protocol):
        """Callable signature for constructing Docfacts coordinators."""

        def __call__(self, *, check_mode: bool) -> DocfactsCoordinator:
            """Return a coordinator configured for the requested mode."""
            ...

else:
    ProblemDetailsBuilder = Callable
    DocfactsCoordinatorFactory = Callable


def _new_error_envelopes() -> list[ErrorEnvelope]:
    return []


def _new_file_reports() -> list[FileReport]:
    return []


def _new_ir_list() -> list[IRDocstring]:
    return []


def _new_preview_list() -> list[tuple[Path, str]]:
    return []


@dataclass(slots=True)
class PipelineState:
    """Mutable state accumulated while processing files."""

    status_counts: Counter[ExitStatus]
    processed_count: int = 0
    skipped_count: int = 0
    changed_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[ErrorEnvelope] = field(default_factory=_new_error_envelopes)
    file_reports: list[FileReport] = field(default_factory=_new_file_reports)
    all_ir: list[IRDocstring] = field(default_factory=_new_ir_list)
    docfacts_checked: bool = False
    docfacts_payload_text: str | None = None
    diff_previews: list[tuple[Path, str]] = field(default_factory=_new_preview_list)


@dataclass(slots=True, frozen=True)
class CliResultContext:
    """Context for CLI result building."""

    exit_status: ExitStatus
    duration: float
    state: PipelineState
    policy_report: PolicyReport
    cache_summary: CacheSummary
    input_hashes: Mapping[str, InputHash]
    diff_links: Mapping[str, str]


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for PipelineRunner dependency injection."""

    request: DocstringBuildRequest
    config: BuilderConfig
    selection: ConfigSelection | None
    options: ProcessingOptions
    cache: BuilderCache
    file_processor: FileProcessor
    record_docfacts: Callable[[Iterable[DocFact], Path], None]
    filter_docfacts: Callable[[], list[DocFact]]
    docfacts_coordinator_factory: DocfactsCoordinatorFactory
    plugin_manager: PluginManager | None
    policy_engine: PolicyEngine
    metrics: MetricsRecorder
    diff_manager: DiffManager
    logger: LoggerLike
    status_from_exit: Callable[[ExitStatus], RunStatus]
    status_labels: Mapping[ExitStatus, str]
    build_problem_details: ProblemDetailsBuilder
    success_status: ExitStatus
    violation_status: ExitStatus
    config_status: ExitStatus
    error_status: ExitStatus


class PipelineRunner:
    """Coordinate docstring builder execution across helper components."""

    __slots__ = ("_cfg",)

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the pipeline runner with dependency configuration."""
        self._cfg = config

    def run(self, files: Iterable[Path]) -> DocstringBuildResult:
        """Execute the pipeline for the provided file set."""
        files_list = list(files)
        jobs = self._resolve_jobs(self._cfg.request.jobs)
        state = PipelineState(status_counts=Counter())
        start = time.perf_counter()

        self._process_files(files_list, jobs, state)
        self._maybe_reconcile_docfacts(state)
        policy_report = self._cfg.policy_engine.finalize()
        self._apply_policy_report(policy_report, state)

        duration = time.perf_counter() - start
        exit_status = self._resolve_exit_status(state.status_counts)
        command = self._cfg.request.command or "unknown"
        self._cfg.metrics.observe_cli_duration(
            command=command,
            status=exit_status,
            duration_seconds=duration,
        )

        return self._assemble_result(exit_status, duration, state, policy_report, files_list)

    @staticmethod
    def _resolve_jobs(jobs: int) -> int:
        """Determine the number of worker jobs to use."""
        if jobs <= 0:
            return max(1, os.cpu_count() or 1)
        return jobs

    @staticmethod
    def _repo_label(path: Path) -> str:
        """Return ``path`` relative to ``REPO_ROOT`` when possible."""
        return _repo_relative_path(path)

    def _process_files(self, files: Sequence[Path], jobs: int, state: PipelineState) -> None:
        """Process files using thread pool or serial execution."""
        for file_path, outcome in self._ordered_outcomes(files, jobs):
            state.status_counts[outcome.status] += 1
            if outcome.skipped:
                state.skipped_count += 1
            else:
                state.processed_count += 1
            if outcome.changed:
                state.changed_count += 1
            if outcome.cache_hit:
                state.cache_hits += 1
            else:
                state.cache_misses += 1

            if outcome.status is not self._cfg.success_status:
                state.errors.append(
                    ErrorEnvelope(
                        file=self._repo_label(file_path),
                        status=self._cfg.status_from_exit(outcome.status),
                        message=outcome.message or "",
                    )
                )

            self._cfg.record_docfacts(outcome.docfacts, file_path)
            if outcome.semantics:
                self._cfg.policy_engine.record(outcome.semantics)
            state.all_ir.extend(outcome.ir)

            if (
                self._cfg.request.command == "check"
                and self._cfg.request.diff
                and outcome.changed
                and outcome.preview is not None
            ):
                state.diff_previews.append((file_path, outcome.preview))

            if self._cfg.request.json_output:
                state.file_reports.append(self._build_file_report(file_path, outcome))

            self._cfg.diff_manager.record_docstring_baseline(file_path, outcome.preview)

        self._cfg.diff_manager.finalize_docstring_drift()

    def _ordered_outcomes(
        self, files: Sequence[Path], jobs: int
    ) -> Iterable[tuple[Path, FileOutcome]]:
        """Process files in order, respecting job count."""
        if jobs <= 1:
            for file_path in files:
                yield file_path, self._cfg.file_processor.process(file_path)
            return

        futures: list[tuple[int, Path, concurrent.futures.Future[FileOutcome]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            for index, candidate in enumerate(files):
                future = executor.submit(self._cfg.file_processor.process, candidate)
                futures.append((index, candidate, future))

            ordered: list[tuple[int, Path, FileOutcome]] = []
            for index, candidate, future in futures:
                exception = future.exception()
                if exception is None:
                    outcome = future.result()
                else:
                    if isinstance(exception, KeyboardInterrupt):  # pragma: no cover - propagate
                        raise exception
                    self._cfg.logger.error(
                        "Processing failed for %s", candidate, exc_info=exception
                    )
                    outcome = FileOutcome(
                        status=self._cfg.error_status,
                        docfacts=[],
                        preview=None,
                        changed=False,
                        skipped=False,
                        message=str(exception),
                    )
                ordered.append((index, candidate, outcome))

        for _, candidate, outcome in sorted(ordered, key=itemgetter(0)):
            yield candidate, outcome

    def _maybe_reconcile_docfacts(self, state: PipelineState) -> None:
        """Reconcile DocFacts if required."""
        if self._cfg.options.skip_docfacts or self._cfg.request.command not in {"update", "check"}:
            return

        filtered = self._cfg.filter_docfacts()
        is_check = self._cfg.request.command == "check"
        coordinator = self._cfg.docfacts_coordinator_factory(check_mode=is_check)
        result = coordinator.reconcile(filtered)
        state.docfacts_checked = True
        exit_status = self._map_docfacts_status(result.status)
        state.status_counts[exit_status] += 1

        if exit_status is not self._cfg.success_status:
            state.errors.append(
                ErrorEnvelope(
                    file="<docfacts>",
                    status=self._cfg.status_from_exit(exit_status),
                    message=result.message or "",
                )
            )
        else:
            try:
                docfacts_payload_text = DOCFACTS_PATH.read_text(encoding="utf-8")
            except FileNotFoundError:
                docfacts_payload_text = None
            state.docfacts_payload_text = docfacts_payload_text
            self._cfg.diff_manager.record_docfacts_baseline_diff(docfacts_payload_text)

    def _apply_policy_report(self, policy_report: PolicyReport, state: PipelineState) -> None:
        """Apply policy violations to error state."""
        for violation in policy_report.violations:
            state.errors.append(
                ErrorEnvelope(
                    file=violation.symbol,
                    status=self._cfg.status_from_exit(self._cfg.violation_status),
                    message=violation.message,
                )
            )
            if violation.fatal:
                state.status_counts[self._cfg.violation_status] += 1

    def _resolve_exit_status(self, status_counts: Counter[ExitStatus]) -> ExitStatus:
        """Determine the exit status from counts."""
        for status in (
            self._cfg.error_status,
            self._cfg.config_status,
            self._cfg.violation_status,
            self._cfg.success_status,
        ):
            if status_counts.get(status, 0):
                return status
        return self._cfg.success_status

    def _map_docfacts_status(self, status: str) -> ExitStatus:
        """Map DocFacts status string to ExitStatus enum."""
        if status == "success":
            return self._cfg.success_status
        if status in {"violation", "failure"}:
            return self._cfg.violation_status
        if status == "config":
            return self._cfg.config_status
        if status == "error":
            return self._cfg.error_status
        return self._cfg.error_status

    def _build_file_report(self, file_path: Path, outcome: FileOutcome) -> FileReport:
        """Build a file report for JSON output."""
        report: FileReport = {
            "path": self._repo_label(file_path),
            "status": self._cfg.status_from_exit(outcome.status),
            "changed": outcome.changed,
            "skipped": outcome.skipped,
            "cacheHit": outcome.cache_hit,
        }
        if outcome.message:
            report["message"] = outcome.message
        if outcome.preview:
            report["preview"] = outcome.preview
        if self._cfg.options.baseline:
            report["baseline"] = self._cfg.options.baseline
        return report

    def _build_run_summary(
        self,
        files: Sequence[Path],
        state: PipelineState,
        duration: float,
    ) -> Mapping[str, object]:
        """Build the run summary for observability."""
        return {
            "considered": len(files),
            "processed": state.processed_count,
            "skipped": state.skipped_count,
            "changed": state.changed_count,
            "duration_seconds": duration,
            "status_counts": {
                "success": state.status_counts.get(self._cfg.success_status, 0),
                "violation": state.status_counts.get(self._cfg.violation_status, 0),
                "config": state.status_counts.get(self._cfg.config_status, 0),
                "error": state.status_counts.get(self._cfg.error_status, 0),
            },
            "cache_hits": state.cache_hits,
            "cache_misses": state.cache_misses,
            "subcommand": str(
                self._cfg.request.invoked_subcommand
                or self._cfg.request.subcommand
                or self._cfg.request.command
                or ""
            ),
            "docfacts_checked": state.docfacts_checked,
        }

    def _assemble_result(
        self,
        exit_status: ExitStatus,
        duration: float,
        state: PipelineState,
        policy_report: PolicyReport,
        files_list: Sequence[Path],
    ) -> DocstringBuildResult:
        """Assemble the final result from accumulated state."""
        command = self._cfg.request.command or "unknown"

        cache_summary = self._build_cache_summary(state)
        if self._cfg.request.command == "update":
            self._cfg.cache.write()

        input_hashes = self._build_input_hashes(files_list)
        dependency_map = self._build_dependency_map(files_list)
        diff_links = self._cfg.diff_manager.collect_diff_links()

        manifest_path = write_manifest(
            ManifestContext(
                request=self._cfg.request,
                options=self._cfg.options,
                files=files_list,
                processed_count=state.processed_count,
                skipped_count=state.skipped_count,
                changed_count=state.changed_count,
                cache_summary=cache_summary,
                input_hashes=input_hashes,
                dependency_map=dependency_map,
                diff_links=diff_links,
                all_ir=state.all_ir,
                plugin_report=self._build_plugin_report(),
                selection=self._cfg.selection,
            )
        )

        errors_payload = [envelope.to_report() for envelope in state.errors]
        summary = self._build_run_summary(files_list, state, duration)
        observability_payload = self._build_observability_payload(
            exit_status=exit_status,
            summary=summary,
            errors=errors_payload,
            policy_report=policy_report,
            diff_links=diff_links,
        )

        OBSERVABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        OBSERVABILITY_PATH.write_text(
            json.dumps(observability_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        cli_result = self._build_cli_result(
            CliResultContext(
                exit_status=exit_status,
                duration=duration,
                state=state,
                policy_report=policy_report,
                cache_summary=cache_summary,
                input_hashes=input_hashes,
                diff_links=diff_links,
            )
        )
        problem_details: ModelProblemDetails | None = None
        if exit_status is not self._cfg.success_status:
            invoked = (
                self._cfg.request.invoked_subcommand
                or self._cfg.request.subcommand
                or self._cfg.request.command
            )
            status_label = self._cfg.status_labels.get(exit_status, "error")
            problem_details = self._cfg.build_problem_details(
                exit_status,
                self._cfg.request,
                f"Docstring builder exited with status {status_label}",
                instance=(f"urn:cli:docbuilder:{command}:{invoked}" if invoked else None),
                errors=errors_payload,
            )
            if cli_result is not None:
                cli_result["problem"] = problem_details

        if cli_result is not None:
            validate_cli_output(cli_result)

        docfacts_report = self._build_docfacts_report(docfacts_checked=state.docfacts_checked)
        if cli_result is not None and docfacts_report is not None:
            cli_result["docfacts"] = docfacts_report

        return DocstringBuildResult(
            exit_status=exit_status,
            errors=errors_payload,
            file_reports=state.file_reports,
            observability_payload=observability_payload,
            cli_payload=cli_result,
            manifest_path=manifest_path,
            problem_details=problem_details,
            config_selection=self._cfg.selection,
            diff_previews=state.diff_previews,
        )

    @staticmethod
    def _build_cache_summary(state: PipelineState) -> CacheSummary:
        """Build the cache summary for the manifest."""
        exists = CACHE_PATH.exists()
        summary: CacheSummary = {
            "path": str(CACHE_PATH),
            "exists": exists,
            "mtime": None,
            "hits": state.cache_hits,
            "misses": state.cache_misses,
        }
        if exists:
            summary["mtime"] = datetime.datetime.fromtimestamp(
                CACHE_PATH.stat().st_mtime,
                tz=datetime.UTC,
            ).isoformat()
        return summary

    @staticmethod
    def _build_input_hashes(files: Sequence[Path]) -> dict[str, InputHash]:
        """Build input file hashes for the manifest."""
        hashes: dict[str, InputHash] = {}
        for path in files:
            rel = _repo_relative_path(path)
            if path.exists():
                hashes[rel] = {
                    "hash": hash_file(path),
                    "mtime": datetime.datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=datetime.UTC,
                    ).isoformat(),
                }
            else:
                hashes[rel] = {"hash": "", "mtime": None}
        return hashes

    @staticmethod
    def _build_dependency_map(files: Sequence[Path]) -> dict[str, list[str]]:
        """Build the dependency map for the manifest."""
        mapping: dict[str, list[str]] = {}
        for path in files:
            key = _repo_relative_path(path)
            dependents = [_repo_relative_path(dependent) for dependent in dependents_for(path)]
            mapping[key] = dependents
        return mapping

    def _build_plugin_report(self) -> PluginReport:
        """Build the plugin report for the manifest."""
        if self._cfg.plugin_manager is None:
            return {"enabled": [], "available": [], "disabled": [], "skipped": []}
        return {
            "enabled": self._cfg.plugin_manager.enabled_plugins(),
            "available": self._cfg.plugin_manager.available,
            "disabled": self._cfg.plugin_manager.disabled,
            "skipped": self._cfg.plugin_manager.skipped,
        }

    def _build_observability_payload(
        self,
        *,
        exit_status: ExitStatus,
        summary: Mapping[str, object],
        errors: Sequence[ErrorReport],
        policy_report: PolicyReport,
        diff_links: Mapping[str, str],
    ) -> dict[str, object]:
        """Build the observability payload for storage."""
        limited_errors = list(errors)[:OBSERVABILITY_MAX_ERRORS]
        payload: dict[str, object] = {
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": self._cfg.status_labels.get(exit_status, "error"),
            "summary": dict(summary),
            "errors": limited_errors,
            "cache": {
                "path": str(CACHE_PATH),
                "exists": CACHE_PATH.exists(),
                "hits": _coerce_int(summary.get("cache_hits", 0)),
                "misses": _coerce_int(summary.get("cache_misses", 0)),
            },
            "policy": {
                "coverage": policy_report.coverage,
                "threshold": policy_report.threshold,
                "violations": len(policy_report.violations),
                "fatal_violations": sum(1 for v in policy_report.violations if v.fatal),
            },
        }
        if diff_links:
            payload["drift_previews"] = dict(diff_links)
        if self._cfg.selection is not None:
            payload["config"] = {
                "path": str(self._cfg.selection.path),
                "source": self._cfg.selection.source,
            }
        return payload

    def _build_cli_result(
        self,
        context: CliResultContext,
    ) -> CliResult | None:
        """Build the CLI result for JSON output."""
        if not self._cfg.request.json_output:
            return None

        cli_result = build_cli_result_skeleton(self._cfg.status_from_exit(context.exit_status))
        invoked = (
            self._cfg.request.invoked_subcommand
            or self._cfg.request.subcommand
            or self._cfg.request.command
            or ""
        )
        cli_result["command"] = self._cfg.request.command or ""
        cli_result["subcommand"] = invoked
        cli_result["durationSeconds"] = context.duration
        cli_result["files"] = context.state.file_reports
        cli_result["errors"] = [e.to_report() for e in context.state.errors]

        status_counts: StatusCounts = {
            "success": context.state.status_counts.get(self._cfg.success_status, 0),
            "violation": context.state.status_counts.get(self._cfg.violation_status, 0),
            "config": context.state.status_counts.get(self._cfg.config_status, 0),
            "error": context.state.status_counts.get(self._cfg.error_status, 0),
        }
        summary_block: RunSummary = {
            "processed": context.state.processed_count,
            "skipped": context.state.skipped_count,
            "changed": context.state.changed_count,
            "status_counts": status_counts,
            "docfacts_checked": context.state.docfacts_checked,
            "cache_hits": context.state.cache_hits,
            "cache_misses": context.state.cache_misses,
            "duration_seconds": context.duration,
            "subcommand": invoked,
        }
        cli_result["summary"] = summary_block

        cli_result["policy"] = {
            "coverage": context.policy_report.coverage,
            "threshold": context.policy_report.threshold,
            "violations": [
                {
                    "rule": v.rule,
                    "symbol": v.symbol,
                    "action": str(v.action),
                    "message": v.message,
                }
                for v in context.policy_report.violations
            ],
        }

        if self._cfg.options.baseline:
            cli_result["baseline"] = self._cfg.options.baseline

        cli_result["cache"] = context.cache_summary
        cli_result["inputs"] = dict(context.input_hashes)
        cli_result["plugins"] = self._build_plugin_report()

        obs_block: ObservabilityReport = {
            "status": self._cfg.status_from_exit(context.exit_status),
            "errors": [e.to_report() for e in context.state.errors[:OBSERVABILITY_MAX_ERRORS]],
        }
        if context.diff_links:
            obs_block["driftPreviews"] = dict(context.diff_links)
        cli_result["observability"] = obs_block

        return cli_result

    @classmethod
    def _build_docfacts_report(cls, *, docfacts_checked: bool) -> DocfactsReport | None:
        """Build the docfacts report for JSON output."""
        if not docfacts_checked:
            return None
        report: DocfactsReport = {
            "path": cls._repo_label(DOCFACTS_PATH),
            "version": DOCFACTS_VERSION,
            "validated": DOCFACTS_PATH.exists(),
        }
        if DOCFACTS_DIFF_PATH.exists():
            report["diff"] = cls._repo_label(DOCFACTS_DIFF_PATH)
        return report
