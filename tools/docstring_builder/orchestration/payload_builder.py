"""Payload building for pipeline observability and CLI output.

This module manages construction of observability and CLI output payloads,
including run summaries, problem details, and complete CLI result envelopes.
It separates payload building concerns from the main orchestration logic.

Ownership: docstring-builder team
Version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools.docstring_builder.models import (
    CacheSummary,
    CliResult,
    DocfactsReport,
    ErrorReport,
    FileReport,
    InputHash,
    ObservabilityReport,
    PluginReport,
    ProblemDetails,
    RunSummary,
    StatusCounts,
)
from tools.docstring_builder.orchestrator import (
    STATUS_LABELS,
    ExitStatus,
    _build_observability_payload,
    _status_from_exit,
    build_cli_result_skeleton,
    build_problem_details,
    validate_cli_output,
)
from tools.docstring_builder.paths import DOCFACTS_DIFF_PATH, DOCFACTS_PATH, REPO_ROOT

if TYPE_CHECKING:
    from tools.docstring_builder.config import ConfigSelection
    from tools.docstring_builder.plugins import PluginManager
    from tools.docstring_builder.policy import PolicyReport

OBSERVABILITY_MAX_ERRORS = 10
DOCFACTS_VERSION = "1.0"


@dataclass(slots=True)
class PayloadBuilderContext:
    """Context for building observability and CLI payloads.

    Parameters
    ----------
    exit_status : ExitStatus
        The final exit status.
    command : str
        The command (update, check, etc.).
    invoked : str
        The invoked subcommand name.
    duration : float
        Execution duration in seconds.
    files_count : int
        Total files considered.
    processed_count : int
        Files fully processed.
    skipped_count : int
        Files skipped.
    changed_count : int
        Files with changes.
    cache_hits : int
        Cache hit count.
    cache_misses : int
        Cache miss count.
    status_counts : StatusCounts
        Status count breakdown.
    cache_payload : CacheSummary
        Cache metadata.
    input_hashes : dict[str, InputHash]
        Input file hashes.
    errors : list[ErrorReport]
        Accumulated errors.
    file_reports : list[FileReport]
        File-level reports.
    policy_report : PolicyReport
        Policy report.
    plugin_manager : PluginManager | None
        Plugin manager.
    selection : ConfigSelection | None
        Config selection.
    docfacts_checked : bool
        Whether docfacts was checked.
    baseline : str | None
        Baseline version.
    diff_links : dict[str, str]
        Diff artifact links.
    """

    exit_status: ExitStatus
    command: str
    invoked: str
    duration: float
    files_count: int
    processed_count: int
    skipped_count: int
    changed_count: int
    cache_hits: int
    cache_misses: int
    status_counts: StatusCounts
    cache_payload: CacheSummary
    input_hashes: dict[str, InputHash]
    errors: list[ErrorReport]
    file_reports: list[FileReport]
    policy_report: PolicyReport
    plugin_manager: PluginManager | None
    selection: ConfigSelection | None
    docfacts_checked: bool
    baseline: str | None
    diff_links: dict[str, str]


@dataclass(slots=True)
class PayloadBuilder:
    """Builds observability and CLI output payloads.

    Encapsulates all payload construction logic, including:
    - Run summary aggregation
    - Observability payload creation with policy and drift info
    - Full CLI result skeleton with all nested blocks
    - Problem details generation for error cases

    This replaces scattered payload building throughout the orchestrator
    with a focused, testable class.
    """

    def build_all(
        self, ctx: PayloadBuilderContext
    ) -> tuple[dict[str, object], CliResult | None, ProblemDetails | None]:
        """Build all payloads in a single pass.

        Parameters
        ----------
        ctx : PayloadBuilderContext
            Context for payload building.

        Returns
        -------
        tuple[dict[str, object], CliResult | None, ProblemDetails | None]
            Tuple of (observability_payload, cli_result, problem_details).
        """
        # Build run summary
        summary = self._build_run_summary(ctx)

        # Build observability payload
        observability_payload = self._build_observability_payload(ctx, summary)

        # Build CLI result if requested
        cli_result = self._build_cli_result(ctx) if ctx.file_reports else None

        # Build problem details if error
        problem_details = (
            self._build_problem_details(ctx) if ctx.exit_status is not ExitStatus.SUCCESS else None
        )

        # Attach problem details to CLI result if both exist
        if cli_result is not None and problem_details is not None:
            cli_result["problem"] = problem_details

        # Validate CLI result
        if cli_result is not None:
            validate_cli_output(cli_result)

        return observability_payload, cli_result, problem_details

    @staticmethod
    def _build_run_summary(ctx: PayloadBuilderContext) -> RunSummary:
        """Build run summary from context.

        Parameters
        ----------
        ctx : PayloadBuilderContext
            Context for building.

        Returns
        -------
        RunSummary
            Complete run summary.
        """
        return {
            "considered": ctx.files_count,
            "processed": ctx.processed_count,
            "skipped": ctx.skipped_count,
            "changed": ctx.changed_count,
            "duration_seconds": ctx.duration,
            "status_counts": ctx.status_counts,
            "cache_hits": ctx.cache_hits,
            "cache_misses": ctx.cache_misses,
            "subcommand": ctx.invoked,
            "docfacts_checked": ctx.docfacts_checked,
        }

    @staticmethod
    def _build_observability_payload(
        ctx: PayloadBuilderContext, summary: RunSummary
    ) -> dict[str, object]:
        """Build observability payload.

        Parameters
        ----------
        ctx : PayloadBuilderContext
            Context for building.
        summary : RunSummary
            Run summary.

        Returns
        -------
        dict[str, object]
            Observability payload.
        """
        observability = _build_observability_payload(
            status=ctx.exit_status,
            summary=summary,
            errors=ctx.errors,
            selection=ctx.selection,
        )

        if ctx.selection is not None:
            observability["config"] = {
                "path": str(getattr(ctx.selection, "path", "")),
                "source": getattr(ctx.selection, "source", ""),
            }

        observability["policy"] = {
            "coverage": ctx.policy_report.coverage,
            "threshold": ctx.policy_report.threshold,
            "violations": len(ctx.policy_report.violations),
            "fatal_violations": sum(
                1 for violation in ctx.policy_report.violations if violation.fatal
            ),
        }

        if ctx.diff_links:
            observability["drift_previews"] = ctx.diff_links

        return observability

    @staticmethod
    def _build_cli_result(ctx: PayloadBuilderContext) -> CliResult:
        """Build complete CLI result.

        Parameters
        ----------
        ctx : PayloadBuilderContext
            Context for building.

        Returns
        -------
        CliResult
            Complete CLI result envelope.
        """
        cli_result = build_cli_result_skeleton(_status_from_exit(ctx.exit_status))
        cli_result["command"] = ctx.command
        cli_result["subcommand"] = ctx.invoked
        cli_result["durationSeconds"] = ctx.duration
        cli_result["files"] = ctx.file_reports
        cli_result["errors"] = ctx.errors

        summary_block = cli_result["summary"]
        summary_block["considered"] = ctx.files_count
        summary_block["processed"] = ctx.processed_count
        summary_block["skipped"] = ctx.skipped_count
        summary_block["changed"] = ctx.changed_count
        summary_block["status_counts"] = ctx.status_counts
        summary_block["docfacts_checked"] = ctx.docfacts_checked
        summary_block["cache_hits"] = ctx.cache_hits
        summary_block["cache_misses"] = ctx.cache_misses
        summary_block["duration_seconds"] = ctx.duration
        summary_block["subcommand"] = ctx.invoked

        # Add policy info
        cli_result["policy"] = {
            "coverage": ctx.policy_report.coverage,
            "threshold": ctx.policy_report.threshold,
            "violations": [
                {
                    "rule": violation.rule,
                    "symbol": violation.symbol,
                    "action": str(violation.action),
                    "message": violation.message,
                }
                for violation in ctx.policy_report.violations
            ],
        }

        if ctx.baseline:
            cli_result["baseline"] = ctx.baseline

        cli_result["cache"] = ctx.cache_payload
        cli_result["inputs"] = ctx.input_hashes

        # Add plugin info
        if ctx.plugin_manager is not None:
            plugin_report: PluginReport = {
                "enabled": ctx.plugin_manager.enabled_plugins(),
                "available": ctx.plugin_manager.available,
                "disabled": ctx.plugin_manager.disabled,
                "skipped": ctx.plugin_manager.skipped,
            }
            cli_result["plugins"] = plugin_report

        # Add docfacts info
        docfacts_report: DocfactsReport = {
            "path": str(DOCFACTS_PATH.relative_to(REPO_ROOT)),
            "version": DOCFACTS_VERSION,
            "validated": ctx.docfacts_checked,
        }
        if DOCFACTS_DIFF_PATH.exists():
            docfacts_report["diff"] = str(DOCFACTS_DIFF_PATH.relative_to(REPO_ROOT))
        cli_result["docfacts"] = docfacts_report

        # Add observability block
        observability_block: ObservabilityReport = {
            "status": _status_from_exit(ctx.exit_status),
            "errors": ctx.errors[:OBSERVABILITY_MAX_ERRORS],
        }
        if ctx.diff_links:
            observability_block["driftPreviews"] = ctx.diff_links
        cli_result["observability"] = observability_block

        return cli_result

    @staticmethod
    def _build_problem_details(ctx: PayloadBuilderContext) -> ProblemDetails:
        """Build problem details for error status.

        Parameters
        ----------
        ctx : PayloadBuilderContext
            Context for building.

        Returns
        -------
        ProblemDetails
            Problem details envelope.
        """
        return build_problem_details(
            ctx.exit_status,
            ctx.command,
            ctx.invoked,
            f"Docstring builder exited with status {STATUS_LABELS[ctx.exit_status]}",
            errors=ctx.errors,
        )
