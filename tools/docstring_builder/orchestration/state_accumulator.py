"""State accumulation for pipeline file processing.

This module manages mutable state accumulated during file processing,
including counters, error tracking, report generation, and outcome
aggregation. It provides a single, typed accumulator for all pipeline
state instead of scattering local variables throughout the orchestrator.

Ownership: docstring-builder team
Version: 1.0
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tools.docstring_builder.io import read_baseline_version
from tools.docstring_builder.models import ErrorReport, FileReport, RunStatus
from tools.docstring_builder.orchestrator import ExitStatus
from tools.docstring_builder.paths import REPO_ROOT
from tools.docstring_builder.pipeline_types import FileOutcome, ProcessingOptions
from tools.drift_preview import DocstringDriftEntry

if TYPE_CHECKING:
    from tools.docstring_builder.ir import IRDocstring


@dataclass(slots=True)
class FileProcessingContext:
    """Context for processing a single file outcome.

    Parameters
    ----------
    file_path : Path
        Path to the file being processed.
    outcome : FileOutcome
        The outcome of processing this file.
    options : ProcessingOptions
        Processing options (baseline, etc.).
    request_command : str | None
        The request command ("update", "check", etc.).
    request_json_output : bool
        Whether JSON output is requested.
    request_diff : bool
        Whether diff previews are requested.
    """

    file_path: Path
    outcome: FileOutcome
    options: ProcessingOptions
    request_command: str | None
    request_json_output: bool
    request_diff: bool


@dataclass(slots=True)
class PipelineStateAccumulator:
    """Accumulates pipeline state from file processing outcomes.

    Manages all mutable state accumulated during pipeline execution,
    including counters, errors, reports, and accumulated results.
    This replaces scattered local variables with a typed, single-source-of-truth
    for pipeline state.

    Attributes
    ----------
    cache_hits : int
        Number of files processed from cache.
    cache_misses : int
        Number of files requiring processing.
    processed_count : int
        Number of files fully processed.
    skipped_count : int
        Number of files skipped.
    changed_count : int
        Number of files with changes.
    status_counts : Counter[ExitStatus]
        Count of each exit status encountered.
    errors : list[ErrorReport]
        Accumulated error reports.
    file_reports : list[FileReport]
        File-level reports (for JSON output).
    docstring_diffs : list[DocstringDriftEntry]
        Docstring drift entries for baseline comparisons.
    diff_previews : list[tuple[Path, str]]
        Diff preview paths and content.
    all_ir : list[IRDocstring]
        Accumulated IR docstring entries.
    """

    cache_hits: int = 0
    cache_misses: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    changed_count: int = 0
    status_counts: Counter[ExitStatus] = field(default_factory=Counter)
    errors: list[ErrorReport] = field(default_factory=list)
    file_reports: list[FileReport] = field(default_factory=list)
    docstring_diffs: list[DocstringDriftEntry] = field(default_factory=list)
    diff_previews: list[tuple[Path, str]] = field(default_factory=list)
    all_ir: list[IRDocstring] = field(default_factory=list)

    def process_file_outcome(self, ctx: FileProcessingContext) -> None:
        """Process a single file outcome and accumulate state.

        Updates all accumulators based on the file outcome, including
        counters, error tracking, report generation, and baseline
        comparisons.

        Parameters
        ----------
        ctx : FileProcessingContext
            Context containing file path, outcome, options, and flags.

        Notes
        -----
        Increments counters, builds error reports, and may generate
        baseline comparison diffs. All operations complete regardless
        of individual failures.
        """
        self._update_counters(ctx.outcome)
        self._track_diff_previews(ctx.file_path, ctx.outcome, ctx.request_command, ctx.request_diff)
        self._track_errors(ctx.file_path, ctx.outcome)
        self._accumulate_ir(ctx.outcome)
        if ctx.request_json_output:
            self._build_file_report(ctx.file_path, ctx.outcome, ctx.options)
        if ctx.options.baseline:
            self._handle_baseline_comparison(ctx.file_path, ctx.outcome, ctx.options)

    def _update_counters(self, outcome: FileOutcome) -> None:
        """Update counter fields from outcome.

        Parameters
        ----------
        outcome : FileOutcome
            The file processing outcome.
        """
        self.status_counts[outcome.status] += 1
        if outcome.skipped:
            self.skipped_count += 1
        else:
            self.processed_count += 1
        if outcome.changed:
            self.changed_count += 1
        if outcome.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def _track_diff_previews(
        self,
        file_path: Path,
        outcome: FileOutcome,
        request_command: str | None,
        request_diff: bool,
    ) -> None:
        """Track diff previews for check mode.

        Parameters
        ----------
        file_path : Path
            The file path.
        outcome : FileOutcome
            The file outcome.
        request_command : str | None
            The command (check, update, etc.).
        request_diff : bool
            Whether diff output is requested.
        """
        if (
            request_command == "check"
            and outcome.changed
            and request_diff
            and outcome.preview is not None
        ):
            self.diff_previews.append((file_path, outcome.preview))

    def _track_errors(self, file_path: Path, outcome: FileOutcome) -> None:
        """Track error reports for failures.

        Parameters
        ----------
        file_path : Path
            The file path.
        outcome : FileOutcome
            The file outcome.
        """
        if outcome.status is not ExitStatus.SUCCESS:
            rel = str(file_path.relative_to(REPO_ROOT))
            self.errors.append(
                {
                    "file": rel,
                    "status": self._status_from_exit(outcome.status),
                    "message": outcome.message or "",
                }
            )

    def _accumulate_ir(self, outcome: FileOutcome) -> None:
        """Accumulate IR entries.

        Parameters
        ----------
        outcome : FileOutcome
            The file outcome.
        """
        self.all_ir.extend(outcome.ir)

    def _build_file_report(
        self,
        file_path: Path,
        outcome: FileOutcome,
        options: ProcessingOptions,
    ) -> None:
        """Build JSON file report.

        Parameters
        ----------
        file_path : Path
            The file path.
        outcome : FileOutcome
            The file outcome.
        options : ProcessingOptions
            Processing options.
        """
        rel_path = str(file_path.relative_to(REPO_ROOT))
        file_report: FileReport = {
            "path": rel_path,
            "status": self._status_from_exit(outcome.status),
            "changed": outcome.changed,
            "skipped": outcome.skipped,
            "cacheHit": outcome.cache_hit,
        }
        if outcome.message:
            file_report["message"] = outcome.message
        if outcome.preview:
            file_report["preview"] = outcome.preview
        if options.baseline:
            file_report["baseline"] = options.baseline
        self.file_reports.append(file_report)

    def _handle_baseline_comparison(
        self,
        file_path: Path,
        outcome: FileOutcome,
        options: ProcessingOptions,
    ) -> None:
        """Handle baseline comparison and diff generation.

        Parameters
        ----------
        file_path : Path
            The file path.
        outcome : FileOutcome
            The file outcome.
        options : ProcessingOptions
            Processing options with baseline.
        """
        if not options.baseline:
            return

        baseline_text = read_baseline_version(options.baseline, file_path)
        if baseline_text is None:
            return

        current_text = outcome.preview or ""
        if not current_text and not outcome.preview:
            try:
                current_text = file_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                current_text = ""
        if baseline_text != current_text:
            self.docstring_diffs.append(
                DocstringDriftEntry(
                    path=str(file_path.relative_to(REPO_ROOT)),
                    before=baseline_text,
                    after=current_text,
                )
            )

    @staticmethod
    def _status_from_exit(exit_status: ExitStatus) -> RunStatus:
        """Convert ExitStatus to RunStatus.

        Parameters
        ----------
        exit_status : ExitStatus
            The exit status to convert.

        Returns
        -------
        RunStatus
            RunStatus enum value corresponding to the exit status.
        """
        status_map: dict[ExitStatus, RunStatus] = {
            ExitStatus.SUCCESS: RunStatus.SUCCESS,
            ExitStatus.VIOLATION: RunStatus.VIOLATION,
            ExitStatus.CONFIG: RunStatus.CONFIG,
            ExitStatus.ERROR: RunStatus.ERROR,
        }
        return status_map.get(exit_status, RunStatus.ERROR)
