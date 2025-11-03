"""Unit tests targeting the orchestrated PipelineRunner."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from tools.docstring_builder.builder_types import (
    STATUS_LABELS,
    DocstringBuildRequest,
    ExitStatus,
    status_from_exit,
)
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.diff_manager import DiffManager
from tools.docstring_builder.docfacts import DocFact
from tools.docstring_builder.docfacts_coordinator import DocfactsCoordinator
from tools.docstring_builder.file_processor import FileProcessor
from tools.docstring_builder.metrics import MetricsRecorder
from tools.docstring_builder.models import ErrorReport, ProblemDetails
from tools.docstring_builder.orchestrator import load_builder_config
from tools.docstring_builder.pipeline import PipelineConfig, PipelineRunner
from tools.docstring_builder.pipeline_types import DocfactsResult, FileOutcome, ProcessingOptions
from tools.docstring_builder.policy import PolicyEngine, PolicyReport
from tools.docstring_builder.semantics import SemanticResult


def _noop_record_docfacts(_facts: Iterable[DocFact], _path: Path) -> None:
    return None


def _empty_docfacts() -> list[DocFact]:
    return []


def _problem_details_stub(
    status: ExitStatus,
    request: DocstringBuildRequest,
    detail: str,
    *,
    instance: str | None = None,
    errors: Sequence[ErrorReport] | None = None,
) -> ProblemDetails:
    return {
        "type": "https://kgfoundry.dev/problems/test",
        "title": "stub",
        "status": int(status),
        "detail": detail,
        "instance": instance or "",
        "extensions": {
            "command": request.command,
            "subcommand": request.subcommand,
            "errorCount": len(errors or []),
        },
    }


class _StubPolicyEngine:
    def record(self, _semantics: Sequence[SemanticResult]) -> None:  # pragma: no cover - no-op
        return

    def finalize(self) -> PolicyReport:
        return PolicyReport(coverage=1.0, threshold=1.0, violations=[])


type DocfactsStatus = Literal["success", "violation", "config", "error"]


class _StubDocfactsCoordinator:
    def __init__(self, status: DocfactsStatus, message: str | None = None) -> None:
        self._status: DocfactsStatus = status
        self._message = message

    def reconcile(self, _docfacts: Iterable[DocFact]) -> DocfactsResult:
        return DocfactsResult(
            status=self._status,
            message=self._message,
            diff_path=None,
        )


class _StubDiffManager:
    def __init__(self) -> None:
        self.baseline_calls: list[tuple[str, str | None]] = []
        self.recorded_docfacts: list[str | None] = []

    def record_docstring_baseline(self, file_path: Path, preview: str | None) -> None:
        """Record the baseline docstring for diff inspection."""
        self.baseline_calls.append((str(file_path), preview))

    def finalize_docstring_drift(self) -> None:  # pragma: no cover - no-op
        return

    def record_docfacts_baseline_diff(self, payload_text: str | None) -> None:  # pragma: no cover
        self.recorded_docfacts.append(payload_text)

    def collect_diff_links(self) -> dict[str, str]:  # pragma: no cover - deterministic
        return {}


@dataclass(slots=True)
class _MetricsHistogram:
    labels_called: list[dict[str, object]] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)

    def labels(self, **labels: object) -> _MetricsHistogram:
        self.labels_called.append(labels)
        return self

    def observe(self, value: float) -> None:
        self.observed.append(value)


@dataclass(slots=True)
class _MetricsCounter:
    labels_called: list[dict[str, object]] = field(default_factory=list)
    increments: list[float] = field(default_factory=list)

    def labels(self, **labels: object) -> _MetricsCounter:
        self.labels_called.append(labels)
        return self

    def inc(self, value: float = 1.0) -> None:
        self.increments.append(value)


def _build_metrics_recorder() -> MetricsRecorder:
    histogram = _MetricsHistogram()
    counter = _MetricsCounter()
    return MetricsRecorder(cli_duration_seconds=histogram, runs_total=counter)


def _build_pipeline_config(
    request: DocstringBuildRequest,
    file_outcomes: Sequence[FileOutcome],
    *,
    docfacts_status: DocfactsStatus = "success",
    docfacts_message: str | None = None,
) -> PipelineConfig:
    def _write_cache() -> None:
        return None

    cache = cast(BuilderCache, SimpleNamespace(write=_write_cache))

    def _process_file(_path: Path) -> FileOutcome:
        return file_outcomes[0]

    file_processor = cast(FileProcessor, SimpleNamespace(process=_process_file))

    def coordinator_factory(check_mode: bool) -> DocfactsCoordinator:
        status = docfacts_status if check_mode else "success"
        return cast(
            DocfactsCoordinator,
            _StubDocfactsCoordinator(status=status, message=docfacts_message),
        )

    config, _ = load_builder_config(None)

    return PipelineConfig(
        request=request,
        config=config,
        selection=None,
        options=ProcessingOptions(
            command=request.command,
            force=False,
            ignore_missing=False,
            missing_patterns=(),
            skip_docfacts=request.command not in {"update", "check"},
            baseline=None,
        ),
        cache=cache,
        file_processor=file_processor,
        record_docfacts=_noop_record_docfacts,
        filter_docfacts=_empty_docfacts,
        docfacts_coordinator_factory=coordinator_factory,
        plugin_manager=None,
        policy_engine=cast(PolicyEngine, _StubPolicyEngine()),
        metrics=_build_metrics_recorder(),
        diff_manager=cast(DiffManager, _StubDiffManager()),
        logger=logging.getLogger(__name__),
        status_from_exit=status_from_exit,
        status_labels=STATUS_LABELS,
        build_problem_details=_problem_details_stub,
        success_status=ExitStatus.SUCCESS,
        violation_status=ExitStatus.VIOLATION,
        config_status=ExitStatus.CONFIG,
        error_status=ExitStatus.ERROR,
    )


def test_pipeline_runner_collects_diff_previews(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = DocstringBuildRequest(command="check", subcommand="check", diff=True)
    outcome = FileOutcome(
        status=ExitStatus.VIOLATION,
        docfacts=_empty_docfacts(),
        preview="diff output",
        changed=True,
        skipped=False,
    )
    config = _build_pipeline_config(request, [outcome])

    dummy_path = tmp_path / "module.py"
    dummy_path.write_text("# placeholder")

    monkeypatch.setattr(
        "tools.docstring_builder.pipeline.CACHE_PATH",
        tmp_path / "cache",
    )
    monkeypatch.setattr(
        "tools.docstring_builder.pipeline.OBSERVABILITY_PATH",
        tmp_path / "observability.json",
    )

    runner = PipelineRunner(config)
    result = runner.run([dummy_path])

    assert result.diff_previews == [(dummy_path, "diff output")]


def test_pipeline_runner_records_docfacts_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = DocstringBuildRequest(command="check", subcommand="check")
    outcome = FileOutcome(
        status=ExitStatus.SUCCESS,
        docfacts=_empty_docfacts(),
        preview=None,
        changed=False,
        skipped=False,
    )
    config = _build_pipeline_config(
        request,
        [outcome],
        docfacts_status="violation",
        docfacts_message="docfacts drift",
    )

    dummy_path = tmp_path / "module.py"
    dummy_path.write_text("# placeholder")

    monkeypatch.setattr(
        "tools.docstring_builder.pipeline.CACHE_PATH",
        tmp_path / "cache",
    )
    monkeypatch.setattr(
        "tools.docstring_builder.pipeline.OBSERVABILITY_PATH",
        tmp_path / "observability.json",
    )

    runner = PipelineRunner(config)
    result = runner.run([dummy_path])

    assert result.exit_status is ExitStatus.VIOLATION
    assert any(error["file"] == "<docfacts>" for error in result.errors)
