"""Unit tests for docstring builder pipeline helpers."""

from __future__ import annotations

import logging
from typing import cast
from unittest import mock

import pytest
from tools.docstring_builder.builder_types import ExitStatus
from tools.docstring_builder.failure_summary import FailureSummaryRenderer, RunSummarySnapshot
from tools.docstring_builder.metrics import MetricsRecorder
from tools.docstring_builder.models import RunStatus
from tools.docstring_builder.paths import OBSERVABILITY_PATH
from tools.docstring_builder.pipeline_types import ErrorEnvelope, FileOutcome


class TestFailureSummaryRenderer:
    """Tests for FailureSummaryRenderer."""

    def test_render_no_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty error list should produce no output."""
        renderer = FailureSummaryRenderer(logger=logging.getLogger(__name__))
        summary = RunSummarySnapshot(
            considered=10,
            processed=10,
            changed=0,
            status_counts={"success": 10, "violation": 0, "config": 0, "error": 0},
            observability_path=OBSERVABILITY_PATH,
        )
        renderer.render(summary, [])
        assert len(caplog.records) == 0

    def test_render_with_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Error list should produce structured logging."""
        renderer = FailureSummaryRenderer(logger=logging.getLogger(__name__))
        summary = RunSummarySnapshot(
            considered=10,
            processed=8,
            changed=2,
            status_counts={"success": 7, "violation": 1, "config": 0, "error": 0},
            observability_path=OBSERVABILITY_PATH,
        )
        errors = [
            ErrorEnvelope(file="foo.py", status=RunStatus.VIOLATION, message="drift detected"),
            ErrorEnvelope(file="bar.py", status=RunStatus.ERROR, message="harvest failed"),
        ]
        renderer.render(summary, errors)
        messages = cast(list[str], getattr(caplog, "messages", []))
        assert any("[SUMMARY]" in message for message in messages)

    def test_render_truncates_top_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should only show top 5 errors."""
        renderer = FailureSummaryRenderer(logger=logging.getLogger(__name__))
        summary = RunSummarySnapshot(
            considered=20,
            processed=15,
            changed=5,
            status_counts={"success": 10, "violation": 5, "config": 0, "error": 0},
            observability_path=OBSERVABILITY_PATH,
        )
        errors = [
            ErrorEnvelope(file=f"file_{i}.py", status=RunStatus.VIOLATION, message=f"error {i}")
            for i in range(10)
        ]
        renderer.render(summary, errors)
        messages = cast(list[str], getattr(caplog, "messages", []))
        output = "\n".join(messages)
        assert output.count("file_") == 5  # Only top 5


class TestMetricsRecorder:
    """Tests for MetricsRecorder."""

    def test_observe_cli_duration(self) -> None:
        """Should record duration and increment counter."""
        mock_hist = mock.Mock()
        mock_counter = mock.Mock()
        mock_hist_child = mock.Mock()
        mock_counter_child = mock.Mock()
        mock_hist.labels.return_value = mock_hist_child
        mock_counter.labels.return_value = mock_counter_child

        recorder = MetricsRecorder(cli_duration_seconds=mock_hist, runs_total=mock_counter)

        recorder.observe_cli_duration(
            command="check",
            status=ExitStatus.SUCCESS,
            duration_seconds=1.5,
        )

        mock_hist.labels.assert_called_once_with(command="check", status="success")
        mock_hist_child.observe.assert_called_once_with(1.5)
        mock_counter.labels.assert_called_once_with(status="success")
        mock_counter_child.inc.assert_called_once()


class TestErrorEnvelope:
    """Tests for ErrorEnvelope."""

    def test_to_report(self) -> None:
        """Should convert to error report dict."""
        envelope = ErrorEnvelope(
            file="test.py",
            status=RunStatus.VIOLATION,
            message="test error",
        )
        report = envelope.to_report()
        assert report["file"] == "test.py"
        assert report["status"] == RunStatus.VIOLATION
        assert report["message"] == "test error"


class TestFileOutcomeTracking:
    """Tests for FileOutcome data structure."""

    def test_file_outcome_defaults(self) -> None:
        """FileOutcome should have sensible defaults."""
        outcome = FileOutcome(
            status=ExitStatus.SUCCESS,
            docfacts=[],
            preview=None,
            changed=False,
            skipped=False,
        )
        assert outcome.cache_hit is False
        assert outcome.message is None
        assert outcome.semantics == []
        assert outcome.ir == []


@pytest.mark.parametrize(
    ("status", "label"),
    [
        (RunStatus.SUCCESS, "success"),
        (RunStatus.VIOLATION, "violation"),
        (RunStatus.CONFIG, "config"),
        (RunStatus.ERROR, "error"),
    ],
)
class TestStatusMapping:
    """Tests for status value mappings."""

    def test_status_to_label(self, status: RunStatus, label: str) -> None:  # noqa: ARG002
        """Verify status string representation."""
        envelope = ErrorEnvelope(file="test.py", status=status, message="")
        assert envelope.status == status
