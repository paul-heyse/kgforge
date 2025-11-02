"""Failure summary rendering for docstring builder runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.docstring_builder.pipeline_types import ErrorEnvelope


@dataclass(slots=True)
class RunSummarySnapshot:
    """Condensed view of pipeline run statistics."""

    considered: int
    processed: int
    changed: int
    status_counts: Mapping[str, int]
    observability_path: Path


@dataclass(slots=True)
class FailureSummaryRenderer:
    """Emit structured failure summaries for CLI output."""

    logger: logging.LoggerAdapter | logging.Logger

    def render(self, summary: RunSummarySnapshot, errors: Sequence[ErrorEnvelope]) -> None:
        """Render a structured summary when the run does not succeed."""
        if not errors:
            return

        lines = ["[SUMMARY] Docstring builder reported issues."]
        lines.append(f"  Considered files: {summary.considered}")
        lines.append(f"  Processed files: {summary.processed}")
        lines.append(f"  Changed files: {summary.changed}")
        lines.append(f"  Status counts: {dict(summary.status_counts)}")
        lines.append(f"  Observability log: {summary.observability_path}")
        lines.append("  Top errors:")
        for entry in errors[:5]:
            status_label = getattr(entry.status, "value", str(entry.status))
            lines.append(
                f"    - {entry.file}: {status_label} ({entry.message or 'no additional details'})"
            )
        for line in lines:
            self.logger.error(line)
