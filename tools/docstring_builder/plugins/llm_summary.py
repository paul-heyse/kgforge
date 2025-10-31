"""Plugin that rewrites docstring summaries using simple LLM heuristics."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import ClassVar

from tools.docstring_builder.plugins.base import PluginContext, PluginStage
from tools.docstring_builder.semantics import SemanticResult

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

MIN_WORD_LENGTH = 4


class LLMSummaryRewritePlugin:
    """Rewrite summaries into imperative mood based on configured LLM mode."""

    name: ClassVar[str] = "llm_summary_rewriter"
    stage: ClassVar[PluginStage] = "transformer"

    def on_start(self, context: PluginContext) -> None:
        """Prepare plugin state for execution (no-op)."""
        del context

    def on_finish(self, context: PluginContext) -> None:
        """Clean up plugin state after execution (no-op)."""
        del context

    def apply(self, context: PluginContext, result: SemanticResult) -> SemanticResult:
        """Rewrite the summary text for ``result`` when configured to do so."""
        mode = getattr(context.config, "llm_summary_mode", "off").lower()
        if mode not in {"apply", "dry-run"}:
            return result
        summary = result.schema.summary.strip()
        if not summary:
            return result
        if not _needs_rewrite(summary):
            return result
        candidate = _rewrite_summary(summary)
        if mode == "dry-run":
            LOGGER.info(
                "[LLM] summary dry-run",
                extra={
                    "operation": "llm_summary",
                    "symbol": result.symbol.qname,
                    "mode": mode,
                    "original_summary": summary,
                    "candidate_summary": candidate,
                },
            )
            return result
        LOGGER.info(
            "[LLM] summary rewritten",
            extra={
                "operation": "llm_summary",
                "symbol": result.symbol.qname,
                "mode": mode,
                "original_summary": summary,
                "candidate_summary": candidate,
            },
        )
        updated_schema = replace(result.schema, summary=candidate)
        return replace(result, schema=updated_schema)


def _needs_rewrite(summary: str) -> bool:
    tokens = summary.strip().split()
    if not tokens:
        return False
    first = tokens[0].strip().lower()
    if first in {"this", "the"}:
        return True
    return bool(first.endswith("s") and len(first) >= MIN_WORD_LENGTH)


def _rewrite_summary(summary: str) -> str:
    tokens = summary.strip().split()
    if not tokens:
        return summary.strip()
    first = tokens[0]
    lowered = first.lower()
    replacements = {
        "returns": "Return",
        "handles": "Handle",
        "provides": "Provide",
        "creates": "Create",
        "initialises": "Initialise",
        "initializes": "Initialize",
    }
    if lowered in replacements:
        tokens[0] = replacements[lowered]
    elif lowered.endswith("s") and len(lowered) >= MIN_WORD_LENGTH:
        tokens[0] = first[:-1].capitalize()
    else:
        tokens[0] = first.capitalize()
    rewritten = " ".join(tokens)
    if not rewritten.endswith("."):
        rewritten += "."
    return rewritten


__all__ = ["LLMSummaryRewritePlugin"]
