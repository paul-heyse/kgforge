"""Plugin that rewrites docstring summaries using simple LLM heuristics."""

from __future__ import annotations

from dataclasses import replace

from tools._shared.logging import get_logger
from tools.docstring_builder.plugins.base import (
    PluginContext,
    PluginStage,
    TransformerPlugin,
)
from tools.docstring_builder.semantics import SemanticResult

LOGGER = get_logger(__name__)

MIN_WORD_LENGTH = 4


class LLMSummaryRewritePlugin(TransformerPlugin):
    """Rewrite summaries into imperative mood based on configured LLM mode."""

    name: str = "llm_summary_rewriter"
    stage: PluginStage = "transformer"

    def on_start(self, context: PluginContext) -> None:
        """Prepare plugin state for execution (no-op)."""
        del self, context

    def on_finish(self, context: PluginContext) -> None:
        """Clean up plugin state after execution (no-op)."""
        del self, context

    def apply(self, context: PluginContext, payload: SemanticResult) -> SemanticResult:
        """Rewrite the summary text for ``payload`` when configured to do so."""
        del self
        mode_attr: object = getattr(context.config, "llm_summary_mode", "off")
        mode = str(mode_attr).lower()
        if mode not in {"apply", "dry-run"}:
            return payload
        summary_attr = payload.schema.summary
        summary = str(summary_attr).strip()
        if not summary:
            return payload
        if not _needs_rewrite(summary):
            return payload
        candidate = _rewrite_summary(summary)
        if mode == "dry-run":
            LOGGER.info(
                "[LLM] summary dry-run",
                extra={
                    "operation": "llm_summary",
                    "symbol": payload.symbol.qname,
                    "mode": mode,
                    "original_summary": summary,
                    "candidate_summary": candidate,
                },
            )
            return payload
        LOGGER.info(
            "[LLM] summary rewritten",
            extra={
                "operation": "llm_summary",
                "symbol": payload.symbol.qname,
                "mode": mode,
                "original_summary": summary,
                "candidate_summary": candidate,
            },
        )
        updated_schema = replace(payload.schema, summary=candidate)
        return replace(payload, schema=updated_schema)


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
