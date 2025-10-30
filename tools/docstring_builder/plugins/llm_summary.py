from __future__ import annotations

import logging
from dataclasses import replace

from tools.docstring_builder.plugins import PluginContext, TransformerPlugin
from tools.docstring_builder.semantics import SemanticResult

LOGGER = logging.getLogger(__name__)


class LLMSummaryRewritePlugin(TransformerPlugin):
    """Rewrite summaries into imperative mood based on configured LLM mode."""

    name = "llm_summary_rewriter"

    def run(self, context: PluginContext, result: SemanticResult) -> SemanticResult:
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
                "[LLM] summary dry-run for %s: %s -> %s",
                result.symbol.qname,
                summary,
                candidate,
            )
            return result
        LOGGER.info(
            "[LLM] summary rewritten for %s: %s -> %s",
            result.symbol.qname,
            summary,
            candidate,
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
    if first.endswith("s") and len(first) > 3:
        return True
    return False


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
    elif lowered.endswith("s") and len(lowered) > 3:
        tokens[0] = first[:-1].capitalize()
    else:
        tokens[0] = first.capitalize()
    rewritten = " ".join(tokens)
    if not rewritten.endswith("."):
        rewritten += "."
    return rewritten


__all__ = ["LLMSummaryRewritePlugin"]
