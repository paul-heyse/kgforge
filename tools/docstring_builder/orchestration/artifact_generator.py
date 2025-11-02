"""Artifact generation for pipeline output.

This module manages generation and persistence of pipeline artifacts,
including docstring drift diffs, schema diffs, manifest files, and
diff link tracking. It separates artifact I/O concerns from the main
orchestration logic.

Ownership: docstring-builder team
Version: 1.0
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools.docstring_builder.ir import write_schema
from tools.docstring_builder.models import CacheSummary, InputHash
from tools.docstring_builder.paths import (
    DOCFACTS_DIFF_PATH,
    DOCSTRINGS_DIFF_PATH,
    MANIFEST_PATH,
    REPO_ROOT,
    SCHEMA_DIFF_PATH,
)
from tools.drift_preview import DocstringDriftEntry, write_docstring_drift, write_html_diff

if TYPE_CHECKING:
    from tools.docstring_builder.ir import IRDocstring
    from tools.docstring_builder.orchestrator import DocstringBuildRequest
    from tools.docstring_builder.policy import PolicyReport

# Constant for IR version in manifests
IR_VERSION = "1.0"


@dataclass(slots=True)
class ArtifactGeneratorContext:
    """Context for artifact generation.

    Parameters
    ----------
    docstring_diffs : list[DocstringDriftEntry]
        Docstring drift entries to write.
    all_ir : list[IRDocstring]
        Accumulated IR docstring entries.
    cache_payload : CacheSummary
        Cache metadata.
    input_hashes : dict[str, InputHash]
        Input file hashes.
    dependency_map : dict[str, list[str]]
        File dependency map.
    policy_report : PolicyReport
        Policy check results.
    request : DocstringBuildRequest
        Original request.
    selection : object | None
        Config selection.
    docfacts_payload_text : str | None
        DocFacts content text.
    baseline : str | None
        Baseline version identifier.
    files_count : int
        Total files considered.
    processed_count : int
        Number of files processed.
    skipped_count : int
        Number of files skipped.
    changed_count : int
        Number of files with changes.
    """

    docstring_diffs: list[DocstringDriftEntry]
    all_ir: list[IRDocstring]
    cache_payload: CacheSummary
    input_hashes: dict[str, InputHash]
    dependency_map: dict[str, list[str]]
    policy_report: PolicyReport
    request: DocstringBuildRequest
    selection: object | None
    docfacts_payload_text: str | None
    baseline: str | None
    files_count: int
    processed_count: int
    skipped_count: int
    changed_count: int


@dataclass(slots=True)
class ArtifactGenerator:
    """Generates pipeline artifacts including diffs, manifest, and links.

    Encapsulates all artifact generation and persistence logic, including:
    - Docstring and DocFacts diff file generation
    - Schema drift detection and diff generation
    - Manifest file creation with complete metadata
    - Diff link tracking for CLI output

    This replaces scattered artifact I/O throughout the orchestrator with
    a focused, testable class.
    """

    def generate_and_persist(self, ctx: ArtifactGeneratorContext) -> dict[str, str]:
        """Generate all artifacts and persist to disk.

        Parameters
        ----------
        ctx : ArtifactGeneratorContext
            Complete context for artifact generation.

        Returns
        -------
        dict[str, str]
            Mapping of diff labels to relative paths that exist.
        """
        # Generate docstring diffs
        write_docstring_drift(ctx.docstring_diffs, DOCSTRINGS_DIFF_PATH)

        # Generate schema diffs
        self._write_schema_diffs()

        # Build and persist manifest
        diff_links = self._collect_diff_links()
        self._write_manifest(ctx, diff_links)

        return diff_links

    @staticmethod
    def _write_schema_diffs() -> None:
        """Write schema drift diffs if schema has changed."""
        schema_path = REPO_ROOT / "docs" / "_build" / "schema_docstrings.json"
        previous_schema = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""
        write_schema(schema_path)
        current_schema = schema_path.read_text(encoding="utf-8")

        if previous_schema and previous_schema != current_schema:
            write_html_diff(previous_schema, current_schema, SCHEMA_DIFF_PATH, "Schema drift")
        else:
            SCHEMA_DIFF_PATH.unlink(missing_ok=True)

    @staticmethod
    def _collect_diff_links() -> dict[str, str]:
        """Collect all existing diff artifact links.

        Returns
        -------
        dict[str, str]
            Mapping of diff labels to relative paths.
        """
        diff_links: dict[str, str] = {}
        for label, path in (
            ("docfacts", DOCFACTS_DIFF_PATH),
            ("docstrings", DOCSTRINGS_DIFF_PATH),
            ("schema", SCHEMA_DIFF_PATH),
        ):
            if path.exists():
                diff_links[label] = str(path.relative_to(REPO_ROOT))
        return diff_links

    @staticmethod
    def _write_manifest(ctx: ArtifactGeneratorContext, diff_links: dict[str, str]) -> None:
        """Write manifest file with complete metadata.

        Parameters
        ----------
        ctx : ArtifactGeneratorContext
            Artifact generation context.
        diff_links : dict[str, str]
            Diff artifact links.
        """
        invoked = str(
            ctx.request.invoked_subcommand or ctx.request.subcommand or ctx.request.command or ""
        )

        manifest_payload: dict[str, object] = {
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "command": ctx.request.command,
            "subcommand": invoked,
            "options": {
                "module": ctx.request.module,
                "since": ctx.request.since,
                "force": ctx.request.force,
                "changed_only": ctx.request.changed_only,
                "skip_docfacts": ctx.request.skip_docfacts,
            },
            "counts": {
                "considered": ctx.files_count,
                "processed": ctx.processed_count,
                "skipped": ctx.skipped_count,
                "changed": ctx.changed_count,
            },
            "cache": ctx.cache_payload,
            "hashes": ctx.input_hashes,
            "dependencies": ctx.dependency_map,
        }

        if diff_links:
            manifest_payload["drift_previews"] = diff_links

        schema_path = REPO_ROOT / "docs" / "_build" / "schema_docstrings.json"
        manifest_payload["ir"] = {
            "version": IR_VERSION,
            "schema": str(schema_path.relative_to(REPO_ROOT)),
            "count": len(ctx.all_ir),
            "symbols": [entry.symbol_id for entry in ctx.all_ir],
        }

        manifest_payload["policy"] = {
            "coverage": ctx.policy_report.coverage,
            "threshold": ctx.policy_report.threshold,
            "violations": [
                {
                    "rule": violation.rule,
                    "symbol": violation.symbol,
                    "action": violation.action,
                    "message": violation.message,
                }
                for violation in ctx.policy_report.violations
            ],
        }

        if ctx.selection is not None:
            # Use getattr to safely access path and source attributes
            manifest_payload["config_source"] = {
                "path": str(getattr(ctx.selection, "path", "")),
                "source": getattr(ctx.selection, "source", ""),
            }

        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
