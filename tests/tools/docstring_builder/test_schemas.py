from __future__ import annotations

import copy
import json
from collections.abc import Callable
from collections.abc import Callable as TypingCallable
from typing import cast

import pytest
from tools.docstring_builder.docfacts import (
    DOCFACTS_VERSION,
    DocFact,
    DocfactsProvenance,
    build_docfacts_document,
)
from tools.docstring_builder.models import (
    CliResult,
    DocfactsDocumentLike,
    DocfactsDocumentPayload,
    RunStatus,
    SchemaViolationError,
    StatusCounts,
    build_cli_result_skeleton,
    build_docfacts_document_payload,
    validate_cli_output,
    validate_docfacts_payload,
)

DocfactsMutator = Callable[[DocfactsDocumentPayload], None]
CliMutator = Callable[[CliResult], None]


def _remove_qname(payload: DocfactsDocumentPayload) -> None:
    payload["entries"][0].pop("qname")


def _downgrade_docfacts_version(payload: DocfactsDocumentPayload) -> None:
    payload["docfactsVersion"] = "0.1"


DOCFACTS_MUTATORS: tuple[DocfactsMutator, ...] = (
    _remove_qname,
    _downgrade_docfacts_version,
)


def _truncate_file_entry(payload: CliResult) -> None:
    payload["files"][0] = {"path": "src/pkg/module.py"}


def _collapse_status_counts(payload: CliResult) -> None:
    payload["summary"]["status_counts"] = cast(StatusCounts, {"success": 1})


CLI_MUTATORS: tuple[CliMutator, ...] = (
    _truncate_file_entry,
    _collapse_status_counts,
)


ParametrizeDocfacts = TypingCallable[
    [TypingCallable[[DocfactsMutator], None]], TypingCallable[[DocfactsMutator], None]
]
parametrize_docfacts = cast(
    ParametrizeDocfacts, pytest.mark.parametrize("mutator", DOCFACTS_MUTATORS)
)

ParametrizeCli = TypingCallable[
    [TypingCallable[[CliMutator], None]], TypingCallable[[CliMutator], None]
]
parametrize_cli = cast(ParametrizeCli, pytest.mark.parametrize("mutator", CLI_MUTATORS))


def _example_docfact() -> DocFact:
    return DocFact(
        qname="pkg.module.func",
        module="pkg.module",
        kind="function",
        filepath="src/pkg/module.py",
        lineno=10,
        end_lineno=20,
        decorators=[],
        is_async=False,
        is_generator=False,
        owned=True,
        parameters=[
            {
                "name": "value",
                "display_name": "value",
                "annotation": "int",
                "optional": False,
                "default": None,
                "kind": "positional_or_keyword",
            }
        ],
        returns=[{"kind": "returns", "annotation": "int", "description": "Number"}],
        raises=[{"exception": "ValueError", "description": "Bad value"}],
        notes=["Example"],
    )


def _example_cli_payload() -> CliResult:
    payload = build_cli_result_skeleton(RunStatus.SUCCESS)
    payload["command"] = "update"
    payload["subcommand"] = "generate"
    payload["durationSeconds"] = 0.5
    payload["files"] = [
        {
            "path": "src/pkg/module.py",
            "status": RunStatus.SUCCESS,
            "changed": False,
            "skipped": False,
            "cacheHit": False,
        }
    ]
    payload["errors"] = []
    summary = payload["summary"]
    summary["considered"] = 1
    summary["processed"] = 1
    summary["skipped"] = 0
    summary["changed"] = 0
    summary["status_counts"] = {
        "success": 1,
        "violation": 0,
        "config": 0,
        "error": 0,
    }
    summary["docfacts_checked"] = True
    summary["cache_hits"] = 0
    summary["cache_misses"] = 0
    summary["duration_seconds"] = 0.5
    summary["subcommand"] = "generate"
    payload["policy"] = {"coverage": 1.0, "threshold": 0.8, "violations": []}
    payload["cache"] = {
        "path": ".cache/docstring_builder.json",
        "exists": False,
        "hits": 0,
        "misses": 0,
        "mtime": None,
    }
    payload["inputs"] = {
        "src/pkg/module.py": {
            "hash": "deadbeef",
            "mtime": "2024-01-01T00:00:00Z",
        }
    }
    payload["plugins"] = {
        "enabled": [],
        "available": [],
        "disabled": [],
        "skipped": [],
    }
    payload["docfacts"] = {
        "path": "docs/_build/docfacts.json",
        "version": DOCFACTS_VERSION,
        "validated": True,
    }
    return payload


def test_validate_docfacts_payload_round_trip() -> None:
    provenance = DocfactsProvenance(
        builder_version="1.0.0",
        config_hash="0" * 64,
        commit_hash="abcdef123456",
        generated_at="2024-01-01T00:00:00Z",
    )
    document = build_docfacts_document([_example_docfact()], provenance, DOCFACTS_VERSION)
    payload = build_docfacts_document_payload(cast(DocfactsDocumentLike, document))
    validate_docfacts_payload(payload)


@parametrize_docfacts
def test_validate_docfacts_payload_failure(
    mutator: DocfactsMutator,
) -> None:
    provenance = DocfactsProvenance(
        builder_version="1.0.0",
        config_hash="0" * 64,
        commit_hash="abcdef123456",
        generated_at="2024-01-01T00:00:00Z",
    )
    document = build_docfacts_document([_example_docfact()], provenance, DOCFACTS_VERSION)
    payload = build_docfacts_document_payload(cast(DocfactsDocumentLike, document))
    mutated = cast(DocfactsDocumentPayload, json.loads(json.dumps(payload)))
    mutator(mutated)
    with pytest.raises(SchemaViolationError) as excinfo:
        validate_docfacts_payload(mutated)
    problem = excinfo.value.problem
    assert problem is not None
    assert problem["status"] == 422


def test_validate_cli_output_round_trip() -> None:
    payload = _example_cli_payload()
    validate_cli_output(payload)


@parametrize_cli
def test_validate_cli_output_failure(mutator: CliMutator) -> None:
    payload = _example_cli_payload()
    mutated = copy.deepcopy(payload)
    mutator(mutated)
    with pytest.raises(SchemaViolationError) as excinfo:
        validate_cli_output(mutated)
    problem = excinfo.value.problem
    assert problem is not None
    assert problem["status"] == 422
