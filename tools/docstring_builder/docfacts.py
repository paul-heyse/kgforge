"""Utilities to emit machine-readable DocFacts alongside docstrings."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .semantics import SemanticResult


@dataclass(slots=True)
class DocFact:
    """Serializable representation of a symbol's documentation facts."""

    qname: str
    module: str
    kind: str
    parameters: list[dict[str, str | bool | None]]
    returns: list[dict[str, str | None]]
    raises: list[dict[str, str]]
    notes: list[str]


def build_docfacts(entries: Iterable[SemanticResult]) -> list[DocFact]:
    """Convert semantic results to DocFacts dataclasses."""

    payload: list[DocFact] = []
    for entry in entries:
        schema = entry.schema
        parameters = [
            {
                "name": param.name,
                "annotation": param.annotation,
                "optional": param.optional,
                "default": param.default,
            }
            for param in schema.parameters
        ]
        returns = [
            {
                "kind": value.kind,
                "annotation": value.annotation,
            }
            for value in schema.returns
        ]
        raises = [
            {
                "exception": value.exception,
                "description": value.description,
            }
            for value in schema.raises
        ]
        payload.append(
            DocFact(
                qname=entry.symbol.qname,
                module=entry.symbol.module,
                kind=entry.symbol.kind,
                parameters=parameters,
                returns=returns,
                raises=raises,
                notes=list(schema.notes),
            )
        )
    return payload


def write_docfacts(path: Path, facts: Iterable[DocFact]) -> None:
    """Persist DocFacts to the provided path as JSON."""

    data = [asdict(fact) for fact in facts]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["DocFact", "build_docfacts", "write_docfacts"]
