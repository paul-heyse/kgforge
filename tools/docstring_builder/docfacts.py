"""Utilities to emit machine-readable DocFacts alongside docstrings."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, SupportsInt, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from tools.docstring_builder.semantics import SemanticResult

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DOCFACTS_VERSION: Final = "2.0"
DOCFACTS_SCHEMA_PATH: Final = REPO_ROOT / "docs" / "_build" / "schema_docfacts.json"


@dataclass(slots=True)
class DocFact:
    """Serializable representation of a symbol's documentation facts."""

    qname: str
    module: str
    kind: str
    filepath: str
    lineno: int
    end_lineno: int | None
    decorators: list[str]
    is_async: bool
    is_generator: bool
    owned: bool
    parameters: list[dict[str, str | bool | None]]
    returns: list[dict[str, str | None]]
    raises: list[dict[str, str | None]]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "qname": self.qname,
            "module": self.module,
            "kind": self.kind,
            "filepath": self.filepath,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "decorators": list(self.decorators),
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "owned": self.owned,
            "parameters": [dict(parameter) for parameter in self.parameters],
            "returns": [dict(value) for value in self.returns],
            "raises": [dict(value) for value in self.raises],
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> DocFact | None:
        """Instantiate a :class:`DocFact` from a mapping if possible."""
        try:
            qname = str(data["qname"])
            module = str(data.get("module", ""))
            kind = str(data.get("kind", "function"))
            filepath = str(data.get("filepath", ""))
        except (KeyError, TypeError, ValueError):
            return None
        lineno_value = data.get("lineno", 1)
        lineno = _to_int_or_none(lineno_value)
        if lineno is None:
            lineno = 1
        end_lineno = _to_int_or_none(data.get("end_lineno"))
        decorators = [str(item) for item in _iterable_values(data.get("decorators"))]
        returns_payload: list[dict[str, str | None]] = [
            {
                "kind": str(value.get("kind", "returns")),
                "annotation": cast(str | None, value.get("annotation")),
                "description": cast(str | None, value.get("description")),
            }
            for value in _mapping_items(data.get("returns"))
        ]
        raises_payload: list[dict[str, str | None]] = [
            {
                "exception": str(value.get("exception", "")),
                "description": cast(str | None, value.get("description")),
            }
            for value in _mapping_items(data.get("raises"))
        ]
        parameters_payload: list[dict[str, str | bool | None]] = [
            {
                "name": cast(str | None, value.get("name")),
                "display_name": cast(str | None, value.get("display_name")),
                "annotation": cast(str | None, value.get("annotation")),
                "optional": cast(bool | None, value.get("optional")),
                "default": cast(str | None, value.get("default")),
                "kind": cast(str | None, value.get("kind")),
            }
            for value in _mapping_items(data.get("parameters"))
        ]
        notes_payload = [str(item) for item in _iterable_values(data.get("notes"))]
        is_async = bool(data.get("is_async", False))
        is_generator = bool(data.get("is_generator", False))
        owned = bool(data.get("owned", False))
        return cls(
            qname=qname,
            module=module,
            kind=kind,
            filepath=filepath,
            lineno=lineno,
            end_lineno=end_lineno,
            decorators=decorators,
            is_async=is_async,
            is_generator=is_generator,
            owned=owned,
            parameters=parameters_payload,
            returns=returns_payload,
            raises=raises_payload,
            notes=notes_payload,
        )


@dataclass(slots=True)
class DocfactsProvenance:
    """Metadata describing how the DocFacts payload was generated."""

    builder_version: str
    config_hash: str
    commit_hash: str
    generated_at: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-serialisable provenance metadata."""
        return {
            "builderVersion": self.builder_version,
            "configHash": self.config_hash,
            "commitHash": self.commit_hash,
            "generatedAt": self.generated_at,
        }


@dataclass(slots=True)
class DocfactsDocument:
    """Structured representation of the DocFacts artifact."""

    docfacts_version: str
    provenance: DocfactsProvenance
    entries: list[DocFact]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON payload for the full DocFacts document."""
        return {
            "docfactsVersion": self.docfacts_version,
            "provenance": self.provenance.to_dict(),
            "entries": [fact.to_dict() for fact in self.entries],
        }


def _relative_filepath(path: Path) -> str:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:  # pragma: no cover - defensive guard
        relative = path
    return relative.as_posix()


def _to_int_or_none(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, SupportsInt):
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value)
        except ValueError:
            return None
    else:
        return None
    if number <= 0:
        return None
    return number


def _iterable_values(value: object | None) -> Iterable[object]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Iterable):
        return value
    return ()


def _mapping_items(value: object | None) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    mappings: list[Mapping[str, object]] = []
    for item in _iterable_values(value):
        if isinstance(item, Mapping):
            mappings.append(item)
    return mappings


def build_docfacts(entries: Iterable[SemanticResult]) -> list[DocFact]:
    """Convert semantic results to DocFacts dataclasses."""
    payload: list[DocFact] = []
    for entry in entries:
        schema = entry.schema
        parameters = [
            {
                "name": param.name,
                "display_name": param.display_name or param.name,
                "annotation": param.annotation,
                "optional": param.optional,
                "default": param.default,
                "kind": param.kind,
            }
            for param in schema.parameters
        ]
        returns: list[dict[str, str | None]] = [
            {
                "kind": value.kind,
                "annotation": value.annotation,
                "description": value.description,
            }
            for value in schema.returns
        ]
        raises: list[dict[str, str | None]] = [
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
                filepath=_relative_filepath(entry.symbol.filepath),
                lineno=entry.symbol.lineno,
                end_lineno=_to_int_or_none(entry.symbol.end_lineno),
                decorators=list(entry.symbol.decorators),
                is_async=entry.symbol.is_async,
                is_generator=entry.symbol.is_generator,
                owned=entry.symbol.owned,
                parameters=parameters,
                returns=returns,
                raises=raises,
                notes=list(schema.notes),
            )
        )
    return payload


def build_docfacts_document(
    entries: Iterable[DocFact], provenance: DocfactsProvenance, version: str | None = None
) -> DocfactsDocument:
    """Create a :class:`DocfactsDocument` from the provided entries."""
    ordered = sorted(entries, key=lambda fact: fact.qname)
    return DocfactsDocument(
        docfacts_version=version or DOCFACTS_VERSION,
        provenance=provenance,
        entries=ordered,
    )


@lru_cache(maxsize=1)
def _load_validator(schema_path: Path = DOCFACTS_SCHEMA_PATH) -> Draft202012Validator:
    schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema_data)


def validate_docfacts_payload(payload: Mapping[str, object]) -> None:
    """Validate a DocFacts payload against the JSON schema."""
    validator = _load_validator()
    validator.validate(payload)


def write_docfacts(path: Path, document: DocfactsDocument) -> None:
    """Persist DocFacts to the provided path as JSON."""
    payload = document.to_dict()
    validate_docfacts_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "DOCFACTS_SCHEMA_PATH",
    "DOCFACTS_VERSION",
    "DocFact",
    "DocfactsDocument",
    "DocfactsProvenance",
    "build_docfacts",
    "build_docfacts_document",
    "validate_docfacts_payload",
    "write_docfacts",
]
