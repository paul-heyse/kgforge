"""Build schema-validated symbol index artifacts for the documentation pipeline."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from docs._scripts import shared
from docs._scripts.validation import validate_against_schema
from tools import (
    ProblemDetailsParams,
    StructuredLoggerAdapter,
    build_problem_details,
    get_logger,
    observe_tool_run,
    render_problem,
)
from tools._shared.proc import ToolExecutionError

from kgfoundry_common.errors import DeserializationError, SchemaValidationError, SerializationError
from kgfoundry_common.optional_deps import OptionalDependencyError, safe_import_griffe


def _resolve_alias_resolution_error() -> type[Exception]:
    try:
        griffe_module = safe_import_griffe()
    except OptionalDependencyError:
        return RuntimeError

    exceptions_module = getattr(griffe_module, "exceptions", None)
    alias_exc = getattr(exceptions_module, "AliasResolutionError", None)
    if isinstance(alias_exc, type) and issubclass(alias_exc, Exception):
        return alias_exc
    return RuntimeError


AliasResolutionErrorType = _resolve_alias_resolution_error()

ENV = shared.detect_environment()
shared.ensure_sys_paths(ENV)
SETTINGS = shared.load_settings()
LOADER = shared.make_loader(ENV)

DOCS_BUILD = SETTINGS.docs_build_dir
SCHEMA_DIR = ENV.root / "schema" / "docs"
SYMBOLS_PATH = DOCS_BUILD / "symbols.json"
BY_FILE_PATH = DOCS_BUILD / "by_file.json"
BY_MODULE_PATH = DOCS_BUILD / "by_module.json"
SYMBOL_INDEX_SCHEMA = SCHEMA_DIR / "symbol-index.schema.json"

BASE_LOGGER = get_logger(__name__)
SYMBOL_LOG = shared.make_logger("symbol_index", artifact="symbols.json", logger=BASE_LOGGER)
BY_FILE_LOG = shared.make_logger("symbol_index", artifact="by_file.json", logger=BASE_LOGGER)
BY_MODULE_LOG = shared.make_logger("symbol_index", artifact="by_module.json", logger=BASE_LOGGER)

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonPayload = Mapping[str, JsonValue] | Sequence[JsonValue] | JsonValue
ProblemDetailsDict = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LineSpan:
    """Start/end line span for a symbol."""

    start: int | None
    end: int | None


@dataclass(frozen=True, slots=True)
class NavLookup:
    """Lookup tables derived from the navmap payload."""

    symbol_meta: Mapping[str, Mapping[str, JsonValue]]
    module_meta: Mapping[str, Mapping[str, JsonValue]]
    sections: Mapping[str, str]

    @classmethod
    def empty(cls) -> NavLookup:
        return cls(symbol_meta={}, module_meta={}, sections={})


@dataclass(frozen=True, slots=True)
class SymbolIndexRow:
    """In-memory representation of a symbol entry."""

    path: str
    canonical_path: str | None
    kind: str
    package: str | None
    module: str | None
    file: str | None
    span: LineSpan
    doc: str
    signature: str | None
    is_async: bool
    is_property: bool
    owner: str | None
    stability: str | None
    since: str | None
    deprecated_in: str | None
    section: str | None
    tested_by: tuple[str, ...]
    source_link: Mapping[str, str]

    def to_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible payload for the row."""
        return {
            "path": self.path,
            "canonical_path": self.canonical_path,
            "kind": self.kind,
            "package": self.package,
            "module": self.module,
            "file": self.file,
            "lineno": self.span.start,
            "endlineno": self.span.end,
            "doc": self.doc,
            "signature": self.signature,
            "is_async": self.is_async,
            "is_property": self.is_property,
            "owner": self.owner,
            "stability": self.stability,
            "since": self.since,
            "deprecated_in": self.deprecated_in,
            "section": self.section,
            "tested_by": list(self.tested_by),
            "source_link": dict(self.source_link),
        }


@dataclass(frozen=True, slots=True)
class SymbolIndexArtifacts:
    """Bundle of symbol index artifacts emitted by this script."""

    rows: tuple[SymbolIndexRow, ...]
    by_file: Mapping[str, tuple[str, ...]]
    by_module: Mapping[str, tuple[str, ...]]

    @property
    def symbol_count(self) -> int:
        return len(self.rows)

    def rows_payload(self) -> list[dict[str, JsonValue]]:
        return [row.to_payload() for row in self.rows]

    def by_file_payload(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in sorted(self.by_file.items())}

    def by_module_payload(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in sorted(self.by_module.items())}


@dataclass(frozen=True, slots=True)
class SchemaValidation:
    """Schema validation metadata applied before writing an artifact."""

    schema: Path


@runtime_checkable
class GriffeNode(Protocol):
    """Subset of Griffe attributes consumed by the symbol index builder."""

    path: str
    canonical_path: object | None
    kind: object
    members: Mapping[str, GriffeNode] | None
    docstring: object | None
    relative_package_filepath: object | None
    relative_filepath: object | None
    lineno: int | float | None
    endlineno: int | float | None
    is_async: bool | None
    is_property: bool | None


def safe_getattr(obj: object, name: str, default: object | None = None) -> object | None:
    """Return ``getattr`` with defensive error handling."""
    try:
        return cast(object, getattr(obj, name, default))
    except (AttributeError, RuntimeError, AliasResolutionErrorType):
        return default


def _normalize_kind(node: GriffeNode) -> str | None:
    kind_obj = safe_getattr(node, "kind")
    if isinstance(kind_obj, str):
        return kind_obj
    value = safe_getattr(kind_obj, "value")
    return value if isinstance(value, str) else None


def _canonical_path(node: GriffeNode) -> str | None:
    canonical = safe_getattr(node, "canonical_path")
    if canonical is None:
        return None
    path_attr = safe_getattr(canonical, "path")
    if isinstance(path_attr, str):
        return path_attr
    if path_attr is not None:
        return str(path_attr)
    return str(canonical)


def _relative_file(node: GriffeNode) -> str | None:
    candidate = safe_getattr(node, "relative_package_filepath")
    if candidate is None:
        candidate = safe_getattr(node, "relative_filepath")
    if isinstance(candidate, Path):
        return candidate.as_posix()
    if isinstance(candidate, str):
        return candidate
    return None


def _normalize_lineno(value: object | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _doc_first_paragraph(node: GriffeNode) -> str:
    doc_obj = safe_getattr(node, "docstring")
    value = safe_getattr(doc_obj, "value")
    if isinstance(value, str):
        text = value.strip()
        if text:
            first = text.split("\n\n", 1)[0]
            return first.strip()
    return ""


def _normalize_tests(value: object | None) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _module_for(path: str | None, kind: str) -> str | None:
    if path is None:
        return None
    if kind in {"module", "package"}:
        return path
    if "." in path:
        return path.rsplit(".", 1)[0]
    return path


def _package_for(module: str | None, path: str | None) -> str | None:
    target = module or path
    if not target:
        return None
    return target.split(".", 1)[0]


def _join_symbol(module: str, symbol: str) -> str:
    if symbol.startswith(module):
        return symbol
    if "." in symbol:
        return symbol
    return f"{module}.{symbol}"


def _meta_value(
    symbol_meta: Mapping[str, JsonValue] | None,
    module_defaults: Mapping[str, JsonValue] | None,
    key: str,
) -> str | None:
    if symbol_meta and key in symbol_meta:
        value = symbol_meta[key]
        if isinstance(value, str):
            return value
    if module_defaults and key in module_defaults:
        fallback = module_defaults[key]
        if isinstance(fallback, str):
            return fallback
    return None


def _row_section(
    nav: NavLookup,
    module: str | None,
    path: str,
    canonical: str | None,
    kind: str,
) -> str | None:
    direct = nav.sections.get(path)
    if direct:
        return direct
    if canonical:
        canonical_section = nav.sections.get(canonical)
        if canonical_section:
            return canonical_section
    if module:
        module_section = nav.sections.get(module)
        if module_section:
            return module_section
    if kind == "module":
        return "module"
    if kind == "class":
        return "class"
    return None


def _build_source_links(file_rel: str | None, span: LineSpan) -> dict[str, str]:
    if not file_rel:
        return {}
    absolute = (ENV.root / file_rel).resolve()
    links: dict[str, str] = {}
    start_line = span.start or 1
    if SETTINGS.link_mode in {"editor", "both"}:
        links["editor"] = f"vscode://file/{absolute}:{start_line}:1"
    if SETTINGS.link_mode in {"github", "both"}:
        github = build_github_permalink(Path(file_rel), span)
        if github:
            links["github"] = github
    return links


def _clean_meta(meta: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    if not isinstance(meta, Mapping):
        return {}
    return {key: value for key, value in meta.items() if value is not None}


def _record_module_defaults(
    module_name: str,
    payload: Mapping[str, JsonValue],
    module_meta: dict[str, dict[str, JsonValue]],
    symbol_meta: dict[str, dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    defaults = _clean_meta(cast(Mapping[str, JsonValue] | None, payload.get("module_meta")))
    module_meta[module_name] = defaults
    if defaults:
        symbol_meta.setdefault(module_name, dict(defaults))
    return defaults


def _record_symbol_meta(
    module_name: str,
    per_symbol_meta: Mapping[str, JsonValue] | None,
    symbol_meta: dict[str, dict[str, JsonValue]],
) -> None:
    if not isinstance(per_symbol_meta, Mapping):
        return
    for name, meta in per_symbol_meta.items():
        if not isinstance(name, str) or not isinstance(meta, Mapping):
            continue
        fq_name = _join_symbol(module_name, name)
        symbol_meta[fq_name] = _clean_meta(meta)


def _record_sections(
    module_name: str,
    sections_payload: Sequence[JsonValue] | None,
    sections: dict[str, str],
) -> None:
    if sections_payload is None:
        return
    for section_obj in sections_payload:
        if not isinstance(section_obj, Mapping):
            continue
        section_id = section_obj.get("id")
        if not isinstance(section_id, str):
            continue
        symbols = section_obj.get("symbols")
        if not isinstance(symbols, Sequence):
            continue
        for symbol in symbols:
            if not isinstance(symbol, str):
                continue
            fq_name = _join_symbol(module_name, symbol)
            sections[fq_name] = section_id


def _navmap_from_payload(data: Mapping[str, JsonValue]) -> NavLookup:
    symbol_meta: dict[str, dict[str, JsonValue]] = {}
    module_meta: dict[str, dict[str, JsonValue]] = {}
    sections: dict[str, str] = {}

    modules_value = data.get("modules")
    if not isinstance(modules_value, Mapping):
        return NavLookup.empty()
    modules_mapping = cast(Mapping[str, JsonValue], modules_value)

    for module_name, payload in modules_mapping.items():
        if not isinstance(module_name, str) or not isinstance(payload, Mapping):
            continue
        payload_mapping = cast(Mapping[str, JsonValue], payload)
        _record_module_defaults(module_name, payload_mapping, module_meta, symbol_meta)
        _record_symbol_meta(
            module_name,
            cast(Mapping[str, JsonValue] | None, payload_mapping.get("meta")),
            symbol_meta,
        )
        _record_sections(
            module_name,
            cast(Sequence[JsonValue] | None, payload_mapping.get("sections")),
            sections,
        )

    return NavLookup(symbol_meta=symbol_meta, module_meta=module_meta, sections=sections)


def load_nav_lookup() -> NavLookup:
    """Return navmap metadata if available."""
    for candidate in SETTINGS.navmap_candidates:
        if not candidate.exists():
            continue
        try:
            payload = cast(JsonPayload, json.loads(candidate.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            BASE_LOGGER.warning(
                "Failed to parse navmap candidate",
                extra={
                    "status": "invalid_navmap",
                    "candidate": str(candidate),
                    "reason": str(exc),
                },
            )
            continue
        if isinstance(payload, Mapping):
            lookup = _navmap_from_payload(payload)
            BASE_LOGGER.info(
                "NavMap metadata loaded",
                extra={
                    "status": "loaded",
                    "candidate": str(candidate),
                    "symbol_meta": len(lookup.symbol_meta),
                    "module_meta": len(lookup.module_meta),
                    "sections": len(lookup.sections),
                },
            )
            return lookup
    BASE_LOGGER.info(
        "NavMap metadata unavailable",
        extra={"status": "missing_navmap"},
    )
    return NavLookup.empty()


def load_test_map() -> dict[str, JsonValue]:
    """Return the optional test map produced earlier in the docs pipeline."""
    path = DOCS_BUILD / "test_map.json"
    if not path.exists():
        return {}
    try:
        payload = cast(JsonPayload, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        SYMBOL_LOG.warning(
            "Failed to parse test_map.json",
            extra={"status": "invalid_test_map", "path": str(path), "reason": str(exc)},
        )
        return {}
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    return {}


def _iter_members(node: GriffeNode) -> Iterable[GriffeNode]:
    members = safe_getattr(node, "members")
    if isinstance(members, Mapping):
        return tuple(cast(GriffeNode, member) for member in members.values())
    return ()


def _row_from_node(
    node: GriffeNode,
    *,
    nav: NavLookup,
    test_map: Mapping[str, JsonValue],
) -> SymbolIndexRow | None:
    path_obj = safe_getattr(node, "path")
    if not isinstance(path_obj, str):
        return None

    kind = _normalize_kind(node)
    if kind is None:
        return None

    module = _module_for(path_obj, kind)
    package = _package_for(module, path_obj)
    file_rel = _relative_file(node)

    lineno = _normalize_lineno(safe_getattr(node, "lineno"))
    endlineno = _normalize_lineno(safe_getattr(node, "endlineno"))
    span = LineSpan(start=lineno, end=endlineno)

    canonical = _canonical_path(node)
    symbol_meta = nav.symbol_meta.get(path_obj)
    if symbol_meta is None and canonical:
        symbol_meta = nav.symbol_meta.get(canonical)
    module_defaults = nav.module_meta.get(module or "")

    tested_by = _normalize_tests(test_map.get(path_obj))
    if not tested_by and canonical:
        tested_by = _normalize_tests(test_map.get(canonical))

    signature_obj = safe_getattr(node, "signature")
    signature = str(signature_obj) if signature_obj is not None else None

    return SymbolIndexRow(
        path=path_obj,
        canonical_path=canonical,
        kind=kind,
        package=package,
        module=module,
        file=file_rel,
        span=span,
        doc=_doc_first_paragraph(node),
        signature=signature,
        is_async=bool(safe_getattr(node, "is_async")),
        is_property=bool(safe_getattr(node, "is_property")),
        owner=_meta_value(symbol_meta, module_defaults, "owner"),
        stability=_meta_value(symbol_meta, module_defaults, "stability"),
        since=_meta_value(symbol_meta, module_defaults, "since"),
        deprecated_in=_meta_value(symbol_meta, module_defaults, "deprecated_in"),
        section=_row_section(nav, module, path_obj, canonical, kind),
        tested_by=tested_by,
        source_link=_build_source_links(file_rel, span),
    )


def _collect_rows(
    node: GriffeNode,
    *,
    nav: NavLookup,
    test_map: Mapping[str, JsonValue],
) -> list[SymbolIndexRow]:
    rows: list[SymbolIndexRow] = []
    stack: list[GriffeNode] = [node]
    while stack:
        current = stack.pop()
        row = _row_from_node(current, nav=nav, test_map=test_map)
        if row is not None:
            rows.append(row)
        stack.extend(_iter_members(current))
    return rows


def _build_reverse_map(
    rows: Sequence[SymbolIndexRow], attribute: str
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, set[str]] = {}
    for row in rows:
        value = cast(object, getattr(row, attribute))
        if isinstance(value, str):
            if value not in mapping:
                mapping[value] = set()
            mapping[value].add(row.path)
    return {key: tuple(sorted(paths)) for key, paths in sorted(mapping.items())}


def generate_index(
    packages: Sequence[str],
    loader: shared.GriffeLoader,
) -> SymbolIndexArtifacts:
    """Produce typed symbol index artifacts for ``packages``."""
    nav_lookup = load_nav_lookup()
    test_map = load_test_map()
    rows: list[SymbolIndexRow] = []
    for package in packages:
        root = cast(GriffeNode, loader.load(package))
        rows.extend(_collect_rows(root, nav=nav_lookup, test_map=test_map))

    def get_row_path(row: SymbolIndexRow) -> str:
        """Extract path for sorting."""
        return row.path

    rows_sorted = tuple(sorted(rows, key=get_row_path))
    by_file = _build_reverse_map(rows_sorted, "file")
    by_module = _build_reverse_map(rows_sorted, "module")
    return SymbolIndexArtifacts(rows=rows_sorted, by_file=by_file, by_module=by_module)


@lru_cache(maxsize=1)
def _git_sha() -> str:
    """Return the Git SHA for permalink construction."""
    return shared.resolve_git_sha(ENV, SETTINGS, logger=BASE_LOGGER)


def build_github_permalink(file: Path, span: LineSpan) -> str | None:
    """Return a commit-stable GitHub permalink for ``file`` when configured."""
    if not (SETTINGS.github_org and SETTINGS.github_repo):
        return None
    sha = _git_sha()
    fragment = ""
    if span.start is not None and span.end is not None and span.end >= span.start:
        fragment = f"#L{span.start}-L{span.end}"
    elif span.start is not None:
        fragment = f"#L{span.start}"
    relative = file.as_posix()
    return (
        "https://github.com/"
        f"{SETTINGS.github_org}/{SETTINGS.github_repo}/blob/{sha}/{relative}{fragment}"
    )


def write_artifact(
    path: Path,
    payload: object,
    *,
    logger: StructuredLoggerAdapter,
    artifact: str,
    validation: SchemaValidation | None = None,
) -> bool:
    """Validate (when configured) and write ``payload`` to ``path`` if it changed."""
    if validation is not None:
        validate_against_schema(
            cast(JsonPayload, payload),
            validation.schema,
            artifact=artifact,
        )
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == serialized:
            logger.info(
                "Artifact already up-to-date",
                extra={"status": "unchanged", "path": str(path)},
            )
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    logger.info(
        "Artifact written",
        extra={"status": "updated", "path": str(path)},
    )
    return True


def iter_packages() -> list[str]:
    """Return packages configured for documentation builds (compatibility helper)."""
    return list(SETTINGS.packages)


def safe_attr(node: object, attr: str, default: object | None = None) -> object | None:
    """Compatibility wrapper delegating to :func:`safe_getattr`."""
    return safe_getattr(node, attr, default)


def _emit_problem(problem: ProblemDetailsDict | None, *, default_message: str) -> None:
    payload = problem or build_problem_details(
        ProblemDetailsParams(
            type="https://kgfoundry.dev/problems/docs-symbol-index",
            title="Symbol index build failed",
            status=500,
            detail=default_message,
            instance="urn:docs:symbol-index",
        )
    )
    sys.stderr.write(render_problem(payload) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Build and validate documentation symbol index artifacts.

    This entry point generates the symbol index by analyzing all packages
    in the repository. It produces three artifacts: symbols.json (complete
    symbol metadata), by_file.json (symbols grouped by file), and
    by_module.json (symbols grouped by module).

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments (reserved for future use). Currently unused.
        Defaults to None.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.

    Raises
    ------
    SystemExit
        When called as __main__, propagates the exit code.

    Examples
    --------
    >>> from docs._scripts.build_symbol_index import main
    >>> exit_code = main()
    >>> exit_code
    0
    """
    # argv is reserved for future CLI argument parsing if needed
    del argv

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    packages = list(SETTINGS.packages or ())
    with observe_tool_run(["docs-symbol-index"], cwd=DOCS_BUILD, timeout=None) as observation:
        try:
            artifacts = generate_index(packages, LOADER)
            rows_payload = artifacts.rows_payload()
            wrote_symbols = write_artifact(
                SYMBOLS_PATH,
                rows_payload,
                logger=SYMBOL_LOG,
                artifact="symbols.json",
                validation=SchemaValidation(schema=SYMBOL_INDEX_SCHEMA),
            )
            wrote_by_file = write_artifact(
                BY_FILE_PATH,
                artifacts.by_file_payload(),
                logger=BY_FILE_LOG,
                artifact="by_file.json",
            )
            wrote_by_module = write_artifact(
                BY_MODULE_PATH,
                artifacts.by_module_payload(),
                logger=BY_MODULE_LOG,
                artifact="by_module.json",
            )
        except ToolExecutionError as exc:  # pragma: no cover - exercised in CLI
            observation.failure("failure", returncode=1)
            _emit_problem(exc.problem, default_message=str(exc))
            return 1
        except KeyboardInterrupt:
            observation.failure("cancelled", returncode=130)
            SYMBOL_LOG.info("Symbol index generation cancelled by user")
            return 130
        except (
            DeserializationError,
            SerializationError,
            SchemaValidationError,
            OSError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            observation.failure("exception", returncode=1)
            problem = build_problem_details(
                ProblemDetailsParams(
                    type="https://kgfoundry.dev/problems/docs-symbol-index",
                    title="Symbol index build failed",
                    status=500,
                    detail=str(exc),
                    instance="urn:docs:symbol-index:unexpected-error",
                    extensions={"packages": list(packages)},
                )
            )
            _emit_problem(problem, default_message=str(exc))
            return 1
        else:
            observation.success(0)

    SYMBOL_LOG.info(
        "Symbol index build complete",
        extra={
            "status": "success",
            "symbols_entries": artifacts.symbol_count,
            "symbols_updated": wrote_symbols,
            "by_file_updated": wrote_by_file,
            "by_module_updated": wrote_by_module,
            "symbols_path": str(SYMBOLS_PATH),
            "by_file_path": str(BY_FILE_PATH),
            "by_module_path": str(BY_MODULE_PATH),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
