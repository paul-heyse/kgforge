"""Render repository documentation into distributable README files.

The script composes package README pages that align with the content hosted on
kgfoundry.dev by invoking the shared documentation rendering utilities. It ensures
local packages contain synchronized summaries, badges, and metadata for publishing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, NoReturn, Protocol, cast
from urllib.parse import urlparse

from tools._shared.logging import get_logger, with_fields
from tools._shared.problem_details import ProblemDetailsDict, build_problem_details
from tools._shared.proc import ToolExecutionError, run_tool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.griffe_utils import GriffeAPI, resolve_griffe  # noqa: E402

LOGGER = get_logger(__name__)


type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class ReadmeGenerationError(RuntimeError):
    """Base exception raised when README generation fails."""

    def __init__(self, message: str, *, problem: ProblemDetailsDict | None = None) -> None:
        super().__init__(message)
        self.problem = problem


class MissingMetadataError(ReadmeGenerationError):
    """Raised when required badge metadata is missing for public symbols."""


class LinkMode(Enum):
    """Enumerate link emission strategies for generated README entries."""

    GITHUB = "github"
    EDITOR = "editor"
    BOTH = "both"


class EditorMode(Enum):
    """Enumerate editor link formats supported by README generation."""

    VSCODE = "vscode"
    RELATIVE = "relative"


class DocstringLike(Protocol):
    """Minimal interface of ``griffe`` docstrings used by this module."""

    value: str | None


class KindLike(Protocol):
    """Subset of the ``griffe`` kind enumeration accessed during rendering."""

    value: str


class BaseLike(Protocol):
    """Shape of base classes referenced when classifying exceptions."""

    full: str | None
    name: str | None


class GriffeObjectLike(Protocol):
    """Structural type describing the ``griffe`` objects we inspect."""

    path: str
    name: str
    docstring: DocstringLike | None
    kind: KindLike | None
    members: Mapping[str, GriffeObjectLike] | None
    relative_package_filepath: str | None
    lineno: int | None
    endlineno: int | None
    bases: Sequence[BaseLike] | None
    is_package: bool | None


class LoaderInstance(Protocol):
    """Protocol describing loader instances returned by :mod:`griffe`."""

    def load(self, name: str) -> GriffeObjectLike:
        """Return the object graph for the package identified by ``name``."""


class LoaderFactory(Protocol):
    """Factory returning loader instances used to inspect packages."""

    def __call__(self, *, search_paths: Sequence[str]) -> LoaderInstance:
        """Return a loader bound to ``search_paths``."""


from tools.detect_pkg import detect_packages, detect_primary  # noqa: E402

_griffe_api: GriffeAPI = resolve_griffe()
GriffeLoader = cast(LoaderFactory, _griffe_api.loader_type)

SRC = ROOT / "src"
NAVMAP_PATH = ROOT / "site" / "_build" / "navmap" / "navmap.json"
TESTMAP_PATH = ROOT / "docs" / "_build" / "test_map.json"

DEFAULT_SYNOPSIS = "Package synopsis not yet documented."
MIN_REMOTE_PARTS: Final[int] = 2
README_WRAP_COLUMN: Final[int] = 80
README_BADGE_INDENT: Final[int] = 4


def detect_repo() -> tuple[str, str]:
    """Return the GitHub ``(owner, repo)`` tuple used for documentation links.

    Environment variables ``DOCS_GITHUB_ORG`` and ``DOCS_GITHUB_REPO`` override
    the detected values so downstream builds can rehost the docs without
    reconfiguring git remotes.
    """
    log_adapter = with_fields(
        LOGGER,
        command=("git", "config", "--get", "remote.origin.url"),
    )
    try:
        result = run_tool(
            (
                "git",
                "config",
                "--get",
                "remote.origin.url",
            ),
            cwd=ROOT,
            timeout=10.0,
        )
        remote = result.stdout.strip()
    except ToolExecutionError as exc:
        log_adapter.debug("Unable to detect git remote: %s", exc)
        remote = ""

    override_owner = os.environ.get("DOCS_GITHUB_ORG")
    override_repo = os.environ.get("DOCS_GITHUB_REPO")
    if override_owner and override_repo:
        return override_owner, override_repo

    if remote.endswith(".git"):
        remote = remote[:-4]
    path = ""
    if remote.startswith("git@"):
        _, remainder = remote.split("@", 1)
        path = remainder.split(":", 1)[1]
    else:
        parsed_remote = urlparse(remote)
        if parsed_remote.path:
            path = parsed_remote.path.lstrip("/")

    if path:
        parts = path.split("/")
        if len(parts) >= MIN_REMOTE_PARTS:
            return parts[0], parts[1]

    return override_owner or "your-org", override_repo or "your-repo"


def git_sha() -> str:
    """Return the Git revision used when composing GitHub source links.

    A ``DOCS_GITHUB_SHA`` environment variable can provide the value when the
    repository is not available locally (for example, in CI artifacts).
    """
    log_adapter = with_fields(LOGGER, command=("git", "rev-parse", "HEAD"))
    try:
        result = run_tool(("git", "rev-parse", "HEAD"), cwd=ROOT, timeout=10.0)
        return result.stdout.strip() or os.environ.get("DOCS_GITHUB_SHA", "main")
    except ToolExecutionError as exc:
        log_adapter.debug("Unable to resolve git SHA: %s", exc)
        return os.environ.get("DOCS_GITHUB_SHA", "main")


OWNER, REPO = detect_repo()
SHA = git_sha()


def gh_url(rel_path: str, start: int, end: int | None) -> str:
    """Return a GitHub ``blob`` URL anchored to the provided line range.

    Parameters
    ----------
    rel_path
        File path relative to the repository root.
    start
        1-based starting line number.
    end
        Optional inclusive end line number. When omitted the URL anchors to
        ``start`` only.

    Returns
    -------
    str
        Fully-qualified GitHub URL pointing at the requested source snippet.
    """
    fragment = f"#L{start}-L{end}" if end and end >= start else f"#L{start}"
    return f"https://github.com/{OWNER}/{REPO}/blob/{SHA}/{rel_path}{fragment}"


def iter_packages() -> list[str]:
    """Return the list of packages documented by default.

    The helper delegates to :func:`tools.detect_pkg.detect_packages` so CLI
    overrides and auto-detection share the same semantics.
    """
    env_pkgs = os.environ.get("DOCS_PKG")
    if env_pkgs:
        return [pkg.strip() for pkg in env_pkgs.split(",") if pkg.strip()]
    return detect_packages() or [detect_primary()]


def summarize(node: GriffeObjectLike) -> str:
    """Return the first sentence of ``node``'s docstring, when present."""
    doc = node.docstring
    if doc is None or not doc.value:
        return ""
    raw = doc.value.strip()
    if not raw:
        return ""
    # Find the first non-empty line before attempting to split into sentences.
    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    if not first_line:
        return ""
    match = re.search(r"(?<=[.!?])\s", first_line)
    if match:
        return first_line[: match.start()].strip()
    return first_line


def is_public(node: GriffeObjectLike) -> bool:
    """Return ``True`` if the symbol name is not private by convention."""
    return not node.name.startswith("_")


def get_open_link(node: GriffeObjectLike, readme_dir: Path) -> str | None:
    """Return a local editor link for ``node`` relative to ``readme_dir``.

    The link points to the source file on disk using the ``./path:line:column``
    format understood by VS Code and other Markdown-aware viewers. ``None`` is
    returned when the source file is not within the README directory tree.
    """
    rel_path = node.relative_package_filepath
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        relative = abs_path.relative_to(readme_dir).as_posix()
    except ValueError:
        return None
    start = int(node.lineno or 1)
    return f"./{relative}:{start}:1"


def get_view_link(node: GriffeObjectLike) -> str | None:
    """Return a GitHub URL for ``node`` when its path maps inside the repo."""
    rel_path = node.relative_package_filepath
    if not rel_path:
        return None
    base = SRC if SRC.exists() else ROOT
    abs_path = (base / rel_path).resolve()
    try:
        rel = abs_path.relative_to(ROOT)
    except ValueError:
        return None
    start = int(node.lineno or 1)
    end = node.endlineno
    return gh_url(str(rel).replace("\\", "/"), start, end)


def iter_public_members(node: GriffeObjectLike) -> list[GriffeObjectLike]:
    """Return ``node`` members that are public based on their name."""
    members = node.members
    if not members:
        return []
    public = [member for member in members.values() if is_public(member)]

    def _member_key(member: GriffeObjectLike) -> str:
        """Return the fully qualified path when available, otherwise the name."""
        if member.path:
            return member.path
        return member.name

    return sorted(public, key=_member_key)


def _load_json(path: Path) -> JsonObject:
    """Return the JSON document stored at ``path`` or ``{}`` when unavailable."""
    if not path.exists():
        return {}
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        with_fields(LOGGER, path=str(path)).debug("Failed to load JSON: %s", exc)
        return {}
    if isinstance(raw, dict):
        return cast(JsonObject, raw)
    with_fields(LOGGER, path=str(path), received=type(raw).__name__).debug(
        "Expected object at JSON root, defaulting to empty document",
    )
    return {}


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    """Badge metadata derived from the navigation map for a single symbol."""

    owner: str | None = None
    stability: str | None = None
    section: str | None = None
    since: str | None = None
    deprecated_in: str | None = None

    def merged(self, override: SymbolMetadata | None) -> SymbolMetadata:
        """Return a metadata instance where ``override`` values take precedence."""
        if override is None:
            return self
        return SymbolMetadata(
            owner=override.owner or self.owner,
            stability=override.stability or self.stability,
            section=override.section or self.section,
            since=override.since or self.since,
            deprecated_in=override.deprecated_in or self.deprecated_in,
        )

    def with_section(self, section: str | None) -> SymbolMetadata:
        """Return metadata with ``section`` applied when provided."""
        if section is None or self.section == section:
            return self
        return SymbolMetadata(
            owner=self.owner,
            stability=self.stability,
            section=section,
            since=self.since,
            deprecated_in=self.deprecated_in,
        )


@dataclass(frozen=True, slots=True)
class NavMatch:
    """Result of attempting to resolve badge metadata for a symbol."""

    symbol_meta: SymbolMetadata | None
    defaults: SymbolMetadata
    matches_symbol_list: bool
    matches_prefix: bool


@dataclass(frozen=True, slots=True)
class NavModuleData:
    """Typed representation of a module entry within the navigation map."""

    identifier: str
    defaults: SymbolMetadata
    overrides: dict[str, SymbolMetadata]
    sections: dict[str, str]
    listed_symbols: frozenset[str]

    def lookup(self, qname: str, symbol: str) -> NavMatch:
        """Return metadata lookup result for ``qname`` within this module."""
        override = self.overrides.get(qname) or self.overrides.get(symbol)
        section = self.sections.get(symbol)
        defaults = self.defaults.with_section(section)
        resolved_override = override.with_section(section) if override is not None else None
        return NavMatch(
            symbol_meta=resolved_override,
            defaults=defaults,
            matches_symbol_list=symbol in self.listed_symbols,
            matches_prefix=qname.startswith(self.identifier),
        )


@dataclass(frozen=True, slots=True)
class NavData:
    """Navigation metadata extracted from ``navmap.json``."""

    modules: dict[str, NavModuleData]

    def lookup(self, qname: str) -> tuple[SymbolMetadata | None, SymbolMetadata]:
        """Return override metadata and defaults for ``qname``."""
        symbol = qname.rsplit(".", maxsplit=1)[-1]
        fallback: SymbolMetadata | None = None
        for module in self.modules.values():
            match = module.lookup(qname, symbol)
            if match.symbol_meta is not None:
                return match.symbol_meta, match.defaults
            if match.matches_symbol_list or (match.matches_prefix and fallback is None):
                fallback = match.defaults
        return None, fallback or SymbolMetadata()


@dataclass(frozen=True, slots=True)
class TestRecord:
    """Association between a symbol and one or more validating tests."""

    file: str
    lines: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TestCatalog:
    """Collection of test coverage metadata derived from ``test_map.json``."""

    records: dict[str, tuple[TestRecord, ...]]

    def lookup(self, qname: str) -> tuple[TestRecord, ...]:
        """Return test records associated with ``qname`` or its short name."""
        symbol = qname.rsplit(".", maxsplit=1)[-1]
        return self.records.get(qname) or self.records.get(symbol) or ()


@dataclass(frozen=True, slots=True)
class Badges:
    """Resolved badge metadata for a public symbol."""

    stability: str | None = None
    owner: str | None = None
    section: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    tested_by: tuple[TestRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadmeConfig:
    """Validated configuration derived from CLI flags and environment variables."""

    packages: tuple[str, ...]
    link_mode: LinkMode
    editor: EditorMode
    fail_on_metadata_miss: bool
    dry_run: bool
    verbose: bool
    run_doctoc: bool

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> ReadmeConfig:
        """Construct configuration from an argparse namespace."""
        packages_arg: str = getattr(namespace, "packages", "")
        packages = tuple(pkg for pkg in (part.strip() for part in packages_arg.split(",")) if pkg)
        if not packages:
            packages = tuple(iter_packages())
        link_mode = LinkMode(cast(str, namespace.link_mode))
        editor = EditorMode(cast(str, namespace.editor))
        return cls(
            packages=packages,
            link_mode=link_mode,
            editor=editor,
            fail_on_metadata_miss=bool(namespace.fail_on_metadata_miss),
            dry_run=bool(namespace.dry_run),
            verbose=bool(namespace.verbose),
            run_doctoc=bool(namespace.run_doctoc),
        )


def _optional_str(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) else None


def _symbol_metadata_from_mapping(payload: Mapping[str, JsonValue]) -> SymbolMetadata:
    return SymbolMetadata(
        owner=_optional_str(payload.get("owner")),
        stability=_optional_str(payload.get("stability")),
        section=_optional_str(payload.get("section")),
        since=_optional_str(payload.get("since")),
        deprecated_in=_optional_str(payload.get("deprecated_in")),
    )


def _parse_symbol_overrides(payload: Mapping[str, JsonValue]) -> dict[str, SymbolMetadata]:
    overrides: dict[str, SymbolMetadata] = {}
    meta_value = payload.get("meta")
    if isinstance(meta_value, Mapping):
        for key, raw_meta in meta_value.items():
            if isinstance(key, str) and isinstance(raw_meta, Mapping):
                overrides[key] = _symbol_metadata_from_mapping(raw_meta)
    return overrides


def _parse_sections_map(payload: Mapping[str, JsonValue]) -> dict[str, str]:
    sections: dict[str, str] = {}
    section_entries = payload.get("sections")
    if isinstance(section_entries, list):
        for entry in section_entries:
            if not isinstance(entry, Mapping):
                continue
            section_id = _optional_str(entry.get("id"))
            if not section_id:
                continue
            symbols_value = entry.get("symbols")
            if not isinstance(symbols_value, list):
                continue
            for symbol in symbols_value:
                if isinstance(symbol, str):
                    sections[symbol] = section_id
    return sections


def _parse_listed_symbols(payload: Mapping[str, JsonValue]) -> frozenset[str]:
    symbol_entries = payload.get("symbols")
    collected: set[str] = set()
    if isinstance(symbol_entries, list):
        for entry in symbol_entries:
            if isinstance(entry, str):
                collected.add(entry)
    return frozenset(collected)


def _parse_nav_module(identifier: str, payload: Mapping[str, JsonValue]) -> NavModuleData:
    module_meta = payload.get("module_meta")
    defaults = (
        _symbol_metadata_from_mapping(module_meta)
        if isinstance(module_meta, Mapping)
        else SymbolMetadata(
            owner=_optional_str(payload.get("owner")),
            stability=_optional_str(payload.get("stability")),
            since=_optional_str(payload.get("since")),
            deprecated_in=_optional_str(payload.get("deprecated_in")),
        )
    )
    return NavModuleData(
        identifier=identifier,
        defaults=defaults,
        overrides=_parse_symbol_overrides(payload),
        sections=_parse_sections_map(payload),
        listed_symbols=_parse_listed_symbols(payload),
    )


def _build_nav_data(document: JsonObject) -> NavData:
    modules_value = document.get("modules")
    modules: dict[str, NavModuleData] = {}
    if isinstance(modules_value, Mapping):
        for identifier, payload in modules_value.items():
            if isinstance(identifier, str) and isinstance(payload, Mapping):
                modules[identifier] = _parse_nav_module(identifier, payload)
    return NavData(modules=modules)


def _build_test_catalog(document: JsonObject) -> TestCatalog:
    records: dict[str, tuple[TestRecord, ...]] = {}
    for key, value in document.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        entries: list[TestRecord] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            file_path = _optional_str(item.get("file"))
            if not file_path:
                continue
            lines_value = item.get("lines")
            lines: tuple[int, ...] = ()
            if isinstance(lines_value, list):
                lines = tuple(number for number in lines_value if isinstance(number, int))
            entries.append(TestRecord(file=file_path, lines=lines))
        records[key] = tuple(entries)
    return TestCatalog(records=records)


_NAV_DATA = _build_nav_data(_load_json(NAVMAP_PATH))
_TEST_CATALOG = _build_test_catalog(_load_json(TESTMAP_PATH))


def parse_config(argv: Sequence[str] | None = None) -> ReadmeConfig:
    """Parse CLI arguments and environment overrides into a :class:`ReadmeConfig`."""
    parser = argparse.ArgumentParser(description="Generate per-package README files.")
    parser.add_argument("--packages", default=os.getenv("DOCS_PKG", ""))
    parser.add_argument(
        "--link-mode",
        default=os.getenv("DOCS_LINK_MODE", "both"),
        choices=["github", "editor", "both"],
    )
    parser.add_argument(
        "--editor",
        default=os.getenv("DOCS_EDITOR", "vscode"),
        choices=["vscode", "relative"],
    )
    parser.add_argument("--fail-on-metadata-miss", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--run-doctoc", action="store_true", default=False)
    args = parser.parse_args(argv)

    return ReadmeConfig.from_namespace(args)


def badges_for(
    qname: str,
    *,
    nav: NavData | None = None,
    tests: TestCatalog | None = None,
) -> Badges:
    """Return badge metadata for ``qname`` using NavMap and test fixtures."""
    nav_data = nav if nav is not None else _NAV_DATA
    test_catalog = tests if tests is not None else _TEST_CATALOG
    symbol_meta, defaults = nav_data.lookup(qname)
    merged = defaults.merged(symbol_meta)
    return Badges(
        stability=merged.stability,
        owner=merged.owner,
        section=merged.section,
        since=merged.since,
        deprecated_in=merged.deprecated_in,
        tested_by=test_catalog.lookup(qname),
    )


def _format_test_badge(entries: Sequence[TestRecord] | None) -> str | None:
    """Format the ``tested-by`` badge snippet when entries exist."""
    if not entries:
        return None
    formatted: list[str] = []
    for entry in entries:
        if entry.lines:
            formatted.append(f"{entry.file}:{entry.lines[0]}")
        else:
            formatted.append(entry.file)
    if not formatted:
        return None
    return "`tested-by: " + ", ".join(formatted) + "`"


def _badge_parts(badge: Badges) -> list[str]:
    """Return textual fragments that make up the badge line."""
    attributes = [
        ("stability", "stability"),
        ("owner", "owner"),
        ("section", "section"),
        ("since", "since"),
        ("deprecated_in", "deprecated"),
    ]
    parts = [
        f"`{label}:{value}`"
        for attr, label in attributes
        if (value := cast(str | None, getattr(badge, attr)))
    ]
    test_badge = _format_test_badge(badge.tested_by)
    if test_badge:
        parts.append(test_badge)
    return parts


def _wrap_badge_parts(parts: Sequence[str]) -> list[str]:
    """Wrap badge fragments to respect the configured line width."""
    wrapped: list[str] = []
    current: list[str] = []
    current_len = 0
    available = README_WRAP_COLUMN - README_BADGE_INDENT
    for part in parts:
        part_len = len(part) + (1 if current else 0)
        if current and current_len + part_len > available:
            wrapped.append(" ".join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += part_len
    if current:
        wrapped.append(" ".join(current))
    return wrapped


def format_badges(qname: str, base_length: int = 0) -> str:
    """Render badge metadata as inline Markdown for the README heading.

    Parameters
    ----------
    qname
        Qualified symbol name used when looking up badges.
    base_length
        Length of the text preceding the badges on the line. Used to decide if
        the badges should wrap.

    Returns
    -------
    str
        Badge snippet prefixed with a space or newline when wrapping is required.
    """
    badge = badges_for(qname)
    parts = _badge_parts(badge)
    if not parts:
        return ""
    badge_line = " ".join(parts)
    if not base_length or base_length + 1 + len(badge_line) <= README_WRAP_COLUMN:
        return " " + badge_line
    wrapped = _wrap_badge_parts(parts)
    return "\n    " + "\n    ".join(wrapped)


def editor_link(abs_path: Path, lineno: int, editor_mode: EditorMode) -> str | None:
    """Return an editor-friendly link for ``abs_path`` based on ``editor_mode``."""
    if editor_mode is EditorMode.VSCODE:
        return f"vscode://file/{abs_path}:{lineno}:1"
    if editor_mode is EditorMode.RELATIVE:
        try:
            rel = abs_path.relative_to(ROOT)
        except ValueError:
            rel = abs_path
        return f"./{rel.as_posix()}:{lineno}:1"
    message = f"Unsupported editor mode: {editor_mode}"
    raise ValueError(message)


def _is_exception(node: GriffeObjectLike) -> bool:
    """Return ``True`` when ``node`` represents an exception type."""
    kind = node.kind.value if node.kind else ""
    if kind != "class":
        return False
    if node.name.endswith(("Error", "Exception")):
        return True
    for base in node.bases or []:
        base_name = base.full or base.name
        if base_name and base_name.endswith(("Error", "Exception")):
            return True
    return False


KINDS = {"module", "package", "class", "function"}


def bucket_for(node: GriffeObjectLike) -> str:
    """Return the README section name that should contain ``node``."""
    kind = node.kind.value if node.kind else ""
    if kind in {"module", "package"}:
        return "Modules"
    if kind == "class":
        return "Exceptions" if _is_exception(node) else "Classes"
    if kind == "function":
        return "Functions"
    return "Other"


def render_line(node: GriffeObjectLike, readme_dir: Path, cfg: ReadmeConfig) -> str | None:
    """Render a Markdown bullet for ``node`` including navigation links.

    The output includes GitHub links, optional editor URIs, and badges derived
    from the NavMap, matching the style published on kgfoundry.dev.
    """
    qname = node.path
    summary = summarize(node)

    link_mode = cfg.link_mode
    open_link = (
        get_open_link(node, readme_dir) if link_mode in {LinkMode.EDITOR, LinkMode.BOTH} else None
    )
    view_link = get_view_link(node) if link_mode in {LinkMode.GITHUB, LinkMode.BOTH} else None

    if link_mode in {LinkMode.EDITOR, LinkMode.BOTH} and node.relative_package_filepath:
        base = SRC if SRC.exists() else ROOT
        abs_path = (base / node.relative_package_filepath).resolve()
        direct = editor_link(abs_path, int(node.lineno or 1), cfg.editor)
        if direct:
            open_link = direct

    if not (open_link or view_link):
        return None

    line = f"- **`{qname}`**"
    if summary:
        line += f" — {summary}"
    badge_text = format_badges(qname, len(line))
    if badge_text:
        line += badge_text

    links: list[str] = []
    if open_link:
        links.append(f"[open]({open_link})")
    if view_link:
        links.append(f"[view]({view_link})")
    tail = f" → {' | '.join(links)}" if links else ""
    return line + tail + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` when the rendered output differs."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    rendered = content.rstrip() + f"\n<!-- agent:readme v1 sha:{SHA} content:{digest} -->\n"
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if previous == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def write_readme(node: GriffeObjectLike, cfg: ReadmeConfig) -> bool:
    """Generate or update the README for the package described by ``node``."""
    pkg_dir = (SRC if SRC.exists() else ROOT) / node.path.replace(".", "/")
    readme = pkg_dir / "README.md"

    buckets: dict[str, list[str]] = {
        name: [] for name in ("Modules", "Classes", "Functions", "Exceptions", "Other")
    }
    children = [
        child for child in iter_public_members(node) if child.kind and child.kind.value in KINDS
    ]

    for child in sorted(children, key=lambda child: child.path):
        line = render_line(child, pkg_dir, cfg)
        if line:
            buckets[bucket_for(child)].append(line)

    lines: list[str] = [f"# `{node.path}`\n\n"]
    synopsis = summarize(node) or DEFAULT_SYNOPSIS
    lines.append(f"{synopsis}\n\n")
    lines.extend(
        [
            "<!-- START doctoc generated TOC please keep comment here to allow auto update -->\n",
            "<!-- END doctoc generated TOC please keep comment here to allow auto update -->\n\n",
        ]
    )

    for section in ("Modules", "Classes", "Functions", "Exceptions", "Other"):
        items = buckets.get(section, [])
        if items:
            lines.append(f"## {section}\n\n")
            lines.extend(items)
            lines.append("\n")

    content = "".join(lines).rstrip() + "\n"
    if cfg.dry_run:
        with_fields(LOGGER, path=str(readme)).info("Dry run: README write skipped")
        return False
    changed = write_if_changed(readme, content)
    if changed:
        with_fields(LOGGER, path=str(readme)).info("README updated")
        _maybe_run_doctoc(readme, cfg)
    else:
        with_fields(LOGGER, path=str(readme)).debug("README already up to date")
    return changed


def _maybe_run_doctoc(readme: Path, cfg: ReadmeConfig) -> None:
    """Run DocToc when enabled via ``--run-doctoc``."""
    if not cfg.run_doctoc:
        return
    doctoc = shutil.which("doctoc")
    if not doctoc:
        with_fields(LOGGER, path=str(readme)).info(
            "docToc executable not available; skipping TOC update"
        )
        return
    log_adapter = with_fields(LOGGER, command=(doctoc, str(readme)))
    try:
        result = run_tool([doctoc, str(readme)], check=False, timeout=30.0)
    except ToolExecutionError as exc:
        log_adapter.warning("docToc invocation failed: %s", exc)
        return
    if cfg.verbose and result.stdout.strip():
        with_fields(log_adapter, stdout=result.stdout.strip()).info("docToc stdout")
    if result.stderr.strip():
        with_fields(log_adapter, stderr=result.stderr.strip()).warning("docToc stderr")
    if result.returncode != 0:
        with_fields(log_adapter, returncode=result.returncode).warning(
            "DocToc exited with non-zero code"
        )


def _collect_missing_metadata(node: GriffeObjectLike, missing: set[str]) -> None:
    """Record symbols missing badge metadata under ``node`` into ``missing``."""
    for child in iter_public_members(node):
        kind = child.kind.value if child.kind else ""
        if kind in KINDS:
            qname = child.path
            badge = badges_for(qname)
            if not badge.stability or not badge.owner:
                missing.add(qname)
        if kind in {"module", "package"}:
            _collect_missing_metadata(child, missing)


def _ensure_packages_selected(packages: Sequence[str]) -> None:
    """Exit when no packages are available for processing."""
    if packages:
        return
    message = "No packages detected; set DOCS_PKG or add packages under src/."
    raise SystemExit(message)


def _warn_missing_inputs() -> None:
    """Emit warnings when auxiliary metadata files are missing."""
    if not NAVMAP_PATH.exists():
        with_fields(LOGGER, path=str(NAVMAP_PATH)).warning("NavMap not found; badges will be empty")
    if not TESTMAP_PATH.exists():
        with_fields(LOGGER, path=str(TESTMAP_PATH)).warning(
            "Test map not found; tested-by badges will be empty"
        )


def _process_module(module: GriffeObjectLike, cfg: ReadmeConfig, missing_meta: set[str]) -> bool:
    """Render README files for ``module`` and its package members."""
    changed = False
    if cfg.fail_on_metadata_miss:
        _collect_missing_metadata(module, missing_meta)
    changed |= write_readme(module, cfg)

    members = module.members
    if not members:
        return changed

    for member in members.values():
        if not member.is_package:
            continue
        if cfg.fail_on_metadata_miss:
            _collect_missing_metadata(member, missing_meta)
        changed |= write_readme(member, cfg)
    return changed


def _report_duration(start: float, changed_any: bool) -> None:
    """Print a timing summary when verbose mode is enabled."""
    duration = time.monotonic() - start
    with_fields(LOGGER, duration_seconds=round(duration, 2), changed=changed_any).info(
        "README generation completed"
    )


def _log_problem_and_exit(problem: ProblemDetailsDict, exit_code: int) -> NoReturn:
    """Log ``problem`` using structured context before exiting."""
    with_fields(
        LOGGER,
        problemType=problem["type"],
        status=problem.get("status"),
        problem=problem,
    ).error(problem["detail"])
    raise SystemExit(exit_code)


def _raise_missing_metadata(detail: str, problem: ProblemDetailsDict) -> NoReturn:
    """Raise ``MissingMetadataError`` with the provided context."""
    raise MissingMetadataError(detail, problem=problem)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate README files for configured packages."""
    try:
        cfg = parse_config(argv)
        _ensure_packages_selected(cfg.packages)
        _warn_missing_inputs()

        loader: LoaderInstance = GriffeLoader(search_paths=(str(SRC if SRC.exists() else ROOT),))
        missing_meta: set[str] = set()
        changed_any = False
        start = time.monotonic()

        for pkg in cfg.packages:
            module = loader.load(pkg)
            changed_any |= _process_module(module, cfg, missing_meta)

        if cfg.fail_on_metadata_miss and missing_meta:
            detail = "Public symbols are missing owner or stability metadata"
            problem = build_problem_details(
                type="https://kgfoundry.dev/problems/readme-metadata-missing",
                title="Missing badge metadata",
                status=422,
                detail=detail,
                instance="urn:tool:gen-readmes:missing-metadata",
                extensions={
                    "packages": list(cfg.packages),
                    "symbols": sorted(missing_meta),
                },
            )
            _raise_missing_metadata(detail, problem)

        if cfg.verbose:
            _report_duration(start, changed_any)

    except MissingMetadataError as exc:
        problem = exc.problem or build_problem_details(
            type="https://kgfoundry.dev/problems/readme-metadata-missing",
            title="Missing badge metadata",
            status=422,
            detail=str(exc),
            instance="urn:tool:gen-readmes:missing-metadata",
        )
        _log_problem_and_exit(problem, exit_code=2)
    except ReadmeGenerationError as exc:
        problem = exc.problem or build_problem_details(
            type="https://kgfoundry.dev/problems/readme-generation-error",
            title="README generation failed",
            status=500,
            detail=str(exc),
            instance="urn:tool:gen-readmes:failure",
        )
        _log_problem_and_exit(problem, exit_code=1)


if __name__ == "__main__":
    main()
