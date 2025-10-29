"""Command line interface for the docstring builder."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from tools.docstring_builder.apply import apply_edits
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config import BuilderConfig, load_config_from_env
from tools.docstring_builder.docfacts import DocFact, build_docfacts, write_docfacts
from tools.docstring_builder.harvest import HarvestResult, harvest_file, iter_target_files
from tools.docstring_builder.normalizer import normalize_docstring
from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult, build_semantic_schemas

LOGGER = logging.getLogger("docstring_builder")
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / ".cache" / "docstring_builder.json"
DOCFACTS_PATH = REPO_ROOT / "docs" / "_build" / "docfacts.json"
DEFAULT_IGNORE_PATTERNS = [
    "tests/e2e/**",
    "tests/mock_servers/**",
    "tests/tools/**",
    "docs/_scripts/**",
    "docs/conf.py",
    "src/__init__.py",
]
MISSING_MODULE_PATTERNS = ["docs/_build/**"]


@dataclasses.dataclass(slots=True)
class ProcessingOptions:
    """Runtime options controlling how a file is processed."""

    command: str
    force: bool
    ignore_missing: bool
    missing_patterns: tuple[str, ...]


def _module_to_path(module: str) -> Path | None:
    if not module:
        return None
    parts = module.split(".")
    relative = Path("src", *parts)
    file_candidate = REPO_ROOT / relative
    if file_candidate.suffix:
        # Already points to a concrete file (e.g., pkg.module).
        return file_candidate
    file_path = file_candidate.with_suffix(".py")
    if file_path.exists():
        return file_path
    package_init = file_candidate / "__init__.py"
    if package_init.exists():
        return package_init
    return file_path


def _resolve_ignore_patterns(config: BuilderConfig) -> list[str]:
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    for pattern in config.ignore:
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def _select_files(config: BuilderConfig, args: argparse.Namespace) -> Iterable[Path]:
    if getattr(args, "paths", None):
        selected: list[Path] = []
        for raw in args.paths:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            if candidate.suffix != ".py" or not candidate.exists():
                continue
            selected.append(candidate)
        return selected

    files = [
        path for path in iter_target_files(config, REPO_ROOT) if not _should_ignore(path, config)
    ]
    if args.module:
        filtered: list[Path] = []
        for candidate in files:
            parts = list(candidate.relative_to(REPO_ROOT).with_suffix("").parts)
            if parts and parts[0] in {"src", "tools", "docs"}:
                parts = parts[1:]
            module_name = ".".join(parts)
            if module_name.startswith(args.module):
                filtered.append(candidate)
        files = filtered
    if args.since:
        changed = set(_changed_files_since(args.since))
        files = [path for path in files if str(path.relative_to(REPO_ROOT)) in changed]
    return [file_path for file_path in files if not _should_ignore(file_path, config)]


def _matches_patterns(path: Path, patterns: Iterable[str]) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:  # pragma: no cover - defensive guard
        rel = path
    return any(rel.match(pattern) for pattern in patterns)


def _should_ignore(path: Path, config: BuilderConfig) -> bool:
    rel = path.relative_to(REPO_ROOT)
    patterns = _resolve_ignore_patterns(config)
    for pattern in patterns:
        if rel.match(pattern):
            LOGGER.debug("Skipping %s because it matches ignore pattern %s", rel, pattern)
            return True
    return False


def _changed_files_since(revision: str) -> set[str]:
    cmd = ["git", "-C", str(REPO_ROOT), "diff", "--name-only", revision, "HEAD", "--"]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        LOGGER.warning("git diff failed: %s", result.stderr.strip())
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _legacy_command_from_flags(args: argparse.Namespace) -> str | None:
    for attr, command in (
        ("diff", "check"),
        ("flag_check", "check"),
        ("flag_update", "update"),
        ("flag_harvest", "harvest"),
    ):
        if getattr(args, attr, False):
            return command
    if getattr(args, "since", "") or getattr(args, "module", ""):
        return "update"
    if getattr(args, "all", False) or getattr(args, "force", False):
        return "update"
    return None


def _assign_command(args: argparse.Namespace) -> None:
    """Normalise legacy flags and attach the appropriate handler."""
    if getattr(args, "all", False):
        args.force = True
    if getattr(args, "flag_diff", False):
        args.diff = True
        args.flag_check = True
    if hasattr(args, "func"):
        return

    for attr, handler in (
        ("flag_harvest", _command_harvest),
        ("flag_update", _command_update),
        ("flag_check", _command_check),
    ):
        if getattr(args, attr, False):
            args.func = handler
            return

    legacy_command = _legacy_command_from_flags(args)
    if legacy_command is not None:
        args.func = LEGACY_COMMAND_HANDLERS[legacy_command]


def _collect_edits(
    result: HarvestResult, config: BuilderConfig
) -> tuple[list[DocstringEdit], list[SemanticResult]]:
    semantics = build_semantic_schemas(result, config)
    edits: list[DocstringEdit] = []
    for entry in semantics:
        text: str
        if config.normalize_sections:
            normalized = normalize_docstring(entry.symbol, config.ownership_marker)
            if normalized is not None:
                text = normalized
            else:
                text = render_docstring(entry.schema, config.ownership_marker)
        else:
            text = render_docstring(entry.schema, config.ownership_marker)
        edits.append(DocstringEdit(qname=entry.symbol.qname, text=text))
    return edits, semantics


def _handle_docfacts(docfacts: list[DocFact], check_mode: bool) -> bool:
    if check_mode:
        if not DOCFACTS_PATH.exists():
            LOGGER.error("DocFacts missing at %s", DOCFACTS_PATH)
            return True
        existing = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
        current = [
            dataclasses.asdict(fact) for fact in sorted(docfacts, key=lambda fact: fact.qname)
        ]
        if existing != current:
            LOGGER.error("DocFacts drift detected; run update mode to refresh.")
            return True
        return False
    ordered = sorted(docfacts, key=lambda fact: fact.qname)
    write_docfacts(DOCFACTS_PATH, ordered)
    return False


def _load_docfacts_from_disk() -> dict[str, DocFact]:
    """Load previously generated DocFacts entries keyed by qualified name."""
    if not DOCFACTS_PATH.exists():
        return {}
    try:
        raw = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("DocFacts cache is not valid JSON; ignoring existing data.")
        return {}
    entries: dict[str, DocFact] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            fact = DocFact(
                qname=item["qname"],
                module=item.get("module", ""),
                kind=item.get("kind", "function"),
                parameters=item.get("parameters", []),
                returns=item.get("returns", []),
                raises=item.get("raises", []),
                notes=item.get("notes", []),
            )
        except KeyError:
            continue
        entries[fact.qname] = fact
    return entries


def _load_docfact_state(config: BuilderConfig) -> tuple[dict[str, DocFact], dict[str, Path]]:
    """Load docfact entries along with best-effort source mapping."""
    entries = _load_docfacts_from_disk()
    sources: dict[str, Path] = {}
    for qname, fact in entries.items():
        source = _module_to_path(fact.module)
        if source is not None:
            sources[qname] = source
    return entries, sources


def _record_docfacts(
    facts: Iterable[DocFact],
    file_path: Path,
    entries: dict[str, DocFact],
    sources: dict[str, Path],
) -> None:
    for fact in facts:
        entries[fact.qname] = fact
        sources[fact.qname] = file_path


def _filter_docfacts_for_output(
    entries: dict[str, DocFact], sources: dict[str, Path], config: BuilderConfig
) -> list[DocFact]:
    filtered: list[DocFact] = []
    for qname, fact in entries.items():
        source = sources.get(qname) or _module_to_path(fact.module)
        if source is not None and _should_ignore(source, config):
            LOGGER.debug("Dropping docfact %s due to ignore rules", qname)
            continue
        filtered.append(fact)
    return filtered


def _process_file(
    file_path: Path,
    config: BuilderConfig,
    cache: BuilderCache,
    options: ProcessingOptions,
) -> tuple[int, list[DocFact], str | None, bool]:
    """Harvest, render, and apply docstrings for a single file."""
    command = options.command
    is_update = command == "update"
    is_check = command == "check"
    exit_code = 0
    docfacts: list[DocFact] = []
    preview: str | None = None
    changed = False
    if (
        command != "harvest"
        and not options.force
        and not cache.needs_update(file_path, config.config_hash)
    ):
        LOGGER.debug("Skipping %s; cache is fresh", file_path)
        return exit_code, docfacts, preview, changed
    try:
        result = harvest_file(file_path, config, REPO_ROOT)
    except ModuleNotFoundError as exc:
        relative = file_path.relative_to(REPO_ROOT)
        if options.ignore_missing and _matches_patterns(file_path, options.missing_patterns):
            LOGGER.info("Skipping %s due to missing dependency: %s", relative, exc)
            return exit_code, docfacts, preview, changed
        LOGGER.exception("Failed to harvest %s", relative)
        return 1, docfacts, preview, changed
    except Exception:  # pragma: no cover - runtime defensive handling
        LOGGER.exception("Failed to harvest %s", file_path)
        return 1, docfacts, preview, changed
    edits, semantics = _collect_edits(result, config)
    if command == "harvest":
        docfacts = build_docfacts(semantics)
    else:
        if not semantics:
            if is_update:
                cache.update(file_path, config.config_hash)
            return exit_code, docfacts, preview, changed
        changed, preview = apply_edits(result, edits, write=is_update)
        if is_check and changed:
            relative = file_path.relative_to(REPO_ROOT)
            LOGGER.error("Docstrings out of date in %s", relative)
        if is_update:
            cache.update(file_path, config.config_hash)
        docfacts = build_docfacts(semantics)
        exit_code = 1 if is_check and changed else 0
    return exit_code, docfacts, preview, changed


def _run(files: Iterable[Path], args: argparse.Namespace, config: BuilderConfig) -> int:
    cache = BuilderCache(CACHE_PATH)
    docfact_entries, docfact_sources = _load_docfact_state(config)
    is_update = args.command == "update"
    is_check = args.command == "check"
    exit_code = 0
    options = ProcessingOptions(
        command=args.command or "",
        force=args.force,
        ignore_missing=getattr(args, "ignore_missing", False),
        missing_patterns=tuple(MISSING_MODULE_PATTERNS),
    )
    for file_path in files:
        file_exit, docfacts, preview, changed = _process_file(
            file_path,
            config,
            cache,
            options,
        )
        exit_code = max(exit_code, file_exit)
        if is_check and changed and args.diff:
            sys.stdout.write(preview or "")
        _record_docfacts(docfacts, file_path, docfact_entries, docfact_sources)
    if args.command in {"update", "check"}:
        filtered = _filter_docfacts_for_output(docfact_entries, docfact_sources, config)
        drift = _handle_docfacts(filtered, check_mode=is_check)
        if drift:
            exit_code = 1
    if is_update:
        cache.write()
    return exit_code


def _command_update(args: argparse.Namespace) -> int:
    config = load_config_from_env()
    files = _select_files(config, args)
    args.command = "update"
    return _run(files, args, config)


def _command_check(args: argparse.Namespace) -> int:
    config = load_config_from_env()
    files = _select_files(config, args)
    args.command = "check"
    return _run(files, args, config)


def _command_harvest(args: argparse.Namespace) -> int:
    config = load_config_from_env()
    files = _select_files(config, args)
    args.command = "harvest"
    args.force = True
    return _run(files, args, config)


def _command_list(args: argparse.Namespace) -> int:
    config = load_config_from_env()
    files = _select_files(config, args)
    for file_path in files:
        result = harvest_file(file_path, config, REPO_ROOT)
        for symbol in result.symbols:
            if symbol.owned:
                print(symbol.qname)
    return 0


def _command_clear_cache(_: argparse.Namespace) -> int:
    """Remove any cached docstring builder metadata."""
    BuilderCache(CACHE_PATH).clear()
    return 0


LEGACY_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "update": _command_update,
    "check": _command_check,
    "harvest": _command_harvest,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the docstring builder CLI."""
    parser = argparse.ArgumentParser(prog="docstring-builder")
    parser.add_argument("--module", help="Restrict to module prefix", default="")
    parser.add_argument("--since", help="Only consider files changed since revision", default="")
    parser.add_argument("--force", action="store_true", help="Ignore cache entries")
    parser.add_argument("--diff", action="store_true", help="Show diffs in check mode")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip modules that raise ModuleNotFoundError (e.g., docs/_build artefacts)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all files, ignoring cache entries"
    )
    parser.add_argument(
        "--update",
        dest="flag_update",
        action="store_true",
        help="Legacy flag: run in update mode",
    )
    parser.add_argument(
        "--check",
        dest="flag_check",
        action="store_true",
        help="Legacy flag: run in check mode",
    )
    parser.add_argument(
        "--harvest",
        dest="flag_harvest",
        action="store_true",
        help="Legacy flag: harvest symbols without writing",
    )
    parser.add_argument(
        "--diff-only",
        dest="flag_diff",
        action="store_true",
        help="Legacy flag: run check mode and show diffs",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    update = subparsers.add_parser("update", help="Synchronize docstrings")
    update.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    update.set_defaults(func=_command_update)

    check = subparsers.add_parser("check", help="Validate docstrings without writing")
    check.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    check.set_defaults(func=_command_check)

    list_cmd = subparsers.add_parser("list", help="List managed symbols")
    list_cmd.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    list_cmd.set_defaults(func=_command_list)

    clear = subparsers.add_parser("clear-cache", help="Clear the builder cache")
    clear.set_defaults(func=_command_clear_cache)
    harvest = subparsers.add_parser("harvest", help="Harvest metadata without applying edits")
    harvest.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    harvest.set_defaults(func=_command_harvest)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the docstring builder CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _assign_command(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
