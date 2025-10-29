"""Command line interface for the docstring builder."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from tools.docstring_builder.apply import apply_edits
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config import BuilderConfig, load_config_from_env
from tools.docstring_builder.docfacts import DocFact, build_docfacts, write_docfacts
from tools.docstring_builder.harvest import HarvestResult, harvest_file, iter_target_files
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
        path
        for path in iter_target_files(config, REPO_ROOT)
        if not _should_ignore(path, config)
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


def _collect_edits(
    result: HarvestResult, config: BuilderConfig
) -> tuple[list[DocstringEdit], list[SemanticResult]]:
    semantics = build_semantic_schemas(result, config)
    edits = [
        DocstringEdit(
            qname=entry.symbol.qname,
            text=render_docstring(entry.schema, config.ownership_marker),
        )
        for entry in semantics
    ]
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


def _process_file(
    file_path: Path,
    config: BuilderConfig,
    cache: BuilderCache,
    *,
    command: str,
    force: bool,
) -> tuple[int, list[DocFact], str | None, bool]:
    """Harvest, render, and apply docstrings for a single file."""
    is_update = command == "update"
    is_check = command == "check"
    if command != "harvest" and not force and not cache.needs_update(file_path, config.config_hash):
        LOGGER.debug("Skipping %s; cache is fresh", file_path)
        return 0, [], None, False
    try:
        result = harvest_file(file_path, config, REPO_ROOT)
    except Exception:  # pragma: no cover - runtime defensive handling
        LOGGER.exception("Failed to harvest %s", file_path)
        return 1, [], None, False
    edits, semantics = _collect_edits(result, config)
    if command == "harvest":
        docfacts = build_docfacts(semantics)
        return 0, docfacts, None, False
    if not semantics:
        if is_update:
            cache.update(file_path, config.config_hash)
        return 0, [], None, False
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
    docfact_entries = _load_docfacts_from_disk()
    docfact_sources: dict[str, Path] = {}
    for qname, fact in docfact_entries.items():
        source = _module_to_path(fact.module)
        if source is not None:
            docfact_sources[qname] = source
    is_update = args.command == "update"
    is_check = args.command == "check"
    exit_code = 0
    for file_path in files:
        file_exit, docfacts, preview, changed = _process_file(
            file_path,
            config,
            cache,
            command=args.command or "",
            force=args.force,
        )
        exit_code = max(exit_code, file_exit)
        if is_check and changed and args.diff:
            sys.stdout.write(preview or "")
        for fact in docfacts:
            docfact_entries[fact.qname] = fact
            docfact_sources[fact.qname] = file_path
    if args.command in {"update", "check"}:
        filtered: list[DocFact] = []
        for qname, fact in docfact_entries.items():
            source = docfact_sources.get(qname) or _module_to_path(fact.module)
            if source is not None and _should_ignore(source, config):
                LOGGER.debug("Dropping docfact %s due to ignore rules", qname)
                continue
            filtered.append(fact)
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


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the docstring builder CLI."""
    parser = argparse.ArgumentParser(prog="docstring-builder")
    parser.add_argument("--module", help="Restrict to module prefix", default="")
    parser.add_argument("--since", help="Only consider files changed since revision", default="")
    parser.add_argument("--force", action="store_true", help="Ignore cache entries")
    parser.add_argument("--diff", action="store_true", help="Show diffs in check mode")
    parser.add_argument("--all", action="store_true", help="Process all files, ignoring cache entries")
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
    if getattr(args, "all", False):
        args.force = True
    if getattr(args, "flag_diff", False):
        args.diff = True
        args.flag_check = True
    if getattr(args, "flag_harvest", False):
        args.func = _command_harvest
    elif getattr(args, "flag_update", False):
        args.func = _command_update
    elif getattr(args, "flag_check", False):
        args.func = _command_check
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
