"""Command line interface for the docstring builder."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .apply import apply_edits
from .cache import BuilderCache
from .config import BuilderConfig, load_config_from_env
from .docfacts import DocFact, build_docfacts, write_docfacts
from .harvest import HarvestResult, harvest_file, iter_target_files
from .render import render_docstring
from .schema import DocstringEdit
from .semantics import SemanticResult, build_semantic_schemas

LOGGER = logging.getLogger("docstring_builder")
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / ".cache" / "docstring_builder.json"
DOCFACTS_PATH = REPO_ROOT / "docs" / "_build" / "docfacts.json"


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

    files = list(iter_target_files(config, REPO_ROOT))
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
    return files


def _changed_files_since(revision: str) -> set[str]:
    cmd = ["git", "-C", str(REPO_ROOT), "diff", "--name-only", revision, "HEAD", "--"]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        LOGGER.warning("git diff failed: %s", result.stderr.strip())
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _collect_edits(result: HarvestResult, config: BuilderConfig) -> tuple[list[DocstringEdit], list[SemanticResult]]:
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
        current = [dataclasses.asdict(fact) for fact in sorted(docfacts, key=lambda fact: fact.qname)]
        if existing != current:
            LOGGER.error("DocFacts drift detected; run update mode to refresh.")
            return True
        return False
    ordered = sorted(docfacts, key=lambda fact: fact.qname)
    write_docfacts(DOCFACTS_PATH, ordered)
    return False


def _run(files: Iterable[Path], args: argparse.Namespace, config: BuilderConfig) -> int:
    cache = BuilderCache(CACHE_PATH)
    docfact_entries: dict[str, DocFact] = {}
    if DOCFACTS_PATH.exists():
        try:
            existing_data = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_data = []
        for item in existing_data:
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
            docfact_entries[fact.qname] = fact
    exit_code = 0
    for file_path in files:
        if not args.force and not cache.needs_update(file_path, config.config_hash):
            LOGGER.debug("Skipping %s; cache is fresh", file_path)
            continue
        try:
            result = harvest_file(file_path, config, REPO_ROOT)
        except Exception as exc:  # pragma: no cover - runtime defensive handling
            LOGGER.error("Failed to harvest %s: %s", file_path, exc)
            exit_code = 1
            continue
        edits, semantics = _collect_edits(result, config)
        if not semantics:
            if args.command == "update":
                cache.update(file_path, config.config_hash)
            continue
        changed, preview = apply_edits(result, edits, write=args.command == "update")
        if args.command == "check" and changed:
            LOGGER.error("Docstrings out of date in %s", file_path.relative_to(REPO_ROOT))
            exit_code = 1
        if args.command == "check" and changed and args.diff:
            sys.stdout.write(preview or "")
        if args.command == "update":
            cache.update(file_path, config.config_hash)
        for fact in build_docfacts(semantics):
            docfact_entries[fact.qname] = fact
    if args.command in {"update", "check"}:
        drift = _handle_docfacts(list(docfact_entries.values()), check_mode=args.command == "check")
        if drift:
            exit_code = 1
    if args.command == "update":
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
    BuilderCache(CACHE_PATH).clear()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docstring-builder")
    parser.add_argument("--module", help="Restrict to module prefix", default="")
    parser.add_argument("--since", help="Only consider files changed since revision", default="")
    parser.add_argument("--force", action="store_true", help="Ignore cache entries")
    parser.add_argument("--diff", action="store_true", help="Show diffs in check mode")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
