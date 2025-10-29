"""Command line interface for the docstring builder."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import enum
import hashlib
import importlib
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml
from tools.docstring_builder.apply import apply_edits
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config import BuilderConfig, load_config_from_env
from tools.docstring_builder.docfacts import DocFact, build_docfacts, write_docfacts
from tools.docstring_builder.harvest import HarvestResult, harvest_file, iter_target_files
from tools.docstring_builder.ir import IRDocstring, IR_VERSION, build_ir, validate_ir, write_schema
from tools.docstring_builder.normalizer import normalize_docstring
from tools.docstring_builder.plugins import (
    PluginConfigurationError,
    PluginManager,
    load_plugins,
)
from tools.docstring_builder.policy import (
    PolicyConfigurationError,
    PolicyEngine,
    PolicyReport,
    load_policy_settings,
)
from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.schema import DocstringEdit
from tools.docstring_builder.semantics import SemanticResult, build_semantic_schemas

LOGGER = logging.getLogger("docstring_builder")
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / ".cache" / "docstring_builder.json"
DOCFACTS_PATH = REPO_ROOT / "docs" / "_build" / "docfacts.json"
MANIFEST_PATH = REPO_ROOT / "docs" / "_build" / "docstrings_manifest.json"
OBSERVABILITY_PATH = REPO_ROOT / "docs" / "_build" / "observability_docstrings.json"
OBSERVABILITY_MAX_ERRORS = 20
DEFAULT_IGNORE_PATTERNS = [
    "tests/e2e/**",
    "tests/mock_servers/**",
    "tests/tools/**",
    "docs/_scripts/**",
    "docs/conf.py",
    "src/__init__.py",
]
MISSING_MODULE_PATTERNS = ["docs/_build/**"]

CommandHandler = Callable[[argparse.Namespace], int]


class ExitStatus(enum.IntEnum):
    """Standardised exit codes for CLI subcommands."""

    SUCCESS = 0
    VIOLATION = 1
    CONFIG = 2
    ERROR = 3


STATUS_LABELS = {
    ExitStatus.SUCCESS: "success",
    ExitStatus.VIOLATION: "violation",
    ExitStatus.CONFIG: "config",
    ExitStatus.ERROR: "error",
}


@dataclasses.dataclass(slots=True)
class ProcessingOptions:
    """Runtime options controlling how a file is processed."""

    command: str
    force: bool
    ignore_missing: bool
    missing_patterns: tuple[str, ...]
    skip_docfacts: bool


@dataclasses.dataclass(slots=True)
class FileOutcome:
    """Result of processing a single file."""

    status: ExitStatus
    docfacts: list[DocFact]
    preview: str | None
    changed: bool
    skipped: bool
    message: str | None = None
    semantics: list[SemanticResult] = dataclasses.field(default_factory=list)
    ir: list[IRDocstring] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class DocfactsOutcome:
    """Outcome of reconciling DocFacts artifacts."""

    status: ExitStatus
    message: str | None = None


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


def _parse_policy_overrides(values: Sequence[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not values:
        return overrides
    for raw in values:
        for chunk in raw.split(","):
            token = chunk.strip()
            if not token:
                continue
            if "=" not in token:
                raise PolicyConfigurationError(f"Invalid policy override '{token}'")
            key, value = token.split("=", 1)
            overrides[key.strip().lower()] = value.strip()
    return overrides


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _parse_plugin_names(values: Sequence[str] | None) -> list[str]:
    names: list[str] = []
    if not values:
        return names
    for raw in values:
        for chunk in raw.split(","):
            name = chunk.strip()
            if name:
                names.append(name)
    return names


def _dependents_for(path: Path) -> set[Path]:
    dependents: set[Path] = set()
    if path.name == "__init__.py":
        for candidate in path.parent.glob("*.py"):
            if candidate != path and candidate.is_file():
                dependents.add(candidate.resolve())
    else:
        init_file = (path.parent / "__init__.py").resolve()
        if init_file.exists():
            dependents.add(init_file)
    return dependents


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
    candidates = [file_path for file_path in files if not _should_ignore(file_path, config)]
    if getattr(args, "changed_only", False) or args.since or getattr(args, "paths", None):
        expanded: dict[Path, None] = {candidate.resolve(): None for candidate in candidates}
        for candidate in list(expanded.keys()):
            for dependent in _dependents_for(candidate):
                if not _should_ignore(dependent, config):
                    expanded.setdefault(dependent, None)
        candidates = sorted(expanded.keys())
    return candidates


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
    if getattr(args, "func", None):
        return

    for attr, handler in (
        ("flag_harvest", _command_harvest),
        ("flag_update", _command_update),
        ("flag_check", _command_check),
    ):
        if getattr(args, attr, False):
            setattr(args, "func", cast(CommandHandler, handler))
            return

    legacy_command = _legacy_command_from_flags(args)
    if legacy_command is not None:
        legacy_handler: CommandHandler = LEGACY_COMMAND_HANDLERS[legacy_command]
        setattr(
            args,
            "func",
            cast(
                CommandHandler,
                getattr(sys.modules[__name__], legacy_handler.__name__),
            ),
        )


def _collect_edits(
    result: HarvestResult,
    config: BuilderConfig,
    plugin_manager: PluginManager | None,
) -> tuple[list[DocstringEdit], list[SemanticResult], list[IRDocstring]]:
    semantics = build_semantic_schemas(result, config)
    if plugin_manager is not None:
        semantics = plugin_manager.apply_transformers(result.filepath, semantics)
    edits: list[DocstringEdit] = []
    ir_entries: list[IRDocstring] = []
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
        ir_entry = build_ir(entry)
        validate_ir(ir_entry)
        ir_entries.append(ir_entry)
    if plugin_manager is not None:
        edits = plugin_manager.apply_formatters(result.filepath, edits)
    return edits, semantics, ir_entries


def _handle_docfacts(docfacts: list[DocFact], check_mode: bool) -> DocfactsOutcome:
    if check_mode:
        if not DOCFACTS_PATH.exists():
            LOGGER.error("DocFacts missing at %s", DOCFACTS_PATH)
            return DocfactsOutcome(ExitStatus.CONFIG, "docfacts missing")
        existing = json.loads(DOCFACTS_PATH.read_text(encoding="utf-8"))
        current = [
            dataclasses.asdict(fact) for fact in sorted(docfacts, key=lambda fact: fact.qname)
        ]
        if existing != current:
            LOGGER.error("DocFacts drift detected; run update mode to refresh.")
            return DocfactsOutcome(ExitStatus.VIOLATION, "docfacts drift")
        return DocfactsOutcome(ExitStatus.SUCCESS)
    ordered = sorted(docfacts, key=lambda fact: fact.qname)
    write_docfacts(DOCFACTS_PATH, ordered)
    return DocfactsOutcome(ExitStatus.SUCCESS)


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


def _default_since_revision() -> str | None:
    """Return a sensible default revision for ``--changed-only`` runs."""
    candidates = [
        ["git", "-C", str(REPO_ROOT), "merge-base", "HEAD", "origin/main"],
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD~1"],
    ]
    for cmd in candidates:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if result.returncode == 0:
            revision = result.stdout.strip()
            if revision:
                return revision
    return None


def _process_file(
    file_path: Path,
    config: BuilderConfig,
    cache: BuilderCache,
    options: ProcessingOptions,
    plugin_manager: PluginManager | None,
) -> FileOutcome:  # noqa: PLR0911
    """Harvest, render, and apply docstrings for a single file."""
    command = options.command
    is_update = command == "update"
    is_check = command == "check"
    docfacts: list[DocFact] = []
    preview: str | None = None
    changed = False
    skipped = False
    message: str | None = None
    if (
        command != "harvest"
        and not options.force
        and not cache.needs_update(file_path, config.config_hash)
    ):
        LOGGER.debug("Skipping %s; cache is fresh", file_path)
        skipped = True
        message = "cache fresh"
        return FileOutcome(ExitStatus.SUCCESS, docfacts, preview, changed, skipped, message)
    try:
        result = harvest_file(file_path, config, REPO_ROOT)
        if plugin_manager is not None:
            result = plugin_manager.apply_harvest(file_path, result)
    except ModuleNotFoundError as exc:
        relative = file_path.relative_to(REPO_ROOT)
        message = f"missing dependency: {exc}"
        if options.ignore_missing and _matches_patterns(file_path, options.missing_patterns):
            LOGGER.info("Skipping %s due to missing dependency: %s", relative, exc)
            return FileOutcome(ExitStatus.SUCCESS, docfacts, preview, False, True, message)
        LOGGER.exception("Failed to harvest %s", relative)
        return FileOutcome(ExitStatus.CONFIG, docfacts, preview, changed, skipped, message)
    except Exception as exc:  # pragma: no cover - runtime defensive handling
        LOGGER.exception("Failed to harvest %s", file_path)
        return FileOutcome(ExitStatus.ERROR, docfacts, preview, changed, skipped, str(exc))

    edits, semantics, ir_entries = _collect_edits(result, config, plugin_manager)
    if command == "harvest":
        docfacts = build_docfacts(semantics)
        return FileOutcome(
            ExitStatus.SUCCESS,
            docfacts,
            preview,
            changed,
            skipped,
            message,
            semantics=list(semantics),
            ir=ir_entries,
        )

    if not semantics:
        if is_update:
            cache.update(file_path, config.config_hash)
        message = "no managed symbols"
        return FileOutcome(
            ExitStatus.SUCCESS,
            docfacts,
            preview,
            changed,
            skipped,
            message,
            semantics=list(semantics),
            ir=ir_entries,
        )

    changed, preview = apply_edits(result, edits, write=is_update)
    status = ExitStatus.SUCCESS
    if is_check and changed:
        relative = file_path.relative_to(REPO_ROOT)
        LOGGER.error("Docstrings out of date in %s", relative)
        status = ExitStatus.VIOLATION
        message = "docstrings drift"
    if is_update:
        cache.update(file_path, config.config_hash)
    docfacts = build_docfacts(semantics)
    return FileOutcome(
        status,
        docfacts,
        preview,
        changed,
        skipped,
        message,
        semantics=list(semantics),
        ir=ir_entries,
    )


def _print_failure_summary(payload: Mapping[str, object]) -> None:  # noqa: C901
    """Emit a concise summary to stderr when the CLI exits non-zero."""
    summary_obj = payload.get("summary")
    summary = summary_obj if isinstance(summary_obj, Mapping) else {}

    errors_obj = payload.get("errors")
    if isinstance(errors_obj, Sequence):
        error_entries = [entry for entry in errors_obj if isinstance(entry, Mapping)]
    else:
        error_entries = []

    def _coerce_int(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    def _coerce_str(value: object, fallback: str) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return fallback
        return str(value)

    considered = _coerce_int(summary.get("considered"))
    processed = _coerce_int(summary.get("processed"))
    changed = _coerce_int(summary.get("changed"))

    status_counts_obj = summary.get("status_counts")
    if isinstance(status_counts_obj, Mapping):
        status_counts = {str(key): _coerce_int(value) for key, value in status_counts_obj.items()}
    else:
        status_counts = {}

    lines = [
        "[SUMMARY] Docstring builder reported issues.",
        f"  Considered files: {considered}",
        f"  Processed files: {processed}",
        f"  Changed files: {changed}",
        f"  Status counts: {status_counts}",
        f"  Observability log: {OBSERVABILITY_PATH}",
    ]
    if error_entries:
        lines.append("  Top errors:")
        for entry in error_entries[:5]:
            file_name = _coerce_str(entry.get("file"), "<unknown>")
            status = _coerce_str(entry.get("status"), "unknown")
            message = _coerce_str(entry.get("message"), "no additional details")
            lines.append(f"    - {file_name}: {status} ({message})")
    for line in lines:
        print(line, file=sys.stderr)


def _run(
    files: Iterable[Path], args: argparse.Namespace, config: BuilderConfig
) -> int:  # noqa: C901, PLR0915
    cache = BuilderCache(CACHE_PATH)
    docfact_entries, docfact_sources = _load_docfact_state(config)
    is_update = args.command == "update"
    is_check = args.command == "check"
    start = time.perf_counter()
    files_list = list(files)
    status_counts: Counter[ExitStatus] = Counter()
    processed_count = 0
    skipped_count = 0
    changed_count = 0
    errors: list[dict[str, str]] = []
    options = ProcessingOptions(
        command=args.command or "",
        force=args.force,
        ignore_missing=getattr(args, "ignore_missing", False),
        missing_patterns=tuple(MISSING_MODULE_PATTERNS),
        skip_docfacts=getattr(args, "skip_docfacts", False),
    )
    try:
        plugin_manager = load_plugins(
    for file_path in files_list:
        outcome = _process_file(
            file_path,
            config,
            REPO_ROOT,
            only=_parse_plugin_names(getattr(args, "only_plugin", None)),
            disable=_parse_plugin_names(getattr(args, "disable_plugin", None)),
        )
    except PluginConfigurationError as exc:
        LOGGER.error("Plugin configuration error: %s", exc)
        return int(ExitStatus.CONFIG)
    try:
        try:
            cli_policy_overrides = _parse_policy_overrides(getattr(args, "policy_override", None))
        except PolicyConfigurationError as exc:
            LOGGER.error("Policy override error: %s", exc)
            return int(ExitStatus.CONFIG)
        try:
            policy_settings = load_policy_settings(REPO_ROOT, cli_overrides=cli_policy_overrides)
        except PolicyConfigurationError as exc:
            LOGGER.error("Policy configuration error: %s", exc)
            return int(ExitStatus.CONFIG)
        policy_engine = PolicyEngine(policy_settings)
        all_ir: list[IRDocstring] = []
        docfacts_checked = False

        for file_path in files_list:
            outcome = _process_file(
                file_path,
                config,
                cache,
                options,
                plugin_manager,
            )
            status_counts[outcome.status] += 1
            if outcome.skipped:
                skipped_count += 1
            else:
                processed_count += 1
            if outcome.changed:
                changed_count += 1
            if is_check and outcome.changed and args.diff:
                sys.stdout.write(outcome.preview or "")
            if outcome.status is not ExitStatus.SUCCESS:
                rel = str(file_path.relative_to(REPO_ROOT))
                errors.append(
                    {
                        "file": rel,
                        "status": STATUS_LABELS[outcome.status],
                        "message": outcome.message or "",
                    }
                )
            _record_docfacts(outcome.docfacts, file_path, docfact_entries, docfact_sources)
            if outcome.semantics:
                policy_engine.record(outcome.semantics)
            all_ir.extend(outcome.ir)

        if args.command in {"update", "check"} and not options.skip_docfacts:
            filtered = _filter_docfacts_for_output(docfact_entries, docfact_sources, config)
            docfacts_result = _handle_docfacts(filtered, check_mode=is_check)
            docfacts_checked = True
            status_counts[docfacts_result.status] += 1
            if docfacts_result.status is not ExitStatus.SUCCESS:
                errors.append(
                    {
                        "file": "<docfacts>",
                        "status": STATUS_LABELS[docfacts_result.status],
                        "message": docfacts_result.message or "",
                    }
                )
        if is_update:
            cache.write()

        policy_report = policy_engine.finalize()
        for violation in policy_report.violations:
            errors.append(
                {
                    "file": violation.symbol,
                    "status": violation.action,
                    "message": violation.message,
                }
            )
            if violation.fatal:
                status_counts[ExitStatus.VIOLATION] += 1

        duration = time.perf_counter() - start
        exit_status = max(
            (status for status, count in status_counts.items() if count), default=ExitStatus.SUCCESS
        )
        status_counts.setdefault(ExitStatus.SUCCESS, 0)
        status_counts_map: dict[str, int] = {
            STATUS_LABELS[key]: value for key, value in status_counts.items() if value
        }
        cache_payload: dict[str, object] = {
            "path": str(CACHE_PATH),
            "exists": CACHE_PATH.exists(),
            "mtime": None,
        }
        input_hashes: dict[str, dict[str, object]] = {}
        for path in files_list:
            rel = str(path.relative_to(REPO_ROOT))
            if path.exists():
                input_hashes[rel] = {
                    "hash": _hash_file(path),
                    "mtime": datetime.datetime.fromtimestamp(
                        path.stat().st_mtime, tz=datetime.UTC
                    ).isoformat(),
                }
            else:
                input_hashes[rel] = {"hash": "", "mtime": None}

        dependency_map = {
            str(path.relative_to(REPO_ROOT)): [
                str(dependent.relative_to(REPO_ROOT)) for dependent in _dependents_for(path)
            ]
            for path in files_list
        }

        manifest_payload: dict[str, object] = {
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "command": args.command,
            "options": {
                "module": args.module,
                "since": args.since,
                "force": args.force,
                "changed_only": getattr(args, "changed_only", False),
                "skip_docfacts": options.skip_docfacts,
            },
            "counts": {
                "considered": len(files_list),
                "processed": processed_count,
                "skipped": skipped_count,
                "changed": changed_count,
            },
            "status_counts": status_counts_map,
            "config_hash": config.config_hash,
            "docfacts_checked": docfacts_checked,
            "cache": cache_payload,
            "processed_files": [str(path.relative_to(REPO_ROOT)) for path in files_list],
            "duration_seconds": duration,
            "inputs": input_hashes,
            "plugins": {
                "enabled": plugin_manager.enabled_plugins(),
                "available": plugin_manager.available,
                "disabled": plugin_manager.disabled,
                "skipped": plugin_manager.skipped,
            },
            "dependencies": dependency_map,
        }
        schema_path = REPO_ROOT / "docs" / "_build" / "schema_docstrings.json"
        write_schema(schema_path)
        manifest_payload["ir"] = {
            "version": IR_VERSION,
            "schema": str(schema_path.relative_to(REPO_ROOT)),
            "count": len(all_ir),
            "symbols": [entry.symbol_id for entry in all_ir],
        }
        manifest_payload["policy"] = {
            "coverage": policy_report.coverage,
            "threshold": policy_report.threshold,
            "violations": [
                {
                    "rule": violation.rule,
                    "symbol": violation.symbol,
                    "action": violation.action,
                    "message": violation.message,
                }
                for violation in policy_report.violations
            ],
        }
        if CACHE_PATH.exists():
            cache_payload["mtime"] = datetime.datetime.fromtimestamp(
                CACHE_PATH.stat().st_mtime, tz=datetime.UTC
            ).isoformat()
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        if exit_status is not ExitStatus.SUCCESS:
            observability_payload: dict[str, object] = {
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "summary": {
                    "considered": len(files_list),
                    "processed": processed_count,
                    "skipped": skipped_count,
                    "changed": changed_count,
                    "duration_seconds": duration,
                    "status_counts": status_counts_map,
                },
                "errors": errors[:OBSERVABILITY_MAX_ERRORS],
            }
            OBSERVABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
            OBSERVABILITY_PATH.write_text(
                json.dumps(observability_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _print_failure_summary(observability_payload)
        else:
            OBSERVABILITY_PATH.unlink(missing_ok=True)

        return int(exit_status)
    finally:
        plugin_manager.finish()


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


def _command_lint(args: argparse.Namespace) -> int:
    config = load_config_from_env()
    files = _select_files(config, args)
    args.command = "check"
    args.skip_docfacts = getattr(args, "skip_docfacts", False)
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


def _command_doctor(_: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0915
    """Run environment and configuration diagnostics."""
    issues: list[str] = []
    try:
        if sys.version_info[:2] < (3, 13):
            version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            issues.append(f"Python 3.13 or newer required; detected {version}.")

        mypy_path = REPO_ROOT / "mypy.ini"
        if mypy_path.exists():
            content = mypy_path.read_text(encoding="utf-8")
            if "mypy_path = src:stubs" not in content:
                issues.append("mypy.ini must set 'mypy_path = src:stubs'.")
        else:
            issues.append("mypy.ini not found; run bootstrap to generate it.")

        for relative in ("stubs/griffe", "stubs/libcst", "stubs/mkdocs_gen_files"):
            path = REPO_ROOT / relative
            if not path.exists():
                issues.append(f"Missing stub package at {relative}.")

        for module_name in ("griffe", "libcst"):
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                issues.append(f"Optional dependency '{module_name}' not importable: {exc}.")

        for directory in (REPO_ROOT / "docs" / "_build", REPO_ROOT / ".cache"):
            try:
                directory.mkdir(parents=True, exist_ok=True)
                probe = directory / ".doctor_probe"
                probe.write_text("", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                issues.append(f"Directory {directory} is not writeable: {exc}.")

        precommit_path = REPO_ROOT / ".pre-commit-config.yaml"
        hook_names: list[str] = []
        if precommit_path.exists():
            data = yaml.safe_load(precommit_path.read_text(encoding="utf-8")) or {}
            for repo in data.get("repos", []):
                for hook in repo.get("hooks", []):
                    hook_names.append(hook.get("name") or hook.get("id", ""))

            def _index(name: str) -> int | None:
                try:
                    return hook_names.index(name)
                except ValueError:
                    issues.append(f"Pre-commit hook '{name}' is missing.")
                    return None

            doc_builder_idx = _index("docstring-builder (check)")
            docs_artifacts_idx = _index("docs: regenerate artifacts")
            navmap_idx = _index("navmap-check")
            pyrefly_idx = _index("pyrefly-check")

            if (
                doc_builder_idx is not None
                and docs_artifacts_idx is not None
                and doc_builder_idx > docs_artifacts_idx
            ):
                issues.append(
                    "'docstring-builder (check)' must run before 'docs: regenerate artifacts'."
                )
            if (
                docs_artifacts_idx is not None
                and navmap_idx is not None
                and navmap_idx < docs_artifacts_idx
            ):
                issues.append("'navmap-check' should run after 'docs: regenerate artifacts'.")
            if pyrefly_idx is None:
                issues.append("Add 'pyrefly-check' to pre-commit to validate dependency typing.")
        else:
            issues.append(".pre-commit-config.yaml not found; install pre-commit hooks.")
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.exception("Doctor encountered an unexpected error.")
        return int(ExitStatus.ERROR)

    if issues:
        print("[DOCTOR] Configuration issues detected:")
        for item in issues:
            print(f"  - {item}")
        return int(ExitStatus.CONFIG)

    print("Docstring builder environment looks good.")
    return int(ExitStatus.SUCCESS)


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
        "--changed-only",
        action="store_true",
        help="Automatically set --since to the latest merge-base for fast checks",
    )
    parser.add_argument(
        "--only-plugin",
        action="append",
        dest="only_plugin",
        default=[],
        help="Enable only the specified plugin names (repeat or comma-separate values)",
    )
    parser.add_argument(
        "--disable-plugin",
        action="append",
        dest="disable_plugin",
        default=[],
        help="Disable the specified plugin names (repeat or comma-separate values)",
    )
    parser.add_argument(
        "--policy",
        dest="policy_override",
        action="append",
        default=[],
        help="Override policy settings, e.g. coverage=0.95,missing-returns=warn",
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

    lint = subparsers.add_parser("lint", help="Alias for check with optional docfacts skip")
    lint.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    lint.add_argument(
        "--no-docfacts",
        dest="skip_docfacts",
        action="store_true",
        help="Skip DocFacts drift verification for speed",
    )
    lint.set_defaults(func=_command_lint)

    list_cmd = subparsers.add_parser("list", help="List managed symbols")
    list_cmd.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    list_cmd.set_defaults(func=_command_list)

    clear = subparsers.add_parser("clear-cache", help="Clear the builder cache")
    clear.set_defaults(func=_command_clear_cache)
    harvest = subparsers.add_parser("harvest", help="Harvest metadata without applying edits")
    harvest.add_argument("paths", nargs="*", help="Optional Python paths to limit processing")
    harvest.set_defaults(func=_command_harvest)

    doctor = subparsers.add_parser("doctor", help="Diagnose environment and configuration")
    doctor.set_defaults(func=_command_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the docstring builder CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _assign_command(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if getattr(args, "changed_only", False) and not args.since:
        revision = _default_since_revision()
        if revision:
            args.since = revision
            LOGGER.info("--changed-only resolved to %s", revision)
        else:
            LOGGER.warning(
                "Unable to determine a merge-base for --changed-only; processing full set."
            )
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    handler = cast(CommandHandler, args.func)
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
