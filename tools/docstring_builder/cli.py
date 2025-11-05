"""Docstring builder CLI for managing generated docstrings across the repo.

This module wires together the tooling that harvests code metadata, renders
managed docstrings, and applies or inspects the resulting edits. Subcommands
cover the full workflow: ``generate`` synchronizes docstrings and DocFacts,
``fix`` forces writes while bypassing the cache, ``diff`` and ``check`` display
drift without mutating files, ``lint`` mirrors ``check`` with optional DocFacts
skips, ``measure`` emits observability metrics, ``schema`` exports the IR
schema, ``doctor`` performs health checks, ``list`` enumerates managed symbols,
``clear-cache`` resets builder state, ``harvest`` collects metadata only, and
``update`` serves project automation.

Usage
-----
The CLI is typically invoked through project tooling, for example via ``uv run
python -m tools.docstring_builder.cli <subcommand>`` or the corresponding Make
targets when regenerating documentation artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from tools._shared.logging import get_logger
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.harvest import harvest_file
from tools.docstring_builder.io import (
    SelectionCriteria,
    default_since_revision,
    select_files,
)
from tools.docstring_builder.ir import write_schema
from tools.docstring_builder.models import DocstringBuilderError
from tools.docstring_builder.orchestrator import (
    DocstringBuildRequest,
    ExitStatus,
    InvalidPathError,
    load_builder_config,
    render_cli_result,
    render_failure_summary,
    run_docstring_builder,
)
from tools.docstring_builder.paths import (
    CACHE_PATH,
    REPO_ROOT,
    REQUIRED_PYTHON_MAJOR,
    REQUIRED_PYTHON_MINOR,
)
from tools.docstring_builder.policy import PolicyConfigurationError, load_policy_settings
from tools.docstring_builder.utils import optional_str
from tools.stubs.drift_check import run as run_stub_drift

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.docstring_builder.orchestrator import (
        DocstringBuildResult,
    )

LOGGER = get_logger(__name__)
CommandHandler = Callable[[argparse.Namespace], int]


CLI_ARGUMENT_DEFINITIONS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (("--config",), {"dest": "config_path", "help": "Override the path to docstring_builder.toml"}),
    (("--module",), {"default": "", "help": "Restrict to module prefix"}),
    (("--since",), {"default": "", "help": "Only consider files changed since revision"}),
    (("--force",), {"action": "store_true", "help": "Ignore cache entries"}),
    (("--diff",), {"action": "store_true", "help": "Show diffs in check mode"}),
    (
        ("--ignore-missing",),
        {
            "action": "store_true",
            "help": "Skip modules that raise ModuleNotFoundError (e.g., docs/_build artefacts)",
        },
    ),
    (
        ("--changed-only",),
        {
            "action": "store_true",
            "help": "Automatically set --since to the latest merge-base for fast checks",
        },
    ),
    (
        ("--only-plugin",),
        {
            "action": "append",
            "dest": "only_plugin",
            "default": [],
            "help": "Enable only the specified plugin names (repeat or comma-separate values)",
        },
    ),
    (
        ("--disable-plugin",),
        {
            "action": "append",
            "dest": "disable_plugin",
            "default": [],
            "help": "Disable the specified plugin names (repeat or comma-separate values)",
        },
    ),
    (
        ("--policy-override",),
        {
            "action": "append",
            "dest": "policy_override",
            "default": [],
            "help": "Override policy settings (key=value, repeat or comma-separate)",
        },
    ),
    (
        ("--baseline",),
        {"default": "", "help": "Reference git revision or path for baseline comparisons"},
    ),
    (("--jobs",), {"type": int, "default": 1, "help": "Number of worker threads for processing"}),
    (("--skip-docfacts",), {"action": "store_true", "help": "Skip DocFacts reconciliation"}),
    (
        ("--json-output",),
        {
            "action": "store_true",
            "dest": "json_output",
            "help": "Emit JSON summary payload to stdout",
        },
    ),
    (
        ("--update",),
        {"dest": "flag_update", "action": "store_true", "help": "Legacy flag: run in update mode"},
    ),
    (
        ("--check",),
        {"dest": "flag_check", "action": "store_true", "help": "Legacy flag: run in check mode"},
    ),
    (
        ("--harvest",),
        {
            "dest": "flag_harvest",
            "action": "store_true",
            "help": "Legacy flag: harvest symbols without writing",
        },
    ),
    (
        ("--diff-only",),
        {
            "dest": "flag_diff",
            "action": "store_true",
            "help": "Legacy flag: run check mode and show diffs",
        },
    ),
)


def _parse_plugin_names(values: Sequence[str] | None) -> tuple[str, ...]:
    names: list[str] = []
    if not values:
        return tuple(names)
    for raw in values:
        for chunk in raw.split(","):
            name = chunk.strip()
            if name:
                names.append(name)
    return tuple(names)


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
                message = f"Invalid policy override '{token}'"
                raise PolicyConfigurationError(message)
            key, value = token.split("=", 1)
            overrides[key.strip().lower()] = value.strip()
    return overrides


def _legacy_command_from_flags(args: argparse.Namespace) -> str | None:
    for attr, command in (
        ("flag_update", "update"),
        ("flag_check", "check"),
        ("flag_harvest", "harvest"),
        ("flag_diff", "diff"),
    ):
        if getattr(args, attr, False):
            return command
    return None


def _assign_command(args: argparse.Namespace) -> None:
    if hasattr(args, "func") and callable(args.func):
        return
    handlers: tuple[tuple[str, CommandHandler], ...] = (
        ("flag_update", _command_update),
        ("flag_check", _command_check),
        ("flag_harvest", _command_harvest),
        ("flag_diff", _command_diff),
    )
    for attr, handler in handlers:
        if getattr(args, attr, False):
            args.func = handler
            args.invoked_subcommand = attr.removeprefix("flag_")
            return
    legacy_command = _legacy_command_from_flags(args)
    if legacy_command is not None:
        args.func = LEGACY_COMMAND_HANDLERS[legacy_command]
        args.invoked_subcommand = legacy_command


def _build_request(
    args: argparse.Namespace,
    *,
    command: str,
    subcommand: str,
) -> DocstringBuildRequest:
    explicit_paths = tuple(getattr(args, "paths", []) or [])
    policy_override_values = getattr(args, "policy_override", None)
    try:
        policy_overrides = _parse_policy_overrides(policy_override_values)
    except PolicyConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    return DocstringBuildRequest(
        command=command,
        subcommand=subcommand,
        module=optional_str(getattr(args, "module", None)),
        since=optional_str(getattr(args, "since", None)),
        changed_only=getattr(args, "changed_only", False),
        explicit_paths=explicit_paths,
        force=getattr(args, "force", False),
        diff=getattr(args, "diff", False),
        ignore_missing=getattr(args, "ignore_missing", False),
        skip_docfacts=getattr(args, "skip_docfacts", False),
        json_output=getattr(args, "json_output", False),
        jobs=getattr(args, "jobs", 1) or 1,
        baseline=optional_str(getattr(args, "baseline", None)),
        only_plugins=_parse_plugin_names(getattr(args, "only_plugin", None)),
        disable_plugins=_parse_plugin_names(getattr(args, "disable_plugin", None)),
        policy_overrides=policy_overrides,
        llm_summary=getattr(args, "llm_summary", False),
        llm_dry_run=getattr(args, "llm_dry_run", False),
        normalize_sections=subcommand == "fmt",
        invoked_subcommand=subcommand,
    )


def _emit_json(result: DocstringBuildResult) -> None:
    payload = render_cli_result(result)
    if payload is None:
        return
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _emit_diffs(result: DocstringBuildResult) -> None:
    for _, preview in result.diff_previews:
        sys.stdout.write(preview)


def _execute_pipeline(args: argparse.Namespace, subcommand: str, command: str) -> int:
    request = _build_request(args, command=command, subcommand=subcommand)
    config_override = getattr(args, "config_path", None)
    result = run_docstring_builder(request, config_override=config_override)
    if request.diff and not request.json_output:
        _emit_diffs(result)
    if request.json_output:
        _emit_json(result)
    render_failure_summary(result)
    return int(result.exit_status)


def _command_generate(args: argparse.Namespace) -> int:
    return _execute_pipeline(args, "generate", "update")


def _command_fix(args: argparse.Namespace) -> int:
    args.force = True
    return _execute_pipeline(args, "fix", "update")


def _command_fmt(args: argparse.Namespace) -> int:
    args.skip_docfacts = True
    return _execute_pipeline(args, "fmt", "fmt")


def _command_update(args: argparse.Namespace) -> int:
    return _execute_pipeline(args, "update", "update")


def _command_check(args: argparse.Namespace) -> int:
    subcommand = getattr(args, "subcommand", None) or "check"
    return _execute_pipeline(args, subcommand, "check")


def _command_diff(args: argparse.Namespace) -> int:
    args.diff = True
    return _execute_pipeline(args, "diff", "check")


def _command_measure(args: argparse.Namespace) -> int:
    args.measure = True
    return _execute_pipeline(args, "measure", "check")


def _command_lint(args: argparse.Namespace) -> int:
    args.skip_docfacts = getattr(args, "skip_docfacts", False)
    return _execute_pipeline(args, "lint", "check")


def _command_harvest(args: argparse.Namespace) -> int:
    args.force = True
    return _execute_pipeline(args, "harvest", "harvest")


def _command_list(args: argparse.Namespace) -> int:
    config, _ = load_builder_config(getattr(args, "config_path", None))
    try:
        selection = SelectionCriteria(
            module=optional_str(getattr(args, "module", None)),
            since=optional_str(getattr(args, "since", None)),
            changed_only=getattr(args, "changed_only", False),
            explicit_paths=getattr(args, "paths", None),
        )
        files = select_files(config, selection)
    except InvalidPathError:
        LOGGER.exception("Invalid path supplied to docstring builder list")
        return int(ExitStatus.CONFIG)
    for file_path in files:
        result = harvest_file(file_path, config, REPO_ROOT)
        for symbol in result.symbols:
            if symbol.owned:
                sys.stdout.write(f"{symbol.qname}\n")
    return int(ExitStatus.SUCCESS)


def _command_clear_cache(_: argparse.Namespace) -> int:
    BuilderCache(CACHE_PATH).clear()
    LOGGER.info("Cleared docstring builder cache at %s", CACHE_PATH)
    return int(ExitStatus.SUCCESS)


def _command_schema(args: argparse.Namespace) -> int:
    load_builder_config(getattr(args, "config_path", None))
    output = getattr(args, "output", None)
    if output:
        target = Path(output)
        if not target.is_absolute():
            target = (REPO_ROOT / target).resolve()
    else:
        target = REPO_ROOT / "docs" / "_build" / "schema_docstrings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_schema(target)
    rel = target.relative_to(REPO_ROOT)
    LOGGER.info("Schema written to %s", rel)
    return int(ExitStatus.SUCCESS)


def _check_python_version() -> list[str]:
    current = sys.version_info
    if current.major < REQUIRED_PYTHON_MAJOR or (
        current.major == REQUIRED_PYTHON_MAJOR and current.minor < REQUIRED_PYTHON_MINOR
    ):
        version = f"{current.major}.{current.minor}.{current.micro}"
        return [f"Python 3.13 or newer required; detected {version}."]
    return []


def _check_pyright_configuration(config_path: Path) -> list[str]:
    if not config_path.exists():
        return ["pyrightconfig.jsonc not found; run bootstrap to generate it."]
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive guard for doctor command
        return [f"Unable to read pyrightconfig.jsonc: {exc}."]
    if '"typeCheckingMode": "strict"' not in content:
        return ['pyrightconfig.jsonc must set "typeCheckingMode": "strict".']
    return []


def _check_stub_packages(relative_paths: Sequence[str]) -> list[str]:
    return [
        f"Missing stub package at {relative}."
        for relative in relative_paths
        if not (REPO_ROOT / relative).exists()
    ]


def _check_optional_dependencies(modules: Sequence[str]) -> list[str]:
    def _module_issue(module_name: str) -> str | None:
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            return f"Optional dependency '{module_name}' not importable: {exc}."
        return None

    return [
        message for module_name in modules if (message := _module_issue(module_name)) is not None
    ]


def _check_writable_directories(directories: Sequence[Path]) -> list[str]:
    def _directory_issue(directory: Path) -> str | None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".doctor_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return f"Directory {directory} is not writeable: {exc}."
        return None

    return [message for directory in directories if (message := _directory_issue(directory))]


def _extract_precommit_hook_names(config_path: Path) -> list[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    repos = data.get("repos", [])
    return [
        name
        for repo in repos
        if isinstance(repo, dict)
        for hook in repo.get("hooks", [])
        if isinstance(hook, dict)
        if (name := hook.get("name") or hook.get("id", ""))
    ]


def _evaluate_precommit_hooks(config_path: Path) -> list[str]:
    if not config_path.exists():
        return [".pre-commit-config.yaml not found; install pre-commit hooks."]

    issues: list[str] = []
    hook_names = _extract_precommit_hook_names(config_path)

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
        issues.append("'docstring-builder (check)' must run before 'docs: regenerate artifacts'.")
    if (
        docs_artifacts_idx is not None
        and navmap_idx is not None
        and navmap_idx < docs_artifacts_idx
    ):
        issues.append("'navmap-check' should run after 'docs: regenerate artifacts'.")
    if pyrefly_idx is None:
        issues.append("Add 'pyrefly-check' to pre-commit to validate dependency typing.")
    return issues


def _load_policy_settings_with_logging() -> list[str]:
    try:
        policy_settings = load_policy_settings(REPO_ROOT)
    except PolicyConfigurationError as exc:
        return [f"Unable to load policy settings: {exc}."]

    LOGGER.info(
        "[DOCTOR] Policy actions: coverage=%s, missing-params=%s, missing-returns=%s, "
        "missing-examples=%s, summary-mood=%s, dataclass-parity=%s",
        policy_settings.coverage_action.value,
        policy_settings.missing_params_action.value,
        policy_settings.missing_returns_action.value,
        policy_settings.missing_examples_action.value,
        policy_settings.summary_mood_action.value,
        policy_settings.dataclass_parity_action.value,
    )
    if policy_settings.exceptions:
        LOGGER.info("[DOCTOR] Policy exceptions: %s", len(policy_settings.exceptions))
        for exception in policy_settings.exceptions:
            LOGGER.info(
                "[DOCTOR]   %s (%s) expires %s: %s",
                exception.symbol,
                exception.rule,
                exception.expires_on.isoformat(),
                exception.justification or "no justification provided",
            )
    return []


def _maybe_run_stub_drift(args: argparse.Namespace, issues: list[str]) -> int:
    if not getattr(args, "stubs", False):
        return int(ExitStatus.SUCCESS)

    LOGGER.info("[DOCTOR] Running stub drift check...")
    drift_status = run_stub_drift()
    if drift_status != 0:
        message = "Stub drift detected; see output above."
        sys.stdout.write(f"{message}\n")
        issues.append(message)
    return drift_status


def _finalize_doctor(issues: Sequence[str], drift_status: int) -> int:
    if issues:
        LOGGER.error("[DOCTOR] Configuration issues detected:")
        for item in issues:
            LOGGER.error("  - %s", item)
        return int(ExitStatus.CONFIG)

    if drift_status != 0:
        return int(ExitStatus.CONFIG)

    LOGGER.info("Docstring builder environment looks good.")
    return int(ExitStatus.SUCCESS)


def _collect_doctor_checks(args: argparse.Namespace) -> tuple[list[str], int]:
    issues: list[str] = []
    issues.extend(_check_python_version())
    issues.extend(_check_pyright_configuration(REPO_ROOT / "pyrightconfig.jsonc"))
    issues.extend(_check_stub_packages(("stubs/griffe", "stubs/libcst", "stubs/mkdocs_gen_files")))
    issues.extend(_check_optional_dependencies(("griffe", "libcst")))
    issues.extend(
        _check_writable_directories((REPO_ROOT / "docs" / "_build", REPO_ROOT / ".cache"))
    )
    issues.extend(_evaluate_precommit_hooks(REPO_ROOT / ".pre-commit-config.yaml"))
    issues.extend(_load_policy_settings_with_logging())
    drift_status = _maybe_run_stub_drift(args, issues)
    return issues, drift_status


def _command_doctor(args: argparse.Namespace) -> int:
    _, selection = load_builder_config(getattr(args, "config_path", None))
    LOGGER.info("[DOCTOR] Active config: %s (%s)", selection.path, selection.source)
    try:
        issues, drift_status = _collect_doctor_checks(args)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        yaml.YAMLError,
        DocstringBuilderError,
    ):  # pragma: no cover - defensive doctor safeguard
        LOGGER.exception("Doctor encountered an unexpected error.")
        return int(ExitStatus.ERROR)

    return _finalize_doctor(issues, drift_status)


LEGACY_COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "update": _command_update,
    "check": _command_check,
    "harvest": _command_harvest,
}


def _apply_cli_arguments(parser: argparse.ArgumentParser) -> None:
    for option_names, options in CLI_ARGUMENT_DEFINITIONS:
        parser.add_argument(*option_names, **options)


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--llm-summary",
        action="store_true",
        dest="llm_summary",
        help="Rewrite summaries into imperative mood using the LLM plugin.",
    )
    llm_group.add_argument(
        "--llm-dry-run",
        action="store_true",
        dest="llm_dry_run",
        help="Preview LLM summary rewrites without mutating files.",
    )


def _add_path_argument(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "paths",
        nargs="*",
        help="Optional Python paths to limit processing",
    )


def _configure_lint_subparser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--no-docfacts",
        dest="skip_docfacts",
        action="store_true",
        help="Skip DocFacts drift verification for speed",
    )


def _configure_schema_subparser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--output", help="Optional output path for the schema JSON")


def _configure_doctor_subparser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--stubs",
        action="store_true",
        help="Run the stub drift checker as part of diagnostics",
    )


def _register_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    spec: dict[str, object],
) -> None:
    name = str(spec["name"])
    help_text = str(spec.get("help_text", ""))
    handler = cast("CommandHandler", spec["handler"])
    include_paths = bool(spec.get("include_paths"))
    configure = cast("Callable[[argparse.ArgumentParser], None] | None", spec.get("configure"))

    subparser = subparsers.add_parser(name, help=help_text)
    if include_paths:
        _add_path_argument(subparser)
    if configure is not None:
        configure(subparser)
    subparser.set_defaults(func=handler)


SUBCOMMAND_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "generate",
        "help_text": "Synchronize managed docstrings and DocFacts",
        "handler": _command_generate,
        "include_paths": True,
    },
    {
        "name": "fix",
        "help_text": "Apply docstring updates while bypassing the cache",
        "handler": _command_fix,
        "include_paths": True,
    },
    {
        "name": "fmt",
        "help_text": "Normalize existing docstring sections without regenerating content",
        "handler": _command_fmt,
        "include_paths": True,
    },
    {
        "name": "diff",
        "help_text": "Show docstring drift without writing changes",
        "handler": _command_diff,
        "include_paths": True,
    },
    {
        "name": "check",
        "help_text": "Validate docstrings without writing",
        "handler": _command_check,
        "include_paths": True,
    },
    {
        "name": "lint",
        "help_text": "Alias for check with optional DocFacts skip",
        "handler": _command_lint,
        "include_paths": True,
        "configure": _configure_lint_subparser,
    },
    {
        "name": "measure",
        "help_text": "Run validation and emit observability metrics",
        "handler": _command_measure,
        "include_paths": True,
    },
    {
        "name": "schema",
        "help_text": "Generate the docstring IR schema JSON",
        "handler": _command_schema,
        "configure": _configure_schema_subparser,
    },
    {
        "name": "doctor",
        "help_text": "Diagnose environment, configuration, and optional stubs",
        "handler": _command_doctor,
        "configure": _configure_doctor_subparser,
    },
    {
        "name": "list",
        "help_text": "List managed docstring symbols",
        "handler": _command_list,
        "include_paths": True,
    },
    {
        "name": "clear-cache",
        "help_text": "Clear the builder cache",
        "handler": _command_clear_cache,
    },
    {
        "name": "harvest",
        "help_text": "Harvest metadata without applying edits",
        "handler": _command_harvest,
        "include_paths": True,
    },
    {
        "name": "update",
        "help_text": argparse.SUPPRESS,
        "handler": _command_update,
        "include_paths": True,
    },
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the docstring builder CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all subcommands registered.
    """
    parser = argparse.ArgumentParser(prog="docstring-builder")
    _apply_cli_arguments(parser)
    _add_llm_arguments(parser)
    subparsers = parser.add_subparsers(dest="subcommand")
    for spec in SUBCOMMAND_SPECS:
        _register_subcommand(subparsers, spec)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the docstring builder CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _assign_command(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if getattr(args, "changed_only", False) and not args.since:
        revision = default_since_revision()
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
    handler = cast("CommandHandler", args.func)
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
