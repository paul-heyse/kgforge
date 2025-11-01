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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import yaml

from tools._shared.logging import get_logger
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.harvest import harvest_file
from tools.docstring_builder.io import (
    default_since_revision,
    select_files,
)
from tools.docstring_builder.ir import write_schema
from tools.docstring_builder.orchestrator import (
    DocstringBuildRequest,
    DocstringBuildResult,
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
from tools.stubs.drift_check import run as run_stub_drift

LOGGER = get_logger(__name__)
CommandHandler = Callable[[argparse.Namespace], int]


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
        module=(args.module or "") or None,
        since=(args.since or "") or None,
        changed_only=getattr(args, "changed_only", False),
        explicit_paths=explicit_paths,
        force=getattr(args, "force", False),
        diff=getattr(args, "diff", False),
        ignore_missing=getattr(args, "ignore_missing", False),
        skip_docfacts=getattr(args, "skip_docfacts", False),
        json_output=getattr(args, "json_output", False),
        jobs=getattr(args, "jobs", 1) or 1,
        baseline=(getattr(args, "baseline", "") or None),
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
        files = select_files(
            config,
            module=(args.module or "") or None,
            since=(args.since or "") or None,
            changed_only=getattr(args, "changed_only", False),
            explicit_paths=getattr(args, "paths", None),
        )
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


def _command_doctor(args: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0914, PLR0915
    _, selection = load_builder_config(getattr(args, "config_path", None))
    LOGGER.info("[DOCTOR] Active config: %s (%s)", selection.path, selection.source)
    issues: list[str] = []
    try:
        current = sys.version_info
        if current.major < REQUIRED_PYTHON_MAJOR or (
            current.major == REQUIRED_PYTHON_MAJOR and current.minor < REQUIRED_PYTHON_MINOR
        ):
            version = f"{current.major}.{current.minor}.{current.micro}"
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
                __import__(module_name)
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
        try:
            policy_settings = load_policy_settings(REPO_ROOT)
        except PolicyConfigurationError as exc:
            issues.append(f"Unable to load policy settings: {exc}.")
        else:
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
    except Exception:  # pragma: no cover
        LOGGER.exception("Doctor encountered an unexpected error.")
        return int(ExitStatus.ERROR)

    drift_status = int(ExitStatus.SUCCESS)
    if getattr(args, "stubs", False):
        LOGGER.info("[DOCTOR] Running stub drift check...")
        drift_status = run_stub_drift()
        if drift_status != 0:
            message = "Stub drift detected; see output above."
            sys.stdout.write(f"{message}\n")
            issues.append(message)

    if issues:
        LOGGER.error("[DOCTOR] Configuration issues detected:")
        for item in issues:
            LOGGER.error("  - %s", item)
        return int(ExitStatus.CONFIG)

    if drift_status != 0:
        return int(ExitStatus.CONFIG)

    LOGGER.info("Docstring builder environment looks good.")
    return int(ExitStatus.SUCCESS)


LEGACY_COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "update": _command_update,
    "check": _command_check,
    "harvest": _command_harvest,
}


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0914, PLR0915
    """Build the top-level argument parser for the docstring builder CLI."""
    parser = argparse.ArgumentParser(prog="docstring-builder")
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Override the path to docstring_builder.toml",
    )
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
    parser.add_argument(
        "--policy-override",
        action="append",
        dest="policy_override",
        default=[],
        help="Override policy settings (key=value, repeat or comma-separate)",
    )
    parser.add_argument(
        "--baseline",
        help="Reference git revision or path for baseline comparisons",
        default="",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker threads for processing",
    )
    parser.add_argument(
        "--skip-docfacts",
        action="store_true",
        help="Skip DocFacts reconciliation",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        dest="json_output",
        help="Emit JSON summary payload to stdout",
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

    def _with_paths(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "paths",
            nargs="*",
            help="Optional Python paths to limit processing",
        )

    generate = subparsers.add_parser("generate", help="Synchronize managed docstrings and DocFacts")
    _with_paths(generate)
    generate.set_defaults(func=_command_generate)

    fix = subparsers.add_parser("fix", help="Apply docstring updates while bypassing the cache")
    _with_paths(fix)
    fix.set_defaults(func=_command_fix)

    fmt = subparsers.add_parser(
        "fmt", help="Normalize existing docstring sections without regenerating content"
    )
    _with_paths(fmt)
    fmt.set_defaults(func=_command_fmt)

    diff_cmd = subparsers.add_parser("diff", help="Show docstring drift without writing changes")
    _with_paths(diff_cmd)
    diff_cmd.set_defaults(func=_command_diff)

    check = subparsers.add_parser("check", help="Validate docstrings without writing")
    _with_paths(check)
    check.set_defaults(func=_command_check)

    lint = subparsers.add_parser("lint", help="Alias for check with optional DocFacts skip")
    _with_paths(lint)
    lint.add_argument(
        "--no-docfacts",
        dest="skip_docfacts",
        action="store_true",
        help="Skip DocFacts drift verification for speed",
    )
    lint.set_defaults(func=_command_lint)

    measure = subparsers.add_parser("measure", help="Run validation and emit observability metrics")
    _with_paths(measure)
    measure.set_defaults(func=_command_measure)

    schema = subparsers.add_parser("schema", help="Generate the docstring IR schema JSON")
    schema.add_argument("--output", help="Optional output path for the schema JSON")
    schema.set_defaults(func=_command_schema)

    doctor = subparsers.add_parser(
        "doctor", help="Diagnose environment, configuration, and optional stubs"
    )
    doctor.add_argument(
        "--stubs",
        action="store_true",
        help="Run the stub drift checker as part of diagnostics",
    )
    doctor.set_defaults(func=_command_doctor)

    list_cmd = subparsers.add_parser("list", help="List managed docstring symbols")
    _with_paths(list_cmd)
    list_cmd.set_defaults(func=_command_list)

    clear = subparsers.add_parser("clear-cache", help="Clear the builder cache")
    clear.set_defaults(func=_command_clear_cache)

    harvest = subparsers.add_parser("harvest", help="Harvest metadata without applying edits")
    _with_paths(harvest)
    harvest.set_defaults(func=_command_harvest)

    update = subparsers.add_parser("update", help=argparse.SUPPRESS)
    _with_paths(update)
    update.set_defaults(func=_command_update)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the docstring builder CLI."""
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
    handler = cast(CommandHandler, args.func)
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
