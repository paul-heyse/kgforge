#!/usr/bin/env python3
"""Run import-linter and emit structured results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools._shared.logging import get_logger, with_fields
from tools._shared.problem_details import build_problem_details

LOGGER = get_logger(__name__)

try:
    from importlinter.application.ports.reporting import Report
    from importlinter.application.use_cases import create_report, read_user_options
except ImportError:
    LOGGER.exception("import-linter not installed")
    LOGGER.info("Install with: uv sync")
    sys.exit(1)

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
type SummaryDict = dict[str, int]
type ProblemDetails = dict[str, JsonValue]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="importlinter.cfg",
        type=str,
        help="Path to the import-linter configuration file (default: importlinter.cfg)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit Problem Details JSON to stdout when violations occur.",
    )
    return parser.parse_args()


def main() -> int:
    """Run import-linter checks with structured logging."""
    args = parse_args()
    config_str: str = args.config
    json_output: bool = args.json
    config_path = Path(config_str)
    if not config_path.exists():
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/import-linter-config-missing",
            title="Import-linter configuration missing",
            status=500,
            detail=f"Configuration file '{config_path}' was not found",
            instance="urn:tool:importlinter:config-missing",
            extensions={"config_path": str(config_path)},
        )
        LOGGER.error(
            "Import-linter configuration missing",
            extra={"problem_details": problem},
        )
        _emit_problem(problem, emit_json=json_output)
        return 1

    try:
        user_options = read_user_options(config_filename=str(config_path))
    except Exception as exc:  # pragma: no cover - configuration parsing errors
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/import-linter-config-invalid",
            title="Failed to parse import-linter configuration",
            status=500,
            detail=str(exc),
            instance="urn:tool:importlinter:config-invalid",
            extensions={"config_path": str(config_path)},
        )
        LOGGER.exception(
            "Error parsing import-linter configuration",
            extra={"problem_details": problem},
        )
        _emit_problem(problem, emit_json=json_output)
        return 1

    try:
        report = create_report(user_options=user_options)
    except Exception as exc:  # pragma: no cover - import-linter execution errors
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/import-linter-execution",
            title="Import-linter execution failed",
            status=500,
            detail=str(exc),
            instance="urn:tool:importlinter:execution",
            extensions={},
        )
        LOGGER.exception(
            "Import-linter execution failed",
            extra={"problem_details": problem},
        )
        _emit_problem(problem, emit_json=json_output)
        return 1

    summary = _build_summary(report)
    if not (report.contains_failures or report.could_not_run):
        _log_success(summary)
        return 0

    violations = _collect_broken_contracts(report)
    invalid_contracts = _collect_invalid_contracts(report)
    problem = _build_failure_problem(summary, violations, invalid_contracts)
    _log_failures(summary, violations, invalid_contracts, problem)
    _emit_problem(problem, emit_json=json_output)
    return 1


def _log_success(summary: SummaryDict) -> None:
    with_fields(LOGGER, **summary).info("✅ Import contracts satisfied")


def _log_failures(
    summary: SummaryDict,
    violations: list[dict[str, JsonValue]],
    invalid_contracts: list[dict[str, JsonValue]],
    problem: ProblemDetails,
) -> None:
    for violation in violations:
        with_fields(
            LOGGER,
            contract=violation.get("name"),
            contract_type=violation.get("type"),
            duration_ms=violation.get("duration_ms"),
        ).error(
            "Import contract violation",
            extra={
                "metadata": violation.get("metadata"),
                "warnings": violation.get("warnings"),
                "contract_options": violation.get("options"),
            },
        )

    for invalid in invalid_contracts:
        with_fields(LOGGER, contract=invalid.get("name")).error(
            "Import contract misconfigured",
            extra={"errors": invalid.get("errors")},
        )

    with_fields(LOGGER, **summary).error(
        "❌ Import contract checks failed",
        extra={"problem_details": problem},
    )


def _build_failure_problem(
    summary: SummaryDict,
    violations: list[dict[str, JsonValue]],
    invalid_contracts: list[dict[str, JsonValue]],
) -> ProblemDetails:
    detail_parts: list[str] = []
    broken_count = summary.get("contracts_broken", 0)
    invalid_count = len(invalid_contracts)
    if broken_count:
        detail_parts.append(f"{broken_count} contract(s) broken")
    if invalid_count:
        detail_parts.append(f"{invalid_count} contract(s) invalid")
    if not detail_parts:
        detail_parts.append("import-linter reported failures")

    extensions: dict[str, JsonValue] = {"summary": _to_jsonable(summary)}
    if violations:
        extensions["violations"] = _to_jsonable(violations)
    if invalid_contracts:
        extensions["invalid_contracts"] = _to_jsonable(invalid_contracts)

    return build_problem_details(
        type="https://kgfoundry.dev/problems/import-linter-violations",
        title="Import-linter reported contract violations",
        status=500,
        detail="; ".join(detail_parts),
        instance="urn:tool:importlinter:violations",
        extensions=extensions,
    )


def _build_summary(report: Report) -> SummaryDict:
    return {
        "contracts_total": len(report.contracts),
        "contracts_broken": report.broken_count,
        "contracts_kept": report.kept_count,
        "invalid_contracts": len(report.invalid_contract_options),
        "warnings_total": report.warnings_count,
        "graph_modules": report.module_count,
        "graph_imports": report.import_count,
        "graph_build_ms": report.graph_building_duration,
    }


def _collect_broken_contracts(report: Report) -> list[dict[str, JsonValue]]:
    violations: list[dict[str, JsonValue]] = []
    for contract, check in report.get_contracts_and_checks():
        if check.kept:
            continue
        violations.append(
            {
                "name": contract.name,
                "type": contract.__class__.__name__,
                "duration_ms": report.get_duration(contract),
                "warnings": _to_jsonable(list(check.warnings)),
                "metadata": _to_jsonable(check.metadata),
                "options": _to_jsonable(contract.contract_options),
            }
        )
    return violations


def _collect_invalid_contracts(report: Report) -> list[dict[str, JsonValue]]:
    invalids: list[dict[str, JsonValue]] = []
    for name, exception in report.invalid_contract_options.items():
        invalids.append({"name": name, "errors": _to_jsonable(exception.errors)})
    return invalids


def _to_jsonable(value: object) -> JsonValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return repr(value)


def _emit_problem(problem: ProblemDetails, *, emit_json: bool) -> None:
    if emit_json:
        sys.stdout.write(json.dumps(problem, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
