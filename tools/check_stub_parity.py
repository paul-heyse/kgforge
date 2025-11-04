"""Check stub file parity with runtime modules.

This script verifies that stub files (.pyi) mirror runtime module exports
accurately, identifying missing symbols and type issues that could break
downstream tooling.

Usage:
    python tools/check_stub_parity.py

Exit codes:
    0: All checks passed
    1: Mismatches found
"""

from __future__ import annotations

import ast
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from kgfoundry_common.errors import ConfigurationError
from tools._shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = get_logger(__name__)

__all__ = (
    "StubEvaluationResult",
    "StubParityAnyUsageEntry",
    "StubParityContext",
    "StubParityIssueEntry",
    "StubParityReport",
    "build_stub_parity_context",
    "check_any_usage",
    "evaluate_stub",
    "get_module_exports",
    "get_stub_exports",
    "main",
    "run_stub_parity_checks",
)


@dataclass(frozen=True, slots=True)
class StubEvaluationResult:
    """Result of stub evaluation."""

    missing_symbols: list[str]
    extra_symbols: list[str]
    any_usages: list[tuple[int, str]]
    has_errors: bool


class StubParityAnyUsageEntry(TypedDict):
    """Representation of an Any usage discovered in a stub."""

    line: int
    preview: str


class StubParityIssueEntry(TypedDict):
    """Structured issue payload for stub parity failures."""

    module: str
    stub_path: str
    missing_symbols: list[str]
    extra_symbols: list[str]
    any_usages: list[StubParityAnyUsageEntry]


class StubParityContext(TypedDict):
    """Problem Details context emitted for stub parity checks."""

    issue_count: int
    error_count: int
    issues: list[StubParityIssueEntry]


@dataclass(frozen=True, slots=True)
class StubParityIssueRecord:
    """Internal representation for a stub parity issue."""

    module: str
    stub_path: Path
    missing_symbols: tuple[str, ...]
    extra_symbols: tuple[str, ...]
    any_usages: tuple[tuple[int, str], ...]

    @property
    def error_count(self) -> int:
        """Return the number of error categories triggered for this issue."""
        count = 0
        if self.missing_symbols:
            count += 1
        if self.any_usages:
            count += 1
        return count


@dataclass(frozen=True, slots=True)
class StubParityReport:
    """Aggregate summary of stub parity evaluation results."""

    issues: tuple[StubParityIssueRecord, ...]

    def __post_init__(self) -> None:
        """Stabilize internal storage for downstream consumers."""
        normalized: tuple[StubParityIssueRecord, ...] = tuple(self.issues)
        object.__setattr__(self, "issues", normalized)

    @property
    def issue_count(self) -> int:
        """Return the number of issue entries."""
        return len(self.issues)

    @property
    def error_count(self) -> int:
        """Return the total error categories across all issues."""
        return sum(issue.error_count for issue in self.issues)

    @property
    def has_issues(self) -> bool:
        """Return ``True`` if any issues were recorded."""
        return self.issue_count > 0

    def to_context(self) -> StubParityContext:
        """Produce a problem details context payload."""
        return build_stub_parity_context(self)


def get_module_exports(module_name: str) -> set[str]:
    """Get public exports from a runtime module.

    Parameters
    ----------
    module_name : str
        Fully qualified module name (e.g., 'kgfoundry.agent_catalog.search').

    Returns
    -------
    set[str]
        Names of public (non-private) exports.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        LOGGER.exception("Could not import %s", module_name)
        return set()

    # Use __all__ if available, otherwise use dir() filtering
    all_attr: object = getattr(module, "__all__", None)
    if isinstance(all_attr, (list, tuple, set)):
        # Build set explicitly with typed iteration to avoid Any in set() call
        result: set[str] = set()
        for item in cast("Iterable[object]", all_attr):
            result.add(str(item))
        return result

    return {name for name in dir(module) if not name.startswith("_") and not name.startswith("__")}


def _is_export_name(name: str) -> bool:
    """Check if a name should be considered an export.

    Parameters
    ----------
    name : str
        The symbol name to check.

    Returns
    -------
    bool
        True if the name is exportable (non-private).
    """
    return not name.startswith("_")


def _extract_import_names(node: ast.ImportFrom) -> set[str]:
    """Extract exported names from an ImportFrom node.

    Parameters
    ----------
    node : ast.ImportFrom
        The import node to process.

    Returns
    -------
    set[str]
        Names that should be exported.
    """
    names = set()
    if node.names:
        for alias in node.names:
            if _is_export_name(alias.name):
                export_name = alias.asname if alias.asname else alias.name
                if _is_export_name(export_name):
                    names.add(export_name)
    return names


def get_stub_exports(stub_path: Path) -> set[str]:
    """Extract public names from a stub file.

    Parameters
    ----------
    stub_path : Path
        Path to the .pyi stub file.

    Returns
    -------
    set[str]
        Names of symbols defined in the stub.
    """
    if not stub_path.exists():
        return set()

    try:
        tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    except SyntaxError:
        LOGGER.exception("Could not parse %s", stub_path)
        return set()

    exports = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _is_export_name(node.name):
                exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_export_name(target.id):
                    exports.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            exports.update(_extract_import_names(node))

    return exports


def check_any_usage(stub_path: Path) -> list[tuple[int, str]]:
    """Find Any type usage in stub file.

    Parameters
    ----------
    stub_path : Path
        Path to the .pyi stub file.

    Returns
    -------
    list[tuple[int, str]]
        List of (line_number, line_content) tuples containing Any usage.
    """
    if not stub_path.exists():
        return []

    any_lines = []
    content = stub_path.read_text(encoding="utf-8")

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip comments and ignore directives
        if "Any" in line and not line.strip().startswith("#"):
            # Check if it's an import or actual usage
            if "from typing import" in line or "import " in line:
                # It's an import; flag it only if line has other type hints too
                if line.count("Any") > 0 and (":" in line or "->" in line):
                    any_lines.append((line_num, line.strip()))
            elif " Any" in line or ": Any" in line or "Any[" in line:
                any_lines.append((line_num, line.strip()))

    return any_lines


def evaluate_stub(module_name: str, stub_path: Path) -> StubEvaluationResult:
    """Evaluate a stub file against its runtime module.

    Parameters
    ----------
    module_name : str
        Fully qualified module name.
    stub_path : Path
        Path to the stub file.

    Returns
    -------
    StubEvaluationResult
        Evaluation result with missing symbols, extra symbols, and Any usages.
    """
    runtime_exports = get_module_exports(module_name)
    stub_exports = get_stub_exports(stub_path)
    any_usages = check_any_usage(stub_path)

    missing_symbols = sorted(runtime_exports - stub_exports)
    extra_symbols = sorted(stub_exports - runtime_exports)

    return StubEvaluationResult(
        missing_symbols=missing_symbols,
        extra_symbols=extra_symbols,
        any_usages=any_usages,
        has_errors=bool(missing_symbols or any_usages),
    )


def run_stub_parity_checks(checks: list[tuple[str, Path]]) -> StubParityReport:
    """Run stub parity checks on multiple module/stub pairs."""
    issues: list[StubParityIssueRecord] = []

    for module_name, stub_path in checks:
        result = evaluate_stub(module_name, stub_path)
        if result.has_errors:
            issues.append(
                StubParityIssueRecord(
                    module=module_name,
                    stub_path=stub_path,
                    missing_symbols=tuple(result.missing_symbols),
                    extra_symbols=tuple(result.extra_symbols),
                    any_usages=tuple(result.any_usages),
                )
            )

    report = StubParityReport(issues=tuple(issues))

    if report.has_issues:
        message = f"Found {report.issue_count} stub parity issue(s)"
        raise ConfigurationError(message, context=report.to_context())

    return report


def build_stub_parity_context(report: StubParityReport) -> StubParityContext:
    """Construct the canonical stub parity context payload."""
    issues: list[StubParityIssueEntry] = [
        {
            "module": issue.module,
            "stub_path": str(issue.stub_path),
            "missing_symbols": list(issue.missing_symbols),
            "extra_symbols": list(issue.extra_symbols),
            "any_usages": [
                {"line": line_num, "preview": preview} for line_num, preview in issue.any_usages
            ],
        }
        for issue in report.issues
    ]

    return {
        "issue_count": report.issue_count,
        "error_count": report.error_count,
        "issues": issues,
    }


def _extract_stub_parity_context(error: ConfigurationError) -> StubParityContext | None:
    """Normalize the context associated with a stub parity failure."""
    context = dict(error.context or {})
    expected_keys = {"issue_count", "error_count", "issues"}
    if not expected_keys.issubset(context):
        return None
    return cast("StubParityContext", context)


def main() -> int:
    """Check parity between stubs and runtime modules."""
    project_root = Path(__file__).parent.parent
    stubs_dir = project_root / "stubs" / "kgfoundry"

    modules_to_check = [
        ("kgfoundry._namespace_proxy", stubs_dir / "_namespace_proxy.pyi"),
        ("kgfoundry.agent_catalog.search", stubs_dir / "agent_catalog" / "search.pyi"),
    ]

    context: StubParityContext
    try:
        run_stub_parity_checks(modules_to_check)
    except ConfigurationError as error:
        extracted_context = _extract_stub_parity_context(error)
        if extracted_context is None:
            LOGGER.exception("FAILED: Stub parity issues detected")
            return 1
        context = extracted_context
    else:
        LOGGER.info("SUCCESS: All checks passed")
        return 0

    LOGGER.error(
        "FAILED: %s issue(s) found",
        context.get("error_count", 0),
        extra={"error_count": context.get("error_count", 0)},
    )

    for issue in context.get("issues", []):
        module_name = issue.get("module", "<unknown>")
        LOGGER.error("Module: %s", module_name, extra={"module_name": module_name})
        missing = issue.get("missing_symbols", [])
        if missing:
            LOGGER.error(
                "  Missing in stub: %s",
                missing,
                extra={"module_name": module_name, "missing_symbols": missing},
            )
        extra_symbols = issue.get("extra_symbols", [])
        if extra_symbols:
            LOGGER.info(
                "  Extra in stub (OK if intentional): %s",
                extra_symbols,
                extra={"module_name": module_name, "extra_symbols": extra_symbols},
            )
        any_usages = issue.get("any_usages", [])
        for usage in any_usages:
            LOGGER.error(
                "  Any at line %s: %s",
                usage.get("line"),
                usage.get("preview"),
                extra={
                    "module_name": module_name,
                    "line": usage.get("line"),
                    "preview": usage.get("preview"),
                },
            )

    return 1


if __name__ == "__main__":
    sys.exit(main())
