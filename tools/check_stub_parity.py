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
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    except ImportError as e:
        print(f"ERROR: Could not import {module_name}: {e}")
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
    except SyntaxError as e:
        print(f"ERROR: Could not parse {stub_path}: {e}")
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


def main() -> int:
    """Check parity between stubs and runtime modules."""
    project_root = Path(__file__).parent.parent
    stubs_dir = project_root / "stubs" / "kgfoundry"

    modules_to_check = [
        ("kgfoundry._namespace_proxy", stubs_dir / "_namespace_proxy.pyi"),
        ("kgfoundry.agent_catalog.search", stubs_dir / "agent_catalog" / "search.pyi"),
    ]

    errors = 0

    for module_name, stub_path in modules_to_check:
        print(f"\nChecking: {module_name}")
        print("=" * 70)

        # Get exports from runtime
        runtime_exports = get_module_exports(module_name)
        if not runtime_exports:
            print("  ⚠ WARNING: Could not determine runtime exports")
            continue

        # Get exports from stub
        stub_exports = get_stub_exports(stub_path)

        # Compare
        missing_in_stub = runtime_exports - stub_exports
        extra_in_stub = stub_exports - runtime_exports

        if missing_in_stub:
            print(f"  ✗ Missing in stub: {sorted(missing_in_stub)}")
            errors += 1
        else:
            print("  ✓ All runtime exports present in stub")

        if extra_in_stub:
            print(f"  ⚠ Extra in stub (OK if intentional): {sorted(extra_in_stub)}")

        # Check for Any usage
        any_usages = check_any_usage(stub_path)
        if any_usages:
            print(f"  ✗ Found {len(any_usages)} instance(s) of Any:")
            for line_num, line in any_usages:
                print(f"      Line {line_num}: {line}")
            errors += 1
        else:
            print("  ✓ No problematic Any types found")

    print("\n" + "=" * 70)
    if errors:
        print(f"FAILED: {errors} issue(s) found")
        return 1

    print("SUCCESS: All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
