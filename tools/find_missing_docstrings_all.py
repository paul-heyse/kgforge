#!/usr/bin/env python3
"""Script to find functions and classes missing docstrings across the entire repo."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def find_missing_docstrings(filepath: Path) -> list[dict[str, str]]:
    """Find functions and classes without docstrings in a Python file."""
    try:
        with Path(filepath).open("r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))
    except Exception as e:
        return [{"type": "error", "name": str(e), "line": 0}]

    missing = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip private functions/classes unless they're dunder methods
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue

            # Skip @overload decorated functions (they shouldn't have docstrings)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    (isinstance(dec, ast.Name) and dec.id == "overload")
                    or (isinstance(dec, ast.Attribute) and dec.attr == "overload")
                    for dec in node.decorator_list
                ):
                    continue

            # Check if docstring exists
            docstring = ast.get_docstring(node)
            if not docstring:
                line_num = node.lineno
                node_type = (
                    "async function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else ("function" if isinstance(node, ast.FunctionDef) else "class")
                )
                missing.append(
                    {
                        "type": node_type,
                        "name": node.name,
                        "line": line_num,
                    }
                )

    return missing


def main() -> None:
    """Main entry point."""
    repo_root = Path()
    if not repo_root.exists():
        print(f"Error: {repo_root} does not exist", file=sys.stderr)
        sys.exit(1)

    # Directories to search
    search_dirs = ["src", "tools", "docs", "tests", "examples"]

    all_missing = []
    for search_dir in search_dirs:
        search_path = repo_root / search_dir
        if not search_path.exists():
            continue

        for py_file in search_path.rglob("*.py"):
            # Skip __pycache__ and build directories
            if "__pycache__" in str(py_file) or "_build" in str(py_file):
                continue
            missing = find_missing_docstrings(py_file)
            if missing:
                rel_path = py_file.relative_to(repo_root)
                for item in missing:
                    item["file"] = str(rel_path)
                    all_missing.append(item)

    # Print results
    for item in sorted(all_missing, key=lambda x: (x["file"], x["line"])):
        print(f"{item['file']}:{item['line']} - {item['type']}: {item['name']}")

    print(f"\nTotal: {len(all_missing)} items missing docstrings", file=sys.stderr)


if __name__ == "__main__":
    main()
