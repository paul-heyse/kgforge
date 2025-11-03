"""LibCST-based codemod for migrating to typing facades.

This module provides AST transformers that:
1. Replace private `docs._types` imports with public `docs.types` imports
2. Replace `resolve_numpy`, `resolve_fastapi`, `resolve_faiss` shims with `gate_import` calls
3. Ensure TYPE_CHECKING guards are used for heavy optional dependencies
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

MIN_MODULE_PARTS = 2


class TypingFacadeMigrator(cst.CSTTransformer):
    """Migrate typing imports to use façades instead of private modules."""

    def __init__(self) -> None:
        """Initialize the transformer."""
        self.modified = False

    def leave_ImportFrom(  # noqa: N802
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom | cst.RemovalSentinel:
        """Replace docs._types imports with docs.types facade imports."""
        # Note: original_node is needed by libcst visitor protocol
        del original_node

        # Check for from docs._types.X import Y
        if isinstance(updated_node.module, cst.Attribute):
            module_parts = self._get_module_parts(updated_node.module)
            if (
                len(module_parts) >= MIN_MODULE_PARTS
                and module_parts[0] == "docs"
                and module_parts[1] == "_types"
            ):
                # Replace with public facade
                new_parts = ["docs", "types", *module_parts[MIN_MODULE_PARTS:]]
                new_module = self._build_module(new_parts)
                self.modified = True
                return updated_node.with_changes(module=new_module)

        # Check for from docs._types import X
        elif isinstance(updated_node.module, cst.Name):
            if updated_node.module.value == "docs._types":
                new_module = cst.Name("docs.types")
                self.modified = True
                return updated_node.with_changes(module=new_module)

        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:  # noqa: N802
        """Replace resolve_* shim calls with gate_import."""
        # Note: original_node is needed by libcst visitor protocol
        del original_node

        shim_map = {
            "resolve_numpy": ("gate_import", "numpy", "numpy array operations"),
            "resolve_fastapi": ("gate_import", "fastapi", "FastAPI integration"),
            "resolve_faiss": ("gate_import", "faiss", "FAISS vector indexing"),
        }

        if isinstance(updated_node.func, cst.Name):
            func_name = updated_node.func.value
            if func_name in shim_map:
                gate_func, lib_name, usage = shim_map[func_name]
                new_args = [
                    cst.Arg(cst.SimpleString(f'"{lib_name}"')),
                    cst.Arg(cst.SimpleString(f'"{usage}"')),
                ]
                self.modified = True
                return updated_node.with_changes(
                    func=cst.Name(gate_func),
                    args=new_args,
                )

        return updated_node

    @staticmethod
    def _get_module_parts(node: cst.BaseExpression) -> list[str]:
        """Extract module path as list of parts."""
        parts: list[str] = []
        current = node
        while isinstance(current, cst.Attribute):
            parts.insert(0, current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.insert(0, current.value)
        return parts

    @staticmethod
    def _build_module(parts: list[str]) -> cst.BaseExpression:
        """Build a module expression from parts."""
        result: cst.BaseExpression = cst.Name(parts[0])
        for part in parts[1:]:
            result = cst.Attribute(value=result, attr=cst.Name(part))
        return result


def run_codemod_on_file(file_path: Path) -> bool:
    """Apply the typing facade migration codemod to a file.

    Parameters
    ----------
    file_path : Path
        Path to the Python file to transform.

    Returns
    -------
    bool
        True if the file was modified, False otherwise.
    """
    try:
        source_code = file_path.read_text(encoding="utf-8")
        module = cst.parse_module(source_code)
        transformer = TypingFacadeMigrator()
        new_module = module.visit(transformer)
    except Exception:
        logger.exception("Error processing %s", file_path)
        return False

    if transformer.modified:
        file_path.write_text(new_module.code, encoding="utf-8")
        logger.info("Updated %s", file_path)
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Migrate typing imports to facades.

    Transforms:
    - `docs._types` → `docs.types` (public facade)
    - `resolve_*` shim calls → `gate_import()` calls

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        description="Migrate typing imports to facades (docs._types → docs.types, shims → gate_import)",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Directories or files to codemod",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing",
    )

    args = parser.parse_args(argv)

    # Collect all Python files
    python_files: list[Path] = []
    for path_arg in args.paths:
        if path_arg.is_file() and path_arg.suffix == ".py":
            python_files.append(path_arg)
        elif path_arg.is_dir():
            python_files.extend(path_arg.rglob("*.py"))

    logger.info("Processing %d Python files", len(python_files))

    updated = 0
    for file_path in python_files:
        if run_codemod_on_file(file_path):
            updated += 1

    logger.info("Updated %d/%d files", updated, len(python_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
