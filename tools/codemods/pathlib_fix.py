"""Codemod to convert os.path and os.makedirs to pathlib.Path.

This script uses LibCST to transform common filesystem operations:
- os.makedirs() → Path.mkdir(parents=True)
- os.path.join() → Path / operator
- open(os.path.join(...)) → Path.open()
- os.path.exists() → Path.exists()
- os.path.dirname() → Path.parent

Requirement: R1 — Pathlib Standardization Across Workflows
Scenario: Index writer uses Path helpers
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import libcst as cst
from libcst import matchers as m

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PathlibTransformer(cst.CSTTransformer):
    """Transform os.path operations to pathlib.Path.

    This transformer handles:
    - os.makedirs(path, exist_ok=True) → Path(path).mkdir(parents=True, exist_ok=True)
    - os.path.join(a, b, ...) → Path(a) / b / ...
    - os.path.exists(path) → Path(path).exists()
    - os.path.dirname(path) → Path(path).parent
    - open(os.path.join(...)) → Path(...).open()
    """

    def __init__(self) -> None:
        """Initialize transformer with change tracking."""
        super().__init__()
        self.changes: list[str] = []
        self.needs_pathlib_import = False

    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802 (LibCST visitor pattern)
        """Track if pathlib import already exists."""
        for alias in node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value == "pathlib":
                self.needs_pathlib_import = True

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Transform function calls to pathlib equivalents."""
        # os.makedirs(path, exist_ok=True) → Path(path).mkdir(parents=True, exist_ok=True)
        if m.matches(
            original_node,
            m.Call(
                func=m.Attribute(value=m.Name("os"), attr=m.Name("makedirs")),
            ),
        ):
            args = original_node.args
            if len(args) >= 1:
                path_arg = args[0].value
                # Check for exist_ok keyword
                exist_ok = False
                for kw in original_node.args:
                    if kw.keyword and kw.keyword.value == "exist_ok":
                        if isinstance(kw.value, cst.Name) and kw.value.value == "True":
                            exist_ok = True
                        elif isinstance(kw.value, cst.Name) and kw.value.value == "False":
                            exist_ok = False
                        break

                self.changes.append("os.makedirs() → Path().mkdir()")
                self.needs_pathlib_import = True

                # Build Path(path).mkdir(parents=True, exist_ok=...)
                mkdir_keywords = [
                    cst.Arg(
                        cst.Name("True"),
                        keyword=cst.Name("parents"),
                    ),
                ]
                if exist_ok:
                    mkdir_keywords.append(
                        cst.Arg(
                            cst.Name("True"),
                            keyword=cst.Name("exist_ok"),
                        ),
                    )

                return cst.Call(
                    func=cst.Attribute(
                        value=cst.Call(
                            func=cst.Name("Path"),
                            args=[cst.Arg(path_arg)],
                        ),
                        attr=cst.Name("mkdir"),
                    ),
                    args=mkdir_keywords,
                )

        # os.path.join(a, b, ...) → Path(a) / b / ...
        if m.matches(
            original_node,
            m.Call(
                func=m.Attribute(
                    value=m.Attribute(value=m.Name("os"), attr=m.Name("path")),
                    attr=m.Name("join"),
                ),
            ),
        ):
            if len(original_node.args) >= 2:
                self.changes.append("os.path.join() → Path / operator")
                self.needs_pathlib_import = True

                # Convert Path(arg0) / arg1 / arg2 / ...
                # Only use positional args (skip keyword args)
                pos_args = [arg for arg in original_node.args if not arg.keyword]
                if len(pos_args) >= 2:
                    base = cst.Call(
                        func=cst.Name("Path"),
                        args=[cst.Arg(pos_args[0].value)],
                    )
                    result: cst.BaseExpression = base
                    for arg in pos_args[1:]:
                        result = cst.BinaryOperation(
                            left=result,
                            operator=cst.Divide(),
                            right=arg.value,
                        )
                    return result

        # os.path.exists(path) → Path(path).exists()
        if m.matches(
            original_node,
            m.Call(
                func=m.Attribute(
                    value=m.Attribute(value=m.Name("os"), attr=m.Name("path")),
                    attr=m.Name("exists"),
                ),
            ),
        ):
            if len(original_node.args) >= 1:
                self.changes.append("os.path.exists() → Path().exists()")
                self.needs_pathlib_import = True
                return cst.Call(
                    func=cst.Attribute(
                        value=cst.Call(
                            func=cst.Name("Path"),
                            args=[cst.Arg(original_node.args[0].value)],
                        ),
                        attr=cst.Name("exists"),
                    ),
                )

        # os.path.dirname(path) → Path(path).parent
        if m.matches(
            original_node,
            m.Call(
                func=m.Attribute(
                    value=m.Attribute(value=m.Name("os"), attr=m.Name("path")),
                    attr=m.Name("dirname"),
                ),
            ),
        ):
            if len(original_node.args) >= 1:
                self.changes.append("os.path.dirname() → Path().parent")
                self.needs_pathlib_import = True
                return cst.Attribute(
                    value=cst.Call(
                        func=cst.Name("Path"),
                        args=[cst.Arg(original_node.args[0].value)],
                    ),
                    attr=cst.Name("parent"),
                )

        return updated_node

    def leave_With(self, original_node: cst.With, updated_node: cst.With) -> cst.With:
        """Transform open(os.path.join(...)) patterns."""
        # Handle open(os.path.join(...)) → Path(...).open()
        for item in original_node.items:
            if isinstance(item, cst.WithItem):
                if m.matches(
                    item.item,
                    m.Call(
                        func=m.Name("open"),
                        args=[
                            m.Arg(
                                value=m.Call(
                                    func=m.Attribute(
                                        value=m.Attribute(
                                            value=m.Name("os"),
                                            attr=m.Name("path"),
                                        ),
                                        attr=m.Name("join"),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ):
                    # Extract path components
                    join_call = item.item
                    if isinstance(join_call, cst.Call) and len(join_call.args) >= 2:
                        self.changes.append("open(os.path.join(...)) → Path(...).open()")
                        self.needs_pathlib_import = True

                        # Build Path(arg0) / arg1 / ... / argN
                        # Only use positional args (skip keyword args)
                        pos_args = [arg for arg in join_call.args if not arg.keyword]
                        if len(pos_args) >= 2:
                            base = cst.Call(
                                func=cst.Name("Path"),
                                args=[cst.Arg(pos_args[0].value)],
                            )
                            result: cst.BaseExpression = base
                            for arg in pos_args[1:]:
                                result = cst.BinaryOperation(
                                    left=result,
                                    operator=cst.Divide(),
                                    right=arg.value,
                                )
                        else:
                            return updated_node

                        # Replace open(os.path.join(...)) with Path(...).open()
                        new_items = []
                        for orig_item in updated_node.items:
                            if orig_item == item:
                                new_open_call = cst.Call(
                                    func=cst.Attribute(
                                        value=result,
                                        attr=cst.Name("open"),
                                    ),
                                    args=join_call.args[len(join_call.args) :],
                                )
                                new_items.append(
                                    cst.WithItem(
                                        item=new_open_call,
                                        asname=item.asname,
                                    ),
                                )
                            else:
                                new_items.append(orig_item)

                        return updated_node.with_changes(items=tuple(new_items))

        return updated_node


def ensure_pathlib_import(
    module: cst.Module,
) -> cst.Module:
    """Add pathlib import if missing."""
    has_pathlib = False
    for stmt in module.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for stmt_body in stmt.body:
                if isinstance(stmt_body, cst.Import):
                    for alias in stmt_body.names:
                        if isinstance(alias.name, cst.Name) and alias.name.value == "pathlib":
                            has_pathlib = True
                elif isinstance(stmt_body, cst.ImportFrom):
                    if (
                        isinstance(stmt_body.module, cst.Name)
                        and stmt_body.module.value == "pathlib"
                    ):
                        has_pathlib = True

    if not has_pathlib:
        # Find insertion point after __future__ imports
        insert_idx = 0
        for i, stmt in enumerate(module.body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for stmt_body in stmt.body:
                    if isinstance(stmt_body, cst.ImportFrom):
                        if (
                            isinstance(stmt_body.module, cst.Name)
                            and stmt_body.module.value == "__future__"
                        ):
                            insert_idx = i + 1
                            break

        new_import = cst.SimpleStatementLine(
            body=[
                cst.Import(
                    names=[
                        cst.ImportAlias(
                            name=cst.Name("pathlib"),
                        ),
                    ],
                ),
            ],
        )
        new_body = list(module.body)
        new_body.insert(insert_idx, new_import)
        return module.with_changes(body=tuple(new_body))

    return module


def transform_file(file_path: Path, dry_run: bool = False) -> list[str]:
    """Transform a single Python file.

    Parameters
    ----------
    file_path : Path
        Path to Python file to transform.
    dry_run : bool, optional
        If True, only report changes without modifying files.
        Defaults to False.

    Returns
    -------
    list[str]
        List of change descriptions.
    """
    try:
        source_code = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.exception("Failed to read %s", file_path)
        return []

    try:
        module = cst.parse_module(source_code)
    except cst.ParserSyntaxError:
        logger.exception("Failed to parse %s", file_path)
        return []

    transformer = PathlibTransformer()
    transformed_module = module.visit(transformer)

    if transformer.needs_pathlib_import:
        transformed_module = ensure_pathlib_import(transformed_module)

    if transformer.changes:
        if not dry_run:
            try:
                file_path.write_text(
                    transformed_module.code,
                    encoding="utf-8",
                )
                logger.info("✓ Transformed %s", file_path)
            except OSError:
                logger.exception("Failed to write %s", file_path)
                return []
        else:
            logger.info("[DRY RUN] Would transform %s", file_path)

    return transformer.changes


def main() -> int:
    """Run codemod on target files or directories.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description="Convert os.path operations to pathlib.Path",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="Files or directories to transform",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying files",
    )
    parser.add_argument(
        "--log",
        help="Write change log to file",
    )

    args = parser.parse_args()

    all_changes: dict[Path, list[str]] = {}
    target_paths: list[Path] = []

    for target in args.targets:
        path = Path(target)
        if path.is_file() and path.suffix == ".py":
            target_paths.append(path)
        elif path.is_dir():
            target_paths.extend(path.rglob("*.py"))
        else:
            logger.warning("Skipping non-Python file: %s", path)

    if not target_paths:
        logger.error("No Python files found")
        return 1

    logger.info("Processing %d file(s)...", len(target_paths))

    for file_path in target_paths:
        changes = transform_file(file_path, dry_run=args.dry_run)
        if changes:
            all_changes[file_path] = changes

    if args.log:
        log_path = Path(args.log)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("Pathlib Codemod Change Log\n")
            f.write("=" * 50 + "\n\n")
            for file_path, changes in sorted(all_changes.items()):
                f.write(f"{file_path}\n")
                f.write("-" * len(str(file_path)) + "\n")
                for change in changes:
                    f.write(f"  {change}\n")
                f.write("\n")
        logger.info("Change log written to %s", log_path)

    total_changes = sum(len(changes) for changes in all_changes.values())
    logger.info("Total transformations: %d in %d file(s)", total_changes, len(all_changes))

    return 0


if __name__ == "__main__":
    sys.exit(main())
