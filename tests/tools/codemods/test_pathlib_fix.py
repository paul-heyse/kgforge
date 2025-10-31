from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import libcst as cst
from tools.codemods import pathlib_fix


def _apply_pathlib_transform(source: str) -> str:
    module = cst.parse_module(dedent(source))
    transformer = pathlib_fix.PathlibTransformer()
    transformed = module.visit(transformer)
    if transformer.needs_pathlib_import:
        transformed = pathlib_fix.ensure_pathlib_import(transformed)
    return transformed.code


def test_pathlib_transformer_converts_join_calls() -> None:
    result = _apply_pathlib_transform(
        """
        import os

        def build(name: str) -> object:
            return os.path.join("a", "b", name)
        """
    )

    assert "import pathlib" in result
    assert "Path(" in result
    assert " / " in result


def test_pathlib_transformer_converts_open_calls() -> None:
    result = _apply_pathlib_transform(
        """
        import os

        def use_file(base: str) -> None:
            with open(os.path.join(base, "data.txt")) as handle:
                handle.read()
        """
    )

    assert "Path(" in result
    assert 'Path(base) / "data.txt"' in result
    assert ".open()" in result


def test_transform_file_writes_changes(tmp_path: Path) -> None:
    source = dedent(
        """
        import os

        def make_dir(path: str) -> None:
            os.makedirs(path, exist_ok=True)
        """
    )
    target = tmp_path / "code.py"
    target.write_text(source, encoding="utf-8")

    changes = pathlib_fix.transform_file(target, dry_run=False)

    updated = target.read_text(encoding="utf-8")

    assert changes
    assert "Path" in updated
    assert "mkdir" in updated


def test_parse_args_returns_typed_values(tmp_path: Path) -> None:
    log_file = tmp_path / "changes.log"
    args = pathlib_fix._parse_args(
        [
            "--dry-run",
            "--log",
            str(log_file),
            "sample.py",
        ]
    )

    assert isinstance(args, pathlib_fix.PathlibArgs)
    assert args.dry_run is True
    assert args.log == log_file
    assert args.targets == (Path("sample.py"),)
