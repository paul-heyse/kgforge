from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import libcst as cst
from tools.codemods.blind_except_fix import BlindExceptTransformer, transform_file


def _apply_transform(source: str) -> str:
    module = cst.parse_module(dedent(source))
    transformer = BlindExceptTransformer()
    transformed = module.visit(transformer)
    return transformed.code


def test_blind_except_assigns_exception_variable() -> None:
    result = _apply_transform(
        """
        try:
            call()
        except Exception:
            handle()
        """
    )

    assert "except Exception as exc" in result


def test_bare_except_assigns_variable() -> None:
    result = _apply_transform(
        """
        try:
            call()
        except:
            fallback()
        """
    )

    assert "except Exception as exc" in result


def test_transform_file_reports_changes(tmp_path: Path) -> None:
    source_path = tmp_path / "example.py"
    source_path.write_text(
        dedent(
            """
            try:
                run()
            except Exception:
                cleanup()
            """
        ),
        encoding="utf-8",
    )

    changes = transform_file(source_path, dry_run=True)

    assert changes
    assert any("except Exception" in change for change in changes)
