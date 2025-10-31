"""Tests for :mod:`tools.check_new_suppressions`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import pytest
from _pytest.mark import MarkDecorator
from tools.check_new_suppressions import SuppressionViolation, check_directory, check_file

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:

    def typed_parametrize(
        *args: object, **kwargs: object
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

else:

    def typed_parametrize(*args: object, **kwargs: object) -> MarkDecorator:
        """Return a typed ``pytest.mark.parametrize`` decorator."""
        return pytest.mark.parametrize(*args, **kwargs)


@typed_parametrize(
    ("source", "expected"),
    [
        (
            "x = 1  # type: ignore\n",
            [(1, "x = 1  # type: ignore")],
        ),
        (
            "y = 2  # TYPE: IGNORE[attr-defined]\n",
            [(1, "y = 2  # TYPE: IGNORE[attr-defined]")],
        ),
        (
            "value = 3  # noqa: F401\n",
            [(1, "value = 3  # noqa: F401")],
        ),
        (
            "value = 4  # NOQA\n",
            [(1, "value = 4  # NOQA")],
        ),
    ],
)
def test_check_file_detects_missing_ticket(
    tmp_path: Path, source: str, expected: list[tuple[int, str]]
) -> None:
    """check_file reports suppressions without ``TICKET:`` tags."""
    file_path = tmp_path / "module.py"
    file_path.write_text(source, encoding="utf-8")

    violations = check_file(file_path)
    observed = [(violation.line_number, violation.line_preview) for violation in violations]

    assert observed == expected


@typed_parametrize(
    "source",
    [
        "x = 1  # type: ignore  # TICKET: BUG-123\n",
        "y = 2  # noqa: F401  # ticket: bug-456\n",
        'text = "# type: ignore"  # benign string\n',
    ],
)
def test_check_file_allows_ticket_and_ignores_strings(tmp_path: Path, source: str) -> None:
    """check_file ignores valid tickets and strings containing suppression text."""
    file_path = tmp_path / "module.py"
    file_path.write_text(source, encoding="utf-8")

    assert check_file(file_path) == []


def test_check_directory_reports_per_file(tmp_path: Path) -> None:
    """check_directory maps file paths to their violations."""
    (tmp_path / "pkg").mkdir()
    file_a = tmp_path / "pkg" / "a.py"
    file_a.write_text("a = 1  # type: ignore\n", encoding="utf-8")
    file_b = tmp_path / "pkg" / "b.py"
    file_b.write_text("b = 2  # noqa\n", encoding="utf-8")

    violations = check_directory(tmp_path)

    assert set(violations) == {file_a, file_b}
    assert [isinstance(item, SuppressionViolation) for item in violations[file_a]] == [True]
    assert [(v.line_number, v.line_preview) for v in violations[file_b]] == [(1, "b = 2  # noqa")]
