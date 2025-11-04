"""Tests for :mod:`tools.docs.build_agent_catalog`."""

from pathlib import Path

import pytest
from tools.docs.build_agent_catalog import parse_args


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("test_map", Path("docs/_build/test_map.json")),
        ("test_map_coverage", Path("docs/_build/test_map_coverage.json")),
        ("test_map_summary", Path("docs/_build/test_map_summary.json")),
    ],
)
def test_parse_args_uses_docs_build_defaults(attribute: str, expected: Path) -> None:
    """Ensure the CLI defaults align with the docs build outputs."""
    args = parse_args(())
    assert getattr(args, attribute) == expected
