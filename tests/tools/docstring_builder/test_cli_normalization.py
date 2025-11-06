"""Tests for docstring builder CLI argument normalization."""

from __future__ import annotations

import argparse

from tools.docstring_builder import cli


def test_normalize_request_options_blank_inputs() -> None:
    args = argparse.Namespace(
        module="",
        since="",
        baseline="",
        paths=[],
        changed_only=False,
    )
    normalized = cli.normalize_request_options(args)

    assert normalized.selection.module is None
    assert normalized.selection.since is None
    assert normalized.selection.changed_only is False
    assert normalized.selection.explicit_paths is None
    assert normalized.baseline is None
    assert normalized.explicit_paths == ()


def test_build_request_from_cli_arguments() -> None:
    parser = cli.build_parser()
    argv = [
        "--module",
        "kgfoundry",
        "--since",
        "main",
        "--baseline",
        "release",
        "--changed-only",
        "generate",
        "src/kgfoundry_common/config.py",
    ]
    args = parser.parse_args(argv)
    request = cli.build_request_from_args(args, command="update", subcommand="generate")

    assert request.module == "kgfoundry"
    assert request.since == "main"
    assert request.baseline == "release"
    assert request.changed_only is True
    assert request.explicit_paths == ("src/kgfoundry_common/config.py",)
