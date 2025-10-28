"""Tests for :mod:`tools.make_importlinter`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_importlinter import main


def test_main_writes_expected_importlinter(tmp_path):
    output_path = tmp_path / ".importlinter"

    result = main(root_package="fakepkg", output_path=output_path)

    expected = (
        "[importlinter]\n"
        "root_package = fakepkg\n\n"
        "[importlinter:contract:layers]\n"
        "name = Respect layered architecture\n"
        "type = layers\n"
        "layers =\n"
        "    fakepkg.presentation\n"
        "    fakepkg.domain\n"
        "    fakepkg.infrastructure\n"
    )

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == expected
