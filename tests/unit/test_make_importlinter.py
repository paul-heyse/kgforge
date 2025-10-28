"""Tests for :mod:`tools.make_importlinter`."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "make_importlinter.py"
spec = importlib.util.spec_from_file_location("tools.make_importlinter", MODULE_PATH)
assert spec and spec.loader  # pragma: no cover - fail fast if script missing
make_importlinter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_importlinter)
main = make_importlinter.main


def test_main_writes_expected_importlinter(tmp_path: Path) -> None:
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
