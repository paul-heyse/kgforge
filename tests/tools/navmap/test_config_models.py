"""Tests for navmap configuration models.

This module verifies that the navmap configuration objects work correctly
and enforce their constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tools.navmap.config import NavmapRepairOptions, NavmapStripOptions

from tests.helpers import assert_frozen_attribute


class TestNavmapRepairOptions:
    """Tests for NavmapRepairOptions configuration."""

    def test_defaults(self) -> None:
        """Verify default values are correct."""
        options = NavmapRepairOptions()
        assert options.root is None
        assert options.apply is False
        assert options.emit_json is False

    def test_custom_values(self) -> None:
        """Verify custom values are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            options = NavmapRepairOptions(root=root, apply=True, emit_json=True)
            assert options.root == root
            assert options.apply is True
            assert options.emit_json is True

    def test_partial_values(self) -> None:
        """Verify partial value specification works."""
        options = NavmapRepairOptions(apply=True)
        assert options.root is None
        assert options.apply is True
        assert options.emit_json is False

    def test_is_frozen(self) -> None:
        """Verify config is immutable."""
        options = NavmapRepairOptions()
        assert_frozen_attribute(options, "apply", value=True)

    def test_both_flags_can_be_combined(self) -> None:
        """Verify apply and emit_json can be used together."""
        options = NavmapRepairOptions(apply=True, emit_json=True)
        assert options.apply is True
        assert options.emit_json is True


class TestNavmapStripOptions:
    """Tests for NavmapStripOptions configuration."""

    def test_defaults(self) -> None:
        """Verify default values are correct."""
        options = NavmapStripOptions()
        assert options.dry_run is True
        assert options.verbose is False

    def test_custom_values(self) -> None:
        """Verify custom values are accepted."""
        options = NavmapStripOptions(dry_run=False, verbose=True)
        assert options.dry_run is False
        assert options.verbose is True

    def test_dry_run_true_by_default(self) -> None:
        """Verify dry_run defaults to True (safe by default)."""
        options = NavmapStripOptions()
        assert options.dry_run is True

    def test_partial_values(self) -> None:
        """Verify partial value specification works."""
        options = NavmapStripOptions(verbose=True)
        assert options.dry_run is True
        assert options.verbose is True

    def test_is_frozen(self) -> None:
        """Verify config is immutable."""
        options = NavmapStripOptions()
        assert_frozen_attribute(options, "dry_run", value=False)

    def test_flags_combination_both_true(self) -> None:
        """Verify both flags can be True simultaneously."""
        options = NavmapStripOptions(dry_run=True, verbose=True)
        assert options.dry_run is True
        assert options.verbose is True

    def test_flags_combination_both_false(self) -> None:
        """Verify both flags can be False simultaneously."""
        options = NavmapStripOptions(dry_run=False, verbose=False)
        assert options.dry_run is False
        assert options.verbose is False


class TestConfigComparison:
    """Tests comparing the two config types."""

    def test_configs_are_distinct_types(self) -> None:
        """Verify the two config types are different."""
        repair = NavmapRepairOptions()
        strip = NavmapStripOptions()
        assert isinstance(repair, NavmapRepairOptions)
        assert isinstance(strip, NavmapStripOptions)
        assert type(repair) is NavmapRepairOptions
        assert type(strip) is NavmapStripOptions

    def test_repair_has_root_attribute(self) -> None:
        """Verify NavmapRepairOptions has root attribute."""
        options = NavmapRepairOptions()
        assert hasattr(options, "root")

    def test_strip_does_not_have_root_attribute(self) -> None:
        """Verify NavmapStripOptions does not have root attribute."""
        options = NavmapStripOptions()
        assert not hasattr(options, "root")

    def test_repair_has_emit_json_attribute(self) -> None:
        """Verify NavmapRepairOptions has emit_json attribute."""
        options = NavmapRepairOptions()
        assert hasattr(options, "emit_json")

    def test_strip_does_not_have_emit_json_attribute(self) -> None:
        """Verify NavmapStripOptions does not have emit_json attribute."""
        options = NavmapStripOptions()
        assert not hasattr(options, "emit_json")


class TestConfigDefaults:
    """Tests for default value semantics."""

    def test_repair_apply_defaults_to_false(self) -> None:
        """Verify repair apply defaults to False (safe by default)."""
        options = NavmapRepairOptions()
        assert options.apply is False

    def test_strip_dry_run_defaults_to_true(self) -> None:
        """Verify strip dry_run defaults to True (safe by default)."""
        options = NavmapStripOptions()
        assert options.dry_run is True

    def test_repair_emit_json_defaults_to_false(self) -> None:
        """Verify repair emit_json defaults to False (human readable by default)."""
        options = NavmapRepairOptions()
        assert options.emit_json is False

    def test_strip_verbose_defaults_to_false(self) -> None:
        """Verify strip verbose defaults to False (quiet by default)."""
        options = NavmapStripOptions()
        assert options.verbose is False
