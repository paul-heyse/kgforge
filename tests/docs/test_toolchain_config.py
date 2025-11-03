"""Tests for docs toolchain configuration models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from docs.toolchain.config import DocsDeltaConfig, DocsSymbolIndexConfig


class TestDocsSymbolIndexConfig:
    """Test DocsSymbolIndexConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify default values are correct."""
        config = DocsSymbolIndexConfig()
        assert config.output_format == "json"
        assert config.include_private is False
        assert config.max_depth is None

    def test_custom_output_format(self) -> None:
        """Verify custom output format is accepted."""
        config = DocsSymbolIndexConfig(output_format="yaml")
        assert config.output_format == "yaml"

    def test_include_private_flag(self) -> None:
        """Verify include_private flag works."""
        config = DocsSymbolIndexConfig(include_private=True)
        assert config.include_private is True

    def test_custom_max_depth(self) -> None:
        """Verify custom max_depth is accepted."""
        config = DocsSymbolIndexConfig(max_depth=5)
        assert config.max_depth == 5

    def test_zero_max_depth(self) -> None:
        """Verify max_depth=0 is valid."""
        config = DocsSymbolIndexConfig(max_depth=0)
        assert config.max_depth == 0

    def test_invalid_output_format_raises(self) -> None:
        """Verify invalid output_format raises ValueError."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            DocsSymbolIndexConfig(output_format="invalid")

    def test_negative_max_depth_raises(self) -> None:
        """Verify negative max_depth raises ValueError."""
        with pytest.raises(ValueError, match="max_depth must be non-negative"):
            DocsSymbolIndexConfig(max_depth=-1)

    def test_immutability(self) -> None:
        """Verify config is immutable (frozen dataclass)."""
        config = DocsSymbolIndexConfig()
        with pytest.raises(FrozenInstanceError):
            config.output_format = "yaml"  # type: ignore[misc]

    def test_all_formats(self) -> None:
        """Verify all valid output formats are accepted."""
        for fmt in ("json", "yaml", "xml"):
            config = DocsSymbolIndexConfig(output_format=fmt)
            assert config.output_format == fmt

    def test_partial_customization(self) -> None:
        """Verify partial customization works."""
        config = DocsSymbolIndexConfig(max_depth=10)
        assert config.output_format == "json"
        assert config.include_private is False
        assert config.max_depth == 10


class TestDocsDeltaConfig:
    """Test DocsDeltaConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify default values are correct."""
        config = DocsDeltaConfig()
        assert config.include_removals is True
        assert config.include_modifications is True
        assert config.severity_threshold == "info"

    def test_disable_removals(self) -> None:
        """Verify include_removals can be disabled."""
        config = DocsDeltaConfig(include_removals=False)
        assert config.include_removals is False

    def test_disable_modifications(self) -> None:
        """Verify include_modifications can be disabled."""
        config = DocsDeltaConfig(include_modifications=False)
        assert config.include_modifications is False

    def test_warning_threshold(self) -> None:
        """Verify warning severity threshold."""
        config = DocsDeltaConfig(severity_threshold="warning")
        assert config.severity_threshold == "warning"

    def test_error_threshold(self) -> None:
        """Verify error severity threshold."""
        config = DocsDeltaConfig(severity_threshold="error")
        assert config.severity_threshold == "error"

    def test_invalid_threshold_raises(self) -> None:
        """Verify invalid severity_threshold raises ValueError."""
        with pytest.raises(ValueError, match="severity_threshold must be one of"):
            DocsDeltaConfig(severity_threshold="invalid")

    def test_immutability(self) -> None:
        """Verify config is immutable (frozen dataclass)."""
        config = DocsDeltaConfig()
        with pytest.raises(FrozenInstanceError):
            config.severity_threshold = "error"  # type: ignore[misc]

    def test_all_thresholds(self) -> None:
        """Verify all valid severity thresholds are accepted."""
        for threshold in ("info", "warning", "error"):
            config = DocsDeltaConfig(severity_threshold=threshold)
            assert config.severity_threshold == threshold

    def test_both_flags_false(self) -> None:
        """Verify both include flags can be false together."""
        config = DocsDeltaConfig(include_removals=False, include_modifications=False)
        assert config.include_removals is False
        assert config.include_modifications is False

    def test_partial_customization(self) -> None:
        """Verify partial customization works."""
        config = DocsDeltaConfig(include_removals=False)
        assert config.include_removals is False
        assert config.include_modifications is True
        assert config.severity_threshold == "info"


class TestConfigComparison:
    """Test that configs are distinct and properly typed."""

    def test_configs_are_distinct_types(self) -> None:
        """Verify configs are different types."""
        index_config = DocsSymbolIndexConfig()
        delta_config = DocsDeltaConfig()
        assert type(index_config) is not type(delta_config)

    def test_index_has_output_format(self) -> None:
        """Verify DocsSymbolIndexConfig has output_format."""
        config = DocsSymbolIndexConfig()
        assert hasattr(config, "output_format")

    def test_index_has_include_private(self) -> None:
        """Verify DocsSymbolIndexConfig has include_private."""
        config = DocsSymbolIndexConfig()
        assert hasattr(config, "include_private")

    def test_index_has_max_depth(self) -> None:
        """Verify DocsSymbolIndexConfig has max_depth."""
        config = DocsSymbolIndexConfig()
        assert hasattr(config, "max_depth")

    def test_delta_has_include_removals(self) -> None:
        """Verify DocsDeltaConfig has include_removals."""
        config = DocsDeltaConfig()
        assert hasattr(config, "include_removals")

    def test_delta_has_severity_threshold(self) -> None:
        """Verify DocsDeltaConfig has severity_threshold."""
        config = DocsDeltaConfig()
        assert hasattr(config, "severity_threshold")

    def test_delta_does_not_have_output_format(self) -> None:
        """Verify DocsDeltaConfig doesn't have output_format."""
        config = DocsDeltaConfig()
        assert not hasattr(config, "output_format")

    def test_index_does_not_have_severity_threshold(self) -> None:
        """Verify DocsSymbolIndexConfig doesn't have severity_threshold."""
        config = DocsSymbolIndexConfig()
        assert not hasattr(config, "severity_threshold")
