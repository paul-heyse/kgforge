"""Tests for orchestration configuration models.

This module verifies that IndexCliConfig and ArtifactValidationConfig are correctly implemented with
proper defaults, immutability, and type safety.
"""

from __future__ import annotations

import tempfile

from orchestration.config import ArtifactValidationConfig, IndexCliConfig
from tests.helpers import assert_frozen_attribute


class TestIndexCliConfig:
    """Tests for IndexCliConfig dataclass."""

    def test_basic_construction(self) -> None:
        """Test basic construction with all required parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            assert config.dense_vectors == "vectors.json"
            assert config.index_path == f"{tmpdir}/index.idx"
            assert config.factory == "Flat"
            assert config.metric == "ip"

    def test_custom_factory(self) -> None:
        """Test with custom FAISS factory string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="OPQ64,IVF8192,PQ64",
                metric="l2",
            )
            assert config.factory == "OPQ64,IVF8192,PQ64"
            assert config.metric == "l2"

    def test_immutability(self) -> None:
        """Test that IndexCliConfig is frozen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            assert_frozen_attribute(config, "dense_vectors", value="other.json")

    def test_equality(self) -> None:
        """Test equality comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            config2 = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            assert config1 == config2

    def test_inequality(self) -> None:
        """Test inequality comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            config2 = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="l2",
            )
            assert config1 != config2


class TestArtifactValidationConfig:
    """Tests for ArtifactValidationConfig dataclass."""

    def test_default_construction(self) -> None:
        """Test construction with default values."""
        config = ArtifactValidationConfig()
        assert config.strict_mode is True
        assert config.fail_on_warnings is False

    def test_custom_construction(self) -> None:
        """Test construction with custom values."""
        config = ArtifactValidationConfig(
            strict_mode=False,
            fail_on_warnings=True,
        )
        assert config.strict_mode is False
        assert config.fail_on_warnings is True

    def test_partial_construction(self) -> None:
        """Test construction with partial overrides."""
        config = ArtifactValidationConfig(strict_mode=False)
        assert config.strict_mode is False
        assert config.fail_on_warnings is False

    def test_immutability(self) -> None:
        """Test that ArtifactValidationConfig is frozen."""
        config = ArtifactValidationConfig()
        assert_frozen_attribute(config, "strict_mode", value=False)

    def test_equality(self) -> None:
        """Test equality comparison."""
        config1 = ArtifactValidationConfig(strict_mode=True, fail_on_warnings=False)
        config2 = ArtifactValidationConfig(strict_mode=True, fail_on_warnings=False)
        assert config1 == config2

    def test_inequality(self) -> None:
        """Test inequality comparison."""
        config1 = ArtifactValidationConfig(strict_mode=True, fail_on_warnings=False)
        config2 = ArtifactValidationConfig(strict_mode=False, fail_on_warnings=False)
        assert config1 != config2


class TestConfigComparison:
    """Tests for comparing different config types."""

    def test_different_types(self) -> None:
        """Test that different config types are distinct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            validation_config = ArtifactValidationConfig()
            assert isinstance(index_config, IndexCliConfig)
            assert isinstance(validation_config, ArtifactValidationConfig)

    def test_attribute_presence(self) -> None:
        """Test that configs have expected attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            assert hasattr(index_config, "dense_vectors")
            assert hasattr(index_config, "index_path")
            assert hasattr(index_config, "factory")
            assert hasattr(index_config, "metric")

            validation_config = ArtifactValidationConfig()
            assert hasattr(validation_config, "strict_mode")
            assert hasattr(validation_config, "fail_on_warnings")


class TestConfigDefaults:
    """Tests for default value semantics."""

    def test_artifact_validation_defaults(self) -> None:
        """Test ArtifactValidationConfig default values."""
        config1 = ArtifactValidationConfig()
        config2 = ArtifactValidationConfig()
        assert config1 == config2
        assert config1.strict_mode is True
        assert config1.fail_on_warnings is False

    def test_both_flags_false(self) -> None:
        """Test with both validation flags set to False."""
        config = ArtifactValidationConfig(strict_mode=False, fail_on_warnings=False)
        assert config.strict_mode is False
        assert config.fail_on_warnings is False

    def test_both_flags_true(self) -> None:
        """Test with both validation flags set to True."""
        config = ArtifactValidationConfig(strict_mode=True, fail_on_warnings=True)
        assert config.strict_mode is True
        assert config.fail_on_warnings is True
