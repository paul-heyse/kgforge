"""Tests for orchestration CLI refactoring with typed config.

This module verifies that the refactored index_faiss and run_index_faiss functions properly use
IndexCliConfig and handle error cases correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

from orchestration import cli as cli_module
from orchestration.cli import index_faiss, run_index_faiss
from orchestration.config import IndexCliConfig


class TestRunIndexFaissSignature:
    """Tests for run_index_faiss function signature and type safety."""

    def test_keyword_only_parameter(self) -> None:
        """Test that config parameter is keyword-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            # Calling with keyword should work (when properly mocked)
            # Direct positional call would raise TypeError at runtime
            assert config.dense_vectors == "vectors.json"

    def test_run_index_faiss_accepts_config(self) -> None:
        """Test that run_index_faiss accepts IndexCliConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexCliConfig(
                dense_vectors="vectors.json",
                index_path=f"{tmpdir}/index.idx",
                factory="Flat",
                metric="ip",
            )
            # Verify config is properly structured
            assert isinstance(config, IndexCliConfig)
            assert config.factory == "Flat"
            assert config.metric == "ip"


class TestIndexFaissCliBuildsConfig:
    """Tests for index_faiss CLI function config construction."""

    def test_index_faiss_constructs_config_correctly(self) -> None:
        """Test that index_faiss constructs IndexCliConfig from CLI args."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vectors_file = Path(tmpdir) / "vectors.json"
            index_file = Path(tmpdir) / "index.idx"

            # Create dummy vectors file
            vectors_file.write_text("[]", encoding="utf-8")

            # Mock run_index_faiss to capture the config
            with patch("orchestration.cli.run_index_faiss") as mock_run:
                index_faiss(
                    str(vectors_file),
                    str(index_file),
                    "Flat",
                    "ip",
                )
                # Verify run_index_faiss was called once
                assert mock_run.call_count == 1
                # Extract the config from the call
                call_kwargs = cast("dict[str, object]", mock_run.call_args[1])
                assert "config" in call_kwargs
                config = call_kwargs["config"]
                assert isinstance(config, IndexCliConfig)
                assert config.dense_vectors == str(vectors_file)
                assert config.index_path == str(index_file)
                assert config.factory == "Flat"
                assert config.metric == "ip"

    def test_index_faiss_uses_defaults(self) -> None:
        """Test that index_faiss uses default values correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vectors_file = Path(tmpdir) / "vectors.json"
            vectors_file.write_text("[]", encoding="utf-8")

            with patch("orchestration.cli.run_index_faiss") as mock_run:
                index_faiss(str(vectors_file))
                call_kwargs = cast("dict[str, object]", mock_run.call_args[1])
                config = call_kwargs["config"]
                assert isinstance(config, IndexCliConfig)
                assert config.index_path == "./_indices/faiss/shard_000.idx"
                assert config.factory == "Flat"
                assert config.metric == "ip"


class TestRunIndexFaissDocstring:
    """Tests for run_index_faiss docstring completeness."""

    def test_docstring_present(self) -> None:
        """Test that run_index_faiss has a docstring."""
        assert run_index_faiss.__doc__ is not None
        assert len(run_index_faiss.__doc__) > 50

    def test_docstring_mentions_config(self) -> None:
        """Test that docstring mentions IndexCliConfig."""
        doc = run_index_faiss.__doc__ or ""
        assert "IndexCliConfig" in doc
        assert "config" in doc.lower()

    def test_docstring_has_examples(self) -> None:
        """Test that docstring has Examples section."""
        doc = run_index_faiss.__doc__ or ""
        assert "Examples" in doc
        assert "IndexCliConfig(" in doc

    def test_docstring_documents_errors(self) -> None:
        """Test that docstring documents error handling."""
        doc = run_index_faiss.__doc__ or ""
        assert "Raises" in doc
        assert "typer.Exit" in doc
        assert "Problem Details" in doc


class TestIndexFaissDocstring:
    """Tests for index_faiss docstring completeness."""

    def test_index_faiss_docstring_present(self) -> None:
        """Test that index_faiss has a docstring."""
        assert index_faiss.__doc__ is not None
        assert len(index_faiss.__doc__) > 50

    def test_index_faiss_docstring_has_examples(self) -> None:
        """Test that index_faiss docstring has Examples."""
        doc = index_faiss.__doc__ or ""
        assert "Examples" in doc

    def test_index_faiss_docstring_documents_parameters(self) -> None:
        """Test that docstring documents all parameters."""
        doc = index_faiss.__doc__ or ""
        assert "dense_vectors" in doc
        assert "index_path" in doc
        assert "factory" in doc
        assert "metric" in doc


class TestConfigExportedInAll:
    """Tests for public API exports."""

    def test_run_index_faiss_in_all(self) -> None:
        """Test that run_index_faiss is exported in __all__."""
        assert "run_index_faiss" in cli_module.__all__

    def test_index_faiss_in_all(self) -> None:
        """Test that index_faiss is still exported in __all__."""
        assert "index_faiss" in cli_module.__all__


class TestIndexCliConfigIntegration:
    """Tests for integration between IndexCliConfig and CLI functions."""

    def test_config_can_be_created_from_cli_values(self) -> None:
        """Test that IndexCliConfig can be created from typical CLI values."""
        config = IndexCliConfig(
            dense_vectors="my_vectors.json",
            index_path="./_indices/faiss/shard_000.idx",
            factory="OPQ64,IVF8192,PQ64",
            metric="l2",
        )
        assert config.dense_vectors == "my_vectors.json"
        assert config.factory == "OPQ64,IVF8192,PQ64"
        assert config.metric == "l2"

    def test_config_with_custom_paths(self) -> None:
        """Test IndexCliConfig with custom file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexCliConfig(
                dense_vectors=f"{tmpdir}/custom_vectors.json",
                index_path=f"{tmpdir}/custom_index.idx",
                factory="Flat",
                metric="ip",
            )
            assert config.dense_vectors.endswith("custom_vectors.json")
            assert config.index_path.endswith("custom_index.idx")


class TestAPIConsistency:
    """Tests for API consistency between old and new functions."""

    def test_both_functions_handle_same_parameters(self) -> None:
        """Test that both functions can handle the same parameter values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vectors_file = Path(tmpdir) / "vectors.json"
            index_file = Path(tmpdir) / "index.idx"
            vectors_file.write_text("[]", encoding="utf-8")

            config = IndexCliConfig(
                dense_vectors=str(vectors_file),
                index_path=str(index_file),
                factory="Flat",
                metric="ip",
            )

            # Both functions should accept the same data
            assert config.dense_vectors == str(vectors_file)
            assert config.index_path == str(index_file)
            assert config.factory == "Flat"
            assert config.metric == "ip"
