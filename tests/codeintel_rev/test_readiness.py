"""Unit tests for ReadinessProbe and health checks."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from codeintel_rev.app.config_context import ApplicationContext, ResolvedPaths
from codeintel_rev.app.readiness import CheckResult, ReadinessProbe
from codeintel_rev.config.settings import Settings, VLLMConfig, load_settings


@pytest.fixture
def mock_context(tmp_path: Path) -> ApplicationContext:
    """Create a mock ApplicationContext for testing.

    Returns
    -------
    ApplicationContext
        Mock application context with test paths and settings.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "data").mkdir()
    (repo_root / "data" / "vectors").mkdir()
    (repo_root / "data" / "faiss").mkdir()
    (repo_root / "data" / "faiss" / "code.ivfpq.faiss").touch()
    (repo_root / "data" / "catalog.duckdb").touch()
    (repo_root / "index.scip").touch()

    paths = ResolvedPaths(
        repo_root=repo_root,
        data_dir=repo_root / "data",
        vectors_dir=repo_root / "data" / "vectors",
        faiss_index=repo_root / "data" / "faiss" / "code.ivfpq.faiss",
        duckdb_path=repo_root / "data" / "catalog.duckdb",
        scip_index=repo_root / "index.scip",
    )

    settings = load_settings()
    context = Mock(spec=ApplicationContext)
    context.paths = paths
    context.settings = settings

    return context


def test_check_result_as_payload_healthy() -> None:
    """Test CheckResult.as_payload() for healthy result."""
    # Arrange
    result = CheckResult(healthy=True)

    # Act
    payload = result.as_payload()

    # Assert
    assert payload == {"healthy": True}


def test_check_result_as_payload_unhealthy() -> None:
    """Test CheckResult.as_payload() for unhealthy result with detail."""
    # Arrange
    result = CheckResult(healthy=False, detail="FAISS index not found")

    # Act
    payload = result.as_payload()

    # Assert
    assert payload == {"healthy": False, "detail": "FAISS index not found"}


@pytest.mark.asyncio
async def test_readiness_probe_initialize(mock_context: ApplicationContext) -> None:
    """Test ReadinessProbe.initialize() calls refresh."""
    # Arrange
    probe = ReadinessProbe(mock_context)

    # Act
    await probe.initialize()

    # Assert
    snapshot = probe.snapshot()
    assert len(snapshot) > 0
    assert "repo_root" in snapshot


@pytest.mark.asyncio
async def test_readiness_probe_all_healthy(mock_context: ApplicationContext) -> None:
    """Test ReadinessProbe when all checks pass."""
    # Arrange
    probe = ReadinessProbe(mock_context)

    # Act
    results = await probe.refresh()

    # Assert
    assert results["repo_root"].healthy is True
    assert results["data_dir"].healthy is True
    assert results["vectors_dir"].healthy is True
    assert results["faiss_index"].healthy is True
    assert results["duckdb_catalog"].healthy is True
    assert results["scip_index"].healthy is True


@pytest.mark.asyncio
async def test_readiness_probe_missing_faiss(mock_context: ApplicationContext) -> None:
    """Test ReadinessProbe when FAISS index is missing."""
    # Arrange
    mock_context.paths.faiss_index.unlink()
    probe = ReadinessProbe(mock_context)

    # Act
    results = await probe.refresh()

    # Assert
    assert results["faiss_index"].healthy is False
    assert results["faiss_index"].detail is not None
    assert "not found" in results["faiss_index"].detail.lower()


@pytest.mark.asyncio
async def test_readiness_probe_vllm_unreachable(mock_context: ApplicationContext) -> None:
    """Test ReadinessProbe when vLLM service is unreachable."""
    # Arrange
    probe = ReadinessProbe(mock_context)

    # Act - mock httpx to raise HTTPError
    with patch("httpx.Client") as mock_client:
        mock_instance = Mock()
        mock_instance.get.side_effect = httpx.HTTPError("Connection refused")
        mock_client.return_value.__enter__.return_value = mock_instance

        results = await probe.refresh()

    # Assert
    assert results["vllm_service"].healthy is False
    assert results["vllm_service"].detail is not None
    assert "unreachable" in results["vllm_service"].detail.lower()


@pytest.mark.asyncio
async def test_readiness_probe_caching(mock_context: ApplicationContext) -> None:
    """Test that ReadinessProbe caches results."""
    # Arrange
    probe = ReadinessProbe(mock_context)
    await probe.initialize()

    # Act - get snapshot without refresh
    snapshot1 = probe.snapshot()

    # Modify context (shouldn't affect cached results)
    mock_context.paths.faiss_index.unlink()

    snapshot2 = probe.snapshot()

    # Assert - cached results should be identical
    assert snapshot1 == snapshot2

    # Refresh should update cache
    await probe.refresh()
    snapshot3 = probe.snapshot()
    assert snapshot3["faiss_index"].healthy is False


@pytest.mark.asyncio
async def test_readiness_probe_shutdown(mock_context: ApplicationContext) -> None:
    """Test ReadinessProbe.shutdown() clears state."""
    # Arrange
    probe = ReadinessProbe(mock_context)
    await probe.initialize()

    # Act
    await probe.shutdown()

    # Assert
    with pytest.raises(RuntimeError, match="Readiness probe not initialized"):
        probe.snapshot()


def test_readiness_probe_check_directory_exists() -> None:
    """Test _check_directory() for existing directory."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Act
        result = ReadinessProbe._check_directory(path)  # noqa: SLF001 - testing private method

        # Assert
        assert result.healthy is True
        assert result.detail is None


def test_readiness_probe_check_directory_create() -> None:
    """Test _check_directory() with create=True."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new_subdir"

        # Act
        result = ReadinessProbe._check_directory(new_dir, create=True)  # noqa: SLF001 - testing private method

        # Assert
        assert result.healthy is True
        assert new_dir.exists()
        assert new_dir.is_dir()


def test_readiness_probe_check_file_exists() -> None:
    """Test _check_file() for existing file."""
    # Arrange
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        path = Path(tmpfile.name)

    try:
        # Act
        result = ReadinessProbe._check_file(path, description="test file")  # noqa: SLF001 - testing private method

        # Assert
        assert result.healthy is True
    finally:
        path.unlink()


def test_readiness_probe_check_file_optional() -> None:
    """Test _check_file() with optional=True for missing file."""
    # Arrange
    path = Path("/nonexistent/file.txt")

    # Act
    result = ReadinessProbe._check_file(path, description="test file", optional=True)  # noqa: SLF001 - testing private method

    # Assert
    assert result.healthy is True  # Optional files don't fail readiness
    assert result.detail is not None
    assert "not found" in result.detail.lower()


def test_readiness_probe_check_file_required() -> None:
    """Test _check_file() with optional=False for missing file."""
    # Arrange
    path = Path("/nonexistent/file.txt")

    # Act
    result = ReadinessProbe._check_file(path, description="test file", optional=False)  # noqa: SLF001 - testing private method

    # Assert
    assert result.healthy is False
    assert result.detail is not None
    assert "not found" in result.detail.lower()


def test_readiness_probe_check_vllm_invalid_url(mock_context: ApplicationContext) -> None:
    """Test _check_vllm_connection() with invalid URL."""
    # Arrange - create new settings with invalid URL
    invalid_vllm = VLLMConfig(base_url="not-a-valid-url")
    new_settings = Settings(
        vllm=invalid_vllm,
        paths=mock_context.settings.paths,
        index=mock_context.settings.index,
        limits=mock_context.settings.limits,
    )
    mock_context.settings = new_settings
    probe = ReadinessProbe(mock_context)

    # Act
    result = probe._check_vllm_connection()  # noqa: SLF001 - testing private method

    # Assert
    assert result.healthy is False
    assert result.detail is not None
    assert "invalid" in result.detail.lower()


def test_readiness_probe_check_vllm_success(mock_context: ApplicationContext) -> None:
    """Test _check_vllm_connection() with successful health check."""
    # Arrange - create new settings with valid URL
    valid_vllm = VLLMConfig(base_url="http://localhost:8001/v1")
    new_settings = Settings(
        vllm=valid_vllm,
        paths=mock_context.settings.paths,
        index=mock_context.settings.index,
        limits=mock_context.settings.limits,
    )
    mock_context.settings = new_settings
    probe = ReadinessProbe(mock_context)

    # Act - mock successful HTTP response
    with patch("httpx.Client") as mock_client:
        mock_response = Mock()
        mock_response.is_success = True
        mock_instance = Mock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_instance

        result = probe._check_vllm_connection()  # noqa: SLF001 - testing private method

    # Assert
    assert result.healthy is True


def test_readiness_probe_check_vllm_http_error(mock_context: ApplicationContext) -> None:
    """Test _check_vllm_connection() with HTTP error."""
    # Arrange - create new settings with valid URL
    valid_vllm = VLLMConfig(base_url="http://localhost:8001/v1")
    new_settings = Settings(
        vllm=valid_vllm,
        paths=mock_context.settings.paths,
        index=mock_context.settings.index,
        limits=mock_context.settings.limits,
    )
    mock_context.settings = new_settings
    probe = ReadinessProbe(mock_context)

    # Act - mock HTTP error
    with patch("httpx.Client") as mock_client:
        mock_instance = Mock()
        mock_instance.get.side_effect = httpx.HTTPError("Connection refused")
        mock_client.return_value.__enter__.return_value = mock_instance

        result = probe._check_vllm_connection()  # noqa: SLF001 - testing private method

    # Assert
    assert result.healthy is False
    assert result.detail is not None
    assert "unreachable" in result.detail.lower()
