"""Unit tests for kgfoundry_common.fs helpers.

These tests verify pathlib-based filesystem operations cover success, error,
and security (traversal prevention) scenarios as required by R1.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from kgfoundry_common.fs import (
    atomic_write,
    ensure_dir,
    read_text,
    safe_join,
    write_text,
)


class TestEnsureDir:
    """Test suite for ensure_dir helper.

    Requirement: R1 — Pathlib Standardization Across Workflows
    Scenario: Index writer uses Path helpers
    """

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Verify directory creation succeeds."""
        target = tmp_path / "new_dir"
        result = ensure_dir(target)
        assert result == target
        assert target.is_dir()

    def test_creates_parents(self, tmp_path: Path) -> None:
        """Verify parent directories are created."""
        target = tmp_path / "a" / "b" / "c"
        ensure_dir(target)
        assert target.is_dir()
        assert (tmp_path / "a" / "b").is_dir()

    def test_existing_directory_ok(self, tmp_path: Path) -> None:
        """Verify existing directory does not raise with exist_ok=True."""
        target = tmp_path / "existing"
        target.mkdir()
        ensure_dir(target, exist_ok=True)
        assert target.is_dir()

    @pytest.mark.parametrize("exist_ok", [True, False])  # type: ignore[misc]
    def test_permission_denied_raises(self, tmp_path: Path, exist_ok: bool) -> None:  # type: ignore[misc]
        """Verify PermissionError on filesystem denial."""
        if not hasattr(stat, "S_IWRITE"):
            pytest.skip("Windows-only test")
        # Make parent read-only
        parent = tmp_path / "readonly"
        parent.mkdir()
        parent.chmod(stat.S_IREAD)
        target = parent / "subdir"
        try:
            with pytest.raises(PermissionError):
                ensure_dir(target, exist_ok=exist_ok)
        finally:
            parent.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)


class TestSafeJoin:
    """Test suite for safe_join helper.

    Requirement: R1 — Pathlib Standardization Across Workflows
    Scenario: Artifact orchestrator avoids os.path
    """

    def test_joins_paths(self, tmp_path: Path) -> None:
        """Verify normal path joining."""
        base = tmp_path.resolve()
        result = safe_join(base, "subdir", "file.txt")
        assert result == base / "subdir" / "file.txt"

    def test_absolute_base_required(self) -> None:
        """Verify ValueError for relative base path."""
        with pytest.raises(ValueError, match="Base path must be absolute"):  # type: ignore[call-arg]
            safe_join(Path("relative"), "file.txt")

    def test_traversal_prevented(self, tmp_path: Path) -> None:
        """Verify directory traversal attempts raise ValueError."""
        base = tmp_path.resolve()
        with pytest.raises(ValueError, match="Path escapes base directory"):  # type: ignore[call-arg]
            safe_join(base, "..", "etc", "passwd")

    def test_dot_dot_in_middle_rejected(self, tmp_path: Path) -> None:
        """Verify .. components anywhere in path are detected."""
        base = tmp_path.resolve()
        with pytest.raises(ValueError, match="Path escapes base directory"):  # type: ignore[call-arg]
            safe_join(base, "subdir", "..", "..", "etc")


class TestReadText:
    """Test suite for read_text helper.

    Requirement: R1 — Pathlib Standardization Across Workflows
    """

    def test_reads_file(self, tmp_path: Path) -> None:
        """Verify successful file read."""
        target = tmp_path / "test.txt"
        target.write_text("hello world", encoding="utf-8")
        assert read_text(target) == "hello world"

    def test_custom_encoding(self, tmp_path: Path) -> None:
        """Verify encoding parameter is respected."""
        target = tmp_path / "test.txt"
        target.write_text("café", encoding="utf-8")
        assert read_text(target, encoding="utf-8") == "café"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for missing file."""
        target = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            read_text(target)


class TestWriteText:
    """Test suite for write_text helper.

    Requirement: R1 — Pathlib Standardization Across Workflows
    """

    def test_writes_file(self, tmp_path: Path) -> None:
        """Verify successful file write."""
        target = tmp_path / "output.txt"
        write_text(target, "content")
        assert target.read_text(encoding="utf-8") == "content"

    def test_creates_parents(self, tmp_path: Path) -> None:
        """Verify parent directories are created."""
        target = tmp_path / "a" / "b" / "file.txt"
        write_text(target, "data")
        assert target.read_text(encoding="utf-8") == "data"

    def test_round_trip(self, tmp_path: Path) -> None:
        """Verify write then read produces original content."""
        target = tmp_path / "roundtrip.txt"
        original = "test content\nwith newlines"
        write_text(target, original)
        assert read_text(target) == original


class TestAtomicWrite:
    """Test suite for atomic_write helper.

    Requirement: R1 — Pathlib Standardization Across Workflows
    """

    def test_writes_text_atomically(self, tmp_path: Path) -> None:
        """Verify text mode atomic write."""
        target = tmp_path / "atomic.txt"
        atomic_write(target, "safe content", mode="text")
        assert read_text(target) == "safe content"

    def test_writes_binary_atomically(self, tmp_path: Path) -> None:
        """Verify binary mode atomic write."""
        target = tmp_path / "atomic.bin"
        data = b"\x00\x01\x02"
        atomic_write(target, data, mode="binary")
        assert target.read_bytes() == data

    def test_text_mode_requires_str(self, tmp_path: Path) -> None:
        """Verify ValueError for wrong data type in text mode."""
        target = tmp_path / "bad.txt"
        with pytest.raises(ValueError, match="text mode requires str"):  # type: ignore[call-arg]
            atomic_write(target, b"bytes", mode="text")

    def test_binary_mode_requires_bytes(self, tmp_path: Path) -> None:
        """Verify ValueError for wrong data type in binary mode."""
        target = tmp_path / "bad.bin"
        with pytest.raises(ValueError, match="binary mode requires bytes"):  # type: ignore[call-arg]
            atomic_write(target, "str", mode="binary")

    def test_creates_parents(self, tmp_path: Path) -> None:
        """Verify parent directories are created."""
        target = tmp_path / "nested" / "atomic.txt"
        atomic_write(target, "content", mode="text")
        assert read_text(target) == "content"
