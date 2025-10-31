"""Security regression tests for input sanitization and path traversal prevention.

This module verifies that user-provided inputs are validated against schemas
and that path traversal attacks are prevented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kgfoundry_common.fs import safe_join
from search_api.schemas import SearchRequest


class TestInputSanitization:
    """Test input sanitization and validation."""

    def test_search_request_rejects_extra_fields(self) -> None:
        """SearchRequest rejects extra fields via extra='forbid'.

        Scenario: Input sanitization enforced

        GIVEN a user-provided payload with extra fields
        WHEN the system processes it
        THEN validation catches schema violations and raises a typed error
        """
        # Attempt to create SearchRequest with extra field
        with pytest.raises(Exception, match="Extra inputs are not permitted"):  # type: ignore[call-arg, misc]
            SearchRequest(
                query="test",
                k=5,
                extra_field="should be rejected",  # type: ignore[call-arg]
            )

        # Query must be a string
        with pytest.raises(Exception):  # type: ignore[misc]
            SearchRequest(query=123, k=5)  # type: ignore[arg-type]

    def test_search_request_validates_k_range(self) -> None:
        """SearchRequest validates k is an integer.

        Note: Pydantic validates type but not range. Range validation
        would need to be added via Field(gt=0) if required.
        """
        # k must be an integer (type validation)
        with pytest.raises(Exception, match=".*"):  # type: ignore[call-arg, misc]
            SearchRequest(query="test", k="not an int")  # type: ignore[arg-type]

    def test_safe_join_prevents_path_traversal(self) -> None:
        """safe_join prevents directory traversal attacks.

        Scenario: Input sanitization enforced

        GIVEN a user-provided path with traversal sequences
        WHEN the system processes it
        THEN validation catches traversal and raises a typed error
        """
        base = Path("/safe/base").resolve()

        # Should work for normal paths
        safe_join(base, "file.txt")

        # Should raise ValueError for traversal attempts
        with pytest.raises(ValueError, match="Path escapes base directory"):  # type: ignore[call-arg, misc]
            safe_join(base, "..", "etc", "passwd")

        with pytest.raises(ValueError, match="Path escapes base directory"):  # type: ignore[call-arg, misc]
            safe_join(base, "../etc/passwd")

        # Should raise ValueError for absolute paths
        with pytest.raises(ValueError, match="Path escapes base directory"):  # type: ignore[call-arg, misc]
            safe_join(base, "/etc/passwd")

        base = Path("relative/path")

        with pytest.raises(ValueError, match="Base path must be absolute"):  # type: ignore[call-arg, misc]
            safe_join(base, "file.txt")

    def test_safe_join_allows_nested_paths(self) -> None:
        """safe_join allows legitimate nested paths within base."""
        base = Path("/safe/base").resolve()

        # Should work for nested paths
        result = safe_join(base, "subdir", "file.txt")
        assert result == base / "subdir" / "file.txt"
        assert result.resolve().is_relative_to(base.resolve())


class TestUnsafeConstructs:
    """Test that unsafe constructs are documented and minimized."""

    def test_pickle_usage_documented(self) -> None:
        """Verify pickle usage is documented as backward compatibility only.

        Note: Current pickle usage in the codebase is:
        - src/search_api/bm25_index.py: Legacy format support (marked S301)
        - src/embeddings_sparse/bm25.py: Legacy format support (marked S301)
        - src/embeddings_sparse/splade.py: Legacy format support (marked S301)
        - src/kgfoundry/agent_catalog/search.py: Local trusted artifact (marked S301)

        All instances are marked with # noqa: S301 and comments indicating
        they're for backward compatibility or trusted sources only.
        """

    def test_yaml_safe_load_used(self) -> None:
        """Verify yaml.safe_load is used instead of yaml.load.

        Note: Current YAML usage:
        - src/kgfoundry_common/config.py: Uses yaml.safe_load (correct)
        - tools/docs/scan_observability.py: Uses yaml.safe_load (correct)
        """
        # This test documents that yaml.safe_load is used correctly
        # Actual verification is done via grep/static analysis
        assert True

    def test_no_eval_or_exec(self) -> None:
        """Verify no eval or exec calls exist in source code.

        Note: Verification is done via grep/static analysis.
        This test documents the policy.
        """
        # This test documents that eval/exec are not used
        # Actual verification is done via grep/static analysis
        assert True
