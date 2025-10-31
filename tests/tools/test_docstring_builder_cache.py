from __future__ import annotations

from pathlib import Path

import pytest
from tools.docstring_builder.cache import BuilderCache


def test_builder_cache_resets_on_invalid_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{ invalid json }", encoding="utf-8")
    target_file = tmp_path / "example.py"
    target_file.write_text("print('hello')\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        cache = BuilderCache(cache_path)

    assert cache.needs_update(target_file, "hash")
    assert "resetting cache" in caplog.text.lower()
    assert not cache_path.exists()
