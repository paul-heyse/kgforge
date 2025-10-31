from __future__ import annotations

from tools import architecture


def test_tooling_layers_respect_domain_boundary() -> None:
    result = architecture.enforce_tooling_layers()
    assert result.is_success, "Tooling layering violations detected: " + ", ".join(
        result.violations
    )
