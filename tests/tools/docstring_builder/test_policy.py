from __future__ import annotations

from collections.abc import Mapping

import pytest
from tools.docstring_builder.policy import (
    PolicyAction,
    PolicyConfigurationError,
    PolicySettings,
    _apply_mapping,
    _apply_overrides,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("coverage-threshold", 0.82),
        ("coverage_threshold", 0.65),
    ],
)
def test_apply_mapping_parses_threshold(key: str, expected: float) -> None:
    settings = PolicySettings()
    _apply_mapping(settings, {key: expected})
    assert settings.coverage_threshold == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "attribute"),
    [
        ("coverage-action", "coverage_action"),
        ("missing-params", "missing_params_action"),
        ("missing-returns-action", "missing_returns_action"),
        ("missing_examples_action", "missing_examples_action"),
        ("summary-mood", "summary_mood_action"),
        ("dataclass_parity", "dataclass_parity_action"),
    ],
)
def test_apply_mapping_accepts_aliases(key: str, attribute: str) -> None:
    settings = PolicySettings()
    _apply_mapping(settings, {key: "warn"})
    assert getattr(settings, attribute) is PolicyAction.WARN


def test_apply_mapping_unknown_key_raises() -> None:
    settings = PolicySettings()
    with pytest.raises(PolicyConfigurationError):
        _apply_mapping(settings, {"unsupported": "warn"})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"coverage": "0.9"}, 0.9),
        ({"coverage_threshold": "0.75"}, 0.75),
    ],
)
def test_apply_overrides_parses_threshold(overrides: Mapping[str, str], expected: float) -> None:
    settings = PolicySettings()
    _apply_overrides(settings, overrides)
    assert settings.coverage_threshold == pytest.approx(expected)


def test_apply_overrides_aliases_actions() -> None:
    settings = PolicySettings()
    overrides = {
        "missing-params-action": "warn",
        "summary_mood": "autofix",
    }
    _apply_overrides(settings, overrides)
    assert settings.missing_params_action is PolicyAction.WARN
    assert settings.summary_mood_action is PolicyAction.AUTOFIX


def test_apply_overrides_unknown_key_raises() -> None:
    settings = PolicySettings()
    with pytest.raises(PolicyConfigurationError):
        _apply_overrides(settings, {"unexpected": "warn"})
