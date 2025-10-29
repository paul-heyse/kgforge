"""Policy engine for enforcing docstring quality gates."""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for limited environments
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from tools.docstring_builder.semantics import SemanticResult


class PolicyConfigurationError(RuntimeError):
    """Raised when policy configuration cannot be parsed."""


class PolicyAction(StrEnum):
    """Enumeration of supported policy actions."""

    ERROR = "error"
    WARN = "warn"
    AUTOFIX = "autofix"

    @classmethod
    def parse(cls, value: str) -> "PolicyAction":
        lowered = value.strip().lower()
        try:
            return cls(lowered)
        except ValueError as exc:
            raise PolicyConfigurationError(f"Unknown policy action: {value}") from exc


@dataclass(slots=True)
class PolicyException:
    """Represents an allowlisted violation."""

    symbol: str
    rule: str
    expires_on: _dt.date
    justification: str

    def is_active(self, today: _dt.date) -> bool:
        return self.expires_on >= today


@dataclass(slots=True)
class PolicySettings:
    """Resolved policy configuration with precedence applied."""

    coverage_threshold: float = 0.9
    coverage_action: PolicyAction = PolicyAction.ERROR
    missing_params_action: PolicyAction = PolicyAction.ERROR
    missing_returns_action: PolicyAction = PolicyAction.ERROR
    exceptions: list[PolicyException] = field(default_factory=list)

    def action_for(self, rule: str) -> PolicyAction:
        mapping: dict[str, PolicyAction] = {
            "coverage": self.coverage_action,
            "missing-params": self.missing_params_action,
            "missing-returns": self.missing_returns_action,
        }
        return mapping.get(rule, PolicyAction.ERROR)


@dataclass(slots=True)
class PolicyViolation:
    """Describes a policy violation detected during evaluation."""

    rule: str
    symbol: str
    action: PolicyAction
    message: str

    @property
    def fatal(self) -> bool:
        return self.action == PolicyAction.ERROR


@dataclass(slots=True)
class PolicyReport:
    """Summary of policy evaluation across the run."""

    coverage: float
    threshold: float
    violations: list[PolicyViolation]


def _read_pyproject_policy(repo_root: Path) -> Mapping[str, Any]:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    tool = data.get("tool", {})
    kgfoundry = tool.get("kgfoundry", {})
    docstrings = kgfoundry.get("docstrings", {})
    policy = docstrings.get("policy", {})
    return policy if isinstance(policy, Mapping) else {}


def _parse_exceptions(entries: Iterable[Mapping[str, Any]]) -> list[PolicyException]:
    parsed: list[PolicyException] = []
    for entry in entries:
        symbol = str(entry.get("symbol", "")).strip()
        rule = str(entry.get("rule", "missing-params")).strip() or "missing-params"
        expires_raw = entry.get("expires-on") or entry.get("expires_on")
        justification = str(entry.get("justification", "")).strip()
        if not symbol or not expires_raw:
            raise PolicyConfigurationError("Policy exception requires symbol and expires-on")
        try:
            expires_on = _dt.date.fromisoformat(str(expires_raw))
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise PolicyConfigurationError(f"Invalid expires-on value: {expires_raw}") from exc
        parsed.append(
            PolicyException(
                symbol=symbol,
                rule=rule,
                expires_on=expires_on,
                justification=justification,
            )
        )
    return parsed


def _apply_mapping(settings: PolicySettings, mapping: Mapping[str, Any]) -> None:
    if "coverage-threshold" in mapping:
        settings.coverage_threshold = float(mapping["coverage-threshold"])
    if "coverage_action" in mapping:
        settings.coverage_action = PolicyAction.parse(str(mapping["coverage_action"]))
    if "coverage-action" in mapping:
        settings.coverage_action = PolicyAction.parse(str(mapping["coverage-action"]))
    if "missing_params_action" in mapping:
        settings.missing_params_action = PolicyAction.parse(str(mapping["missing_params_action"]))
    if "missing-params-action" in mapping:
        settings.missing_params_action = PolicyAction.parse(str(mapping["missing-params-action"]))
    if "missing_returns_action" in mapping:
        settings.missing_returns_action = PolicyAction.parse(str(mapping["missing_returns_action"]))
    if "missing-returns-action" in mapping:
        settings.missing_returns_action = PolicyAction.parse(str(mapping["missing-returns-action"]))
    if "exceptions" in mapping:
        entries = mapping["exceptions"]
        if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
            settings.exceptions = _parse_exceptions(
                cast(Iterable[Mapping[str, Any]], entries)
            )


def _parse_override_pairs(raw: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise PolicyConfigurationError(f"Invalid override '{chunk}'")
        key, value = chunk.split("=", 1)
        overrides[key.strip().lower()] = value.strip()
    return overrides


def _apply_overrides(settings: PolicySettings, overrides: Mapping[str, str]) -> None:
    for key, value in overrides.items():
        if key in {"coverage", "coverage-threshold"}:
            settings.coverage_threshold = float(value)
        elif key in {"coverage-action", "coverage_action"}:
            settings.coverage_action = PolicyAction.parse(value)
        elif key in {"missing-params", "missing-params-action", "missing_params_action"}:
            settings.missing_params_action = PolicyAction.parse(value)
        elif key in {"missing-returns", "missing-returns-action", "missing_returns_action"}:
            settings.missing_returns_action = PolicyAction.parse(value)
        else:
            raise PolicyConfigurationError(f"Unknown policy override: {key}")


def load_policy_settings(
    repo_root: Path,
    *,
    cli_overrides: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
) -> PolicySettings:
    """Load policy settings from configuration, environment, and CLI overrides."""

    settings = PolicySettings()
    _apply_mapping(settings, _read_pyproject_policy(repo_root))
    env_mapping: Mapping[str, str] = env or os.environ
    env_raw = env_mapping.get("KGFOUNDRY_DOCSTRINGS_POLICY", "")
    if env_raw:
        _apply_overrides(settings, _parse_override_pairs(env_raw))
    if cli_overrides:
        _apply_overrides(settings, cli_overrides)
    return settings


class PolicyEngine:
    """Evaluate docstring semantics against configured policy rules."""

    def __init__(self, settings: PolicySettings) -> None:
        self.settings = settings
        self.total_symbols = 0
        self.documented_symbols = 0
        self.violations: list[PolicyViolation] = []
        self._today = _dt.date.today()

    def record(self, semantics: Iterable[SemanticResult]) -> None:
        for entry in semantics:
            self.total_symbols += 1
            documented = True
            missing_params: list[str] = []
            for parameter in entry.schema.parameters:
                description = parameter.description.strip()
                if not description or description.lower().startswith("todo"):
                    missing_params.append(parameter.name)
            if missing_params:
                recorded = self._register_violation(
                    rule="missing-params",
                    symbol=entry.symbol.qname,
                    detail=f"parameters missing descriptions: {', '.join(missing_params)}",
                )
                if recorded:
                    documented = False
            missing_returns = False
            for ret in entry.schema.returns:
                description = ret.description.strip()
                if not description or description.lower().startswith("todo"):
                    missing_returns = True
                    break
            if missing_returns:
                recorded = self._register_violation(
                    rule="missing-returns",
                    symbol=entry.symbol.qname,
                    detail="return values missing descriptions",
                )
                if recorded:
                    documented = False
            summary = entry.schema.summary.strip()
            if not summary or summary.lower().startswith("todo"):
                documented = False
            if documented:
                self.documented_symbols += 1

    def _register_violation(self, rule: str, symbol: str, detail: str) -> bool:
        exception = self._match_exception(symbol, rule)
        if exception and exception.is_active(self._today):
            return False
        action = self.settings.action_for(rule)
        if exception and not exception.is_active(self._today):
            detail = (
                f"{detail}; exception expired {exception.expires_on.isoformat()}"
                f" ({exception.justification or 'no justification provided'})"
            )
        message = f"{symbol}: {detail}"
        self.violations.append(PolicyViolation(rule=rule, symbol=symbol, action=action, message=message))
        return True

    def _match_exception(self, symbol: str, rule: str) -> PolicyException | None:
        for exception in self.settings.exceptions:
            if exception.symbol == symbol and exception.rule == rule:
                return exception
        return None

    def finalize(self) -> PolicyReport:
        coverage = 1.0 if self.total_symbols == 0 else self.documented_symbols / self.total_symbols
        if coverage + 1e-9 < self.settings.coverage_threshold:
            shortfall = f"coverage {coverage:.1%} below threshold {self.settings.coverage_threshold:.1%}"
            self._register_violation("coverage", "<aggregate>", shortfall)
        return PolicyReport(coverage=coverage, threshold=self.settings.coverage_threshold, violations=self.violations)


__all__ = [
    "PolicyAction",
    "PolicyConfigurationError",
    "PolicyEngine",
    "PolicyException",
    "PolicyReport",
    "PolicySettings",
    "PolicyViolation",
    "load_policy_settings",
]

