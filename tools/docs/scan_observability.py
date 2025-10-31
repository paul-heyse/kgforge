#!/usr/bin/env python3
"""Overview of scan observability.

This module bundles scan observability logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

from tools._shared.logging import StructuredLoggerAdapter, get_logger, with_fields
from tools._shared.proc import ToolExecutionError, run_tool

LOGGER: StructuredLoggerAdapter = get_logger(__name__)


def _optional_import(name: str) -> ModuleType | None:
    """Import module if available.

    Parameters
    ----------
    name : str
        Description.

    Returns
    -------
    Any
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _optional_import(...)
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


yaml = _optional_import("yaml")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG_MD = OUT / "config.md"

# Linking
G_ORG = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA = os.getenv("DOCS_GITHUB_SHA")
LINK_MODE = os.getenv("DOCS_LINK_MODE", "both").lower()  # editor|github|both


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    """Configuration guardrails for metric instrumentation."""

    name_regex: str
    allowed_units: tuple[str, ...]
    counter_suffix: str
    require_unit_suffix: bool


@dataclass(frozen=True, slots=True)
class LabelsPolicy:
    """Policy for metric label usage."""

    reserved: tuple[str, ...]
    high_cardinality_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LogsPolicy:
    """Policy options for structured logging."""

    require_structured: bool = True


@dataclass(frozen=True, slots=True)
class TracesPolicy:
    """Policy constraints for tracing spans."""

    name_regex: str


@dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    """Typed observability scanning policy derived from YAML overrides."""

    metric: MetricPolicy
    labels: LabelsPolicy
    logs: LogsPolicy
    traces: TracesPolicy
    error_taxonomy_json: str | None = None


LintSeverity = Literal["error", "warning"]
LintKind = Literal["metric", "log", "trace"]


@dataclass(frozen=True, slots=True)
class LintFinding:
    """Structured lint finding produced during observability scans."""

    severity: LintSeverity
    kind: LintKind
    name: str
    rule: str
    message: str
    file: str
    lineno: int


def _rel(p: Path) -> str:
    """Rel.

    Parameters
    ----------
    p : Path
        Description.

    Returns
    -------
    str
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _rel(...)
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _sha() -> str:
    """Sha.

    Returns
    -------
    str
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _sha(...)
    """
    if G_SHA:
        return G_SHA
    log_adapter = with_fields(LOGGER, command=("git", "rev-parse", "HEAD"))
    try:
        result = run_tool(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=10.0)
        return result.stdout.strip() or "HEAD"
    except ToolExecutionError as exc:
        log_adapter.debug("Unable to resolve git SHA: %s", exc)
        return "HEAD"


def _gh_link(path: Path, start: int | None) -> str | None:
    """Gh link.

    Parameters
    ----------
    path : Path
        Description.
    start : int | None
        Description.

    Returns
    -------
    str | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _gh_link(...)
    """
    if not (G_ORG and G_REPO):
        return None
    frag = f"#L{start}" if start else ""
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{_sha()}/{_rel(path)}{frag}"


def _editor_link(path: Path, line: int | None) -> str:
    """Editor link.

    Parameters
    ----------
    path : Path
        Description.
    line : int | None
        Description.

    Returns
    -------
    str
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _editor_link(...)
    """
    ln = max(1, int(line or 1))
    return f"vscode://file/{_rel(path)}:{ln}:1"


# ---------- Policy ------------------------------------------------------------

DEFAULT_POLICY = ObservabilityPolicy(
    metric=MetricPolicy(
        name_regex=r"^[a-z][a-z0-9_]*$",
        allowed_units=(
            "seconds",
            "bytes",
            "meters",
            "grams",
            "joules",
            "volts",
            "amperes",
            "ratio",
        ),
        counter_suffix="_total",
        require_unit_suffix=True,
    ),
    labels=LabelsPolicy(
        reserved=("le", "quantile", "job", "instance"),
        high_cardinality_patterns=(
            r"user(_)?id",
            r"session(_)?id",
            r"request(_)?id",
            r"trace(_)?id",
            r"email",
            r"url",
            r"path",
        ),
    ),
    logs=LogsPolicy(require_structured=True),
    traces=TracesPolicy(name_regex=r"^[a-z0-9_.]+$"),
    error_taxonomy_json="docs/_build/error_taxonomy.json",
)

POLICY_PATH = ROOT / "docs" / "policies" / "observability.yml"


def _deep_merge_dicts(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    """Return a deep merge of ``override`` into ``base`` without mutating either mapping."""
    merged: dict[str, object] = dict(base)
    for key, override_value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _deep_merge_dicts(
                cast(Mapping[str, object], existing),
                cast(Mapping[str, object], override_value),
            )
            continue
        merged[key] = override_value
    return merged


def _coerce_str(value: object, fallback: str) -> str:
    return value if isinstance(value, str) else fallback


def _coerce_bool(value: object, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


def _coerce_optional_str(value: object, fallback: str | None) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return fallback


def _coerce_str_tuple(value: object, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            else:
                return fallback
        return tuple(items)
    return fallback


def _build_metric_policy(data: Mapping[str, object], default: MetricPolicy) -> MetricPolicy:
    return MetricPolicy(
        name_regex=_coerce_str(data.get("name_regex"), default.name_regex),
        allowed_units=_coerce_str_tuple(data.get("allowed_units"), default.allowed_units),
        counter_suffix=_coerce_str(data.get("counter_suffix"), default.counter_suffix),
        require_unit_suffix=_coerce_bool(
            data.get("require_unit_suffix"),
            default.require_unit_suffix,
        ),
    )


def _build_labels_policy(data: Mapping[str, object], default: LabelsPolicy) -> LabelsPolicy:
    return LabelsPolicy(
        reserved=_coerce_str_tuple(data.get("reserved"), default.reserved),
        high_cardinality_patterns=_coerce_str_tuple(
            data.get("high_cardinality_patterns"),
            default.high_cardinality_patterns,
        ),
    )


def _build_logs_policy(data: Mapping[str, object], default: LogsPolicy) -> LogsPolicy:
    return LogsPolicy(
        require_structured=_coerce_bool(
            data.get("require_structured"),
            default.require_structured,
        )
    )


def _build_traces_policy(data: Mapping[str, object], default: TracesPolicy) -> TracesPolicy:
    return TracesPolicy(name_regex=_coerce_str(data.get("name_regex"), default.name_regex))


def _build_policy_from_mapping(data: Mapping[str, object]) -> ObservabilityPolicy:
    metric_data = data.get("metric")
    labels_data = data.get("labels")
    logs_data = data.get("logs")
    traces_data = data.get("traces")

    if not isinstance(metric_data, Mapping) or not isinstance(labels_data, Mapping):
        raise ValueError("Policy overrides must include metric and labels mappings")
    if not isinstance(logs_data, Mapping) or not isinstance(traces_data, Mapping):
        raise ValueError("Policy overrides must include logs and traces mappings")

    metric = _build_metric_policy(cast(Mapping[str, object], metric_data), DEFAULT_POLICY.metric)
    labels = _build_labels_policy(cast(Mapping[str, object], labels_data), DEFAULT_POLICY.labels)
    logs = _build_logs_policy(cast(Mapping[str, object], logs_data), DEFAULT_POLICY.logs)
    traces = _build_traces_policy(cast(Mapping[str, object], traces_data), DEFAULT_POLICY.traces)
    error_taxonomy = _coerce_optional_str(
        data.get("error_taxonomy_json"),
        DEFAULT_POLICY.error_taxonomy_json,
    )
    return ObservabilityPolicy(
        metric=metric,
        labels=labels,
        logs=logs,
        traces=traces,
        error_taxonomy_json=error_taxonomy,
    )


def load_policy() -> ObservabilityPolicy:
    """Load the observability policy from disk, falling back to defaults on failure."""
    if yaml is None or not POLICY_PATH.exists():
        return DEFAULT_POLICY

    try:
        text = POLICY_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        with_fields(LOGGER, policy_path=str(POLICY_PATH)).warning(
            "Failed to read observability policy: %s", exc
        )
        return DEFAULT_POLICY

    try:
        overrides_raw: object = yaml.safe_load(text)
    except Exception as exc:
        yaml_error_type: type[Exception] = cast(
            type[Exception],
            getattr(yaml, "YAMLError", Exception),
        )
        if isinstance(exc, yaml_error_type):
            with_fields(LOGGER, policy_path=str(POLICY_PATH)).warning(
                "Failed to parse observability policy YAML: %s", exc
            )
            return DEFAULT_POLICY
        raise

    if overrides_raw is None:
        return DEFAULT_POLICY
    if not isinstance(overrides_raw, Mapping):
        with_fields(LOGGER, policy_path=str(POLICY_PATH)).warning(
            "Observability policy override must be a mapping, got %s",
            type(overrides_raw).__name__,
        )
        return DEFAULT_POLICY

    merged_data = _deep_merge_dicts(
        cast(Mapping[str, object], asdict(DEFAULT_POLICY)),
        cast(Mapping[str, object], overrides_raw),
    )
    try:
        return _build_policy_from_mapping(merged_data)
    except ValueError as exc:
        with_fields(LOGGER, policy_path=str(POLICY_PATH)).warning(
            "Observability policy overrides are invalid: %s", exc
        )
        return DEFAULT_POLICY


# ---------- Data models -------------------------------------------------------


@dataclass
class MetricRow:
    """Model the MetricRow.

    Represent the metricrow data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    name: str
    type: str | None
    unit: str | None
    labels: list[str]
    file: str
    lineno: int
    call: str
    recommended_aggregation: str | None
    source_link: dict[str, str]


@dataclass
class LogRow:
    """Model the LogRow.

    Represent the logrow data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    logger: str | None
    level: str
    message_template: str
    structured_keys: list[str]
    file: str
    lineno: int
    source_link: dict[str, str]
    runbook: str | None = None


@dataclass
class TraceRow:
    """Model the TraceRow.

    Represent the tracerow data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    span_name: str | None
    attributes: list[str]
    file: str
    lineno: int
    call: str
    source_link: dict[str, str]


# ---------- Heuristics & helpers ---------------------------------------------

_METRIC_CALL_TYPES = {
    # prometheus_client + helpers
    "Counter": "counter",
    "Gauge": "gauge",
    "Histogram": "histogram",
    "Summary": "summary",
    "create_counter": "counter",
    "create_gauge": "gauge",
    "create_histogram": "histogram",
}
_PROM_UNITS = set(DEFAULT_POLICY.metric.allowed_units)


def _first_str(node: ast.AST) -> str | None:
    """Return the first string literal value found in node's args, else None."""
    if isinstance(node, ast.Call) and node.args:
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            return arg0.value
        if isinstance(arg0, ast.JoinedStr):
            # f-string -> treat as dynamic
            return None
    return None


def _keywords_map(node: ast.Call, text: str) -> dict[str, str]:
    """Keywords map.

    Parameters
    ----------
    node : ast.Call
        Description.
    text : str
        Description.

    Returns
    -------
    dict[str, str]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _keywords_map(...)
    """
    out: dict[str, str] = {}
    for kw in node.keywords or []:
        k = kw.arg
        if k is None:
            continue
        try:
            vsrc = ast.get_source_segment(text, kw.value) or ""
        except (OSError, TypeError, ValueError) as exc:
            with_fields(LOGGER, keyword=k).debug("Unable to read source segment: %s", exc)
            vsrc = ""
        out[k] = vsrc.strip()
    return out


def _extract_labels_from_kw(kw_map: dict[str, str]) -> list[str]:
    """Extract labels from kw.

    Parameters
    ----------
    kw_map : dict[str, str]
        Description.

    Returns
    -------
    list[str]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _extract_labels_from_kw(...)
    """
    # prometheus_client: labelnames=(), namespace/subsystem help omitted here
    # common: labelnames=["method","status"], or .labels("method","status")—we only see construction here
    s = kw_map.get("labelnames") or kw_map.get("label_names") or ""
    # naive parse: find quoted tokens
    names = re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", s)
    return list(dict.fromkeys(names))


def _infer_unit_from_name(name: str) -> str | None:
    """Infer unit from name.

    Parameters
    ----------
    name : str
        Description.

    Returns
    -------
    str | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _infer_unit_from_name(...)
    """
    # Prometheus best-practice: include base unit in metric name (seconds, bytes, meters, grams, joules, volts, amperes, ratio)
    # and counters end with _total. We'll pull the suffix token.
    parts = name.split("_")
    suffix = parts[-1] if parts else ""
    return suffix if suffix in _PROM_UNITS else None


def _recommended_aggregation(mtype: str | None) -> str | None:
    """Recommended aggregation.

    Parameters
    ----------
    mtype : str | None
        Description.

    Returns
    -------
    str | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _recommended_aggregation(...)
    """
    if mtype == "counter":
        return "rate(sum by (...) (__metric__[5m]))"
    if mtype == "histogram":
        return "histogram_quantile(0.95, sum by (..., le) (rate(__metric___bucket[5m])))"
    if mtype == "summary":
        return "quantile_over_time(0.95, __metric__[5m])"
    return None


def _is_structured_logging(call: ast.Call, text: str) -> tuple[list[str], bool]:
    """Return (structured_keys, is_structured).

    Structured if kwargs or 'extra={'...'}' carries keys.
    Warn if %-format or f-string is used with positional args (unstructured).
    """
    keys: list[str] = []
    # capture kwargs
    for kw in call.keywords or []:
        if kw.arg and kw.arg not in {"exc_info", "stack_info"}:
            keys.append(kw.arg)
        if kw.arg == "extra":
            # try to parse dict keys
            src = ast.get_source_segment(text, kw.value) or ""
            keys += re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']\s*:", src)
    # detect f-string or % formatting in arg0 with additional args
    unstructured = False
    if call.args:
        a0 = call.args[0]
        if isinstance(a0, ast.JoinedStr):  # f-string
            unstructured = True
        # %-format style: "..." % (...)
        # (hard to detect reliably here; we mark as unstructured if there are extra positional args)
        if len(call.args) > 1:
            unstructured = True
    return (list(dict.fromkeys(keys)), not unstructured)


# ---------- Lint engine -------------------------------------------------------


def _lint_metric(policy: ObservabilityPolicy, row: MetricRow) -> list[LintFinding]:
    """Lint metric.

    Parameters
    ----------
    policy : dict
        Description.
    row : MetricRow
        Description.

    Returns
    -------
    list[dict]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _lint_metric(...)
    """
    errs: list[LintFinding] = []
    name_rx = re.compile(policy.metric.name_regex)
    if not name_rx.match(row.name or ""):
        errs.append(
            LintFinding(
                severity="error",
                kind="metric",
                name=row.name,
                rule="name_regex",
                message=f"Metric '{row.name}' must match regex {name_rx.pattern}",
                file=row.file,
                lineno=row.lineno,
            )
        )
    if policy.metric.require_unit_suffix and row.type in (
        "counter",
        "gauge",
        "histogram",
        "summary",
    ):
        unit = _infer_unit_from_name(row.name)
        if unit is None and row.type != "counter":
            errs.append(
                LintFinding(
                    severity="warning",
                    kind="metric",
                    name=row.name,
                    rule="unit_suffix",
                    message=(
                        "Metric should include base unit suffix (e.g., _seconds, _bytes). See "
                        "Prometheus naming."
                    ),
                    file=row.file,
                    lineno=row.lineno,
                )
            )
        if row.type == "counter" and not row.name.endswith(policy.metric.counter_suffix):
            errs.append(
                LintFinding(
                    severity="error",
                    kind="metric",
                    name=row.name,
                    rule="counter_total",
                    message="Counter names should end with '_total' in Prometheus exposition format.",
                    file=row.file,
                    lineno=row.lineno,
                )
            )
    # Reserved labels
    reserved = set(policy.labels.reserved)
    hc_rx = [re.compile(pat, re.IGNORECASE) for pat in policy.labels.high_cardinality_patterns]
    for lab in row.labels or []:
        if lab in reserved:
            errs.append(
                LintFinding(
                    severity="error",
                    kind="metric",
                    name=row.name,
                    rule="reserved_label",
                    message=f"Label '{lab}' is reserved (Prometheus/internal). Avoid defining it in instrumentation.",
                    file=row.file,
                    lineno=row.lineno,
                )
            )
        if any(rx.search(lab) for rx in hc_rx):
            errs.append(
                LintFinding(
                    severity="warning",
                    kind="metric",
                    name=row.name,
                    rule="high_cardinality_label",
                    message=(
                        f"Label '{lab}' frequently causes cardinality explosion; "
                        "reconsider (user_id/request_id/url/path…)."
                    ),
                    file=row.file,
                    lineno=row.lineno,
                )
            )
    return errs


def _lint_log(policy: ObservabilityPolicy, row: LogRow) -> list[LintFinding]:
    """Lint log.

    Parameters
    ----------
    policy : dict
        Description.
    row : LogRow
        Description.

    Returns
    -------
    list[dict]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _lint_log(...)
    """
    errs: list[LintFinding] = []
    if policy.logs.require_structured and not row.structured_keys:
        errs.append(
            LintFinding(
                severity="warning",
                kind="log",
                name=row.message_template[:50],
                rule="structured_logging",
                message="Prefer structured logging (key=value/extra=…) over %-format or f-strings.",
                file=row.file,
                lineno=row.lineno,
            )
        )
    return errs


def _lint_trace(policy: ObservabilityPolicy, row: TraceRow) -> list[LintFinding]:
    """Lint trace.

    Parameters
    ----------
    policy : dict
        Description.
    row : TraceRow
        Description.

    Returns
    -------
    list[dict]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _lint_trace(...)
    """
    errs: list[LintFinding] = []
    rx = re.compile(policy.traces.name_regex)
    if row.span_name and not rx.match(row.span_name):
        errs.append(
            LintFinding(
                severity="warning",
                kind="trace",
                name=row.span_name,
                rule="span_name",
                message=f"Span name should match regex {rx.pattern} (OTel naming).",
                file=row.file,
                lineno=row.lineno,
            )
        )
    return errs


# ---------- Scanner -----------------------------------------------------------


def read_ast(path: Path) -> tuple[str, ast.AST | None]:
    """Compute read ast.

    Carry out the read ast operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    Tuple[str, ast.AST | None]
        Description of return value.

    Examples
    --------
    >>> from tools.docs.scan_observability import read_ast
    >>> result = read_ast(...)
    >>> result  # doctest: +ELLIPSIS
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ("", None)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return (text, None)
    return (text, tree)


def scan_file(
    path: Path, policy: ObservabilityPolicy
) -> tuple[list[LogRow], list[MetricRow], list[TraceRow]]:
    """Compute scan file.

    Carry out the scan file operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    path : Path
        Description for ``path``.
    policy : collections.abc.Mapping
        Description for ``policy``.

    Returns
    -------
    Tuple[List[LogRow], List[MetricRow], List[TraceRow]]
        Description of return value.

    Examples
    --------
    >>> from tools.docs.scan_observability import scan_file
    >>> result = scan_file(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    del policy
    text, tree = read_ast(path)
    if not text or tree is None:
        return ([], [], [])
    logs: list[LogRow] = []
    metrics: list[MetricRow] = []
    traces: list[TraceRow] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            base = getattr(node.func, "value", None)
            base_name = base.id if isinstance(base, ast.Name) else None
            kwmap = _keywords_map(node, text)

            # --- LOGS (stdlib logging.*)
            if attr in {"debug", "info", "warning", "error", "exception", "critical"}:
                msg = ""
                if node.args:
                    seg = ast.get_source_segment(text, node.args[0])
                    msg = (seg or "").strip()[:240]
                keys, _ = _is_structured_logging(node, text)
                log_row = LogRow(
                    logger=base_name,
                    level=attr,
                    message_template=msg,
                    structured_keys=list(dict.fromkeys(keys)),
                    file=_rel(path),
                    lineno=node.lineno,
                    source_link=_links_for(path, node.lineno),
                )
                logs.append(log_row)

            # --- METRICS (prometheus_client.* or helper factories)
            if base_name in {"prometheus_client", "metrics", "stats"} or attr in _METRIC_CALL_TYPES:
                mtype = _METRIC_CALL_TYPES.get(attr)
                name = _first_str(node)
                labels = _extract_labels_from_kw(kwmap)
                unit = _infer_unit_from_name(name or "") if name else None
                metric_row = MetricRow(
                    name=name or "<dynamic>",
                    type=mtype,
                    unit=unit,
                    labels=labels,
                    file=_rel(path),
                    lineno=node.lineno,
                    call=(ast.get_source_segment(text, node) or "").strip()[:240],
                    recommended_aggregation=_recommended_aggregation(mtype),
                    source_link=_links_for(path, node.lineno),
                )
                metrics.append(metric_row)

            # Trace instrumentation (OpenTelemetry)
            if attr in {"start_span", "start_as_current_span"}:
                span_name = _first_str(node)
                trace_row = TraceRow(
                    span_name=span_name,
                    attributes=[],
                    file=_rel(path),
                    lineno=node.lineno,
                    call=(ast.get_source_segment(text, node) or "").strip()[:240],
                    source_link=_links_for(path, node.lineno),
                )
                traces.append(trace_row)
            if attr in {"set_attribute", "add_event"} and traces:
                # attach attributes to the last span in this file list if any
                seg = ast.get_source_segment(text, node) or ""
                key = _first_str(node) or ""
                traces[-1].attributes.append(key or seg[:80])

    return (logs, metrics, traces)


def _links_for(path: Path, lineno: int) -> dict[str, str]:
    """Links for.

    Parameters
    ----------
    path : Path
        Description.
    lineno : int
        Description.

    Returns
    -------
    dict[str, str]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _links_for(...)
    """
    out: dict[str, str] = {}
    if LINK_MODE in {"editor", "both"}:
        out["editor"] = _editor_link(path, lineno)
    if LINK_MODE in {"github", "both"}:
        gh = _gh_link(path, lineno)
        if gh:
            out["github"] = gh
    return out


def _write_config_summary(
    metrics: list[MetricRow], logs: list[LogRow], traces: list[TraceRow]
) -> None:
    """Emit a Markdown summary that Sphinx includes in observability docs."""
    if not metrics and not logs and not traces:
        CONFIG_MD.write_text(
            "No observability instrumentation detected for this repository.\n",
            encoding="utf-8",
        )
        return

    lines: list[str] = ["# Observability Instrumentation", ""]
    if metrics:
        lines.append("## Metrics")
        lines.append(
            f"Discovered {len(metrics)} metric definition(s); see `docs/_build/metrics.json`."
        )
        lines.append("")
    if logs:
        lines.append("## Logs")
        lines.append(
            f"Collected {len(logs)} structured log template(s); see `docs/_build/log_events.json`."
        )
        lines.append("")
    if traces:
        lines.append("## Traces")
        lines.append(f"Detected {len(traces)} span definition(s); see `docs/_build/traces.json`.")
        lines.append("")

    CONFIG_MD.write_text("\n".join(lines), encoding="utf-8")


def _scan_repository(
    policy: ObservabilityPolicy,
) -> tuple[list[LogRow], list[MetricRow], list[TraceRow], list[LintFinding]]:
    """Traverse the source tree and collect observability artefacts."""
    all_logs: list[LogRow] = []
    all_metrics: list[MetricRow] = []
    all_traces: list[TraceRow] = []
    lints: list[LintFinding] = []

    for py in SRC.rglob("*.py"):
        file_logs, file_metrics, file_traces = scan_file(py, policy)
        all_logs.extend(file_logs)
        all_metrics.extend(file_metrics)
        all_traces.extend(file_traces)

        for metric in file_metrics:
            lints.extend(_lint_metric(policy, metric))
        for log_event in file_logs:
            lints.extend(_lint_log(policy, log_event))
        for trace in file_traces:
            lints.extend(_lint_trace(policy, trace))

    return all_logs, all_metrics, all_traces, lints


def _load_runbooks(policy: ObservabilityPolicy) -> dict[str, str]:
    """Load optional runbook mappings from the error taxonomy."""
    taxonomy = policy.error_taxonomy_json or ""
    tax_path = ROOT / taxonomy if taxonomy else ROOT / ""
    if not tax_path.exists():
        return {}

    try:
        tax_text = tax_path.read_text(encoding="utf-8")
    except OSError as exc:
        with_fields(LOGGER, taxonomy=str(tax_path)).warning(
            "Failed to read error taxonomy JSON: %s", exc
        )
        return {}

    try:
        tax_data_raw = json.loads(tax_text)
    except json.JSONDecodeError as exc:
        with_fields(LOGGER, taxonomy=str(tax_path)).warning(
            "Failed to parse error taxonomy JSON: %s", exc
        )
        return {}

    if not isinstance(tax_data_raw, Mapping):
        return {}

    runbooks: dict[str, str] = {}
    errors_field = tax_data_raw.get("errors")
    if isinstance(errors_field, Sequence):
        for item in errors_field:
            if not isinstance(item, Mapping):
                continue
            message = item.get("message")
            extensions = item.get("extensions")
            if not isinstance(message, str) or not isinstance(extensions, Mapping):
                continue
            runbook_value = extensions.get("runbook")
            if isinstance(runbook_value, str):
                runbooks[message] = runbook_value
    return runbooks


def _apply_runbooks(logs: list[LogRow], runbooks: dict[str, str]) -> None:
    """Attach runbook URLs to log rows when taxonomy entries match."""
    if not runbooks:
        return

    for row in logs:
        if row.runbook:
            continue
        template = row.message_template or ""
        for message, runbook in runbooks.items():
            if message and message in template:
                row.runbook = runbook
                break


def _write_outputs(
    metrics: list[MetricRow],
    logs: list[LogRow],
    traces: list[TraceRow],
    lints: list[LintFinding],
) -> None:
    """Persist JSON artefacts used by downstream documentation builders."""
    metrics_payload = [cast(dict[str, object], asdict(x)) for x in metrics]
    (OUT / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    log_payload = [cast(dict[str, object], asdict(x)) for x in logs]
    (OUT / "log_events.json").write_text(
        json.dumps(log_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    trace_payload = [cast(dict[str, object], asdict(x)) for x in traces]
    (OUT / "traces.json").write_text(
        json.dumps(trace_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lint_payload = [cast(dict[str, object], asdict(finding)) for finding in lints]
    (OUT / "observability_lint.json").write_text(
        json.dumps(lint_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_config_summary(metrics, logs, traces)


def _summarize_exit(
    metrics: list[MetricRow],
    logs: list[LogRow],
    traces: list[TraceRow],
    lints: list[LintFinding],
) -> int:
    """Report a summary to stdout and return the desired exit code."""
    fail = os.getenv("OBS_FAIL_ON_LINT", "0") == "1"
    error_count = sum(1 for item in lints if item.severity == "error")
    warning_count = sum(1 for item in lints if item.severity == "warning")
    if fail and error_count:
        LOGGER.error(
            "[obs] %d error(s), %d warning(s) — failing (OBS_FAIL_ON_LINT=1)",
            error_count,
            warning_count,
        )
        return 2

    LOGGER.info(
        "[obs] metrics=%d logs=%d traces=%d; lint errors=%d warnings=%d",
        len(metrics),
        len(logs),
        len(traces),
        error_count,
        warning_count,
    )
    return 0


def main() -> int:
    """Coordinate observability scanning."""
    policy = load_policy()
    if not SRC.exists():
        return 0

    logs, metrics, traces, lints = _scan_repository(policy)
    runbooks = _load_runbooks(policy)
    _apply_runbooks(logs, runbooks)
    _write_outputs(metrics, logs, traces, lints)
    exit_code = _summarize_exit(metrics, logs, traces, lints)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
