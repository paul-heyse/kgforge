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
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


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
    except Exception:
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
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
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

DEFAULT_POLICY: dict[str, Any] = {
    "metric": {
        # snake_case, must end with base unit or _total (counters) when using Prometheus exposition format
        "name_regex": r"^[a-z][a-z0-9_]*$",
        "allowed_units": [
            "seconds",
            "bytes",
            "meters",
            "grams",
            "joules",
            "volts",
            "amperes",
            "ratio",
        ],  # Prometheus base units
        "counter_suffix": "_total",
        "require_unit_suffix": True,  # applies to Prometheus-style metrics
    },
    "labels": {
        "reserved": ["le", "quantile", "job", "instance"],  # Prometheus specifics
        "high_cardinality_patterns": [
            r"user(_)?id",
            r"session(_)?id",
            r"request(_)?id",
            r"trace(_)?id",
            r"email",
            r"url",
            r"path",
        ],
    },
    "logs": {"require_structured": True},  # prefer structured keys over %-format/f-strings
    "traces": {"name_regex": r"^[a-z0-9_.]+$"},  # OTel-style dotted lowercase names
    "error_taxonomy_json": "docs/_build/error_taxonomy.json",  # optional; map messages/codes to runbooks
}

POLICY_PATH = ROOT / "docs" / "policies" / "observability.yml"


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a deep merge of ``override`` into ``base`` without mutating either mapping."""
    merged: dict[str, Any] = {}
    for key, base_value in base.items():
        if key in override:
            override_value = override[key]
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = _deep_merge_dicts(base_value, override_value)
            else:
                merged[key] = override_value
        else:
            merged[key] = base_value
    for key, override_value in override.items():
        if key not in base:
            merged[key] = override_value
    return merged


def load_policy() -> dict[str, Any]:
    """Compute load policy.

    Carry out the load policy operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    collections.abc.Mapping
        Description of return value.

    Examples
    --------
    >>> from tools.docs.scan_observability import load_policy
    >>> result = load_policy()
    >>> result  # doctest: +ELLIPSIS
    """
    if yaml is None or not POLICY_PATH.exists():
        return DEFAULT_POLICY
    try:
        overrides = yaml.safe_load(POLICY_PATH.read_text()) or {}
        if not isinstance(overrides, dict):
            return DEFAULT_POLICY
        return _deep_merge_dicts(DEFAULT_POLICY, overrides)
    except Exception:
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
_PROM_UNITS = set(DEFAULT_POLICY["metric"]["allowed_units"])


def _first_str(node: ast.AST, text: str) -> str | None:
    """Return the first string literal value found in node's args, else None."""
    if isinstance(node, ast.Call) and node.args:
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            return arg0.value
        if isinstance(arg0, ast.JoinedStr):
            # f-string -> treat as dynamic
            return None
    return None


def _keywords_map(node: ast.Call, text: str) -> dict[str, Any]:
    """Keywords map.

    Parameters
    ----------
    node : ast.Call
        Description.
    text : str
        Description.

    Returns
    -------
    dict[str, Any]
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
        except Exception:
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


def _lint_metric(policy: dict[str, Any], row: MetricRow) -> list[dict[str, Any]]:
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
    errs: list[dict[str, Any]] = []
    name_rx = re.compile(policy["metric"]["name_regex"])
    if not name_rx.match(row.name or ""):
        errs.append(
            {
                "severity": "error",
                "kind": "metric",
                "name": row.name,
                "rule": "name_regex",
                "message": f"Metric '{row.name}' must match regex {name_rx.pattern}",
                "file": row.file,
                "lineno": row.lineno,
            }
        )
    if policy["metric"]["require_unit_suffix"] and row.type in (
        "counter",
        "gauge",
        "histogram",
        "summary",
    ):
        unit = _infer_unit_from_name(row.name)
        if unit is None and row.type != "counter":
            errs.append(
                {
                    "severity": "warning",
                    "kind": "metric",
                    "name": row.name,
                    "rule": "unit_suffix",
                    "message": "Metric should include base unit suffix (e.g., _seconds, _bytes). See Prometheus naming.",
                    "file": row.file,
                    "lineno": row.lineno,
                }
            )
        if row.type == "counter" and not row.name.endswith(policy["metric"]["counter_suffix"]):
            errs.append(
                {
                    "severity": "error",
                    "kind": "metric",
                    "name": row.name,
                    "rule": "counter_total",
                    "message": "Counter names should end with '_total' in Prometheus exposition format.",
                    "file": row.file,
                    "lineno": row.lineno,
                }
            )
    # Reserved labels
    reserved = set(policy["labels"]["reserved"])
    hc_rx = [
        re.compile(pat, re.IGNORECASE) for pat in policy["labels"]["high_cardinality_patterns"]
    ]
    for lab in row.labels or []:
        if lab in reserved:
            errs.append(
                {
                    "severity": "error",
                    "kind": "metric",
                    "name": row.name,
                    "rule": "reserved_label",
                    "message": f"Label '{lab}' is reserved (Prometheus/internal). Avoid defining it in instrumentation.",
                    "file": row.file,
                    "lineno": row.lineno,
                }
            )
        if any(rx.search(lab) for rx in hc_rx):
            errs.append(
                {
                    "severity": "warning",
                    "kind": "metric",
                    "name": row.name,
                    "rule": "high_cardinality_label",
                    "message": f"Label '{lab}' frequently causes cardinality explosion; reconsider (user_id/request_id/url/path…).",
                    "file": row.file,
                    "lineno": row.lineno,
                }
            )
    return errs


def _lint_log(policy: dict[str, Any], row: LogRow) -> list[dict[str, Any]]:
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
    errs: list[dict[str, Any]] = []
    if policy["logs"].get("require_structured", True) and not row.structured_keys:
        errs.append(
            {
                "severity": "warning",
                "kind": "log",
                "name": row.message_template[:50],
                "rule": "structured_logging",
                "message": "Prefer structured logging (key=value/extra=…) over %-format or f-strings.",
                "file": row.file,
                "lineno": row.lineno,
            }
        )
    return errs


def _lint_trace(policy: dict[str, Any], row: TraceRow) -> list[dict[str, Any]]:
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
    errs: list[dict[str, Any]] = []
    rx = re.compile(policy["traces"]["name_regex"])
    if row.span_name and not rx.match(row.span_name):
        errs.append(
            {
                "severity": "warning",
                "kind": "trace",
                "name": row.span_name,
                "rule": "span_name",
                "message": f"Span name should match regex {rx.pattern} (OTel naming).",
                "file": row.file,
                "lineno": row.lineno,
            }
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
    path: Path, policy: dict[str, Any]
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
                name = _first_str(node, text)
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
                span_name = _first_str(node, text)
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
                key = _first_str(node, text) or ""
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
    policy: dict[str, Any],
) -> tuple[list[LogRow], list[MetricRow], list[TraceRow], list[dict[str, Any]]]:
    """Traverse the source tree and collect observability artefacts."""
    all_logs: list[LogRow] = []
    all_metrics: list[MetricRow] = []
    all_traces: list[TraceRow] = []
    lints: list[dict[str, Any]] = []

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


def _load_runbooks(policy: dict[str, Any]) -> dict[str, str]:
    """Load optional runbook mappings from the error taxonomy."""
    taxonomy = policy.get("error_taxonomy_json") or ""
    tax_path = ROOT / taxonomy if taxonomy else ROOT / ""
    if not tax_path.exists():
        return {}

    try:
        tax_data = json.loads(tax_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(tax_data, dict):
        return {}

    runbooks: dict[str, str] = {}
    for item in tax_data.get("errors", []):
        if (
            isinstance(item, dict)
            and "message" in item
            and "extensions" in item
            and isinstance(item.get("extensions"), dict)
            and "runbook" in item["extensions"]
        ):
            runbooks[item["message"]] = item["extensions"]["runbook"]
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
    lints: list[dict[str, Any]],
) -> None:
    """Persist JSON artefacts used by downstream documentation builders."""
    (OUT / "metrics.json").write_text(
        json.dumps([asdict(x) for x in metrics], indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "log_events.json").write_text(
        json.dumps([asdict(x) for x in logs], indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "traces.json").write_text(
        json.dumps([asdict(x) for x in traces], indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "observability_lint.json").write_text(
        json.dumps(lints, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_config_summary(metrics, logs, traces)


def _summarize_exit(
    metrics: list[MetricRow],
    logs: list[LogRow],
    traces: list[TraceRow],
    lints: list[dict[str, Any]],
) -> int:
    """Report a summary to stdout and return the desired exit code."""
    fail = os.getenv("OBS_FAIL_ON_LINT", "0") == "1"
    error_count = sum(1 for item in lints if item.get("severity") == "error")
    warning_count = sum(1 for item in lints if item.get("severity") == "warning")
    if fail and error_count:
        print(
            f"[obs] {error_count} error(s), {warning_count} warning(s) — failing (OBS_FAIL_ON_LINT=1)"
        )
        return 2

    print(
        f"[obs] metrics={len(metrics)} logs={len(logs)} traces={len(traces)}; "
        f"lint errors={error_count} warnings={warning_count}"
    )
    return 0


def main() -> int:
    """Coordinate observability scanning."""
    policy = load_policy()
    if not SRC.exists():
        sys.exit(0)
        return 0

    logs, metrics, traces, lints = _scan_repository(policy)
    runbooks = _load_runbooks(policy)
    _apply_runbooks(logs, runbooks)
    _write_outputs(metrics, logs, traces, lints)
    exit_code = _summarize_exit(metrics, logs, traces, lints)
    sys.exit(exit_code)
    return exit_code


if __name__ == "__main__":
    main()
