#!/usr/bin/env python3
"""
Observability scanner: logs, metrics, traces (+ policy lints).

Outputs (all under docs/_build/):
  - metrics.json           # [{name, type, unit, labels[], file, lineno, call, recommended_aggregation, source_link{editor,github}}]
  - log_events.json        # [{logger?, level, message_template, structured_keys[], file, lineno, source_link{...}}]
  - traces.json            # [{span_name, attributes[], file, lineno, call, source_link{...}}]
  - observability_lint.json# [{severity, kind, name, rule, message, file, lineno}]

Policy file:
  docs/policies/observability.yml  # naming rules, reserved labels, high-cardinality keys, etc.

Exit behavior:
  - default: prints counts and writes JSON files; does not fail the build
  - OBS_FAIL_ON_LINT=1 -> exit(2) if any error-level lint is present

Environment for links (consistent with your other tools):
  DOCS_LINK_MODE=editor|github|both (default: both)
  DOCS_GITHUB_ORG / DOCS_GITHUB_REPO / DOCS_GITHUB_SHA
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # policy
except Exception:
    yaml = None  # handled below

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build"
OUT.mkdir(parents=True, exist_ok=True)

# Linking
G_ORG = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA = os.getenv("DOCS_GITHUB_SHA")
LINK_MODE = os.getenv("DOCS_LINK_MODE", "both").lower()  # editor|github|both


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def _sha() -> str:
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
    if not (G_ORG and G_REPO):
        return None
    frag = f"#L{start}" if start else ""
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{_sha()}/{_rel(path)}{frag}"


def _editor_link(path: Path, line: int | None) -> str:
    ln = max(1, int(line or 1))
    return f"vscode://file/{_rel(path)}:{ln}:1"


# ---------- Policy ------------------------------------------------------------

DEFAULT_POLICY = {
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
    "logs": {
        "require_structured": True  # prefer structured keys over %-format/f-strings
    },
    "traces": {
        "name_regex": r"^[a-z0-9_.]+$"  # OTel-style dotted lowercase names
    },
    "error_taxonomy_json": "docs/_build/error_taxonomy.json",  # optional; map messages/codes to runbooks
}

POLICY_PATH = ROOT / "docs" / "policies" / "observability.yml"


def load_policy() -> dict[str, Any]:
    if yaml is None or not POLICY_PATH.exists():
        return DEFAULT_POLICY
    try:
        return {**DEFAULT_POLICY, **(yaml.safe_load(POLICY_PATH.read_text()) or {})}
    except Exception:
        return DEFAULT_POLICY


# ---------- Data models -------------------------------------------------------


@dataclass
class MetricRow:
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
    out = {}
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
    # prometheus_client: labelnames=(), namespace/subsystem help omitted here
    # common: labelnames=["method","status"], or .labels("method","status")—we only see construction here
    s = kw_map.get("labelnames") or kw_map.get("label_names") or ""
    # naive parse: find quoted tokens
    names = re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", s)
    return list(dict.fromkeys(names))


def _infer_unit_from_name(name: str) -> str | None:
    # Prometheus best-practice: include base unit in metric name (seconds, bytes, meters, grams, joules, volts, amperes, ratio)
    # and counters end with _total. We'll pull the suffix token.
    parts = name.split("_")
    suffix = parts[-1] if parts else ""
    return suffix if suffix in _PROM_UNITS else None


def _recommended_aggregation(mtype: str | None) -> str | None:
    if mtype == "counter":
        return "rate(sum by (...) (__metric__[5m]))"
    if mtype == "histogram":
        return "histogram_quantile(0.95, sum by (..., le) (rate(__metric___bucket[5m])))"
    if mtype == "summary":
        return "quantile_over_time(0.95, __metric__[5m])"
    return None


def _is_structured_logging(call: ast.Call, text: str) -> tuple[list[str], bool]:
    """
    Return (structured_keys, is_structured). Structured if kwargs or 'extra={'...'}' carries keys.
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


def _lint_metric(policy: dict, row: MetricRow) -> list[dict]:
    errs = []
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


def _lint_log(policy: dict, row: LogRow) -> list[dict]:
    errs = []
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


def _lint_trace(policy: dict, row: TraceRow) -> list[dict]:
    errs = []
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
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ("", None)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return (text, None)
    return (text, tree)


def scan_file(path: Path, policy: dict) -> tuple[list[LogRow], list[MetricRow], list[TraceRow]]:
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
                keys, ok_struct = _is_structured_logging(node, text)
                row = LogRow(
                    logger=base_name,
                    level=attr,
                    message_template=msg,
                    structured_keys=list(dict.fromkeys(keys)),
                    file=_rel(path),
                    lineno=node.lineno,
                    source_link=_links_for(path, node.lineno),
                )
                logs.append(row)

            # --- METRICS (prometheus_client.* or helper factories)
            if base_name in {"prometheus_client", "metrics", "stats"} or attr in _METRIC_CALL_TYPES:
                mtype = _METRIC_CALL_TYPES.get(attr)
                name = _first_str(node, text)
                labels = _extract_labels_from_kw(kwmap)
                unit = _infer_unit_from_name(name or "") if name else None
                row = MetricRow(
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
                metrics.append(row)

            # --- TRACES (OpenTelemetry)
            if attr in {"start_span", "start_as_current_span"}:
                span_name = _first_str(node, text)
                row = TraceRow(
                    span_name=span_name,
                    attributes=[],
                    file=_rel(path),
                    lineno=node.lineno,
                    call=(ast.get_source_segment(text, node) or "").strip()[:240],
                    source_link=_links_for(path, node.lineno),
                )
                traces.append(row)
            if attr in {"set_attribute", "add_event"}:
                # attach attributes to the last span in this file list if any
                if traces:
                    seg = ast.get_source_segment(text, node) or ""
                    key = _first_str(node, text) or ""
                    traces[-1].attributes.append(key or seg[:80])

    return (logs, metrics, traces)


def _links_for(path: Path, lineno: int) -> dict[str, str]:
    out = {}
    if LINK_MODE in {"editor", "both"}:
        out["editor"] = _editor_link(path, lineno)
    if LINK_MODE in {"github", "both"}:
        gh = _gh_link(path, lineno)
        if gh:
            out["github"] = gh
    return out


def main() -> None:
    policy = load_policy()
    if not SRC.exists():
        return

    all_logs: list[LogRow] = []
    all_metrics: list[MetricRow] = []
    all_traces: list[TraceRow] = []
    lints: list[dict] = []

    for py in SRC.rglob("*.py"):
        file_logs, file_metrics, file_traces = scan_file(py, policy)
        all_logs.extend(file_logs)
        all_metrics.extend(file_metrics)
        all_traces.extend(file_traces)

        for r in file_metrics:
            lints.extend(_lint_metric(policy, r))
        for r in file_logs:
            lints.extend(_lint_log(policy, r))
        for r in file_traces:
            lints.extend(_lint_trace(policy, r))

    # Optional runbook linking (error taxonomy)
    tax_path = ROOT / (policy.get("error_taxonomy_json") or "")
    if tax_path.exists():
        try:
            tax = json.loads(tax_path.read_text(encoding="utf-8"))
        except Exception:
            tax = {}
        # naïve example: map by message substring to extensions.runbook
        rb = {}
        for item in tax.get("errors", []):
            if "message" in item and "extensions" in item and "runbook" in item["extensions"]:
                rb[item["message"]] = item["extensions"]["runbook"]
        for row in all_logs:
            if not row.runbook:
                for k, v in rb.items():
                    if k and k in (row.message_template or ""):
                        row.runbook = v
                        break

    # Write outputs
    (OUT / "metrics.json").write_text(
        json.dumps([asdict(x) for x in all_metrics], indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "log_events.json").write_text(
        json.dumps([asdict(x) for x in all_logs], indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "traces.json").write_text(
        json.dumps([asdict(x) for x in all_traces], indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "observability_lint.json").write_text(
        json.dumps(lints, indent=2) + "\n", encoding="utf-8"
    )

    # Strict mode for CI
    fail = os.getenv("OBS_FAIL_ON_LINT", "0") == "1"
    if fail and any(x["severity"] == "error" for x in lints):
        print(
            f"[obs] {sum(1 for x in lints if x['severity'] == 'error')} error(s), "
            f"{sum(1 for x in lints if x['severity'] == 'warning')} warning(s) — failing (OBS_FAIL_ON_LINT=1)"
        )
        sys.exit(2)
    else:
        print(
            f"[obs] metrics={len(all_metrics)} logs={len(all_logs)} traces={len(all_traces)}; "
            f"lint errors={sum(1 for x in lints if x['severity'] == 'error')} warnings={sum(1 for x in lints if x['severity'] == 'warning')}"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
