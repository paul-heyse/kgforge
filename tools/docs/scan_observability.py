"""Scan Observability utilities."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build"


def read_ast(path: Path) -> tuple[str, ast.AST | None]:
    """Compute read ast.

    Carry out the read ast operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    Tuple[str, ast.AST | None]
        Description of return value.
    """




















    try:
        text = path.read_text("utf-8")
    except OSError:
        return ("", None)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return (text, None)
    return (text, tree)


def scan_file(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], bool]:
    """Compute scan file.

    Carry out the scan file operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    Tuple[List[dict[str, object]], List[dict[str, object]], List[dict[str, object]], bool]
        Description of return value.
    """




















    text, tree = read_ast(path)
    if not text:
        return ([], [], [], False)

    logs: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    config_hit = False

    if tree is None:
        return (logs, metrics, traces, config_hit)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            base = getattr(node.func, "value", None)

            if attr in {"debug", "info", "warning", "error", "exception", "critical"}:
                message = None
                if node.args:
                    message = ast.get_source_segment(text, node.args[0])
                logs.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "lineno": node.lineno,
                        "level": attr,
                        "message_template": (message or "").strip()[:240],
                    }
                )

            if isinstance(base, ast.Name):
                base_name = base.id
            else:
                base_name = None

            if base_name in {"prometheus_client", "metrics", "stats"} or attr in {
                "Counter",
                "Gauge",
                "Histogram",
                "Summary",
                "create_counter",
                "create_histogram",
                "create_gauge",
            }:
                metrics.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "lineno": node.lineno,
                        "call": (ast.get_source_segment(text, node) or "").strip()[:240],
                    }
                )

            if attr in {"start_span", "start_as_current_span", "set_attribute", "add_event"}:
                traces.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "lineno": node.lineno,
                        "call": (ast.get_source_segment(text, node) or "").strip()[:240],
                    }
                )

        if isinstance(node, ast.Name) and node.id in {"BaseSettings", "SettingsConfigDict"}:
            config_hit = True
        if isinstance(node, ast.Attribute) and getattr(node.attr, "lower", lambda: "")().startswith(
            "env"
        ):
            config_hit = config_hit or "environ" in ast.get_source_segment(text, node) or ""

    if "BaseSettings" in text or "pydantic_settings" in text:
        config_hit = True

    return (logs, metrics, traces, config_hit)


def main() -> None:
    """Compute main.

    Carry out the main operation.
    """




















    OUT.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    configs: list[str] = []

    if not SRC.exists():
        return

    for pyfile in SRC.rglob("*.py"):
        file_logs, file_metrics, file_traces, config_hit = scan_file(pyfile)
        if file_logs:
            logs.extend(file_logs)
        if file_metrics:
            metrics.extend(file_metrics)
        if file_traces:
            traces.extend(file_traces)
        if config_hit:
            configs.append(str(pyfile.relative_to(ROOT)))

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (OUT / "log_events.json").write_text(json.dumps(logs, indent=2) + "\n", encoding="utf-8")
    (OUT / "traces.json").write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")

    config_md = "# Config surfaces (quick index)\n\n"
    config_md += "\n".join(f"- `{path}`" for path in sorted(set(configs)))
    (OUT / "config.md").write_text(
        config_md + ("\n" if not config_md.endswith("\n") else ""), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
