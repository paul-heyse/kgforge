# Observability Facade

## Overview

This note captures the contract for repository tooling that emits metrics,
structured logs, and traces. All call sites MUST rely on the shared typed
helpers in `kgfoundry_common.prometheus` (re-exported via
`tools._shared.prometheus`) and the structured logging utilities in
`kgfoundry_common.logging`. Doing so keeps optional dependencies safe while
maintaining static guarantees (no `type: ignore[...]`).

## Typed Metrics API

- Use `build_counter`, `build_gauge`, and `build_histogram` to materialise
  Prometheus primitives. The helpers return implementations of
  `CounterLike`, `GaugeLike`, and `HistogramLike` that degrade to no-op stubs
  automatically when `prometheus_client` is unavailable.
- Always provide explicit label names; prefer semantic tag sets such as
  `component`, `operation`, `status`, and `reason`.

```python
>>> from tools._shared.prometheus import build_counter
>>> counter = build_counter("demo_runs_total", "Demo invocations", ("status",))
>>> counter.labels(status="success").inc()
```

## Optional Dependency Fallbacks

The helpers are safe to import even when Prometheus is missing. When the
dependency is unavailable they return no-op stubs that satisfy the same
protocol surface.

```python
>>> from unittest.mock import patch
>>> import kgfoundry_common.prometheus as prom
>>> with patch("kgfoundry_common.prometheus._COUNTER_CONSTRUCTOR", None):
...     noop = prom.build_counter("noop_runs_total", "No-op counter")
...     noop.inc()
...     noop.labels(status="success").inc()
```

## Structured Logging & Correlation IDs

- Obtain loggers via `kgfoundry_common.logging.get_logger`.
- Bind structured fields (operation, status, duration_ms, correlation_id)
  using `kgfoundry_common.logging.with_fields`.
- Correlation IDs propagate through `ContextVar`s; use
  `set_correlation_id()` for request-scoped logging.

```python
>>> from kgfoundry_common.logging import get_logger, set_correlation_id, with_fields
>>> set_correlation_id("req-123")
>>> logger = get_logger(__name__)
>>> with with_fields(logger, operation="docs.build", status="success") as adapter:
...     adapter.info("Build completed", extra={"duration_ms": 42.0})
```

## Tracing Helpers

- Use `kgfoundry_common.observability.start_span` as the canonical context
  manager for OpenTelemetry spans. It records exceptions and sets error status
  codes automatically.
- Only attach lightweight attributes (strings, numbers, booleans) to avoid
  serialisation surprises.

## Extension Guidelines

1. Prefer adding new helper functions to `kgfoundry_common.observability`
   rather than introducing bespoke metrics modules.
2. Keep any new public API fully typed; expose `__all__` explicitly.
3. Avoid direct imports from `prometheus_client`; construct metrics through the
   shared builders to preserve fallbacks.
4. When introducing new log fields, document them here and ensure they are
   reflected in JSON Schema examples (e.g., Problem Details payloads or metrics
   manifests under `schema/`).

## Observability Payload Schema

Structured log extras for tooling SHOULD conform to the following JSON Schema
fragment (2020-12) when persisted or shipped cross-process:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ToolingOperationLog",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "operation": {"type": "string"},
    "status": {"type": "string", "enum": ["success", "warning", "error"]},
    "duration_ms": {"type": "number", "minimum": 0},
    "correlation_id": {"type": "string"}
  },
  "required": ["operation", "status", "duration_ms"]
}
```

Consumers that emit new fields MUST update the schema fragment and regenerate
artifacts via `make artifacts` so downstream documentation stays in sync.

