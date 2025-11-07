# ADR 002: Observability with Prometheus Metrics

**Status**: Accepted  
**Date**: 2024-11-07  
**Context**: CodeIntel Enhancement Implementation

## Context

CodeIntel serves as a critical code intelligence layer for KGFoundry. To maintain production reliability, we need:
1. Real-time visibility into tool performance
2. Capacity planning metrics (index size, parse times)
3. Failure tracking and alerting
4. Future OpenTelemetry integration path

## Decision

We implemented a **dedicated observability module** (`codeintel/observability.py`) with Prometheus metrics and decorators for automatic instrumentation.

### Metrics Defined

#### Tool Execution Metrics
```python
TOOL_CALLS_TOTAL = Counter("codeintel_tool_calls_total", ["tool", "status"])
TOOL_DURATION_SECONDS = Histogram("codeintel_tool_duration_seconds", ["tool"])
TOOL_ERRORS_TOTAL = Counter("codeintel_tool_errors_total", ["tool", "error_type"])
```

#### Index Metrics
```python
INDEX_SIZE_SYMBOLS = Gauge("codeintel_index_symbols_total", ["lang"])
INDEX_SIZE_REFS = Gauge("codeintel_index_refs_total")
INDEX_SIZE_FILES = Gauge("codeintel_index_files_total")
INDEX_BUILD_DURATION_SECONDS = Histogram("codeintel_index_build_duration_seconds")
```

#### Parse Metrics
```python
PARSE_DURATION_SECONDS = Histogram("codeintel_parse_duration_seconds", ["lang"])
```

#### Rate Limiting
```python
RATE_LIMIT_REJECTIONS_TOTAL = Counter("codeintel_rate_limit_rejections_total")
```

### Decorator Pattern

```python
@instrument_tool("code.getOutline")
async def _tool_get_outline(self, payload: dict[str, Any]) -> dict[str, Any]:
    # Metrics automatically recorded: duration, status, errors
    pass
```

### Structured Logging Integration

```python
with log_operation("index_build", lang="python", files=100):
    # Operation logged with duration and context
    pass
```

### OpenTelemetry Placeholder

```python
with trace_span("parse_file", language="python", size=1024):
    # Future: OpenTelemetry span created
    # Current: Structured log emitted
    pass
```

## Consequences

### Positive

- ✅ **Automatic instrumentation**: Decorators eliminate boilerplate
- ✅ **Prometheus-native**: Standard metrics format for alerting/graphing
- ✅ **Low overhead**: Metrics are in-memory counters/gauges
- ✅ **Debuggable**: Metrics + structured logs provide full visibility
- ✅ **Future-proof**: OpenTelemetry integration ready

### Negative

- ⚠️ **Metrics storage**: Requires Prometheus deployment (optional)
- ⚠️ **Label cardinality**: Must avoid high-cardinality labels (e.g., file paths)

### Neutral

- 📊 **Testing**: Metrics can be asserted in tests for behavior verification
- 📊 **Standards compliance**: Follows AGENTS.md Principle 9 (Observability)

## Alternatives Considered

### 1. Logging Only
**Rejected**: No aggregation, hard to alert on

### 2. Custom Metrics Format
**Rejected**: Prometheus is industry standard

### 3. OpenTelemetry Only
**Rejected**: Too heavy for initial implementation; we added placeholder instead

## Implementation Details

### Helper Functions

- `record_parse(language, size_bytes, duration_s)` - Track parse operations
- `update_index_metrics(symbol_counts, ref_count, file_count)` - Update index gauges
- `log_operation(operation, **context)` - Context manager for structured logging

### Usage Example

```python
from codeintel.observability import instrument_tool, record_parse

# In tscore.py
duration = time.monotonic() - start
record_parse(lang.name, len(data), duration)

# In server.py
@instrument_tool("code.searchSymbols")
async def _tool_search_symbols(self, payload):
    # Automatic metrics
    pass
```

## References

- Prometheus best practices: https://prometheus.io/docs/practices/naming/
- OpenTelemetry: https://opentelemetry.io/
- AGENTS.md Principle 9: Observability (logs • metrics • traces)

