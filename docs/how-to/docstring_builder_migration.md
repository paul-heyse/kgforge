# Docstring Builder Typed IR Migration Guide

This guide helps plugin authors and downstream consumers migrate to the typed
intermediate representation (IR) pipeline introduced in docstring builder 2.0.

## Overview

The docstring builder now uses typed data models (`DocstringIR`, `DocfactsDocumentPayload`,
`CliResult`) instead of `dict[str, Any]` payloads. This change improves type safety,
enables schema validation, and provides better error messages via RFC 9457 Problem Details.

## Feature Flag

The typed pipeline is enabled by default (`DOCSTRINGS_TYPED_IR=1`). To temporarily
disable it during migration:

```bash
DOCSTRINGS_TYPED_IR=0 uv run python -m tools.docstring_builder.cli generate
```

When disabled, schema validation runs in dry-run mode and logs warnings instead of
aborting. This allows incremental migration while maintaining backward compatibility.

## Plugin Migration

### Legacy Plugin Interface

Legacy plugins used the following signature:

```python
def apply(symbol: SymbolHarvest, ir: dict[str, Any]) -> dict[str, Any]:
    # Process ir dict
    return modified_ir
```

### New Plugin Protocol

Plugins must implement the `DocstringBuilderPlugin` protocol:

```python
from tools.docstring_builder.plugins.base import DocstringBuilderPlugin
from tools.docstring_builder.models import DocstringIR, PluginContext, PluginResultEnvelope

class MyPlugin:
    name: ClassVar[str] = "my_plugin"

    def supports(self, symbol: SymbolHarvest) -> bool:
        return symbol.kind == "function"

    def apply(
        self,
        symbol: SymbolHarvest,
        ir: DocstringIR,
        context: PluginContext,
    ) -> PluginResultEnvelope:
        # Process typed IR
        ir.notes.append("Processed by my_plugin")
        return PluginResultEnvelope(ir=ir, metadata={"plugin": self.name})
```

### Compatibility Shim

A compatibility shim wraps legacy plugins and emits deprecation warnings:

```python
from tools.docstring_builder.plugins.base import LegacyPluginAdapter

legacy_plugin = LegacyPluginAdapter(MyLegacyPlugin())
# Wraps legacy apply(symbol, ir) and converts to PluginResultEnvelope
```

The shim will be removed in a future release; migrate plugins to the Protocol
interface as soon as possible.

## Schema Validation

### DocFacts Schema

DocFacts payloads are validated against `docs/_build/schema_docfacts.json`:

```python
from tools.docstring_builder.models import (
    DocfactsDocumentPayload,
    SchemaViolationError,
    validate_docfacts_payload,
)

try:
    validate_docfacts_payload(payload)
except SchemaViolationError as exc:
    if exc.problem:
        print(f"Schema violation: {exc.problem['detail']}")
```

### CLI Output Schema

CLI JSON outputs are validated against `schema/tools/docstring_builder_cli.json`:

```python
from tools.docstring_builder.models import CliResult, validate_cli_output

cli_result = build_cli_result_skeleton(RunStatus.SUCCESS)
# ... populate fields ...
validate_cli_output(cli_result)
```

## Error Handling

### Problem Details

Errors now include RFC 9457 Problem Details payloads:

```python
from tools.docstring_builder.models import PROBLEM_DETAILS_EXAMPLE

# Example payload structure
{
    "type": "https://kgfoundry.dev/problems/docbuilder/schema-mismatch",
    "title": "DocFacts schema validation failed",
    "status": 422,
    "detail": "Field anchors.endLine is missing",
    "instance": "urn:docbuilder:run:2025-10-30T12:00:00Z",
    "extensions": {
        "schemaVersion": "2.0.0",
        "docstringBuilderVersion": "1.6.0",
        "symbol": "kg.module.function",
    },
}
```

See `docs/examples/docstring_builder_problem_details.json` for the canonical example.

## Observability

### Metrics

Prometheus metrics are emitted for all operations:

- `docbuilder_runs_total{status}` - Total builder runs
- `docbuilder_plugin_failures_total{plugin_name, error_type}` - Plugin failures
- `docbuilder_harvest_duration_seconds{status}` - Harvest operation duration
- `docbuilder_policy_duration_seconds{status}` - Policy engine duration
- `docbuilder_render_duration_seconds{status}` - Render operation duration
- `docbuilder_cli_duration_seconds{command, status}` - CLI operation duration

### Structured Logging

Logs include correlation IDs and structured fields:

```python
from tools._shared.logging import with_fields, get_logger

logger = get_logger(__name__)
logger = with_fields(logger, correlation_id=correlation_id, operation="harvest")
logger.info("Processing symbol", extra={"symbol_id": "kg.module.function"})
```

## Troubleshooting

### Schema Validation Failures

If validation fails, check:

1. **DocFacts version mismatch**: Ensure `docfactsVersion` matches the builder version
2. **Missing required fields**: Review `docs/_build/schema_docfacts.json` for required fields
3. **Type mismatches**: Ensure enum values match schema definitions

### Plugin Errors

If plugins fail:

1. **Protocol conformance**: Ensure plugins implement all required methods
2. **Type errors**: Use `pyright` and `pyrefly` to check plugin types
3. **Compatibility shim**: Check deprecation warnings for migration guidance

### Performance

If performance degrades:

1. **Disable validation temporarily**: Set `DOCSTRINGS_TYPED_IR=0`
2. **Check metrics**: Review `docbuilder_*_duration_seconds` histograms
3. **Profile operations**: Use `record_operation_metrics` context manager

## Migration Checklist

- [ ] Review plugin signatures and update to `DocstringBuilderPlugin` protocol
- [ ] Test plugins with `DOCSTRINGS_TYPED_IR=1` enabled
- [ ] Update error handling to use `SchemaViolationError` and Problem Details
- [ ] Verify schema validation passes for all payloads
- [ ] Update tests to use typed models (`DocstringIR`, `CliResult`, etc.)
- [ ] Remove compatibility shim usage after migration complete
- [ ] Update documentation to reference typed APIs

## Further Reading

- `tools/docstring_builder/models.py` - Typed model definitions
- `schema/tools/docstring_builder_cli.json` - CLI output schema
- `docs/_build/schema_docfacts.json` - DocFacts schema
- `tools/docstring_builder/observability.py` - Observability helpers
- `docs/examples/docstring_builder_problem_details.json` - Problem Details example

