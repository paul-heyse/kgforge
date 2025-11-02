# Testing & Observability Strategy

## Overview

This document describes the comprehensive testing and observability approach for kgfoundry, implementing **table-driven tests**, **doctest integration**, and **structured observability validation**.

---

## 1. Table-Driven Testing (Principle 3: Testing Strategy)

### Architecture

- **Fixtures**: Shared in `tests/conftest.py` with scoped setup/teardown
- **Parametrization**: All variants and edge cases covered via `@pytest.mark.parametrize`
- **Naming**: Clear IDs for each case (e.g., `multiple_facets`, `large_pool`)
- **Assertions**: Happy path + edges + failure modes for each test class

### Example: Search Options Coverage

```python
@pytest.mark.parametrize(
    "facets,candidate_pool,alpha,description",
    [
        (["document", "section"], 1000, 0.5, "multiple_facets"),
        (["document"], 5000, 0.7, "large_pool"),
        ([], 1000, 0.5, "no_facets"),
    ],
    ids=["multiple_facets", "large_pool", "no_facets"],
)
def test_search_options_valid(...):
    """Verify valid SearchOptions configurations are accepted."""
```

**Benefits**:
- ✅ Each variant tested independently
- ✅ Clear IDs for debugging
- ✅ Easy to add new cases
- ✅ Failures show exactly which parameter combination broke

---

## 2. Observability Assertions (Principle 5 + 9: Logging & Observability)

### Structured Logging

**Requirements**: Every operation must emit:
- `operation` (string) — operation name
- `status` (string) — "success" or "error"
- `error_type` (string, on failure) — exception class name
- `correlation_id` (string) — request/task ID

**Test Pattern**:

```python
def test_operation_logs_structure(caplog, structured_log_asserter):
    caplog.set_level(logging.INFO)
    
    # Do operation
    result = my_operation()
    
    # Assert structured fields
    for record in caplog.records:
        if record.__dict__.get("operation") == "my_op":
            structured_log_asserter(
                record,
                {"operation", "status", "correlation_id"}
            )
```

### Prometheus Metrics

**Requirements**: Counters and histograms for:
- `<operation>_total` — total operation count
- `<operation>_errors_total` — error count
- `<operation>_duration_seconds` — operation latency

**Test Pattern**:

```python
def test_operation_increments_metrics(prometheus_registry, metrics_asserter):
    # Do operation
    my_operation()
    
    # Assert metrics
    metrics_asserter("my_op_total", value=1)
    metrics_asserter("my_op_errors_total", value=0)
```

### OpenTelemetry Traces

**Requirements**: Spans must include:
- Span name: `<operation_name>`
- Attributes: `operation`, `status`, `error_type` (if error)
- Event on error: `exception` with message

**Test Pattern**:

```python
def test_operation_emits_trace(trace_exporter):
    # Do operation
    my_operation()
    
    # Assert trace
    spans = trace_exporter.get_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "my_operation"
    assert span.attributes["operation"] == "my_op"
```

---

## 3. Idempotency & Retry Tests (Principle 15: Idempotency)

### Idempotency Verification

**Pattern**: Run operation twice; assert identical results.

```python
def test_operation_idempotent(temp_index_dir):
    # Run 1
    result_1 = index_command(input_file, output_1)
    
    # Run 2 (with different output path, same input)
    result_2 = index_command(input_file, output_2)
    
    # Compare
    assert output_1.stat().st_size == output_2.stat().st_size
    assert open(output_1).read() == open(output_2).read()
```

### Retry Semantics

**Pattern**: Verify retry doesn't cause duplicate side effects.

```python
def test_retry_no_duplicates(mock_backend):
    # Call succeeds after one retry
    result = client.search_with_retry(query, max_retries=3)
    
    # Assert backend was called exactly once (or retried minimally)
    assert mock_backend.call_count <= 3
    
    # Assert result unchanged across retries
    assert result == expected_result
```

---

## 4. Doctest Integration (Principle 3 + 13: Testing & Documentation)

### Configuration

Add to `pytest.ini`:

```ini
[pytest]
addopts = --doctest-modules
testpaths = tests src
```

### Docstring Examples

**Pattern**:

```python
def build_search_options(embedding_model: str, facets: list[str]) -> SearchOptions:
    """Build SearchOptions from configuration.
    
    Parameters
    ----------
    embedding_model : str
        Name of embedding model.
    facets : list[str]
        Filter facets.
    
    Returns
    -------
    SearchOptions
        Configured options object.
    
    Examples
    --------
    >>> options = build_search_options("bert", ["document"])
    >>> options.embedding_model
    'bert'
    >>> options.facets
    ['document']
    """
```

**Testing**: `uv run pytest --doctest-modules src/`

---

## 5. Problem Details Validation (Principle 1 + 2 + 10: Clarity & Data Contracts)

### Schema Compliance

All HTTP error responses must follow [RFC 9457 Problem Details](https://tools.ietf.org/html/rfc9457).

**Example**:

```json
{
  "type": "https://kgfoundry.dev/problems/search/missing-index",
  "title": "Search Index Not Found",
  "status": 404,
  "detail": "Index 'catalog-v1' does not exist",
  "instance": "urn:operation:search:req-abc123",
  "correlation_id": "req-abc123"
}
```

**Test Pattern**:

```python
def test_error_returns_problem_details(problem_details_loader):
    response = client.search(query="missing")
    
    assert response.status_code == 404
    
    problem = response.json()
    assert problem["type"] == "https://kgfoundry.dev/problems/search/missing-index"
    assert problem["status"] == 404
    assert "correlation_id" in problem
    
    # Compare to schema
    expected = problem_details_loader("search-missing-index")
    assert problem.keys() == expected.keys()
```

---

## 6. Fixture Organization

### Root Fixtures (`tests/conftest.py`)

- `temp_index_dir` — Temporary directory for test artifacts
- `prometheus_registry` — Isolated Prometheus registry
- `problem_details_loader` — Load Problem Details examples
- `structured_log_asserter` — Assert structured log fields
- `metrics_asserter` — Assert Prometheus metrics

### Package-Specific Fixtures

**`tests/agent_catalog/conftest.py`**:
- `search_options_factory` — Create SearchOptions variants

**`tests/orchestration/conftest.py`**:
- `cli_runner` — Execute CLI commands with captured output

**`tests/kgfoundry_common/conftest.py`**:
- `logging_config` — Configure structured logging for tests

---

## 7. Quality Gates

### Local Testing

```bash
# Unit tests with coverage
uv run pytest -q tests/ --cov=src --cov-report=html

# Doctests
uv run pytest --doctest-modules src/

# Type checking (no errors allowed)
uv run pyrefly check && uv run mypy --config-file mypy.ini

# Linting (no suppressions)
uv run ruff format && uv run ruff check --fix
```

### CI Gates

- **Precommit**: `ruff format` + `ruff check`
- **Lint**: Full Ruff suite
- **Types**: `pyrefly check` + `mypy --strict`
- **Tests**: `pytest -q` with coverage ≥ 90%
- **Docs**: `make artifacts && git diff --exit-code`

---

## 8. Design Principles Alignment

| Principle | How Addressed |
|-----------|--------------|
| **1. Clarity** | Public APIs have clear signatures; examples in doctests |
| **2. Data Contracts** | Problem Details examples in schema/; schema validation tests |
| **3. Testing Strategy** | Table-driven parametrization; doctests; edge cases |
| **4. Type Safety** | All fixtures typed; no `Any` in test code |
| **5. Logging & Errors** | Structured logs asserted; Problem Details validated |
| **9. Observability** | Logs, metrics, traces captured and validated |
| **13. Documentation** | Doctest examples are copy-ready and runnable |
| **15. Idempotency** | Explicit idempotency tests prove retry safety |

---

## 9. Running Tests

### Execute All Tests

```bash
uv run pytest -q tests/
```

### Execute by Module

```bash
# Search options
uv run pytest -q tests/agent_catalog/test_search_options_table.py

# CLI idempotency
uv run pytest -q tests/orchestration/test_index_cli_idempotency.py

# Doctest examples
uv run pytest --doctest-modules src/kgfoundry/agent_catalog/search.py
```

### With Detailed Output

```bash
uv run pytest -v tests/ --tb=short
```

---

## 10. Adding New Tests

### Checklist

- [ ] Add test class with clear docstring (what's being tested)
- [ ] Use parametrization for variants (avoid duplication)
- [ ] Parametrize with clear IDs (not just `test_0`, `test_1`)
- [ ] Assert structure matches spec (requirements → scenarios)
- [ ] Include happy path + edges + failure modes
- [ ] Assert structured logs / metrics / traces
- [ ] Add doctest example if public API
- [ ] Run `pytest -v` to confirm

### Template

```python
@pytest.mark.parametrize(
    "input_value,expected_output,description",
    [
        ("case1_input", "case1_output", "success"),
        ("case2_input", "case2_output", "edge"),
        ("invalid", None, "failure"),
    ],
    ids=["success", "edge", "failure"],
)
def test_my_function(input_value, expected_output, description, caplog):
    """Verify my_function handles {description} case correctly."""
    caplog.set_level(logging.INFO)
    
    if expected_output is None:
        with pytest.raises(ValueError):
            my_function(input_value)
    else:
        result = my_function(input_value)
        assert result == expected_output
    
    # Assert logging
    assert any(record.levelname == "INFO" for record in caplog.records)
```

---

## References

- [Pytest Parametrization](https://docs.pytest.org/en/stable/how-to-parametrize.html)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [RFC 9457 Problem Details](https://tools.ietf.org/html/rfc9457)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [OpenTelemetry Python](https://open-telemetry.github.io/opentelemetry-python/)
