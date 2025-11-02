# Vector Ingestion Contract

Dense vector ingestion is governed by a shared contract that combines a JSON Schema, typed Python helpers, RFC 9457 Problem Details, and Prometheus telemetry. This page summarises everything you need to validate payloads, troubleshoot CLI failures, and monitor FAISS index builds.

## JSON Schema

Ingestion payloads must conform to [`vector-ingestion.vector-batch.v1.json`](schemas/vector-ingestion.vector-batch.v1.json). Each entry contains a non-empty `key` and a `vector` array of numeric values with consistent dimensionality.

```{literalinclude} ../../schema/vector-ingestion/vector-batch.v1.schema.json
:language: json
:caption: Draft 2020-12 schema for FAISS vector ingestion payloads
```

Validation happens before any numpy coercion:

1. `_load_vectors_from_json` loads the raw JSON file.
2. `_validate_vector_payload` applies the schema via `jsonschema.Draft202012Validator`.
3. On success the payload is converted into a typed `VectorBatch` (`kgfoundry_common.vector_types`).

Any schema violation raises `VectorValidationError` with detailed error messages that are surfaced to callers.

## Problem Details & CLI Behaviour

The FAISS CLI (`orchestration.cli:index_faiss`) wraps ingestion failures in an RFC 9457 Problem Details payload and exits with code 1. The emitted JSON includes the schema ID, the offending file, the correlation ID, and a capped list of validation errors.

```{literalinclude} ../examples/problem_details/vector-ingestion-invalid-vector.json
:language: json
:caption: Example Problem Details response for invalid vector payloads
```

Operators can rerun the CLI with corrected data or use the correlation ID to trace logs and metrics.

## Metrics & Telemetry

`FaissVectorstoreFactory` records ingestion metrics via Prometheus:

- `kgfoundry_vector_ingestion_total{stage="ingestion",operation,status}` counts build/load/save attempts.
- `kgfoundry_vector_ingestion_duration_seconds{stage="ingestion",operation}` records latency by operation.

Both counters include the optional correlation ID in structured logs, making it easy to connect CLI failures with backend telemetry.

## Related APIs

- `kgfoundry_common.vector_types` – typed helpers (`VectorBatch`, `VectorValidationError`, `coerce_vector_batch`).
- `search_api.vectorstore_factory` – telemetry-enabled FAISS factory that consumes `VectorBatch` and exposes build/load/save instrumentation.
- `orchestration.cli:index_faiss` – CLI entry point that validates payloads, emits Problem Details, and forwards batches to the factory.

