# Vector Ingestion Contract

Dense vector ingestion is governed by a shared contract that combines a JSON Schema, typed Python helpers, RFC 9457 Problem Details, and Prometheus telemetry. This page summarises everything you need to validate payloads, troubleshoot CLI failures, and monitor FAISS index builds.

## JSON Schema

Ingestion payloads must conform to [`vector-ingestion.vector-batch.v1.json`](schemas/vector-ingestion.vector-batch.v1.json). Each entry contains a non-empty `key` and a `vector` array of numeric values with consistent dimensionality.

```{literalinclude} ../../schema/vector-ingestion/vector-batch.v1.schema.json
:language: json
:caption: Draft 2020-12 schema for FAISS vector ingestion payloads
```

Validation happens before any numpy coercion:

1. `load_vector_batch_from_json` loads the raw JSON file.
2. `