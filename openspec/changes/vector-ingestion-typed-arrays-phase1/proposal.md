## Why
Vector ingestion surfaces still rely on `NDArray[Any, dtype[Any]]` annotations and ad-hoc JSON parsing, forcing mypy suppressions (`# type: ignore[type-arg]`) in `orchestration.cli` and surfacing `Any` throughout `search_api.vectorstore_factory`. The absence of a shared vector contract permits ragged payloads, silently coerces wrong dtypes, and prevents Problem Details emission when ingestion fails. Without a canonical JSON Schema or structured validation helpers, operators cannot diagnose malformed datasets and static analyzers cannot guarantee safety.

## What Changes
- [x] **ADDED**: Capability spec `vector-ingestion/typed-arrays` defining typed vector batch contracts, schema validation, Problem Details surfaces, and regression coverage expectations.
- [ ] **ADDED**: Shared runtime + stub module `kgfoundry_common.vector_types` exposing immutable vector batch dataclasses, validation helpers, and TypeGuards without `Any` leakage.
- [ ] **ADDED**: JSON Schema `schema/vector-ingestion/vector-batch.v1.schema.json` and Problem Details example `schema/examples/problem_details/vector-ingestion-invalid-vector.json`, with runtime enforcement in orchestrator and factory entry points.
- [ ] **MODIFIED**: `orchestration.cli` vector loading flow to invoke schema validation, construct `VectorBatch` objects, propagate typed errors, and remove `# type: ignore` directives.
- [ ] **MODIFIED**: `search_api.vectorstore_factory` (and related adapters) to consume the shared contracts, enforce shape invariants, and log structured metadata for telemetry.
- [ ] **ADDED**: Regression suites covering helper coercion, CLI behaviour, schema validation, and Problem Details output, alongside doc updates that describe the new contracts and examples.

## Impact
- **Capabilities**: Introduces `vector-ingestion/typed-arrays` spec governing ingestion contracts; future vector capabilities depend on this baseline.
- **Runtime scope**: Touches `kgfoundry_common`, `search_api`, and `orchestration` packages to share types, remove ignores, and standardise validation. Impacts any feature building FAISS indexes from JSON payloads.
- **Schemas & artifacts**: Adds a canonical vector ingestion schema plus Problem Details example, requiring `make artifacts` and schema index updates.
- **Tooling**: Type-checking becomes stricter—calls that previously relied on implicit numpy casting must now satisfy `float32` requirements or handle raised exceptions.
- **Operations**: CLI failure output transitions to structured Problem Details, aiding on-call remediation and observability dashboards.

## Out of Scope
- Replacing FAISS adapters, BM25 pipelines, or vectorstore persistence logic beyond ingestion boundaries.
- GPU-specific performance optimisations or new accelerator backends.
- Broader schema governance for downstream services (only the ingestion payload is addressed here).

## Risks / Mitigations
- **Risk**: Legacy pipelines may supply float64 or ragged vectors and now fail fast.  
  **Mitigation**: Provide clear remediation guidance in Problem Details, document migration steps, and offer an opt-in dry-run mode for staging validation.
- **Risk**: Schema validation could slow large ingestions.  
  **Mitigation**: Use vectorised checks where possible, cache validated payload metadata, and document performance envelopes; add benchmarks in follow-up if needed.
- **Risk**: Introducing new shared modules can break layering tests.  
  **Mitigation**: Keep contracts in `kgfoundry_common` (approved shared package) and update `tools/check_imports.py` baselines if necessary after review.

## Alternatives Considered
- Maintain current numpy annotations and add targeted `typing.cast` calls—rejected because it perpetuates duplicated validation logic and lacks schema-first governance.
- Use Pydantic models directly for vector payloads—rejected to avoid per-record overhead and because numpy arrays integrate poorly with Pydantic without custom serializers.
- Keep validation inside orchestration only—rejected since factories and other consumers would still accept malformed batches and leak `Any` types downstream.


