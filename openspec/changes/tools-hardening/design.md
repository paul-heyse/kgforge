## Context
The `tools/` directory orchestrates documentation, catalog generation, navmap maintenance, lint helpers, and CLI automation. These scripts evolved organically, mixing dynamic dictionaries, `print` debugging, blind `except Exception`, and direct `subprocess` usage. Ruff, mypy, and pyrefly collectively report >1.8k violations, blocking adoption of strict quality gates and creating risk of silent data corruption or insecure command execution. The docstring builder remediation (docstring-builder-hardening) established typed IRs and CLI schemas; this change extends the same rigor across the remainder of the tools stack, aligning with production standards for clarity, type safety, observability, and schema governance.

## Goals / Non-Goals
- **Goals:**
  - Establish shared typed infrastructure for logging, subprocess management, and Problem Details.
  - Adopt typed dataclasses/TypedDicts across docstring builder, documentation generators, and navmap utilities.
  - Secure subprocess invocations with allowlists, absolute paths, timeouts, and structured error handling.
  - Emit JSON Schema–validated machine outputs for CLIs and documentation pipelines.
  - Provide comprehensive pytest coverage (table-driven) for schemas, CLI behaviors, plugin regression flows, and navmap transformations.
  - Instrument tooling with structured logs, Prometheus metrics, and trace spans where applicable.
- **Non-Goals:**
  - Changing upstream schemas beyond required fields (DocFacts remains source of truth).
  - Migrating Agent Portal or docs site architecture (only inputs/outputs hardened).
  - Replacing existing templating engines (Jinja stays, but configuration hardens).
  - Implementing new navigation features; focus is on robustness and observability.

## Architecture Overview

| Layer | Responsibility | Deliverables |
| --- | --- | --- |
| Shared Infrastructure | Logging, subprocess, Problem Details, schema helpers | `_shared/logging.py`, `_shared/proc.py`, `_shared/problem_details.py`, tests |
| Domain Models | Typed dataclasses/TypedDicts representing DocFacts, navmap entries, analytics, CLI envelopes | `tools/docstring_builder/models.py`, new modules under `tools/docs/typed_models.py`, `tools/navmap/models.py` |
| Adapters | Conversion between legacy structures and typed models, schema validation wrappers | Adapter modules with functions `convert_*`, `validate_*` |
| I/O Boundaries | CLI entry points, file reads/writes, subprocess invocations | CLI refactors, environment configuration (`pydantic_settings` usage where needed) |
| Observability | Structured logging adapters, Prometheus metrics, Problem Details emission | Metrics provider classes, logging configuration, JSON examples |

### Logging Unification
`tools/_shared/logging.py` SHALL delegate to `src/kgfoundry_common/logging.py` for JSON formatting, correlation ID propagation, and `NullHandler` behavior. Libraries remain handler-agnostic; CLIs configure handlers. This ensures a consistent structured log envelope across packages and tools.

### Error Taxonomy
- `ToolError` (base) — extends `Exception`, ensures `raise ... from e` for causality.
- `ConfigurationError`, `SchemaViolationError`, `SubprocessError`, `PluginExecutionError`, `RenderingError`, `NavmapError`, `DocumentationBuildError` — precise variants mapped to modules.
- CLI surfaces translate errors to RFC 9457 Problem Details JSON using `_shared.problem_details.build_problem_details()`.

### Schema Strategy
- `schema/tools/docstring_builder_cli.json` (existing) — validated in CLI integration tests.
- **New:**
  - `schema/tools/cli_envelope.json` — base versioned envelope emitted by all tooling CLIs when `--json` is used.
  - `schema/tools/doc_analytics.json` — analytics summary emitted by `build_agent_analytics.py`.
  - `schema/tools/doc_graph_manifest.json` — graph outputs from `build_graphs.py`.
  - `schema/tools/navmap_document.json` — navmap documents and repair outputs.
  - `schema/tools/gallery_validation.json` — gallery validator machine outputs.
- Each schema has version constants, example fixtures under `docs/examples/`, and round-trip tests in `tests/tools/`.

### Base CLI Envelope (canonical)
Fields (JSON Schema lives at `schema/tools/cli_envelope.json`):
- `schemaVersion` (string, SemVer; e.g., "1.0.0")
- `schemaId` (string, URI; e.g., `https://kgfoundry.dev/schema/cli-envelope.json`)
- `generatedAt` (string, `date-time` in UTC)
- `status` (enum: `success | violation | config | error`)
- `command` (string)
- `subcommand` (string)
- `durationSeconds` (number ≥ 0)
- `files` (array of per-file objects with `path`, `status`, optional `message`, `problem`)
- `errors` (array of error entries)
- `problem` (RFC 9457 Problem Details object on failure)

Derivation/usage:
- Tools MAY extend the base envelope with domain sections (e.g., `policy`, `cache`).
- Docstring builder retains its richer schema; other tools adopt the base as a minimum.

## Detailed Implementation Plan

| Step | Description | Notes |
| --- | --- | --- |
| 1 | Harden `_shared` modules (logging, proc, problem details) | Align with pyrefly feedback: correct adapter signatures, typed streams, JSON helpers |
| 2 | Update docstring builder to consume `models.py` end-to-end | Introduce adapters, refactor `normalizer`, `policy`, `render`, fix observability stubs |
| 3 | Formalize plugin Protocols and compatibility shims | Provide typed discovery registry, doc updates for plugin authors |
| 4 | Integrate schema validation & CLI envelope across docstring builder | Validate on `--json` and `--baseline` outputs, expose Problem Details examples |
| 5 | Refactor docs generators (`build_agent_catalog`, `build_graphs`, `build_test_map`, `export_schemas`, `render_agent_portal`) | Break down complex functions, apply typed models, secure subprocesses |
| 6 | Introduce navmap typed models, logging, schema validation | Update `build_navmap`, `check_navmap`, repair/migrate scripts, tests |
| 7 | Harden lint/generation CLIs (`detect_pkg.py`, `generate_docstrings.py`, `hooks/docformatter.py`, etc.) | Provide typed `main()` functions, structured logs, safe subprocess |
| 8 | Testing & observability | Add pytest modules for schemas, CLI integration, plugin regression, navmap transformations; ensure metrics/log coverage |

## Data Contracts
- `tools/_shared/problem_details.py` ensures RFC 9457 compliance with sample JSON at `docs/examples/tools_problem_details.json`.
- CLI outputs must validate against schemas before writing; failure triggers `SchemaViolationError` with Problem Details payload.
- Navmap and docs builders produce JSON/HTML artifacts; JSON outputs get schemas, HTML generation logging includes sanitized file paths only.

## Configuration & 12-factor
CLIs SHALL use typed settings via `pydantic_settings` (env-only inputs, fast fail for required variables). Libraries accept explicit parameters and avoid global configuration.

Defaults and timeouts (apply unless overridden by CLI options):
- git: 10.0s; graphviz/dot: 30.0s; doctoc/formatters: 20.0s; python -m invocations: 20.0s.
- Environment allowlist: PATH, HOME, LANG/LC_*, PYTHON*, TZ, and CI/GIT/UV prefixes.

Input validation:
- Paths resolved via `Path(...).resolve(strict=True)` and must remain under repo root; reject non-`.py` where applicable.
- YAML must use `yaml.safe_load`.

## Observability
- Structured logging via `_shared.logging.get_logger()`; logs include `event`, `status`, `duration_ms`, `command`, `path_count`, etc.
- Prometheus metrics via `tools.docstring_builder.observability.MetricsProvider` and analogous providers for docs/navmap (counters + histograms with labels). Stubs implement `labels()` returning `self` to satisfy type checkers.
- Optional OpenTelemetry instrumentation points at CLI boundaries (span created with command name, status).

Metrics naming (minimum):
- `<tool>_runs_total{status}` (counter)
- `<tool>_cli_duration_seconds{command,status}` (histogram)

Logging fields (minimum): `correlation_id`, `operation`, `status`, `command`, `duration_ms`.

## Risks / Mitigations
- **Risk:** Large refactor may regress outputs. **Mitigation:** Table-driven regression tests, schema validation, feature flags, fallback mode with explicit warning logs.
- **Risk:** Third-party plugins break. **Mitigation:** Compatibility shim, documentation updates, deprecation notice with removal schedule.
- **Risk:** Performance regression due to extra validation. **Mitigation:** Keep validation optional in dry-run mode; add micro-bench tests and monitor metrics.
- **Risk:** Schema drift. **Mitigation:** Source-of-truth schemas under `schema/tools/`, round-trip tests, task requiring `make artifacts` and schema validation.
- **Risk:** Supply-chain vulnerabilities. **Mitigation:** Add `pip-audit` gate in CI; pin versions with sensible ranges; use safe deserializers (e.g., YAML safe_load).

## Traceability (Requirements → Tasks)
- Structured logging replaces prints → Tasks 1.1, 3.4, 4.2, Appendix A per-file steps.
- Subprocess execution is secured → Tasks 1.2, 3.2, 4.2, Appendix A replacements to `run_tool`.
- CLI emits versioned JSON envelope → Tasks 1.5, 3.4, 4.4, Appendix A Base CLI envelope tasks.
- Tests cover edge and failure cases → Tasks 1.4, 2.5, 3.4, 5.2–5.4, Appendix A tests to add.

## Rollout Plan
1. Merge shared infrastructure changes; ensure legacy paths continue functioning (feature flag default `0`).
2. Deploy docstring builder typed pipeline behind `DOCSTRINGS_TYPED_IR=1`; monitor metrics, compare outputs via regression tests.
3. Switch docs/navmap utilities to new logging + schema validation gradually, verifying outputs with schema tests and manual spot checks.
4. After metrics show stability (no increase in failures for two cycles), flip feature flag default and remove compatibility logs.
5. Document rollout and update changelog/README accordingly.

## Migration / Backout
- Backout by toggling `DOCSTRINGS_TYPED_IR=0` and `TOOLS_STRUCTURED_LOGGING=legacy`. Legacy code paths remain until post-rollout cleanup (tracked in tasks).
- Keep previous CLI JSON format accessible via `--legacy-json` for one release cycle; after removal, major version bump documented.
- Schema version increments require bumping constants and updating fixtures; maintain backward compatibility notes in docs.

