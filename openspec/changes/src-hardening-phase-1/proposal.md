## Why
Shared infrastructure in `src/kgfoundry_common` currently blocks strict Ruff/mypy/pyrefly adoption: the error taxonomy exports missing symbols, Prometheus helpers break type contracts, structured logging diverges from the repo standard, serialization utilities rely on `Any`, and settings constructors accept arbitrary kwargs. These weaknesses propagate `Any` throughout higher layers, hide observability failures, and violate our Problem Details contract. Phase 1 delivers typed, deterministic foundations so subsequent phases (search, embeddings, registry, orchestration) can build on a clean base.

## What Changes
- [x] **ADDED**: JSON Schemas for Problem Details and infrastructure payloads under `schema/common/`, with fixtures and validation helpers.
- [x] **ADDED**: Typed helper modules (`problem_details.py`, `observability.py` shims) exposing structured logs, metrics, and trace hooks with safe fallbacks.
- [x] **MODIFIED**: `kgfoundry_common/errors/**` to publish a consistent taxonomy, fix `to_problem_details` signatures, and remove unused `type: ignore`s.
- [x] **MODIFIED**: `kgfoundry_common/logging.py` to delegate to the repo-standard structured logger (`tools/_shared/logging` analog) and surface correlation IDs.
- [x] **MODIFIED**: `kgfoundry_common/serialization.py` to use typed schema validation (jsonschema exceptions), raising explicit `SerializationError` variants.
- [x] **MODIFIED**: `kgfoundry_common/config.py` & `settings.py` to adopt `pydantic_settings` models with explicit fields, environment-only configuration, and fast-fail.
- [ ] **REMOVED**: Deprecated logging adapters and blind exception helpers once phase 1 completes (tracked in rollout section).
- [ ] **RENAMED**: _None._
- [ ] **BREAKING**: External APIs remain source-compatible; Problem Details JSON envelope gains optional `title` field derived from taxonomy (documented in changelog).

## Impact
- **Affected specs:** `src-common-infrastructure` (new capability requirements under `specs/src-common-infrastructure/spec.md`).
- **Affected code paths:**
  - `src/kgfoundry_common/errors/__init__.py`, `errors/exceptions.py`, `errors/http.py`
  - `src/kgfoundry_common/logging.py`, `observability.py`
  - `src/kgfoundry_common/serialization.py`, `serialization/*.py`
  - `src/kgfoundry_common/config.py`, `settings.py`
  - `docs/examples/` Problem Details fixtures; `schema/common/*.json`
  - Shared tests under `tests/kgfoundry_common/`
- **Rollout:** Implement behind feature flag `KGFOUNDRY_LOGGING_V2=0|1` and `KGFOUNDRY_PROBLEM_DETAILS_V2=0|1`. Default remains legacy until telemetry shows 7 days of stability. Provide migration notes for downstream services consuming internal helpers.
- **Risks:** Potential logging format drift (mitigated via golden-file tests), metrics cardinality changes (monitored via dashboards), schema validation introducing performance overhead (guarded via caching and optional dry-run mode).

## Acceptance
- All quality gates pass for `src/kgfoundry_common/**` (Ruff, pyrefly, mypy, pytest, doctests/xdoctests, import linter, pip-audit, artifacts, openspec validate).
- Problem Details JSON produced by common errors validates against `schema/common/problem_details.json`; fixtures updated.
- Structured logging uses unified adapter with correlation IDs and no `print` statements remain.
- Observability helpers expose typed counters/histograms, with tests covering stub + real registry paths.
- Settings classes reject unexpected kwargs and document required env vars; configuration doctests run.
- Public APIs have PEP 257 docstrings with one-line summaries; doctests/xdoctests run.
- Exceptions use `raise ... from e` to preserve cause chains; tests verify `__cause__`.
- Feature flags documented; legacy paths removed after rollout metrics confirm stability.
- Packaging passes: `pip wheel .` succeeds; `pip install .[obs,schema]` works in a clean venv; metadata correct.
- Benchmarks recorded for hot paths (validation/logging); regressions noted.

