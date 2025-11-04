## ADDED Requirements
### Requirement: Deterministic Tooling Package Surface
Tooling packages SHALL expose explicit, documented public APIs whose exports match their stubs and forbid private namespace bridging.

#### Scenario: Curated exports match stubs
- **GIVEN** `tools/__init__.py`, `tools/docs/__init__.py`, and related package initialisers
- **WHEN** `uv run pyrefly check` and `uv run pyright --warnings --pythonversion=3.13` execute
- **THEN** all public symbols originate from curated `__all__` lists, align with `stubs/tools/**`, and no `PLC2701` private-import violations appear

#### Scenario: Namespace bridge is confined
- **GIVEN** modules that previously imported private tooling names (`kgfoundry/namespace_bridge.py`, search clients, vectorstore)
- **WHEN** Ruff linting runs with default rules
- **THEN** imports resolve through sanctioned adapters exposed by `kgfoundry/_namespace_proxy`, and no module imports private `_namespace_proxy` members directly

#### Scenario: Optional tooling extra installs cleanly
- **GIVEN** the project wheel built via `python -m build`
- **WHEN** the wheel is installed with `pip install kgfoundry-*.whl[tools]` inside a clean environment
- **THEN** `import tools; tools.run_tool(['python','--version'])` succeeds without mutating `sys.path`, and `tools/py.typed` is present so type checkers load exported annotations

### Requirement: Layered Tooling Architecture
The tooling suite SHALL maintain a one-directional dependency flow from domain logic to orchestrators to adapters to CLI entry points, enforced by automated checks.

#### Scenario: Import-linter enforces layers
- **GIVEN** `tools/make_importlinter.py`
- **WHEN** `python tools/make_importlinter.py --check` runs in CI
- **THEN** contracts confirm that CLI/adapter modules do not import domain or `_shared` internals directly, failing the build if a violation exists

#### Scenario: Orchestrators isolate domain logic
- **GIVEN** refactored modules like `tools/docstring_builder/orchestrator.py` and `tools/docs/catalog_orchestrator.py`
- **WHEN** orchestration functions are invoked during lint/type gates
- **THEN** they depend only on domain modules and injected IO utilities, while CLI modules remain thin wrappers that satisfy Ruff complexity thresholds

### Requirement: Context-Aware Tool Execution
Subprocess orchestration SHALL propagate correlation context, enforce structured telemetry, and provide typed retries rooted in RFC 9457 Problem Details.

#### Scenario: Operation IDs propagate through telemetry
- **GIVEN** `tools._shared.proc.run_tool`
- **WHEN** orchestration layers call it with default settings under `uv run pyrefly check`
- **THEN** the function attaches a `ContextVar` operation ID to structured logs, metrics, and Problem Details, and all exceptions preserve causes via `raise ... from`

#### Scenario: Retry helper governs idempotent flows
- **GIVEN** `tools._shared.proc.run_tool_with_retry`
- **WHEN** docstring builder or navmap repair orchestrators enable retries
- **THEN** retries honour configured backoff, surface attempt metadata in telemetry, and raise the same typed exception with correlation context when exhaustion occurs

### Requirement: Typed Tooling Payload Contracts
Tooling data crossing module boundaries SHALL use typed models capable of emitting and validating JSON Schemas while supporting legacy payload upgrades.

#### Scenario: Models replace dictionary payloads
- **GIVEN** docstring edits, navmap documents, analytics envelopes, and CLI responses
- **WHEN** orchestration modules construct or consume these payloads
- **THEN** they do so via msgspec or frozen dataclass models, without relying on `dict[str, Any]`, and schema emission helpers in `tools._shared.schema` stay in sync with runtime types

#### Scenario: Legacy payloads upgrade deterministically
- **GIVEN** previously persisted cache or navmap payloads
- **WHEN** the refactored tooling loads them
- **THEN** dedicated conversion helpers validate the legacy structure, upgrade it to the new model, and surface structured Problem Details if the payload cannot be reconciled

### Requirement: Static Checker Guardrails
Tooling development SHALL include shared lint/type helpers, automated pre-commit checks, and contributor guidance so Ruff and pyrefly guardrails remain enforced.

#### Scenario: Lint defaults module eliminates Ruff regressions
- **GIVEN** `tools/_shared/linting.py`
- **WHEN** modules adopt its helpers for union ordering, typing aliases, and static/class method conversions
- **THEN** Ruff `RUF036`, `UP035`, `UP040`, and `PLR6301` violations are absent without introducing bespoke ignores, confirmed by `uv run ruff check src tools`.

#### Scenario: Pre-commit hooks catch tooling regressions
- **GIVEN** `.pre-commit-config.yaml`
- **WHEN** the hook stack runs (`pre-commit run --all-files`)
- **THEN** `ruff-format`, `ruff --fix`, and `pyrefly check` execute over `tools/**`, failing fast if lint or type errors are introduced.

#### Scenario: Contributor guidance reflects guardrails
- **GIVEN** `openspec/AGENTS.md`
- **WHEN** a contributor reviews the tooling quality gates section
- **THEN** it instructs them to run `uv run ruff format && uv run ruff check --fix && uv run pyrefly check` and references the shared linting helpers for resolving common warnings.

