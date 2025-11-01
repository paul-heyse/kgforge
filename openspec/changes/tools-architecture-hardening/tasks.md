## Phase 0 – Source-of-truth & packaging hygiene

- [x] **P0.1 Promote `tools` as a packaged optional extra**
  - [x] Update `pyproject.toml` with `[project.optional-dependencies.tools]`, enumerating Prometheus/OpenTelemetry/msgspec/yaml dependencies required at runtime.
  - [x] Configure `tool.hatch.build.targets.wheel.packages` (and matching sdist include list) to ship the `tools` package, templates, and schema assets; add a `tools/py.typed` marker if missing.
  - [x] Remove any residual `sys.path` manipulations in scripts/tests that existed to compensate for missing packaging, replacing them with `pip install .[tools]` guidance where necessary.
  - [x] Add a packaging smoke-test script (e.g., `scripts/test_tools_package.sh`) that builds the wheel and imports `tools` after installing with the new extra.

- [x] **P0.2 Curate public exports and typings**
  - [x] Audit `tools/__init__.py`, `tools/docs/__init__.py`, and sibling `__init__` files to ensure `PUBLIC_EXPORTS` only includes supported orchestrator/adaptor entry points with precise type hints and NumPy-style docstrings.
  - [x] Regenerate `__all__` lists from the curated mappings and remove unused imports or wildcard exports.
  - [x] Mirror the curated exports in `stubs/tools/__init__.pyi` and `stubs/tools/docs/__init__.pyi`, replacing lax `ModuleType` annotations with concrete Callable/Protocol signatures where feasible.
  - [x] Update module docstrings to reference RFC 9457 Problem Details examples (`schema/examples/tools/problem_details/*`) and the shared exception taxonomy.

- [x] **P0.3 Replace private namespace bridging**
  - [x] Introduce a public adapter module (e.g., `src/kgfoundry/tooling_bridge.py`) that wraps helpers from `kgfoundry._namespace_proxy` and exports supported bridge functions.
  - [x] Refactor `src/kgfoundry/namespace_bridge.py`, `src/kgfoundry/search_client/client.py`, and `src/kgfoundry/vectorstore_faiss/__init__.py` to depend on the new adapter instead of importing `_namespace_proxy` directly; update docstrings accordingly.
  - [x] Harden `kgfoundry/_namespace_proxy.py` with improved typing, NumPy-style docstrings, and an internal-only docstring warning; keep `__all__` limited to helper names used by the adapter.
  - [x] Add a Ruff/Pyrefly regression guard (e.g., `ruff` forbid rule or targeted unit test) ensuring no modules import `_namespace_proxy` outside the adapter.

- [ ] **P0.4 Validate packaging flow**
  - [ ] Run `uv run python -m build --wheel` and install the resulting wheel with `pip install dist/kgfoundry-*.whl[tools] --force-reinstall` inside a clean venv.
  - [ ] Execute a smoke command (`python -c "import tools; tools.run_tool(['python','--version'])"`) to confirm the packaged exports work without repo-relative paths.
  - [ ] Capture the command outputs and add them to the Phase 0 PR checklist.

## Phase 1 – Static checker guardrails

- [ ] **P1.1 Shared lint defaults module**
  - [ ] Implement `tools/_shared/linting.py` with utilities for enforcing union ordering (`RUF036`), detecting deprecated typing aliases, and providing helper decorators (`as_staticmethod`, `as_classmethod`).
  - [ ] Refactor high-noise modules (docstring builder orchestrator, metrics helpers, navmap utilities) to use the new helpers or convert low-cohesion instance methods to module-level functions.
  - [ ] Add comprehensive NumPy docstrings and typing to the new linting helpers; expose any public helpers via `tools/__init__` if desired.

- [ ] **P1.2 Ruff configuration adjustments**
  - [ ] Review `pyproject.toml` Ruff settings to remove obsolete ignores now covered by the helper module; ensure `RUF036`, `UP035`, and `UP040` remain enforced.
  - [ ] Add any additional lint rules (e.g., forbidding `_namespace_proxy` imports) required to complement Phase 0; document rationale in commit messages.

- [ ] **P1.3 Pre-commit workflow updates**
  - [ ] Ensure `.pre-commit-config.yaml` runs `ruff-format`, `ruff --fix`, and `pyrefly check` over `tools/**` (tighten hook globs if necessary).
  - [ ] Add a lightweight self-test hook or script for the linting helper module if needed to validate decorator behaviour.

- [ ] **P1.4 Contributor guidance**
  - [ ] Update `openspec/AGENTS.md` (or `docs/contributing/quality.md`) with instructions to run `uv run ruff format && uv run ruff check --fix && uv run pyrefly check` for tooling changes and tips on using the new linting helpers.
  - [ ] Reference the new workflow in the PR template/test plan checklist for tooling-related work.

## Phase 2 – Module boundaries

- [ ] Split oversized adapters (docstring builder CLI, docs builders) into orchestrator/IO/adapter layers with fully annotated entry points and documented exception contracts.

## Phase 3 – Reliability & context propagation

- [ ] Harden `tools._shared.proc` with `ContextVar` operation IDs, structured logging, retry helpers, and namespaced settings objects that validate configuration on load.

## Phase 4 – Modularity & complexity reduction

- [ ] Refactor high-complexity functions (BM25, FAISS adapters, agent catalog search) into composable strategies and regenerate import-linter contracts to enforce one-way dependencies.

## Phase 5 – Typed data contracts

- [ ] Replace dictionary payloads with msgspec/frozen dataclass models, add legacy conversion helpers, and expose schema emission/validation utilities in `tools._shared.schema`.

## Verification

- [ ] After completing each phase, run `uv run ruff check --fix`, `uv run pyrefly check`, and `uv run mypy --config-file mypy.ini` before starting the next tranche.
- [ ] Once layering changes land, ensure `python tools/make_importlinter.py --check` passes as part of the local gate.

