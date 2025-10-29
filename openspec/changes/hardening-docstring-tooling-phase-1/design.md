## Context
The documentation tooling layer comprises `docstring_parser` (external library), our compatibility shim (`src/sitecustomize.py`), and the internal docstring builder (`tools/docstring_builder`). Strict mypy settings reveal structural weaknesses:
- Monkey-patched attributes (`attrs`, `yields`, `many_yields`, `size`) are applied to `docstring_parser.common.Docstring` without static types, triggering mypy errors.
- The docstring builder imports `griffe`, `libcst`, and `mkdocs_gen_files`, none of which ship type stubs. mypy treats their identifiers as runtime values, leading to “invalid type” errors.
- Nuanced helper logic for harvesting metadata is unannotated and lacks descriptive docstrings, making it hard for new contributors to reason about behaviour.

By focusing on a “Phase 1” hardening effort, we aim to make these modules type-safe and lint-clean without yet tackling CLI compatibility or workflow automation (reserved for later phases).

## Goals / Non-Goals
- **Goals**
  - G1: Ensure `src/sitecustomize.py` passes mypy strictly by representing docstring monkey patches via Protocols and guarded helper functions.
  - G2: Provide minimal stub packages so mypy can type-check `tools/docstring_builder` imports without using `Any`.
  - G3: Annotate docstring builder internals with precise types and thorough docstrings so junior developers can follow the flow.
  - G4: Introduce a plugin architecture for harvest/transform/format stages with entry-point discovery.
  - G5: Define a versioned Intermediate Representation (IR) for docstrings and publish a JSON Schema.
  - G6: Add a policy engine for quality gates (coverage, completeness, exceptions) with CI enforcement.
  - G7: Implement incremental builds with a content-addressed manifest and change detection.
  - G8: Improve CLI ergonomics with clear subcommands and deterministic exit codes.
  - G9: Provide observability (metrics/traces) and an HTML drift preview for PRs.
  - G10: Establish robust testing (golden snapshots, property, contract, and E2E CLI tests).
  - G11: Plan deprecation for `sitecustomize` monkey patches with a feature flag and timeline.
- **Non-Goals**
  - N1: No CLI changes (legacy flag support handled in later phases).
  - N2: No docstring regeneration logic changes beyond replacing placeholder docstrings with accurate descriptions.
  - N3: No modifications to navmap/test-map/schema workflows.
  - N4: No content rewriting of existing docs unrelated to policy-driven fixes.
  - N5: No external service dependencies (keep all artifacts local to the repo).

## Decisions
1. **Introduce Protocol-based shims**
   - Define Protocols capturing traits of docstring objects we rely on.
   - Provide helper functions that encapsulate `setattr` logic, returning success status.
   - Optionally log via `logging.getLogger(__name__)` to aid debugging when upstream adds native support.
2. **Add stub packages**
   - Place `.pyi` files in `stubs/` for `griffe`, `libcst`, `mkdocs_gen_files`.
   - Keep stubs minimal and document how to extend them if future imports require additional attributes.
3. **Annotate and document docstring builder internals**
   - Use typed dataclasses for harvested items.
   - Provide accurate docstrings and update tests to assert new typing behaviours.

4. **Plugin architecture and discovery**
   - Choose Python entry points for discovery under group `kgfoundry.docstrings.plugins`.
   - Define typed Protocols for `Harvester`, `Transformer`, `Formatter` with `run(context)` entry methods and optional lifecycle hooks (`on_start`, `on_finish`).
   - Require a `api_version: str` attribute on plugins for compatibility checks; mismatch triggers a helpful error.
   - Provide inclusion/exclusion filters (`--only-plugin`, `--disable-plugin`) at CLI and config layer.

5. **Intermediate Representation (IR)**
   - Model IR with stdlib `dataclasses` (frozen where possible) for `IRSymbol`, `IRParameter`, `IRDocstring`, and related types; include `ir_version` and `source_path`/`symbol_id` for traceability.
   - Generate JSON Schema at build-time and write to `docs/_build/schema_docstrings.json`. Validation happens during pipeline execution before rendering artifacts.
   - Ensure backwards-compatible evolution rules; breaking changes increment `ir_version` and update migration docs.

6. **Policy engine**
   - Define a declarative configuration schema (`pyproject.toml` or YAML) with rules, thresholds, and actions (`error`, `warn`, `autofix`).
   - Allow overrides at package/module level. Provide an exceptions allowlist with `expires_on` and `justification` fields.
   - Evaluation order: global rules → package overrides → module overrides → CLI overrides; first-match-wins with explicit precedence.

7. **Incremental execution and manifest**
   - Store a manifest `docs/_build/docstrings_manifest.json` containing: input file hashes, dependency edges (module → dependents), active plugin names/versions, tool version fingerprints, config fingerprint, produced outputs (paths + hashes).
   - Recompute only changed nodes and their transitive dependents. Invalidate when fingerprinted inputs change.
   - Provide `--since <rev>` and `--changed-only` to restrict the scope in CI.

8. **CLI**
   - Subcommands: `generate` (build artifacts), `lint` (report issues), `fix` (apply autofixes), `diff` (show changes), `check` (validate IR/schema/policy), `schema` (print schema), `doctor` (environment checks), `measure` (timing and cache stats).
   - Exit codes: `0` success, `1` policy/lint failures, `2` config/validation errors, `3` internal/unexpected errors.
   - Config precedence: CLI > env vars > config file (pyproject/YAML) > defaults. Document all options.

9. **Observability and drift preview**
   - Emit metrics/traces JSON: counts (modules processed, symbols, plugins active), timings (per stage), cache stats (hits/misses), and error classes.
   - Produce HTML diffs for doc artifacts drift (docfacts/navmap/schema) per changed module; link file paths in CI output for fast navigation.

10. **Security and safety**
   - Disallow arbitrary code execution while parsing docstrings. Avoid `eval`/`exec`; if unavoidable for a specific feature, run inside a controlled sandbox with strict input validation.
   - Path handling: normalise and resolve input roots; reject path traversal and unexpected symlinks.

11. **Deprecation of `sitecustomize`**
   - Introduce `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE` feature flag (default on during transition). Emit `DeprecationWarning` when shims apply.
   - Provide a kill switch for CI and a deprecation timeline documented in the README.

## Near-term priorities and sequencing (operational focus)
1. Typed stubs + drift checker
   - Rationale: Stabilises mypy immediately and protects against upstream changes.
   - Design notes: Drift checker introspects selected attributes only (keep scope minimal to avoid churn); outputs unified diffs and suggested stub edits.

2. Pre-commit lint/check for docstrings
   - Rationale: Catch violations before full artifact rebuild; preserve fast edit loop.
   - Design notes: Deterministic exit codes; `lint` runs fast static validations; `check` performs light IR/schema checks without writing artifacts.

3. Minimal policy gates
   - Rationale: Guarantee continuous compliance with coverage and parity.
   - Design notes: Policy config lives in `pyproject.toml`; CLI overrides for CI experimentation; first-match-wins precedence.

4. Manifest + incremental rebuilds
   - Rationale: Speed drives adherence; faster hooks mean they’re used.
   - Design notes: Fingerprint includes input hashes, tool/plugin versions, config hash; `--changed-only` short-circuits traversal to touched nodes + dependents.

5. Pre-commit consolidation (ruff/mypy/pyrefly)
   - Rationale: One predictable enforcement pipeline reduces surprises.
   - Design notes: Optional deps for pyrefly moved behind extras; validation degrades gracefully without extras.

6. Golden tests
   - Rationale: Prevent subtle regressions lint can’t catch.
   - Design notes: Stable fixtures with an `--update` mode; diff-friendly output and file layout.

7. Minimal observability
   - Rationale: Fast triage when a hook fails.
   - Design notes: Small JSON summary and concise console output; no heavy tracing yet.

8. Doctor command
   - Rationale: Eliminate environment/config confusion quickly.
   - Design notes: Read-only checks that complete in <1s; prints suggested fixes.

## Implementation notes (file-level guidance for near term)
- Stubs and drift checker
  - Create/extend stubs under `stubs/`: `griffe/__init__.pyi`, `griffe/loader.pyi`, `griffe/dataclasses.pyi`, `libcst/__init__.pyi`, `mkdocs_gen_files/__init__.pyi`.
  - Add `tools/stubs/drift_check.py` to introspect runtime modules (`griffe`, `libcst`, `mkdocs_gen_files`) and assert presence of symbols we use; print diffs and exit non-zero on mismatch.
  - Ensure `mypy.ini` includes `mypy_path = src:stubs` (already present); keep stubs minimal and focused on attributes accessed in `tools/docstring_builder/**` and `docs/_scripts/mkdocs_gen_api.py`.

- Docstring builder CLI
  - File: `tools/docstring_builder/cli.py`.
  - Add subcommands:
    - `lint`: thin alias of `check` for pre-commit fast path; wire deterministic exit codes (0 ok, 1 policy/lint failures, 2 config errors, 3 internal errors).
    - `doctor`: environment/config quick checks; verify Python version, stubs presence, import health, write permissions for `docs/_build/` and `.cache/`, and presence/order of relevant pre-commit hooks.
  - Manifest: after `_run()` completes, write `docs/_build/docstrings_manifest.json` including processed files, `config_hash`, CLI args, cache path/mtime, and counts.
  - Flags: add `--changed-only` to compute a default `--since` using `origin/main` (fallback to `HEAD~1`).

- Pre-commit integration
  - File: `.pre-commit-config.yaml`.
  - Ensure ordering: ruff (imports → fix → format) → mypy → docformatter/pydoclint/interrogate → docstring-builder (check) → docs: regenerate artifacts → navmap-check → pyrefly validate.
  - Add new repo-local hook `pyrefly-check` with `entry: uv run pyrefly check`, `pass_filenames: false`.

- Makefile and targets
  - Add `stubs-check` target calling the drift checker.
  - Keep `lint-docs` as the fast path for devs; optionally call `docstring-builder cli check --diff`.

- Tests
  - Extend `tests/tools/test_docstring_builder_cli.py` to cover exit codes for `check`, `lint`, and `doctor` happy/failure paths.
  - Add `tests/docs/test_docstring_golden.py` with 3–5 representative modules and golden outputs under `tests/docs/goldens/`.
  - Keep existing tests: `tests/docs/test_docfacts.py`, `tests/docs/test_docstring_normalizer.py`.

- Observability
  - On non-zero exit from CLI, write `docs/_build/observability_docstrings.json` with summary counts/timings and top error messages; also print a brief console summary.

## Risks / Trade-offs
- **Stub drift**: External libraries may update APIs, invalidating our stubs. *Mitigation*: Keep stubs intentionally small and align them with the specific modules we import.
- **Runtime vs static mismatch**: Protocols might miss attributes added at runtime. *Mitigation*: guard helper functions with `hasattr` and broaden Protocols only when needed.
- **Developer overhead**: Additional typing discipline may raise the barrier for quick patches. *Mitigation*: supply “how to update stubs” guidelines and code comments linking to relevant docs.
 - **Plugin complexity**: Plugin lifecycle and ordering can add mental overhead. *Mitigation*: provide sensible defaults and a built-in plugin set.
 - **IR stability**: Freezing schema too early may impede evolution. *Mitigation*: follow semantic versioning and supply migration notes.
 - **Cache correctness**: Incorrect fingerprints can serve stale results. *Mitigation*: include all relevant inputs (source, config, plugins, tool versions) in the fingerprint and add `doctor` to verify cache assumptions.

## Migration Plan
1. Implement Protocol helpers and update `sitecustomize.py`.
2. Add stub packages and adjust mypy configuration.
3. Refactor `tools/docstring_builder` modules with typed dataclasses and docstrings.
4. Introduce plugin interfaces and a reference plugin; wire entry-point discovery.
5. Define IR dataclasses; generate and validate JSON Schema; add `schema` CLI.
6. Implement policy engine with config precedence and simple rules; integrate into `lint`/`check`.
7. Add manifest and incremental executor; implement `--since` and `--changed-only`.
8. Restructure CLI into subcommands; implement exit codes and `doctor`/`measure`.
9. Emit observability metrics and generate HTML drift previews.
10. Harden security checks (path normalisation, sandboxing decisions).
11. Add tests (golden/property/contract/E2E) and CI gates; document deprecation path for `sitecustomize`.
12. Run mypy, Ruff, and `openspec validate --strict` for the change.

## Open Questions
- Should stub files live in `stubs/` or co-located `typings/` directories? (Default: centralise in `stubs/`.)
- Is there a preferred logging level for helper functions when they detect existing attributes (DEBUG vs INFO)?
- Do we need to support Python versions earlier than 3.12 for tomllib imports (affects config typing)?
 - Which JSON Schema generator should we use for IR (hand-rolled vs. library)?
 - Preferred plugin ordering semantics (declared order vs. priority numbers vs. topological by dependencies)?
 - Threshold defaults for coverage/quality gates and their variance across packages?
