## Why
Our current documentation tooling stack trips over strict mypy and Ruff settings. The monkey patches in `src/sitecustomize.py` add properties to `docstring_parser.common.Docstring`, but there are no accompanying type stubs, so mypy reports missing attributes and invalid type assignments. Inside `tools/docstring_builder/harvest.py` (and neighbouring modules) we rely on `griffe` objects whose APIs are only defined at runtime; mypy treats identifiers like `Module`, `Function`, and `Object` as values, not types. Missing stubs for `libcst`, `griffe`, and `mkdocs_gen_files` compound the problem. At the same time, the docstring builder emits docstrings that NumPy-style linting deems incomplete because generated parameter descriptions are mismatched or missing. We need a focused “phase 1” effort that stabilises our core tooling so static analysis passes and generated docstrings meet lint expectations.

Beyond immediate hardening, we want a structural foundation that is modular, policy-driven, and fast. The follow-up should formalise a plugin architecture, define a stable intermediate representation (IR), enable incremental builds with a cache/manifest, introduce policy-based quality gates, and improve developer experience with clear CLI surfaces, observability, and safety controls. This sets us up for long-term maintainability and easy extensibility.


### Completed groundwork
- Typed stubs for `griffe`, `libcst`, and `mkdocs_gen_files` now live under `stubs/`, with a drift checker wired into CI.
- The docstring-builder CLI exposes `lint`/`check` commands with deterministic exit codes, and pre-commit invokes them before artefact regeneration.
- Policy gates for coverage and docstring parity integrate with the builder; manifests and incremental rebuilds keep runs fast on changed files.
- Plugin architecture, versioned IR + JSON Schema, and the policy engine are already available, giving us the platform to extend.
- Incremental executor, manifest rewriting, and pre-commit consolidation (including pyrefly) are in place.

### Next-focus areas (remaining scope)
- Restructure the CLI around explicit subcommands (`generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`) with clear exit codes and configuration precedence.
- Add stub governance tooling (drift checks surfaced via CLI and optional PEP-561 packaging).
- Provide observability artefacts (metrics JSON, HTML drift previews) and developer ergonomics (editor tasks/snippets).
- Harden safety (path normalisation, removal of dangerous evaluation) and prepare a deprecation path for `sitecustomize` patches.
- Expand documentation for plugin authors, stub maintainers, CLI usage, and troubleshooting (`doctor` guidance).



### Implementation focus (remaining scope)
- **CLI restructure:** Implement dedicated subcommands (`generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`), share common runner functions, and standardise exit codes and configuration precedence.
- **Stub governance:** Expose the drift checker through the CLI (e.g., `docstring-builder doctor --stubs`), integrate it into CI, and optionally publish stubs as PEP-561 extras with maintenance docs.
- **Observability & developer experience:** Emit `docs/_build/observability_docstrings.json`, generate HTML drift previews for docfacts/navmap/schema changes, and provide editor tasks/snippets for frequent commands.
- **Security hardening:** Audit builder and CLI for unsafe evaluation, normalise input paths, and add tests guarding against traversal/symlink attacks.
- **Sitecustomize deprecation:** Introduce a feature flag, emit `DeprecationWarning`, and prove the pipeline works with patches disabled in CI.
- **Documentation refresh:** Expand contributor docs covering plugin authoring, stub maintenance, config schema, CLI usage, troubleshooting, and interpreting `doctor` output.

## What Changes
Phase 1 work spans two buckets:

**Already delivered**
- Introduced Protocol-based helpers in `src/sitecustomize.py` and converted dynamic `setattr` calls into typed utilities with tests.
- Added local stub packages for `griffe`, `libcst`, and `mkdocs_gen_files`, plus a CI drift checker.
- Annotated core docstring-builder modules, added plugin architecture, versioned IR + schema, policy engine, manifest, and incremental executor.

**Remaining deliverables**
- **CLI restructure:** ship explicit subcommands (`generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`), expose shared runner functions, and codify exit codes/config precedence.
- **Stub governance:** surface the drift checker through the CLI (e.g., `docstring-builder doctor --stubs`), integrate it into CI, and decide on optional PEP-561 packaging/documentation.
- **Observability & developer experience:** produce metrics JSON, HTML drift previews, and editor tasks/snippets to speed up common workflows.
- **Security hardening:** normalise/validate input paths, forbid unsafe evaluation, and add regression tests covering traversal and sandbox scenarios.
- **Sitecustomize deprecation:** add a feature flag, emit `DeprecationWarning`, and verify CI passes with patches disabled so we can eventually drop them.
- **Documentation refresh:** expand contributor docs covering plugin authoring, stub maintenance, CLI usage, policy configuration, troubleshooting, and interpreting `doctor` output.


## Impact
- **Specs affected**: Developer Tooling, Documentation Automation, Type System Compatibility, Observability, CLI, Policy & Governance.
- **Code touched**: `src/sitecustomize.py`, `tools/docstring_builder/**`, `tools/docs/build_artifacts.py`, new plugin modules under `tools/docstring_builder/plugins/`, schema generation utilities, drift checker under `tools/stubs/`, and new/updated stub files under `stubs/`.
- **Artifacts produced**: `docs/_build/schema_docstrings.json`, `docs/_build/docstrings_manifest.json`, `docs/_build/observability_docstrings.json`, HTML diffs for documentation drift.
