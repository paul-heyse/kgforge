## Why
Our current documentation tooling stack trips over strict mypy and Ruff settings. The monkey patches in `src/sitecustomize.py` add properties to `docstring_parser.common.Docstring`, but there are no accompanying type stubs, so mypy reports missing attributes and invalid type assignments. Inside `tools/docstring_builder/harvest.py` (and neighbouring modules) we rely on `griffe` objects whose APIs are only defined at runtime; mypy treats identifiers like `Module`, `Function`, and `Object` as values, not types. Missing stubs for `libcst`, `griffe`, and `mkdocs_gen_files` compound the problem. At the same time, the docstring builder emits docstrings that NumPy-style linting deems incomplete because generated parameter descriptions are mismatched or missing. We need a focused “phase 1” effort that stabilises our core tooling so static analysis passes and generated docstrings meet lint expectations.

Beyond immediate hardening, we want a structural foundation that is modular, policy-driven, and fast. The follow-up should formalise a plugin architecture, define a stable intermediate representation (IR), enable incremental builds with a cache/manifest, introduce policy-based quality gates, and improve developer experience with clear CLI surfaces, observability, and safety controls. This sets us up for long-term maintainability and easy extensibility.


### Near-term adherence details (file-level plan)
- Typed stubs + drift checker
  - Create stubs:
    - `stubs/griffe/__init__.pyi`: expose `Object`, `Module`, `Function`, `GriffeLoader` used by `tools/docstring_builder/harvest.py` and `docs/_scripts/mkdocs_gen_api.py`.
    - `stubs/griffe/loader.pyi`, `stubs/griffe/dataclasses.pyi`: alias `GriffeLoader`, `Class`, `Function`, `Module` for `tools/griffe_utils.py`.
    - `stubs/libcst/__init__.pyi`: define `CSTVisitor`, `Module`, `ClassDef`, `FunctionDef`, `parse_module(text) -> Module` with `.visit`.
    - `stubs/mkdocs_gen_files/__init__.pyi`: `open(path, mode='w')` context manager returning text IO.
  - Add `tools/stubs/drift_check.py`: import `griffe`, `libcst`, `mkdocs_gen_files`, assert expected symbols, print missing/extra diffs, exit non‑zero on mismatch.
  - CI: run drift checker then `uv run mypy --strict src tools/docstring_builder`.

- Docstring builder lint/check in pre-commit
  - Update `tools/docstring_builder/cli.py`:
    - Add `lint` subcommand delegating to `check`; support exit codes: `0` ok, `1` violations, `2` config, `3` internal.
    - Keep `check --diff` for PR convenience.
  - Ensure `.pre-commit-config.yaml` keeps `docstring-builder (check)` before docs artifact generation (present) and optional `docstring-builder (diff)` (present).

- Minimal policy gates
  - Use existing hooks: `pydoclint --style numpy src`, `pydocstyle src`, `interrogate -i src --fail-under 90`.
  - Optionally surface a short summary via `docstring-builder lint` invoking these tools.

- Manifest + incremental rebuilds
  - In `tools/docstring_builder/cli.py`, after `_run()` writes cache, write `docs/_build/docstrings_manifest.json` with:
    - processed files, counts, `config.config_hash`, CLI args (`since/module/force`), cache path/mtime.
  - Add `--changed-only` flag as sugar to compute a default `--since` (use `origin/main`, fallback `HEAD~1`).

- Pre-commit consolidation (ruff/mypy/doc tools/pyrefly)
  - In `.pre-commit-config.yaml`, order hooks: ruff imports → ruff fix → ruff format → mypy strict → docformatter/pydoclint/interrogate → docstring-builder (check) → docs: regenerate artifacts → navmap-check → pyrefly validate.
  - Add `pyrefly-check` hook: `entry: uv run pyrefly check`, `pass_filenames: false`.
  - Treat optional deps as extras in `pyrefly.toml`; ensure it doesn’t fail when extras are absent.

- Golden tests for builder
  - Add `tests/docs/test_docstring_golden.py` and `tests/docs/goldens/**`.
  - Implement `UPDATE_GOLDENS=1` path to refresh goldens intentionally.

- Minimal observability
  - On non‑zero exit in CLI, emit `docs/_build/observability_docstrings.json` with counts, timings, and top errors; print a brief console SUMMARY.

- Doctor command
  - Add `doctor` subcommand to `tools/docstring_builder/cli.py` that checks: Python version, `mypy.ini` (`mypy_path = src:stubs`), presence of key stubs, import health for `griffe`/`libcst`, writeability of `docs/_build/` and `.cache/`, and pre-commit hook presence/order. Exit `0`/`2` with suggestions.



## What Changes
- **Type-safe docstring shim**
  - Create Protocols representing the subset of `docstring_parser` types we monkey patch (`DocstringProto`, `DocstringMetaProto`, `DocstringAttrProto`, `DocstringYieldsProto`).
  - Introduce helper functions (e.g., `ensure_docstring_attrs(doc_cls: type[DocstringProto]) -> bool`) that check for existing attributes, perform `setattr`, and return a boolean so we can log what changed. These helpers wrap every property injection in `sitecustomize.py` and replace ad-hoc `# type: ignore` usage.
  - Add unit tests or integration checks that import `sitecustomize` both with the real library present and with a shimmed version to prove the Protocols behave as expected.
- **Local stub packages for third-party dependencies**
  - Add minimal `.pyi` files under `stubs/griffe`, `stubs/libcst`, and `stubs/mkdocs_gen_files` covering only the attributes we use. For example, expose `griffe.Module`, `griffe.Function`, `griffe.loader.GriffeLoader` with their constructor signatures and key properties.
  - Update `pyproject.toml` / `mypy.ini` to include the `stubs/` directory and remove inline `# type: ignore` comments that become unnecessary.
- **Strongly typed docstring-builder internals**
  - Annotate `tools/docstring_builder/harvest.py` using the new stubs, converting `ParameterHarvest`, `SymbolHarvest`, etc., to typed dataclasses.
  - Refactor helper functions like `_resolve_object`, `_resolve_callable`, `_module_name`, `_collect_symbols` to return precise types (`object | None`, `Callable[..., object] | None`) and document their behaviour with NumPy-style docstrings.
  - Ensure imports reference the stubbed types (e.g., `from griffe.loader import GriffeLoader`), not `Any`.
- **Docstring quality for tooling modules**
  - Update module/class/function docstrings within `tools/docstring_builder` and `src/sitecustomize.py` to meet NumPy standard: accurate parameter descriptions, updated Returns/Raises sections, and removal of placeholder “TODO” text.

- **Pluginized pipeline (harvest → transform → format)**
  - Define typed plugin interfaces (Protocols/ABCs) for `Harvester`, `Transformer`, and `Formatter` stages, with lifecycle hooks (`on_start`, `on_finish`) and context access.
  - Discover plugins via entry points group `kgfoundry.docstrings.plugins`; support `--disable-plugin`/`--only-plugin` filters.
  - Version the plugin API; on mismatch, fail fast with a clear error linking to docs.
  - Provide a sample plugin (e.g., `normalize_numpy_params`) as a reference implementation.

- **Versioned Intermediate Representation (IR) + JSON Schema**
  - Introduce IR dataclasses (e.g., `IRSymbol`, `IRParameter`, `IRDocstring`) with `ir_version` and stable field names.
  - Generate and publish a JSON Schema to `docs/_build/schema_docstrings.json`; validate all IR instances during builds.
  - Add CLI support to print the current schema (`schema` subcommand) and to validate IR-only (`check` subcommand).

- **Policy-driven quality gates**
  - Configure rules in `pyproject.toml` (e.g., `[tool.kgfoundry.docstrings.policy]`) or a YAML file; resolve with precedence: CLI > env > config file.
  - Support thresholds (coverage %, missing params/returns, allowed exceptions), actions (`error`, `warn`, `autofix`), and per-package overrides.
  - Implement an exceptions allowlist with `expires_on` and `justification`; fail CI when exceptions expire.

- **Incremental builds with manifest and change detection**
  - Maintain a content-addressed manifest in `docs/_build/docstrings_manifest.json` capturing inputs (file hashes), active plugins and versions, resolved config, and produced artifacts.
  - Rebuild only changed files and their dependents; support `--since <rev>` and `--changed-only` for CI.
  - Invalidate cache when the plugin set, tool versions, or config fingerprints change.

- **CLI surface and configuration**
  - Split into subcommands: `generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`.
  - Provide `--config` to load an explicit config file; document precedence: CLI > env vars > pyproject/YAML > defaults.
  - Ensure exit codes are deterministic: `0` success/no changes, `1` policy/lint failures, `2` configuration errors, `3` internal errors.

- **Stub governance and drift checks**
  - Add a drift checker that introspects runtime objects vs. local `.pyi` to detect missing/extra attributes; report actionable diffs in CI.
  - Optionally package stubs as PEP-561-compatible typed stub extras; document how to extend stubs safely.

- **Observability and developer experience**
  - Emit metrics and traces to `docs/_build/observability_docstrings.json` (counts, timings, error classes); include a human-readable SUMMARY for PR notes.
  - Produce HTML diffs for doc artifact drift (docfacts/navmap/schema) for changed modules; link from CI logs.
  - Ship editor tasks/snippets for common flows (`generate`, `fix`, `watch`).

- **Testing strategy and gates**
  - Add golden-file tests (rendered docstrings and IR snapshots) with `--update` support.
  - Property tests for docstring roundtrips (parse → IR → render) on representative samples.
  - E2E CLI tests across subcommands and a matrix with/without optional dependencies.

- **Security and safety**
  - Sandbox any evaluation of embedded code in docstrings; restrict import resolution to project roots.
  - Strictly validate input globs/paths; reject path traversal or unexpected symlinks.

- **Deprecation plan for `sitecustomize` monkey patches**
  - Introduce a feature flag (`KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE=0/1`) and emit `DeprecationWarning` when patches are used.
  - Document a timeline for deprecation and a kill switch to disable patches in CI.

## Near-term adherence milestone (focus for next 1–2 weeks)
- Typed stubs + drift checker in CI: finalize `.pyi` for `griffe`, `libcst`, `mkdocs_gen_files`; add runtime vs. stub drift checker with actionable diffs; run mypy strict on PRs.
- Docstring builder lint/check in pre-commit: add subcommands with deterministic exit codes; block commits on doc issues without full artifact rebuild.
- Minimal policy gates: set coverage ≥90% and params/returns parity; wire into `lint`.
- Manifest + incremental rebuilds: write `docs/_build/docstrings_manifest.json`; implement `--changed-only` to process only touched files + dependents.
- Pre-commit consolidation: enforce order ruff→mypy→docformatters→docstring-builder lint/check→artifacts→navmap-check→pyrefly validate; treat optional deps as extras.
- Golden tests: 3–5 golden renders with `--update` path for intentional changes.
- Minimal observability: write `docs/_build/observability_docstrings.json` + console SUMMARY on failure (top errors, timings).
- Doctor command: quick environment/config/stub/manifest checks with suggested fixes.


## Impact
- **Specs affected**: Developer Tooling, Documentation Automation, Type System Compatibility, Observability, CLI, Policy & Governance.
- **Code touched**: `src/sitecustomize.py`, `tools/docstring_builder/**`, `tools/docs/build_artifacts.py`, new plugin modules under `tools/docstring_builder/plugins/`, schema generation utilities, drift checker under `tools/stubs/`, and new/updated stub files under `stubs/`.
- **Artifacts produced**: `docs/_build/schema_docstrings.json`, `docs/_build/docstrings_manifest.json`, `docs/_build/observability_docstrings.json`, HTML diffs for documentation drift.
