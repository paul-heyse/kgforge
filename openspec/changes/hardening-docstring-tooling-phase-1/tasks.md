## 0. Near-term Adherence Milestone (next 1–2 weeks)
- [ ] 0.1 Typed stubs + drift checker in CI
    - [ ] 0.1.1 Finalize `.pyi` for `griffe`, `libcst`, `mkdocs_gen_files`.
    - [ ] 0.1.2 Implement a runtime vs. stubs drift checker with actionable diff output.
    - [ ] 0.1.3 Add CI job: run drift checker then `uv run mypy --strict src tools/docstring_builder`.
    - Details:
        - Create stubs:
            - `stubs/griffe/__init__.pyi`: expose `Object`, `Module`, `Function`, and `GriffeLoader` with minimal attributes used in `tools/docstring_builder/harvest.py` and `docs/_scripts/mkdocs_gen_api.py`.
            - `stubs/griffe/loader.pyi` and `stubs/griffe/dataclasses.pyi`: alias `GriffeLoader`, `Class`, `Function`, `Module` as needed by `tools/griffe_utils.py`.
            - `stubs/libcst/__init__.pyi`: define `CSTVisitor`, `Module`, `ClassDef`, `FunctionDef`, and `parse_module(text: str) -> Module` with `.visit(visitor)`.
            - `stubs/mkdocs_gen_files/__init__.pyi`: define `def open(path: str | os.PathLike[str], mode: str = 'w'): ...` as a context manager returning a `TextIO`-like object.
        - Drift checker:
            - Add `tools/stubs/drift_check.py` that imports `griffe`, `libcst`, `mkdocs_gen_files` and asserts the stubbed symbols exist (e.g., `hasattr(griffe, 'GriffeLoader')`).
            - Print missing/extra attribute diffs compared to a hardcoded expected set used by our code; exit non-zero on mismatch.
        - Integration:
            - Ensure `mypy.ini` has `mypy_path = src:stubs` (already present).
            - Add a Makefile target `stubs-check` that runs the drift checker.
    - AC:
        - CI fails on stub mismatches with clear missing/extra members listed.
        - No `Any` leaks reported by mypy strict.

- [ ] 0.2 Docstring builder lint/check in pre-commit
    - [ ] 0.2.1 Implement `docstring-builder lint` and `docstring-builder check` with exit codes: `0` ok, `1` policy violations, `2` config errors, `3` internal errors.
    - [ ] 0.2.2 Add pre-commit hooks to run `lint`/`check` before artifact generation.
    - Details:
        - Update `tools/docstring_builder/cli.py`:
            - Add a `lint` subcommand that delegates to `check` (fast path) and may skip DocFacts drift comparison when `--no-docfacts` is provided.
            - Standardise exit codes in `main()` to propagate handler status.
        - Pre-commit:
            - Ensure `.pre-commit-config.yaml` has the hook `docstring-builder (check)` before docs artifact generation (already present); add a second optional `docstring-builder (diff)` hook for PR convenience (already present as `docstring-builder-diff`).
    - AC:
        - Commits with doc violations are blocked; no full rebuild is required.

- [ ] 0.3 Minimal policy gates
    - [ ] 0.3.1 Add `[tool.kgfoundry.docstrings.policy]` with coverage ≥90% and params/returns parity.
    - [ ] 0.3.2 Wire policy evaluation into `docstring-builder lint`.
    - Details:
        - Rely on existing hooks in `.pre-commit-config.yaml`:
            - `pydoclint --style numpy src` (parameter/returns parity)
            - `pydocstyle src` (numpydoc convention)
            - `interrogate -i src --fail-under 90` (coverage)
        - Optional: Update `tools/docstring_builder/cli.py` `lint` to surface a short summary of violations if these tools are invoked via subprocess (defer if unnecessary).
    - AC:
        - Violations show symbol/file and cause; hook exits with code `1`.

- [ ] 0.4 Manifest + incremental rebuilds
    - [ ] 0.4.1 Write `docs/_build/docstrings_manifest.json` with input hashes, tool/plugin versions, and config fingerprint.
    - [ ] 0.4.2 Add `--changed-only` to process only touched files and dependents.
    - Details:
        - Implement manifest writer in `tools/docstring_builder/cli.py` after `_run()` completes:
            - Include: list of processed files, `config.config_hash`, CLI args (`since/module/force`), builder cache path and mtime, and counts of symbols/docfacts.
            - Write to `docs/_build/docstrings_manifest.json` (ensure parent exists).
        - Add `--changed-only` flag as sugar for computing a default `--since` (use `origin/main` or last merge-base); reuse `_changed_files_since()`.
        - Do not implement dependency graph yet; scope is “changed files only”.
    - AC:
        - Touching one file triggers <1s rebuild on warm cache for that scope.

- [ ] 0.5 Pre-commit consolidation (ruff/mypy/doc tools/pyrefly)
    - [ ] 0.5.1 Order hooks: ruff imports → ruff fix → ruff format → mypy strict → docformatter/pydoclint/interrogate → docstring-builder lint/check → make artifacts → navmap-check → pyrefly validate.
    - [ ] 0.5.2 Ensure pyrefly treats optional deps as extras and passes without them.
    - Details:
        - Update `.pre-commit-config.yaml`:
            - Move `docstring-builder (check)` above `docs: regenerate artifacts`.
            - Add a new hook `pyrefly-check` with `entry: uv run pyrefly check`, `pass_filenames: false`.
        - Confirm `pyrefly.toml` search paths include `src` and that optional imports are listed under `replace-imports-with-any` or extras.
    - AC:
        - `pre-commit run --all-files` surfaces all violations deterministically.

- [ ] 0.6 Golden tests for the builder
    - [ ] 0.6.1 Add 3–5 representative modules with golden renders.
    - [ ] 0.6.2 Implement `--update` to refresh goldens intentionally.
    - Details:
        - Create `tests/docs/test_docstring_golden.py` to call `python -m tools.docstring_builder.cli check --diff` and capture expected diffs for selected files under `src/`.
        - Store golden snapshots under `tests/docs/goldens/` (JSON or text) and compare with normalized whitespace.
        - Add `--update` support by writing current diff to golden when `UPDATE_GOLDENS=1` is set.
    - AC:
        - CI fails with readable diff on unintended doc output changes.

- [ ] 0.7 Minimal observability for failures
    - [ ] 0.7.1 Emit `docs/_build/observability_docstrings.json` with counts, timings, and first 20 errors.
    - [ ] 0.7.2 Print a console SUMMARY on failures (top errors, slowest stages).
    - Details:
        - In `tools/docstring_builder/cli.py`, collect simple metrics: total files considered, processed, skipped by cache, changed vs unchanged, and elapsed seconds per major step (harvest/apply/write).
        - On non-zero exit, write a JSON summary under `docs/_build/observability_docstrings.json` and print a short console summary (max 10 lines).
    - AC:
        - On failure, developers see where/why without inspecting logs deeply.

- [ ] 0.8 Doctor command
    - [ ] 0.8.1 Implement `docstring-builder doctor` for Python version, config alignment, stubs discoverable, optional deps status, and manifest writeability.
    - Details:
        - Add a `doctor` subcommand in `tools/docstring_builder/cli.py` that checks:
            - Python ≥ 3.13, `mypy.ini` includes `mypy_path = src:stubs`.
            - Presence of expected stubs (`stubs/griffe`, `stubs/libcst`, `stubs/mkdocs_gen_files`).
            - Import health for `griffe` and `libcst`; guidance if missing.
            - Write permissions for `docs/_build/` and `.cache/`.
            - Presence/order of `.pre-commit-config.yaml` hooks for docstring builder.
        - Exit `0` on pass, `2` on configuration issues; print human-readable suggestions.
    - AC:
        - `doctor` completes <1s with clear suggestions when misconfigurations exist.

## 1. Implementation
- [ ] 1.1 Build typed docstring shims
    - [ ] 1.1.1 Define Protocols in `src/sitecustomize.py` for docstring objects (`DocstringProto`, `DocstringMetaProto`, `DocstringAttrProto`, `DocstringYieldsProto`).
    - [ ] 1.1.2 Implement helper functions (`ensure_docstring_attrs`, `ensure_docstring_yields`, `ensure_docstring_size`) that perform guarded `setattr` calls and return status booleans.
    - [ ] 1.1.3 Replace direct monkey patches with the new helpers; log debug messages when attributes are newly created vs. already present.
    - AC:
        - Protocols compile under strict mypy.
        - Helpers are used in all shim locations and covered by tests.

- [ ] 1.2 Provide third-party stubs
    - [ ] 1.2.1 Create `stubs/griffe/__init__.pyi` etc., covering the symbols we import (`Module`, `Function`, `Object`, `GriffeLoader`).
    - [ ] 1.2.2 Add stubs for `libcst` (classes we interact with) and `mkdocs_gen_files` (functions used in generator scripts).
    - [ ] 1.2.3 Update `pyproject.toml` / `mypy.ini` to include the `stubs/` directory; remove redundant ignores.
    - AC:
        - `uv run mypy` finds no missing attributes/types for these libs.
        - Stubs are documented with a short “how to extend” guideline.

- [ ] 1.3 Annotate docstring builder core modules
    - [ ] 1.3.1 Convert data containers (`ParameterHarvest`, `SymbolHarvest`, etc.) to typed dataclasses, referencing stubbed types.
    - [ ] 1.3.2 Add explicit return types to `_resolve_object`, `_resolve_callable`, `_module_name`, and `_collect_symbols`.
    - [ ] 1.3.3 Write/refine module/function docstrings per NumPy style.
    - AC:
        - No `Any` leaks in public surfaces (checked via mypy strict).
        - Docstrings pass pydocstyle/pydoclint/interrogate gates.

- [ ] 1.4 Introduce plugin architecture
    - [ ] 1.4.1 Define Protocols/ABCs for `Harvester`, `Transformer`, `Formatter` with `run(context)` and optional `on_start`/`on_finish`.
    - [ ] 1.4.2 Implement entry-point discovery under `kgfoundry.docstrings.plugins`.
    - [ ] 1.4.3 Add CLI/config filters: `--only-plugin`, `--disable-plugin`.
    - [ ] 1.4.4 Ship a sample plugin `normalize_numpy_params`.
    - AC:
        - Plugins can be enabled/disabled without code changes.
        - A sample plugin runs in `generate` and affects output deterministically.

- [ ] 1.5 Define versioned Intermediate Representation (IR)
    - [ ] 1.5.1 Create IR dataclasses: `IRSymbol`, `IRParameter`, `IRDocstring`.
    - [ ] 1.5.2 Add `ir_version` and stable identifiers (`symbol_id`, `source_path`).
    - [ ] 1.5.3 Generate JSON Schema at `docs/_build/schema_docstrings.json`.
    - [ ] 1.5.4 Validate IR during pipeline execution.
    - AC:
        - Schema file exists and validates IR for a sample run.
        - Breaking schema changes bump `ir_version`.

- [ ] 1.6 Implement policy engine and quality gates
    - [ ] 1.6.1 Define config schema in `pyproject.toml` or YAML.
    - [ ] 1.6.2 Support rules: coverage %, missing params/returns, exceptions.
    - [ ] 1.6.3 Support actions: `error`, `warn`, `autofix` with precedence: CLI > env > config > defaults.
    - [ ] 1.6.4 Implement exceptions allowlist with `expires_on` and `justification`.
    - AC:
        - Failing a threshold exits with code `1` and a clear report.
        - Allowlist expiry is enforced in CI.

- [ ] 1.7 Add incremental executor and manifest
    - [ ] 1.7.1 Create `docs/_build/docstrings_manifest.json` with inputs (hashes), plugins, config fingerprint, outputs, and timings.
    - [ ] 1.7.2 Rebuild only changed files and dependents.
    - [ ] 1.7.3 Add `--since <rev>` and `--changed-only` flags.
    - AC:
        - Touching 1 file only rebuilds that file and dependents.
        - Changing config or plugin version invalidates cache.

- [ ] 1.8 Restructure CLI
    - [ ] 1.8.1 Implement subcommands: `generate`, `lint`, `fix`, `diff`, `check`, `schema`, `doctor`, `measure`.
    - [ ] 1.8.2 Document exit codes: `0` success, `1` policy/lint failures, `2` config errors, `3` internal errors.
    - [ ] 1.8.3 Add `--config` and document config precedence.
    - AC:
        - `generate` produces artifacts; `check` validates IR/schema.
        - `doctor` prints environment/stub status; `measure` reports timings.

- [ ] 1.9 Stub governance and drift checks
    - [ ] 1.9.1 Implement a drift checker that inspects runtime objects vs. local `.pyi`.
    - [ ] 1.9.2 Integrate drift check into CI and fail with actionable diff.
    - [ ] 1.9.3 (Optional) Package stubs as PEP-561 extras; document update steps.
    - AC:
        - CI reports missing/extra stub members with file/line hints.

- [ ] 1.10 Observability and developer experience
    - [ ] 1.10.1 Emit metrics/traces JSON: counts, timings, cache hits/misses, errors to `docs/_build/observability_docstrings.json`.
    - [ ] 1.10.2 Generate HTML drift previews for docfacts/navmap/schema per changed module.
    - [ ] 1.10.3 Provide editor tasks/snippets for `generate`, `fix`, `watch`.
    - AC:
        - Observability JSON exists after runs and includes timings per stage.
        - HTML diff opens locally and highlights changes by symbol.

- [ ] 1.11 Security and safety
    - [ ] 1.11.1 Remove/avoid unsafe evaluation of embedded code in docstrings.
    - [ ] 1.11.2 Normalise and validate input paths; reject traversal/symlinks.
    - AC:
        - Static checks confirm no `eval`/`exec` in parser paths.
        - Path validation unit tests cover traversal attempts.

- [ ] 1.12 Deprecation plan for `sitecustomize`
    - [ ] 1.12.1 Add `KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE` flag (default `1`).
    - [ ] 1.12.2 Emit `DeprecationWarning` when patching occurs; document removal timeline.
    - [ ] 1.12.3 Add CI kill switch to disable patches and ensure pipeline still succeeds.
    - AC:
        - Warning is visible in test logs; disabling works in CI.

- [ ] 1.13 Documentation updates
    - [ ] 1.13.1 Update docs for plugin authoring and stub extension workflow.
    - [ ] 1.13.2 Document config schema, policies, and CLI usage with examples.
    - [ ] 1.13.3 Add troubleshooting and "doctor" guide.
    - AC:
        - Docs build cleanly; examples run end-to-end.

## 2. Validation
- [ ] 2.1 Run `uv run mypy src/sitecustomize.py tools/docstring_builder` to ensure no type errors remain.
- [ ] 2.2 Execute `uv run ruff check src/sitecustomize.py tools/docstring_builder` to confirm lint compliance.
- [ ] 2.3 Add regression tests: import `sitecustomize` with and without `docstring_parser` installed (use `pytest.mark.importorskip`).
- [ ] 2.4 Run E2E CLI tests for each subcommand; assert exit codes and key outputs.
- [ ] 2.5 Validate IR against JSON Schema; include a negative test with deliberate mismatch.
- [ ] 2.6 Verify incremental executor by touching a single file and inspecting manifest deltas.
- [ ] 2.7 Run `make artifacts` to regenerate doc artifacts; verify drift previews and observability JSON.
- [ ] 2.8 Add CI jobs for stub drift checker and policy gates; ensure failures are clear and actionable.
