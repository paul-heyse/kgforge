# Quality automation runbook

Maintainers can use this checklist to keep generated typing overlays, docstrings,
and docfacts in sync before raising a pull request. The commands below assume
`uv` is installed and available on your `PATH`.

## 1. Refresh typing stubs and overrides

We vendor third-party typings under `stubs/` to keep mypy in strict mode even
when upstream packages do not ship annotations. Regenerate the overlays before
upgrading dependencies or when `mypy` starts warning about missing attributes:

```bash
# Generate fresh stubs into a scratch directory
uv run python -m mypy.stubgen \
  --package networkx --package pytest \
  --output stubs/_generated

# Review and copy only the modules you need into stubs/
rsync -a stubs/_generated/ stubs/
rm -rf stubs/_generated
```

The `rsync` step mirrors the generated files over the maintained overlays so
that custom edits (for example, `typing.Any` fallbacks) remain reviewable in the
diff. Commit the resulting changes under `stubs/` alongside any manual fixes
required to satisfy `mypy --strict`.

## 2. Regenerate documentation artefacts

Run the consolidated target whenever you touch documentation inputs:

```bash
make artifacts
```

The helper script invokes the docstring builder (with
`--ignore-missing` to sidestep `docs._build` imports), navmap rebuild,
test-map generation, observability scan, and schema exporter. Each phase prints
a status line such as `[navmap] regenerated site/_build/navmap/navmap.json` so
you immediately know which subsystem needs attention if the command exits with
an error.

| When you change…                                   | Why `make artifacts` matters                         | Notes |
| -------------------------------------------------- | ---------------------------------------------------- | ----- |
| Public API signatures or docstrings                | Updates rendered docstrings and `docfacts.json`      | Uses `tools.docstring_builder.cli generate --all`
| Navigation metadata (`tools/navmap/**`)             | Rebuilds `site/_build/navmap/navmap.json`            | Keeps navmaps/test maps in sync |
| Observability policies or coverage annotations     | Refreshes observability reports under `docs/_build`  | Fails fast if new lint errors appear |
| Pydantic models / schema exports                   | Writes JSON schema artefacts and drift reports       | Inspect `docs/_build/schema_drift.json` when it changes |

### Artefact FAQ

- **DocFacts still drift after running `make artifacts`?** Ensure you committed
  the regenerated docstrings first. The command writes deterministic JSON; rerun
  it after resolving merge conflicts to realign ownership markers.
- **`ModuleNotFoundError: docs._build…` during manual runs?** Call
  `python -m tools.docstring_builder.cli generate --all --ignore-missing` or just
  `make artifacts`. The compatibility flag suppresses transient imports from
  generated directories.
- **`[schemas] drift detected` keeps appearing?** Open
  `docs/_build/schema_drift.json` to review the diff and commit the updates when
  the changes look correct. The exporter leaves the drift file behind for
  troubleshooting.

## 3. Validate locally before pushing

Use the chained lint target to replicate the documentation checks that run in
CI after you regenerate artefacts:

```bash
make lint-docs            # pydoclint + docstring-builder diff + mypy strict
RUN_DOCS_TESTS=1 make lint-docs  # (optional) exercises tests/docs via pytest
```

`make lint-docs` fails fast on docstring drift, DocFacts mismatches, or typing
regressions. Enable `RUN_DOCS_TESTS=1` to execute the `tests/docs` suite when
you need additional safety.

To keep the feedback loop tight, enable the optional pre-commit hook named
`docstring-builder (diff, optional)`. It runs the docstring diff in lightweight
mode and honours the `SKIP_DOC_BUILDER_DIFF=1` and
`DOC_BUILDER_SINCE=<rev>` environment flags.

## GPU-dependent tests

- Guard GPU suites with the shared header emitted by
  `tools/lint/add_gpu_header.py`. The pre-commit hook `Ensure GPU tests are gated`
  fails when a test imports `torch`, `vllm`, `faiss`, or other GPU extras without
  that header.
- Import individual accelerators via
  `require_modules(["torch"], minversions={"torch": "2.9"})` to get consistent
  skip messaging. Use `@pytest.mark.requires("torch>=2.9", "vllm")` when you want
  declarative gating; the plugin auto-applies the `gpu` marker and skips on
  CPU-only agents.
- Semantic index tooling falls back to an in-memory FAISS shim when the GPU
  stack is missing. Set `KGF_FAISS_MODULE` to choose a specific import target or
  `KGF_DISABLE_FAISS_FALLBACK=1` to enforce the real dependency in CI.
- Local helpers: `make test-gpu` runs only GPU suites,
  `make test-cpu` skips them, and `make lint-gpu-gates` reuses the
  enforcement check.
- Cloud runners should export `PYTEST_ADDOPTS="-m 'not gpu'"` so the default
  workflow omits GPU tests unless explicitly requested.

## 4. Docstring builder CLI reference

The CLI now exposes explicit subcommands so you can pick the right behaviour
without juggling legacy flags:

- `generate` — synchronise managed docstrings, update DocFacts, and refresh the
  manifest.
- `fix` — force a regeneration for files touched since the last merge-base via
  `--changed-only` (ideal for CI smoke checks).
- `diff` — report drift without writing changes; pairs nicely with
  `--changed-only` and the optional pre-commit hook.
- `check` — validate docstrings and DocFacts without applying edits.
- `lint` — alias for `check` that skips DocFacts comparisons when you only need
  policy enforcement.
- `measure` — run the pipeline and emit metrics even when everything is clean.
- `schema` — write the JSON schema describing the docstring IR.
- `doctor` — verify the environment, stub packages, and pre-commit ordering; add
  `--stubs` to surface drift in vendored typings.

Configuration precedence is now explicit: `--config` overrides
`KGF_DOCSTRINGS_CONFIG`, which in turn falls back to `docstring_builder.toml`
discovered relative to the repository root. The `doctor` command prints the
active configuration source to simplify debugging.

Every run writes metrics to `docs/_build/observability_docstrings.json` and a
manifest to `docs/_build/docstrings_manifest.json`. When drift is detected the
CLI also produces HTML previews under `docs/_build/drift/` (for example,
`docfacts.html`, `navmap.html`, and `schema.html`). Link targets are listed in
both the manifest and observability payloads so you can jump straight to the
rendered preview.

VS Code users can trigger common workflows from `.vscode/tasks.json`:

- **Docstrings: Generate** runs the full regeneration with
  `--ignore-missing`.
- **Docstrings: Fix changed files** performs a focused regeneration scoped to
  the current diff.
- **Docstrings: Watch** relies on `uvx watchfiles` to rerun
  `docstring-builder check --changed-only --diff` when `src/` or `tools/`
  modules change.

## 5. Extending the builder

- **Plugin authoring.** New transformers or formatters live in
  `tools/docstring_builder/plugins/`. Register them via the
  `kgfoundry.docstrings.plugins` entry point group and implement the typed
  `Harvester`, `Transformer`, or `Formatter` protocols. The CLI honours
  `--only-plugin` and `--disable-plugin` so you can test modules in isolation.
- **Stub maintenance.** Keep vendored typings current by running
  `python -m tools.docstring_builder.cli doctor --stubs`. The command fails with
  actionable output when runtime modules diverge from `stubs/`.
- **Troubleshooting.** `doctor` checks Python version, write permissions for
  `.cache/` and `docs/_build/`, pre-commit ordering, and optional dependency
  imports. Pair it with the HTML drift previews for quick triage.

## 6. Rely on CI for final verification

The primary CI workflow now includes a **Docs Artifacts** job that calls
`make artifacts` and asserts the working tree stays clean. The existing
documentation job still runs `make lint-docs`, rechecks DocFacts, and surfaces
failures as actionable annotations. A green build means both the human-facing
and machine-readable documentation artefacts are up to date.
