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
| Public API signatures or docstrings                | Updates rendered docstrings and `docfacts.json`      | Uses `tools.docstring_builder.cli update --all`
| Navigation metadata (`tools/navmap/**`)             | Rebuilds `site/_build/navmap/navmap.json`            | Keeps navmaps/test maps in sync |
| Observability policies or coverage annotations     | Refreshes observability reports under `docs/_build`  | Fails fast if new lint errors appear |
| Pydantic models / schema exports                   | Writes JSON schema artefacts and drift reports       | Inspect `docs/_build/schema_drift.json` when it changes |

### Artefact FAQ

- **DocFacts still drift after running `make artifacts`?** Ensure you committed
  the regenerated docstrings first. The command writes deterministic JSON; rerun
  it after resolving merge conflicts to realign ownership markers.
- **`ModuleNotFoundError: docs._build…` during manual runs?** Call
  `python -m tools.docstring_builder.cli update --all --ignore-missing` or just
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

## 4. Rely on CI for final verification

The primary CI workflow now includes a **Docs Artifacts** job that calls
`make artifacts` and asserts the working tree stays clean. The existing
documentation job still runs `make lint-docs`, rechecks DocFacts, and surfaces
failures as actionable annotations. A green build means both the human-facing
and machine-readable documentation artefacts are up to date.
