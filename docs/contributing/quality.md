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

## 2. Normalise docstrings

The docstring builder ensures every public API stays aligned with our NumPy
style requirements and updates the DocFacts sidecar JSON used by documentation
bots. Run it in update mode whenever you touch Python interfaces:

```bash
uv run python -m tools.docstring_builder.cli update --diff
```

The command rewrites docstrings in-place, refreshes the builder cache, and
regenerates `docs/_build/docfacts.json`. Re-run it after resolving merge
conflicts so the ownership markers stay consistent.

## 3. Validate locally before pushing

Use the chained lint target to replicate the documentation checks that run in
CI:

```bash
make lint-docs            # pydoclint + docstring-builder diff + mypy strict
RUN_DOCS_TESTS=1 make lint-docs  # (optional) exercises tests/docs via pytest
```

`make lint-docs` fails fast on docstring drift, DocFacts mismatches, or typing
regressions. Enable `RUN_DOCS_TESTS=1` to execute the `tests/docs` suite when
you need additional safety.

To keep the feedback loop tight, enable the optional pre-commit hook named
`docstring-builder (diff, optional)`. It runs the same docstring diff in
lightweight mode and honours the `SKIP_DOC_BUILDER_DIFF=1` and
`DOC_BUILDER_SINCE=<rev>` environment flags.

## 4. Rely on CI for final verification

The primary CI workflow invokes `make lint-docs` after the standard linting
passes. The job reruns the docstring builder in check mode, confirms
`docs/_build/docfacts.json` matches the regenerated metadata, and surfaces
failures as actionable annotations. A green build means both the human-facing
and machine-readable documentation artefacts are up to date.
