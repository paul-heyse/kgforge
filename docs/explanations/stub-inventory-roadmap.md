# Stub Inventory Roadmap

This note tracks the remaining stub files under `stubs/` and outlines the plan
for retiring them in favour of the real, typed modules wherever possible.

The audit below is based on the current `uv` environment (Python 3.13.9) and the
versions constrained in `pyproject.toml`. Packages that expose a `py.typed` file
are good candidates for removing our local `.pyi` shims once we validate that
their inline annotations cover the symbols we use.

## 1. Ready to Drop (runtime already ships typing)

| Package | Stub paths | Runtime typing status | Next steps |
| --- | --- | --- | --- |
| `msgspec>=0.19.0` | `stubs/msgspec/**` | `py.typed` present | Delete stubs and rely on upstream annotations; run mypy/pyrefly smoke afterwards. |
| `packaging>=24` | `stubs/packaging/version.pyi` | `py.typed` present | Remove stub and let typeshed cover the rest of the package. |
| `jinja2>=3.1.4` | `stubs/jinja2/__init__.pyi` | `py.typed` present | Drop stub; ensure `select_autoescape` import still resolves for docstring builder. |
| `prometheus-client>=0.23.1` | `stubs/prometheus_client/*.pyi` | `py.typed` present | Remove stubs and adjust any Protocol helpers in tests to target concrete classes. |
| `fastapi>=0.115.0` | `stubs/fastapi/*.pyi` | `py.typed` present | Delete stubs; re-run test suite to confirm `TestClient` usage still type-checks. |
| `starlette` (via FastAPI) | `stubs/starlette/**` | `py.typed` present | Drop stubs after FastAPI check (Starlette is bundled). |
| `duckdb>=1.4.1` | `stubs/duckdb/__init__.pyi` | `py.typed` present | Remove stub; confirm `duckdb.connect` annotations align with our wrappers. |
| `griffe` (unversioned) | `stubs/griffe/**` | `py.typed` present | Delete stubs and lean on upstream (griffe is fully typed). |
| `mkdocs-gen-files` | `stubs/mkdocs_gen_files/__init__.pyi` | `py.typed` present | Remove stub once doc build passes with upstream module. |
| `prefect>=3.4.25` | `stubs/prefect/logging/*.pyi` | `py.typed` present | Drop stubs; verify `prefect.logging.configuration` imports remain typed. |
| `docstring-parser` | `stubs/docstring_parser/**` | `py.typed` present | Delete stubs; docstring builder tests cover this path. |
| `libcst>=1.4.0` | `stubs/libcst/__init__.pyi` | `py.typed` present | Remove shim; LibCST already publishes complete typing metadata. |
| `msgspec`, `prometheus-client` (tool extra) | same as above | see above | Align tool extras after runtime clean-up. |

## 2. Needs Follow-up (runtime lacks py.typed)

| Package | Stub paths | Status | Suggested action |
| --- | --- | --- | --- |
| `jsonschema>=4.25.1` | `stubs/jsonschema/*.pyi` | no `py.typed` | Keep stubs for now; re-evaluate once upstream merges Pydantic-style typing or `types-jsonschema` arrives. |
| `pyarrow>=21.0.0` | `stubs/pyarrow/**` | no `py.typed` | Maintain stubs; consider switching to `types-pyarrow` if community stub matures. |
| `networkx` | `stubs/networkx/__init__.pyi` | no `py.typed` | Keep stub; upstream typing is still experimental. |
| `pyserini>=1.2.0` | `stubs/pyserini/**` | no `py.typed` | Preserve stubs (package is optional and untyped). |
| `autoapi`, `sphinx-autoapi` | `stubs/autoapi/**`, `stubs/sphinx_autoapi/*.pyi` | no `py.typed` | Keep; open upstream issue to request typing or explore replacing with slim custom Protocols. |
| `astroid>=4.0.0` | `stubs/astroid/__init__.pyi` | no `py.typed` | Retain stub until astroid ships types (planned but not released). |
| `faiss` (GPU optional) | `stubs/faiss/__init__.pyi`, `stubs/vectorstore_faiss/` | binary wheels only | Keep; required for optional FAISS ingestion path. |
| `libcuvs` | `stubs/libcuvs/__init__.pyi` | project-specific wrapper | Keep; compiled library lacks typing. |
| `pytestarch>=4.0.1` | `stubs/pytestarch/**` | no types | Keep; upstream typing backlog. |
| `importlinter` | `stubs/importlinter/**` | minimal runtime typing | Keep stub but consider contributing upstream hints. |
| `auto-generated tempfile wrapper` | `stubs/tempfile/__init__.pyi` | overrides stdlib | Keep for now (captures extra `delete_on_close` kw); revisit once callers are audited. |
| `opentelemetry-*` | `stubs/opentelemetry/**` | our curated API surface | Keep; upstream annotations are incomplete for the symbols we consume. |
| `prometheus_client/metrics_core.pyi` | complements runtime | Keep until we verify upstream metrics-core typing covers builder protocols. |

## 3. Local-only / Intentional Stubs

| Stub | Purpose | Notes |
| --- | --- | --- |
| `stubs/pytest/__init__.pyi` | Extended fixture / parametrize overloads | Required for strict test typing (suppresses Ruff UP047 globally). |
| `stubs/conftest.pyi` | Declares dynamic `pytest_plugins`, `HAS_GPU_STACK` | Keep to satisfy test namespace imports. |
| `stubs/docs/scripts/**` | Provides stubs for docs helper scripts | Keep while docs tooling lacks shipping types. |
| Empty directories (`search_api/`, `vectorstore_faiss/`) | Residual from previous clean-up | Remove once corresponding files are deleted to avoid confusion. |

## 4. Execution Plan

1. **Batch removal PR** for the “Ready to drop” set. Delete stubs, run full static gates, and backfill any missing Protocols (e.g., counter/histogram helpers) with small local helper protocols if needed.
2. **Open upstream issues** for key “Needs follow-up” packages (jsonschema, autoapi, astroid, networkx) referencing our use cases. Track outcomes in this document.
3. **Monitor dependency bumps**: add a lint/check that warns if a stub exists alongside a dependency that now exposes `py.typed`.
4. **Doc updates**: document the policy in `docs/contributing/typing.md` so future contributors evaluate new stubs against this inventory.

We can revisit this roadmap after the first clean-up batch to prune the “Ready to drop” entries and promote any upgraded packages out of the follow-up list.
