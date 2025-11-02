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
| `msgspec>=0.19.0` | `stubs/msgspec/**` | `py.typed` present | ✅ Removed on 2025-11-02 — repo now relies on upstream annotations. |
| `packaging>=24` | `stubs/packaging/version.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — covered by typeshed bundled typing. |
| `jinja2>=3.1.4` | `stubs/jinja2/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — docstring builder uses runtime typing successfully. |
| `prometheus-client>=0.23.1` | `stubs/prometheus_client/*.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — tests target concrete metrics classes. |
| `fastapi>=0.115.0` | `stubs/fastapi/*.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — runtime middleware helpers validate against upstream types. |
| `starlette` (via FastAPI) | `stubs/starlette/**` | `py.typed` present | ✅ Removed on 2025-11-02 alongside FastAPI stubs. |
| `duckdb>=1.4.1` | `stubs/duckdb/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — `registry.duckdb_helpers` now type-checks against upstream hints. |
| `griffe` (unversioned) | `stubs/griffe/**` | `py.typed` present | ✅ Removed on 2025-11-02 — documentation build passes with upstream typing. |
| `mkdocs-gen-files` | `stubs/mkdocs_gen_files/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — doc build validated without local shim. |
| `prefect>=3.4.25` | `stubs/prefect/logging/*.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — observability helpers import upstream module directly. |
| `docstring-parser` | `stubs/docstring_parser/**` | `py.typed` present | ✅ Removed on 2025-11-02 — docstring builder suite covers upstream typing. |
| `libcst>=1.4.0` | `stubs/libcst/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — codemods run against inline typing. |
| `msgspec`, `prometheus-client` (tool extra) | same as above | see above | ✅ Removed on 2025-11-02 — tooling extras aligned with runtime clean-up. |
| `opentelemetry` suite | `stubs/opentelemetry/**` | `py.typed` present | ✅ Removed on 2025-11-02 — tracing helpers now type-check against upstream SDK annotations. |
| `faiss>=1.12.0` | `stubs/faiss/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — ingestion pipeline relies on upstream binary wheel typing. |
| `importlinter>=2.0` | `stubs/importlinter/**` | `py.typed` present | ✅ Removed on 2025-11-02 — architecture checks import upstream types directly. |
| `pytest>=8.3` | `stubs/pytest/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — fixture overloads now covered by upstream typing. |
| `tempfile` shim | `stubs/tempfile/__init__.pyi` | stdlib typing sufficient | ✅ Removed on 2025-11-02 — reliance on extra keyword dropped in favour of upstream signature. |

We now rely on the community wheels `types-jsonschema` and `types-networkx` (added to the
tooling extras on 2025-11-02), so the local `jsonschema` and `networkx` stubs have been
retired in favour of the upstream typings. Likewise, `pyarrow-stubs` ships comprehensive
type information, allowing us to drop the local `pyarrow` shims.

## 2. Needs Follow-up (runtime lacks py.typed)

| Package | Stub paths | Status | Suggested action |
| --- | --- | --- | --- |
| `pyserini>=1.2.0` | `stubs/pyserini/**` | no `py.typed` | Preserve stubs (package is optional and untyped); align Protocols with runtime surface. |
| `faiss` (GPU optional) | `stubs/faiss/__init__.pyi` | binary wheels only | Keep; required for optional FAISS ingestion path and cuVS integration. |
| `libcuvs` | `stubs/libcuvs/__init__.pyi` | project-specific wrapper | Keep; compiled library lacks typing. |
| `autoapi`, `sphinx-autoapi` | `stubs/autoapi/**`, `stubs/sphinx_autoapi/*.pyi` | no `py.typed` | Keep; local stubs trimmed to the parser/setup surface we exercise while we track upstream typing. |
| `astroid>=4.0.0` | `stubs/astroid/**` | no `py.typed` | Retain slim manager/builder stubs until astroid ships types (planned but not released). |
| `pytestarch>=4.0.1` | `stubs/pytestarch/**` | no types | Keep; upstream typing backlog. |
| `importlinter` | `stubs/importlinter/**` | minimal runtime typing | Keep stub but consider contributing upstream hints. |
| `auto-generated tempfile wrapper` | `stubs/tempfile/__init__.pyi` | overrides stdlib | Keep for now (captures extra `delete_on_close` kw); revisit once callers are audited. |
| `opentelemetry-*` | `stubs/opentelemetry/**` | our curated API surface | Keep; upstream annotations are incomplete for the symbols we consume. |

## 3. Local-only / Intentional Stubs

| Stub | Purpose | Notes |
| --- | --- | --- |
| `stubs/conftest.pyi` | Declares dynamic `pytest_plugins`, `HAS_GPU_STACK` | Keep to satisfy test namespace imports. |
| `stubs/docs/scripts/**` | Provides stubs for docs helper scripts | Keep while docs tooling lacks shipping types. |
| Empty directories (`search_api/`) | Residual from previous clean-up | ✅ Removed 2025-11-02 alongside stub retirement. |

## 4. Execution Plan

1. **Batch removal PR** for the “Ready to drop” set. Delete stubs, run full static gates, and backfill any missing Protocols (e.g., counter/histogram helpers) with small local helper protocols if needed.
2. **Open upstream issues** for key “Needs follow-up” packages (autoapi, astroid, pyserini, etc.) referencing our use cases. Track outcomes in this document.
3. **Monitor dependency bumps**: add a lint/check that warns if a stub exists alongside a dependency that now exposes `py.typed`.
4. **Doc updates**: document the policy in `docs/contributing/typing.md` so future contributors evaluate new stubs against this inventory.

We can revisit this roadmap after the first clean-up batch to prune the “Ready to drop” entries and promote any upgraded packages out of the follow-up list.
