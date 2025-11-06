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
| `griffe` (unversioned) | `stubs/griffe/**` | `py.typed` present | ✅ Removed on 2025-11-06 — documentation build passes with upstream typing. |
| `mkdocs-gen-files` | `stubs/mkdocs_gen_files/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — doc build validated without local shim. |
| `prefect>=3.4.25` | `stubs/prefect/logging/*.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — observability helpers import upstream module directly. |
| `docstring-parser` | `stubs/docstring_parser/**` | `py.typed` present | ✅ Removed on 2025-11-02 — docstring builder suite covers upstream typing. |
| `libcst>=1.4.0` | `stubs/libcst/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — codemods run against inline typing. |
| `msgspec`, `prometheus-client` (tool extra) | same as above | see above | ✅ Removed on 2025-11-02 — tooling extras aligned with runtime clean-up. |
| `opentelemetry` suite | `stubs/opentelemetry/**` | `py.typed` present | ✅ Removed on 2025-11-03 — replaced with `kgfoundry_common.opentelemetry_types` Protocols and lazy loaders. |
| `importlinter>=2.0` | `stubs/importlinter/**` | `py.typed` present | ✅ Removed on 2025-11-02 — architecture checks import upstream types directly. |
| `pytest>=8.3` | `stubs/pytest/__init__.pyi` | `py.typed` present | ✅ Removed on 2025-11-02 — fixture overloads now covered by upstream typing. |
| `tempfile` shim | `stubs/tempfile/__init__.pyi` | stdlib typing sufficient | ✅ Removed on 2025-11-02 — reliance on extra keyword dropped in favour of upstream signature. |

Faiss-related coverage:

- `faiss>=1.12.0` — Stub removed on 2025-11-03. Runtime bindings remain untyped, so `search_api.types.wrap_faiss_module` now adapts the module to the [`FaissModuleProtocol`](../../src/search_api/types.py) used across the vector store. GPU helpers continue to flow through the typed helper layer.
- `libcuvs` — Stub removed on 2025-11-03. The optional import in `search_api.faiss_adapter` now casts the loaded callable to `Callable[..., None]`, eliminating the placeholder stub while preserving type safety.
- `opentelemetry` — Suite of stubs removed on 2025-11-03. Optional tracing helpers now rely on `kgfoundry_common.opentelemetry_types`, which exposes Protocols and safe loaders aligned with the runtime SDK/APIs.
- `autoapi` / `sphinx-autoapi` — Stubs removed on 2025-11-03. Sphinx now loads the runtime parser through `docs._types.autoapi_parser.coerce_parser_class`, and optional dependency wiring in `docs._types.sphinx_optional` resolves the Parser type directly from `autoapi._parser`.
- `astroid` — Stubs removed on 2025-11-03. The docs build now uses `docs._types.astroid_facade` to coerce runtime manager/builder classes into typed facades, so Sphinx integration is stub-free.
- `pytestarch>=4.0.1` — Stubs removed on 2025-11-03. Architecture enforcement now depends on `tools.types_facade` which loads the runtime `pytestarch` helpers with typed protocols.

We now rely on the community wheels `types-jsonschema` and `types-networkx` (added to the
tooling extras on 2025-11-02), so the local `jsonschema` and `networkx` stubs have been
retired in favour of the upstream typings. Likewise, `pyarrow-stubs` ships comprehensive
type information, allowing us to drop the local `pyarrow` shims.

## 2. Needs Follow-up (runtime lacks py.typed)

None — the final follow-up items (Pyserini and the bespoke tempfile shim) were migrated on
2025-11-03. Lucene adapters in `embeddings_sparse` now cast the runtime modules to local
Protocols, and filesystem helpers rely on the standard library `tempfile` annotations.

## 3. Local-only / Intentional Stubs

None — as of 2025-11-03 all remaining shim directories have been removed. The `stubs/`
tree now contains only documentation (`README.md`).

## 4. Execution Plan

1. **Batch removal PR** for the “Ready to drop” set. Delete stubs, run full static gates, and backfill any missing Protocols (e.g., counter/histogram helpers) with small local helper protocols if needed.
2. **Open upstream issues** for key “Needs follow-up” packages (pyserini, pytestarch, etc.) referencing our use cases. Track outcomes in this document.
3. **Monitor dependency bumps**: add a lint/check that warns if a stub exists alongside a dependency that now exposes `py.typed`.
4. **Doc updates**: document the policy in `docs/contributing/typing.md` so future contributors evaluate new stubs against this inventory.

We can revisit this roadmap after the first clean-up batch to prune the “Ready to drop” entries and promote any upgraded packages out of the follow-up list.
