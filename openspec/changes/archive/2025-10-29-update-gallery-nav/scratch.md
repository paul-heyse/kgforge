# Warning inventory

## Sphinx build (Sphinx-Gallery + AutoAPI)
- ✅ No remaining `ref.ref` gallery warnings after refreshing metadata and index docnames.
- `docs/gallery/00_quickstart.rst`: `py:mod` target `kgfoundry` still unresolved (pre-existing).
- `docs/autoapi/src/vectorstore_faiss/gpu/index.rst`: duplicate `NDArray[np.float32]` description vs. `search_api/faiss_adapter` persists.
- `docs/reference/schemas/index.md`: toctree glob `*.json` continues to miss generated schema files.
- Auto-generated API pages emit numerous numpydoc `ES01` and `py:class` reference warnings (unchanged baseline).
- AutoAPI sections remain outside any toctree, triggering `toc.not_included` notices.

## MkDocs build
- `not_in_nav` now accepted as a literal block; `mkdocs build` completes without "not included in nav" spam.
