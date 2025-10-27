# `orchestration`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`orchestration.cli`** — Module for orchestration.cli → [open](./cli.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/cli.py#L1)
  - **`orchestration.cli.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`orchestration.cli.api`** — Run the FastAPI app → [open](./cli.py:119:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/cli.py#L119-L124)
  - **`orchestration.cli.e2e`** — Execute the skeleton Prefect flow and print completed stages → [open](./cli.py:128:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/cli.py#L128-L143)
  - **`orchestration.cli.index_bm25`** — Build a BM25 index from chunk fixtures (id, title, section, body) → [open](./cli.py:43:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/cli.py#L43-L85)
  - **`orchestration.cli.index_faiss`** — Train & build FAISS index from fixture dense vectors → [open](./cli.py:89:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/cli.py#L89-L115)
- **`orchestration.fixture_flow`** — Module for orchestration.fixture_flow → [open](./fixture_flow.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L1)
  - **`orchestration.fixture_flow.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`orchestration.fixture_flow.fixture_pipeline`** — Run the full fixture pipeline and register the generated artefacts → [open](./fixture_flow.py:173:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L173-L182)
  - **`orchestration.fixture_flow.t_prepare_dirs`** — Create the directory layout consumed by subsequent fixture tasks → [open](./fixture_flow.py:56:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L56-L75)
  - **`orchestration.fixture_flow.t_register_in_duckdb`** — Insert generated fixture runs and datasets into the DuckDB registry → [open](./fixture_flow.py:127:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L127-L169)
  - **`orchestration.fixture_flow.t_write_fixture_chunks`** — Write a minimal chunk parquet dataset for fixture usage → [open](./fixture_flow.py:79:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L79-L97)
  - **`orchestration.fixture_flow.t_write_fixture_dense`** — Write dense embedding vectors for the fixture chunk → [open](./fixture_flow.py:101:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L101-L109)
  - **`orchestration.fixture_flow.t_write_fixture_splade`** — Write SPLADE-style sparse vectors for the fixture chunk → [open](./fixture_flow.py:113:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/fixture_flow.py#L113-L123)
- **`orchestration.flows`** — Module for orchestration.flows → [open](./flows.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/flows.py#L1)
  - **`orchestration.flows.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`orchestration.flows.e2e_flow`** — Run the high-level kgfoundry demo flow and return step names → [open](./flows.py:41:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/flows.py#L41-L59)
  - **`orchestration.flows.t_echo`** — Return ``msg`` unmodified; handy for flow scaffolding and tests → [open](./flows.py:34:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/orchestration/flows.py#L34-L37)
