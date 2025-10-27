# `orchestration`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`orchestration.cli`** — Module for orchestration.cli → [open](./cli.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/cli.py#L1)
  - **`orchestration.cli.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`orchestration.cli.api`** — Run the FastAPI app → [open](./cli.py:119:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/cli.py#L119-L124)
  - **`orchestration.cli.e2e`** — Execute the skeleton Prefect flow and print completed stages → [open](./cli.py:128:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/cli.py#L128-L143)
  - **`orchestration.cli.index_bm25`** — Build a BM25 index from chunk fixtures (id, title, section, body) → [open](./cli.py:43:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/cli.py#L43-L85)
  - **`orchestration.cli.index_faiss`** — Train & build FAISS index from fixture dense vectors → [open](./cli.py:89:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/cli.py#L89-L115)
- **`orchestration.fixture_flow`** — Module for orchestration.fixture_flow → [open](./fixture_flow.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L1)
  - **`orchestration.fixture_flow.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`orchestration.fixture_flow.fixture_pipeline`** — Run the end-to-end fixture pipeline → [open](./fixture_flow.py:215:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L215-L232)
  - **`orchestration.fixture_flow.t_prepare_dirs`** — T prepare dirs → [open](./fixture_flow.py:55:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L55-L74)
  - **`orchestration.fixture_flow.t_register_in_duckdb`** — T register in duckdb → [open](./fixture_flow.py:159:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L159-L211)
  - **`orchestration.fixture_flow.t_write_fixture_chunks`** — T write fixture chunks → [open](./fixture_flow.py:78:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L78-L107)
  - **`orchestration.fixture_flow.t_write_fixture_dense`** — T write fixture dense → [open](./fixture_flow.py:111:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L111-L130)
  - **`orchestration.fixture_flow.t_write_fixture_splade`** — T write fixture splade → [open](./fixture_flow.py:134:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/fixture_flow.py#L134-L155)
- **`orchestration.flows`** — Module for orchestration.flows → [open](./flows.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/flows.py#L1)
  - **`orchestration.flows.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`orchestration.flows.e2e_flow`** — E2e flow → [open](./flows.py:51:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/flows.py#L51-L69)
  - **`orchestration.flows.t_echo`** — T echo → [open](./flows.py:33:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/orchestration/flows.py#L33-L47)
