# `orchestration`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`orchestration.cli`** — Module for orchestration.cli → [open](./cli.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/cli.py#L1)
  - **`orchestration.cli.api`** — Run the FastAPI app → [open](./cli.py:97:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/cli.py#L97-L102)
  - **`orchestration.cli.e2e`** — Execute the skeleton Prefect flow and print completed stages → [open](./cli.py:105:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/cli.py#L105-L120)
  - **`orchestration.cli.index_bm25`** — Build a BM25 index from chunk fixtures (id, title, section, body) → [open](./cli.py:23:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/cli.py#L23-L65)
  - **`orchestration.cli.index_faiss`** — Train & build FAISS index from fixture dense vectors → [open](./cli.py:68:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/cli.py#L68-L94)
- **`orchestration.fixture_flow`** — Module for orchestration.fixture_flow → [open](./fixture_flow.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L1)
  - **`orchestration.fixture_flow.fixture_pipeline`** — Run the end-to-end fixture pipeline → [open](./fixture_flow.py:177:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L177-L194)
  - **`orchestration.fixture_flow.t_prepare_dirs`** — T prepare dirs → [open](./fixture_flow.py:22:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L22-L41)
  - **`orchestration.fixture_flow.t_register_in_duckdb`** — T register in duckdb → [open](./fixture_flow.py:122:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L122-L174)
  - **`orchestration.fixture_flow.t_write_fixture_chunks`** — T write fixture chunks → [open](./fixture_flow.py:44:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L44-L73)
  - **`orchestration.fixture_flow.t_write_fixture_dense`** — T write fixture dense → [open](./fixture_flow.py:76:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L76-L95)
  - **`orchestration.fixture_flow.t_write_fixture_splade`** — T write fixture splade → [open](./fixture_flow.py:98:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/fixture_flow.py#L98-L119)
- **`orchestration.flows`** — Module for orchestration.flows → [open](./flows.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/flows.py#L1)
  - **`orchestration.flows.e2e_flow`** — E2e flow → [open](./flows.py:30:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/flows.py#L30-L48)
  - **`orchestration.flows.t_echo`** — T echo → [open](./flows.py:13:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/orchestration/flows.py#L13-L27)
