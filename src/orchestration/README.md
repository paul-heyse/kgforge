# `orchestration`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`orchestration.NavMap`** — Describe NavMap → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/navmap_types.py#L32-L45)
- **`orchestration.cli`** — Cli utilities → [open](./cli.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/cli.py#L1)
  - **`orchestration.cli.NavMap`** — Describe NavMap → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/navmap_types.py#L32-L45)
  - **`orchestration.cli.api`** — Compute api → [open](./cli.py:156:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/cli.py#L156-L182)
  - **`orchestration.cli.e2e`** — Compute e2e → [open](./cli.py:186:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/cli.py#L186-L222)
  - **`orchestration.cli.index_bm25`** — Compute index bm25 → [open](./cli.py:35:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/cli.py#L35-L102)
  - **`orchestration.cli.index_faiss`** — Compute index faiss → [open](./cli.py:106:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/cli.py#L106-L152)
- **`orchestration.fixture_flow`** — Fixture Flow utilities → [open](./fixture_flow.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L1)
  - **`orchestration.fixture_flow.Doc`** — Describe Doc → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/models.py#L30-L45)
  - **`orchestration.fixture_flow.NavMap`** — Describe NavMap → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/navmap_types.py#L32-L45)
  - **`orchestration.fixture_flow.ParquetChunkWriter`** — Describe ParquetChunkWriter → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/parquet_io.py#L275-L391)
  - **`orchestration.fixture_flow.ParquetVectorWriter`** — Describe ParquetVectorWriter → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/parquet_io.py#L35-L271)
  - **`orchestration.fixture_flow.fixture_pipeline`** — Compute fixture pipeline → [open](./fixture_flow.py:268:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L268-L301)
  - **`orchestration.fixture_flow.t_prepare_dirs`** — Compute t prepare dirs → [open](./fixture_flow.py:46:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L46-L76)
  - **`orchestration.fixture_flow.t_register_in_duckdb`** — Compute t register in duckdb → [open](./fixture_flow.py:194:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L194-L264)
  - **`orchestration.fixture_flow.t_write_fixture_chunks`** — Compute t write fixture chunks → [open](./fixture_flow.py:80:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L80-L120)
  - **`orchestration.fixture_flow.t_write_fixture_dense`** — Compute t write fixture dense → [open](./fixture_flow.py:124:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L124-L154)
  - **`orchestration.fixture_flow.t_write_fixture_splade`** — Compute t write fixture splade → [open](./fixture_flow.py:158:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/fixture_flow.py#L158-L190)
- **`orchestration.flows`** — Flows utilities → [open](./flows.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/flows.py#L1)
  - **`orchestration.flows.NavMap`** — Describe NavMap → [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/kgfoundry_common/navmap_types.py#L32-L45)
  - **`orchestration.flows.e2e_flow`** — Compute e2e flow → [open](./flows.py:61:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/flows.py#L61-L100)
  - **`orchestration.flows.t_echo`** — Compute t echo → [open](./flows.py:28:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/326cba7089fce0d0bc5d078ad95af075ddc7117d/src/orchestration/flows.py#L28-L57)
