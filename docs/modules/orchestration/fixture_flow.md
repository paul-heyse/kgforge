# orchestration.fixture_flow

Typer-powered orchestration command suite covering indexing flows, API bootstrapping,
and end-to-end demonstrations. Each command maps to a generated OpenAPI operation
consumed by the MkDocs suite.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/fixture_flow.py)

## Hierarchy

- **Parent:** [orchestration](../orchestration.md)

## Sections

- **Public API**

## Contents

### orchestration.fixture_flow._fixture_pipeline_impl

::: orchestration.fixture_flow._fixture_pipeline_impl

### orchestration.fixture_flow._t_prepare_dirs_impl

::: orchestration.fixture_flow._t_prepare_dirs_impl

### orchestration.fixture_flow._t_register_in_duckdb_impl

::: orchestration.fixture_flow._t_register_in_duckdb_impl

### orchestration.fixture_flow._t_write_fixture_chunks_impl

::: orchestration.fixture_flow._t_write_fixture_chunks_impl

### orchestration.fixture_flow._t_write_fixture_dense_impl

::: orchestration.fixture_flow._t_write_fixture_dense_impl

### orchestration.fixture_flow._t_write_fixture_splade_impl

::: orchestration.fixture_flow._t_write_fixture_splade_impl

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.models.Doc`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.parquet_io.ChunkRow`, `kgfoundry_common.parquet_io.ParquetChunkWriter`, `kgfoundry_common.parquet_io.ParquetVectorWriter`, `pathlib.Path`, `prefect.flow`, `prefect.task`, `registry.helper.DuckDBRegistryHelper`, `typing.TYPE_CHECKING`

## Autorefs Examples

- [orchestration.fixture_flow._fixture_pipeline_impl][]
- [orchestration.fixture_flow._t_prepare_dirs_impl][]
- [orchestration.fixture_flow._t_register_in_duckdb_impl][]

## Neighborhood

```d2
direction: right
"orchestration.fixture_flow": "orchestration.fixture_flow" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/fixture_flow.py" }
"__future__.annotations": "__future__.annotations"
"orchestration.fixture_flow" -> "__future__.annotations"
"kgfoundry_common.models.Doc": "kgfoundry_common.models.Doc"
"orchestration.fixture_flow" -> "kgfoundry_common.models.Doc"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"orchestration.fixture_flow" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.parquet_io.ChunkRow": "kgfoundry_common.parquet_io.ChunkRow"
"orchestration.fixture_flow" -> "kgfoundry_common.parquet_io.ChunkRow"
"kgfoundry_common.parquet_io.ParquetChunkWriter": "kgfoundry_common.parquet_io.ParquetChunkWriter"
"orchestration.fixture_flow" -> "kgfoundry_common.parquet_io.ParquetChunkWriter"
"kgfoundry_common.parquet_io.ParquetVectorWriter": "kgfoundry_common.parquet_io.ParquetVectorWriter"
"orchestration.fixture_flow" -> "kgfoundry_common.parquet_io.ParquetVectorWriter"
"pathlib.Path": "pathlib.Path"
"orchestration.fixture_flow" -> "pathlib.Path"
"prefect.flow": "prefect.flow"
"orchestration.fixture_flow" -> "prefect.flow"
"prefect.task": "prefect.task"
"orchestration.fixture_flow" -> "prefect.task"
"registry.helper.DuckDBRegistryHelper": "registry.helper.DuckDBRegistryHelper"
"orchestration.fixture_flow" -> "registry.helper.DuckDBRegistryHelper"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"orchestration.fixture_flow" -> "typing.TYPE_CHECKING"
"orchestration": "orchestration" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/__init__.py" }
"orchestration" -> "orchestration.fixture_flow" { style: dashed }
```

