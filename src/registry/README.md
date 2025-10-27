# `registry`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`registry.api`** — Module for registry.api → [open](./api.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/api.py#L1)
  - **`registry.api.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`registry.api.Registry`** — Registry protocol describing persistence operations → [open](./api.py:34:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/api.py#L34-L77)
- **`registry.duckdb_registry`** — Module for registry.duckdb_registry → [open](./duckdb_registry.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/duckdb_registry.py#L1)
  - **`registry.duckdb_registry.DuckDBRegistry`** — Persist pipeline artefacts and events inside a DuckDB catalog → [open](./duckdb_registry.py:37:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/duckdb_registry.py#L37-L166)
  - **`registry.duckdb_registry.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
- **`registry.helper`** — Module for registry.helper → [open](./helper.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/helper.py#L1)
  - **`registry.helper.DuckDBRegistryHelper`** — Open short-lived DuckDB connections for registry operations → [open](./helper.py:37:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/helper.py#L37-L177)
  - **`registry.helper.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
- **`registry.migrate`** — Module for registry.migrate → [open](./migrate.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/migrate.py#L1)
  - **`registry.migrate.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`registry.migrate.apply`** — Return apply → [open](./migrate.py:36:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/migrate.py#L36-L49)
  - **`registry.migrate.main`** — Run the CLI entry point for migration commands → [open](./migrate.py:53:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/registry/migrate.py#L53-L62)
