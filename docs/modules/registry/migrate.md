# registry.migrate

Migration helpers for DuckDB registry schemas

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/migrate.py)

## Hierarchy

- **Parent:** [registry](../registry.md)

## Sections

- **Public API**

## Contents

### registry.migrate.apply

::: registry.migrate.apply

### registry.migrate.main

::: registry.migrate.main

## Relationships

**Imports:** `__future__.annotations`, `argparse`, `contextlib.closing`, `kgfoundry_common.errors.RegistryError`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `pathlib`, [registry.duckdb_helpers](duckdb_helpers.md), `registry.duckdb_helpers.DuckDBQueryOptions`, `typing.cast`

## Autorefs Examples

- [registry.migrate.apply][]
- [registry.migrate.main][]

## Neighborhood

```d2
direction: right
"registry.migrate": "registry.migrate" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/migrate.py" }
"__future__.annotations": "__future__.annotations"
"registry.migrate" -> "__future__.annotations"
"argparse": "argparse"
"registry.migrate" -> "argparse"
"contextlib.closing": "contextlib.closing"
"registry.migrate" -> "contextlib.closing"
"kgfoundry_common.errors.RegistryError": "kgfoundry_common.errors.RegistryError"
"registry.migrate" -> "kgfoundry_common.errors.RegistryError"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"registry.migrate" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"pathlib": "pathlib"
"registry.migrate" -> "pathlib"
"registry.duckdb_helpers": "registry.duckdb_helpers" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/duckdb_helpers.py" }
"registry.migrate" -> "registry.duckdb_helpers"
"registry.duckdb_helpers.DuckDBQueryOptions": "registry.duckdb_helpers.DuckDBQueryOptions"
"registry.migrate" -> "registry.duckdb_helpers.DuckDBQueryOptions"
"typing.cast": "typing.cast"
"registry.migrate" -> "typing.cast"
"registry": "registry" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/__init__.py" }
"registry" -> "registry.migrate" { style: dashed }
```

