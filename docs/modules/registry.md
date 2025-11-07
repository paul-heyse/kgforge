# registry

DuckDB-backed registry APIs and helpers

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/__init__.py)

## Hierarchy

- **Children:** [registry.api](registry/api.md), [registry.duckdb_helpers](registry/duckdb_helpers.md), [registry.duckdb_registry](registry/duckdb_registry.md), [registry.helper](registry/helper.md), [registry.migrate](registry/migrate.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"registry": "registry" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/__init__.py" }
"__future__.annotations": "__future__.annotations"
"registry" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"registry" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"registry" -> "kgfoundry_common.navmap_types.NavMap"
"registry.api": "registry.api" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/api.py" }
"registry" -> "registry.api" { style: dashed }
"registry.duckdb_helpers": "registry.duckdb_helpers" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/duckdb_helpers.py" }
"registry" -> "registry.duckdb_helpers" { style: dashed }
"registry.duckdb_registry": "registry.duckdb_registry" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/duckdb_registry.py" }
"registry" -> "registry.duckdb_registry" { style: dashed }
"registry.helper": "registry.helper" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/helper.py" }
"registry" -> "registry.helper" { style: dashed }
"registry.migrate": "registry.migrate" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/registry/migrate.py" }
"registry" -> "registry.migrate" { style: dashed }
```

