# orchestration

Typer-powered orchestration command suite covering indexing flows, API bootstrapping,
and end-to-end demonstrations. Each command maps to a generated OpenAPI operation
consumed by the MkDocs suite.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/orchestration/__init__.py)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"orchestration": "orchestration" { link: "./orchestration.md" }
"__future__.annotations": "__future__.annotations"
"orchestration" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"orchestration" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"orchestration" -> "kgfoundry_common.navmap_types.NavMap"
"orchestration_code": "orchestration code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/orchestration/__init__.py" }
"orchestration" -> "orchestration_code" { style: dashed }
```

