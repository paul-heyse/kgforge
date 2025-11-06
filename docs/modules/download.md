# download

Downloader command suite that sources external corpora (currently OpenAlex) using the shared
CLI tooling contracts. Emits structured envelopes and metadata so downstream tooling (OpenAPI,
diagrams, documentation) remains in sync without bespoke glue.

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"download": "download" { link: "download.md" }
"__future__.annotations": "__future__.annotations"
"download" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"download" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"download" -> "kgfoundry_common.navmap_types.NavMap"
```

